"""Evaluation runner for FrankaKitchen-v1 (gymnasium-robotics, low-dim)."""

import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 - registers envs
import numpy as np
import torch
import tqdm
import wandb
from termcolor import cprint

import flow_policy_3d.common.logger_util as logger_util
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.env_runner.base_runner import BaseRunner
from flow_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from flow_policy_3d.policy.base_policy import BasePolicy


gym.register_envs(gymnasium_robotics)


def _sorted_goal_keys(goal_dict) -> List[str]:
    return sorted(goal_dict.keys())


def _flatten_goal_step(goal_dict, keys: Sequence[str]) -> np.ndarray:
    parts = [np.asarray(goal_dict[k], dtype=np.float32).reshape(-1) for k in keys]
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _obs_to_agent_pos(obs_dict, goal_keys: Sequence[str]) -> np.ndarray:
    """Convert a single env observation dict (or frame-stacked dict) to the
    (To, D) agent_pos tensor matching KitchenDataset."""
    observation = np.asarray(obs_dict['observation'], dtype=np.float32)
    desired_goal = obs_dict['desired_goal']

    if observation.ndim == 1:
        goal_flat = _flatten_goal_step(desired_goal, goal_keys)
        return np.concatenate([observation, goal_flat], axis=-1).astype(np.float32)

    # frame-stacked: observation shape (To, D), desired_goal[k] shape (To, d_k)
    To = observation.shape[0]
    goals = []
    for k in goal_keys:
        v = np.asarray(desired_goal[k], dtype=np.float32)
        if v.ndim == 1:
            v = np.broadcast_to(v[None, :], (To, v.shape[0])).copy()
        v = v.reshape(To, -1)
        goals.append(v)
    goal_flat = np.concatenate(goals, axis=-1) if goals else np.zeros((To, 0), np.float32)
    return np.concatenate([observation, goal_flat], axis=-1).astype(np.float32)


class KitchenRunner(BaseRunner):
    """FrankaKitchen-v1 evaluation runner.

    Success metric: fraction of `tasks_to_complete` that were completed during
    the episode (averaged over eval_episodes).
    """

    def __init__(self,
                 output_dir,
                 eval_episodes: int = 20,
                 max_steps: int = 280,
                 n_obs_steps: int = 2,
                 n_action_steps: int = 4,
                 fps: int = 10,
                 crf: int = 22,
                 tqdm_interval_sec: float = 5.0,
                 task_name: str = 'kitchen_complete',
                 tasks_to_complete: Sequence[str] = (
                     'microwave', 'kettle', 'light switch', 'slide cabinet'),
                 render_mode: str = 'rgb_array',
                 record_video: bool = True,
                 max_video_episodes: int = 2,
                 save_videos_dir: Optional[str] = None,
                 wandb_log: bool = True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.tasks_to_complete = list(tasks_to_complete)
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.record_video = record_video
        self.max_video_episodes = max_video_episodes
        self.save_videos_dir = save_videos_dir
        self.wandb_log = wandb_log
        if self.save_videos_dir:
            Path(self.save_videos_dir).mkdir(parents=True, exist_ok=True)

        def env_fn():
            base_env = gym.make(
                'FrankaKitchen-v1',
                tasks_to_complete=list(tasks_to_complete),
                render_mode=render_mode,
                max_episode_steps=max_steps,
            )
            return MultiStepWrapper(
                base_env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_fn = env_fn

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    @staticmethod
    def _save_video_mp4(frames: np.ndarray, out_path: str, fps: int) -> None:
        """frames: (T, H, W, C) uint8."""
        import imageio.v2 as imageio
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        for codec in ("libx264", "mpeg4"):
            try:
                writer = imageio.get_writer(out_path, fps=fps, codec=codec, quality=7)
                try:
                    for frame in frames:
                        writer.append_data(frame)
                finally:
                    writer.close()
                return
            except Exception:
                if Path(out_path).exists():
                    Path(out_path).unlink(missing_ok=True)
        raise RuntimeError("Could not write MP4 with imageio (tried libx264, mpeg4)")

    def _render_frame(self, env) -> np.ndarray:
        """Render the underlying env as (H, W, C) uint8 frame."""
        try:
            frame = env.unwrapped.render()
        except Exception:
            frame = None
        if frame is None:
            return None
        return np.asarray(frame, dtype=np.uint8)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env_fn()
        goal_keys_cache = None

        all_success_fraction = []
        all_completed_tasks = []
        all_time = []
        all_frames: List[np.ndarray] = []
        saved_video_paths: List[str] = []

        for episode_idx in tqdm.tqdm(
                range(self.eval_episodes),
                desc=f"Eval FrankaKitchen-v1 [{self.task_name}]",
                leave=False,
                mininterval=self.tqdm_interval_sec):

            obs, info = env.reset()
            policy.reset()

            if goal_keys_cache is None:
                # Use tasks_to_complete order so the goal-vector layout matches
                # KitchenDataset (which also uses this order). Fallback to
                # alphabetical if tasks_to_complete is empty.
                if self.tasks_to_complete:
                    goal_keys_cache = [
                        k for k in self.tasks_to_complete
                        if k in obs['desired_goal']
                    ]
                    # Append any extra keys present in obs but missing from the
                    # task list, keeping them in alphabetical order for safety.
                    extras = sorted(
                        k for k in obs['desired_goal']
                        if k not in goal_keys_cache)
                    goal_keys_cache.extend(extras)
                else:
                    goal_keys_cache = _sorted_goal_keys(obs['desired_goal'])
                cprint(f"[KitchenRunner] goal keys order: {goal_keys_cache}", 'cyan')

            done = False
            total_time = 0.0
            step_count = 0
            episode_frames: List[np.ndarray] = []
            last_episode_completions: List[str] = []
            record_this_episode = (
                self.record_video and episode_idx < self.max_video_episodes)

            while not done:
                agent_pos = _obs_to_agent_pos(obs, goal_keys_cache)
                obs_dict_input = {
                    'agent_pos': torch.from_numpy(agent_pos).to(device=device).unsqueeze(0)
                }

                with torch.no_grad():
                    t0 = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    t1 = time.time()
                    total_time += t1 - t0

                action = action_dict['action'].detach().to('cpu').numpy().squeeze(0)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated) or bool(truncated)
                step_count += 1

                raw_info = info.get('raw_info', {}) if isinstance(info, dict) else {}
                if isinstance(raw_info, dict):
                    comp = raw_info.get('episode_task_completions')
                    if comp is not None:
                        last_episode_completions = list(comp)

                if record_this_episode:
                    frame = self._render_frame(env)
                    if frame is not None:
                        episode_frames.append(frame)

            n_target = max(len(self.tasks_to_complete), 1)
            n_completed = len(last_episode_completions)
            all_success_fraction.append(n_completed / n_target)
            all_completed_tasks.append(n_completed)
            all_time.append(total_time / max(step_count, 1))

            if record_this_episode and len(episode_frames) > 0:
                vid = np.stack(episode_frames, axis=0)
                all_frames.append(vid)
                if self.save_videos_dir:
                    out_mp4 = os.path.join(
                        self.save_videos_dir, f"episode_{episode_idx:03d}.mp4")
                    try:
                        self._save_video_mp4(vid, out_mp4, self.fps)
                        saved_video_paths.append(out_mp4)
                        cprint(f"[KitchenRunner] saved video: {out_mp4}", 'cyan')
                    except Exception as exc:
                        cprint(f"[KitchenRunner] video save failed: {exc}", 'red')

        env.close()

        log_data = dict()
        log_data['mean_n_completed_tasks'] = float(np.mean(all_completed_tasks))
        log_data['mean_success_rates'] = float(np.mean(all_success_fraction))
        log_data['test_mean_score'] = float(np.mean(all_success_fraction))
        log_data['mean_time'] = float(np.mean(all_time))
        log_data['mean_inference_latency_ms'] = float(np.mean(all_time) * 1000.0)
        if len(all_time) > 1:
            log_data['std_inference_latency_ms'] = float(
                np.std(all_time, ddof=0) * 1000.0)
        else:
            log_data['std_inference_latency_ms'] = 0.0

        completed = np.asarray(all_completed_tasks, dtype=np.int32)
        n_target = max(len(self.tasks_to_complete), 1)
        for k in range(1, 5):
            thr = min(k, n_target)
            log_data[f'success_rate_k{k}'] = float(np.mean(completed >= thr))

        cprint(f"test_mean_score (kitchen success frac): "
               f"{log_data['test_mean_score']:.4f}", 'green')
        cprint(f"mean completed tasks per ep: "
               f"{log_data['mean_n_completed_tasks']:.2f} / "
               f"{len(self.tasks_to_complete)}", 'green')

        self.logger_util_test.record(log_data['test_mean_score'])
        self.logger_util_test10.record(log_data['test_mean_score'])
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        if saved_video_paths:
            log_data['saved_video_paths'] = saved_video_paths

        if self.wandb_log and self.record_video and len(all_frames) > 0:
            try:
                # pad episodes to equal length then stack (N, T, H, W, C)
                T_max = max(v.shape[0] for v in all_frames)
                padded = []
                for v in all_frames:
                    if v.shape[0] < T_max:
                        pad = np.repeat(v[-1:], T_max - v.shape[0], axis=0)
                        v = np.concatenate([v, pad], axis=0)
                    padded.append(v)
                videos = np.stack(padded, axis=0)
                # wandb expects (N, T, C, H, W) uint8
                videos = np.transpose(videos, (0, 1, 4, 2, 3))
                log_data['sim_video_eval'] = wandb.Video(
                    videos, fps=self.fps, format='mp4')
            except Exception as exc:
                cprint(f"[KitchenRunner] video log skipped: {exc}", 'yellow')

        return log_data
