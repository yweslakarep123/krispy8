"""Minari-backed dataset for Franka Kitchen (low-dim state + goal).

Loads `kitchen-complete-v1` via Minari, concatenates the robot state
observation with a deterministic flattening of the (per-task) desired_goal
dict, and exposes sequences through the same ReplayBuffer + SequenceSampler
pipeline used by the original FlowPolicy code.

Features on top of the original implementation:

1. Sub-task ordering. `goal_keys` follow the task yaml `tasks_to_complete`
   (default: microwave, kettle, light switch, slide cabinet) instead of
   alphabetical order so that dataset and runtime runner share the exact
   same vector layout.

2. Episode filter. Only keep demonstrations whose task completion order
   matches `tasks_to_complete` exactly. For `D4RL/kitchen/complete-v2` all
   19 episodes pass, but the filter is a safeguard (and enables using
   mixed/partial datasets later).

3. Optional sliding-window preprocessing (`preprocess.enabled=true`):
   each demonstration of length T is sliced into overlapping windows of
   length `round(window_ratio * T)` with the configured `stride`. Each
   window becomes an independent "mini-episode" in the replay buffer, and
   the whole collection of windows is shuffled then split train/val/test
   (default 70/20/10). When `preprocess.enabled=false`, the original
   episode-level pipeline is used (single `val_ratio` episode split, no
   test split).

4. Optional augmentation (training split only, after split):
   - `obs_noise_std`: Gaussian noise on the 59-dim `observation` part of
     `agent_pos` (not applied to the goal suffix which is static).
   - `action_noise_std`: Gaussian noise on the action target.

4. ``preprocessing_profile`` (Hydra ``task.dataset.preprocessing_profile``):
   - ``standard`` (default): sliding windows (stride 1), train/val split via
     ``val_ratio`` **or** explicit ``train_episode_indices`` /
     ``val_episode_indices``, augmentation from ``obs_noise_std`` /
     ``action_noise_std``.
   - ``minimal``: sliding stride 1, same splits as ``standard``, **no**
     Gaussian augmentation (noise std forced to 0).
   - ``legacy_minimal``: ablation lama — stride = ``horizon`` (jendela tidak
     overlap), ``val_ratio`` dipaksa 0, tanpa noise. Tidak kompatibel dengan
     daftar episode eksplisit (akan memicu ``ValueError``).
   - ``raw``: stride = ``horizon`` (tanpa sliding overlap), tanpa noise,
     tanpa split acak (``val_ratio`` efektif 0 bila tidak pakai daftar episode).
     Boleh dipakai bersama ``train_episode_indices`` / ``val_episode_indices``
     (mis. CV): stride horizon tetap dipakai.

5. **Episode-level CV (opsional):** jika ``train_episode_indices`` dan
   ``val_episode_indices`` keduanya diset (daftar indeks episode buffer 0..N-1),
   ``val_ratio`` diabaikan dan mask diambil dari daftar tersebut.
"""

from typing import Dict, List, Optional, Sequence
import copy
import numpy as np
import torch
from termcolor import cprint

from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.replay_buffer import ReplayBuffer
from flow_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.dataset.base_dataset import BaseDataset


def _sorted_goal_keys(goal_dict: Dict) -> List[str]:
    """Deterministic (alphabetical) ordering for desired_goal keys."""
    return sorted(goal_dict.keys())


def _flatten_goal_step(goal_dict: Dict, keys: Sequence[str]) -> np.ndarray:
    """Concatenate per-task goal vectors for a single step."""
    parts = []
    for k in keys:
        v = np.asarray(goal_dict[k], dtype=np.float32).reshape(-1)
        parts.append(v)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)


def _flatten_goal_seq(goal_dict: Dict, keys: Sequence[str], length: int) -> np.ndarray:
    """Flatten goal dict whose values have shape (T, d_k) into (T, sum(d_k)).

    Also tolerates constant goals stored as scalars / (d_k,) by broadcasting.
    """
    parts = []
    for k in keys:
        v = np.asarray(goal_dict[k], dtype=np.float32)
        if v.ndim == 1:
            # (d_k,) -> (T, d_k)
            v = np.broadcast_to(v[None, :], (length, v.shape[0])).copy()
        elif v.ndim == 0:
            v = np.broadcast_to(v.reshape(1, 1), (length, 1)).copy()
        elif v.ndim >= 2:
            # (T, d_k) - take the first `length` entries
            v = v[:length].astype(np.float32, copy=False).reshape(length, -1)
        parts.append(v)
    if not parts:
        return np.zeros((length, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1)


def _build_windowed_buffer(orig_buffer: ReplayBuffer,
                           window_ratio: float,
                           stride: int,
                           train_ratio: float,
                           val_ratio: float,
                           test_ratio: float,
                           split_seed: int):
    """Turn each episode in `orig_buffer` into a set of overlapping windows
    (size = round(window_ratio * T_episode), stride = `stride`). Every
    window becomes a new "episode" in the returned `ReplayBuffer`. The
    full collection of windows is then shuffled with `split_seed` and
    split into disjoint train / val / test masks according to the given
    ratios.
    """
    assert 0.0 < window_ratio <= 1.0, \
        f"window_ratio must be in (0, 1], got {window_ratio}"
    assert stride >= 1, f"stride must be >= 1, got {stride}"
    total = train_ratio + val_ratio + test_ratio
    assert total > 0, "train+val+test ratios must be > 0"
    train_ratio, val_ratio, test_ratio = (
        train_ratio / total, val_ratio / total, test_ratio / total)

    new_buffer = ReplayBuffer.create_empty_numpy()
    keys = list(orig_buffer.keys())
    episode_ends = orig_buffer.episode_ends[:]
    n_src_episodes = len(episode_ends)

    for ep_idx in range(n_src_episodes):
        ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        ep_end = int(episode_ends[ep_idx])
        T = ep_end - ep_start
        if T <= 1:
            continue
        W = max(2, int(round(window_ratio * T)))
        if W > T:
            W = T
        for s in range(0, T - W + 1, stride):
            window_data = {
                k: np.asarray(
                    orig_buffer[k][ep_start + s: ep_start + s + W]
                ).copy()
                for k in keys
            }
            new_buffer.add_episode(window_data)

    n_windows = new_buffer.n_episodes
    assert n_windows > 0, \
        f"sliding-window preprocessing produced 0 windows " \
        f"(window_ratio={window_ratio}, stride={stride})"

    rng = np.random.default_rng(seed=split_seed)
    perm = rng.permutation(n_windows)
    n_train = int(round(train_ratio * n_windows))
    n_val = int(round(val_ratio * n_windows))
    n_train = min(max(n_train, 1), n_windows - 1 if n_windows > 1 else 1)
    n_val = min(max(n_val, 0), n_windows - n_train)
    n_test = n_windows - n_train - n_val

    train_mask = np.zeros(n_windows, dtype=bool)
    val_mask = np.zeros(n_windows, dtype=bool)
    test_mask = np.zeros(n_windows, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val: n_train + n_val + n_test]] = True
    return new_buffer, (train_mask, val_mask, test_mask)


def _extract_completion_order(episode, tasks_to_complete: Sequence[str]) -> Optional[List[str]]:
    """Recover the chronological order in which `tasks_to_complete` were
    completed during the episode, by watching the per-step
    `info['tasks_to_complete']` shrink.

    Returns a list of task names in completion order, or None if the info
    schema cannot be parsed.
    """
    infos = getattr(episode, "infos", None)
    if infos is None:
        return None

    # Minari may store info either as a list of dicts (one per step) or as
    # a dict-of-arrays. Normalise to list-of-dicts.
    per_step_remaining: List[List[str]] = []
    try:
        if isinstance(infos, dict):
            if "tasks_to_complete" not in infos:
                return None
            remaining_arr = infos["tasks_to_complete"]
            for step in remaining_arr:
                per_step_remaining.append([str(t) for t in list(step)])
        else:
            for step_info in infos:
                if not isinstance(step_info, dict):
                    return None
                if "tasks_to_complete" not in step_info:
                    return None
                per_step_remaining.append(
                    [str(t) for t in list(step_info["tasks_to_complete"])])
    except Exception:
        return None

    if not per_step_remaining:
        return None

    order: List[str] = []
    prev = set(per_step_remaining[0])
    targets = set(tasks_to_complete)
    for remaining in per_step_remaining[1:]:
        curr = set(remaining)
        just_finished = prev - curr
        # Only track tasks we care about
        for t in just_finished & targets:
            if t not in order:
                order.append(t)
        prev = curr
    return order


class KitchenDataset(BaseDataset):
    """Minari-backed FrankaKitchen low-dim dataset.

    agent_pos[t] = concat(observation[t], flatten(desired_goal[t]))  (no point cloud).
    """

    def __init__(self,
                 dataset_id: str = 'kitchen-complete-v1',
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 max_train_episodes=None,
                 task_name=None,
                 download: bool = True,
                 goal_keys: Optional[Sequence[str]] = None,
                 tasks_to_complete: Optional[Sequence[str]] = None,
                 enforce_task_order: bool = True,
                 obs_noise_std: float = 0.0,
                 action_noise_std: float = 0.0,
                 obs_noise_dim: Optional[int] = None,
                 normalizer_mode: str = 'limits',
                 preprocessing_profile: str = 'standard',
                 train_episode_indices: Optional[Sequence[int]] = None,
                 val_episode_indices: Optional[Sequence[int]] = None,
                 preprocess: Optional[Dict] = None,
                 ):
        super().__init__()
        import minari

        self.task_name = task_name
        self.dataset_id = dataset_id
        self.preprocessing_profile = str(preprocessing_profile).strip().lower()
        if self.preprocessing_profile not in (
                'standard', 'minimal', 'legacy_minimal', 'raw'):
            raise ValueError(
                "preprocessing_profile must be 'standard', 'minimal', "
                f"'legacy_minimal', or 'raw', got {preprocessing_profile!r}")

        def _as_int_list(seq) -> Optional[List[int]]:
            if seq is None:
                return None
            return [int(x) for x in list(seq)]

        self._train_epi = _as_int_list(train_episode_indices)
        self._val_epi = _as_int_list(val_episode_indices)
        use_episode_lists = (
            self._train_epi is not None and len(self._train_epi) > 0
            and self._val_epi is not None and len(self._val_epi) > 0)
        tr_nonempty = self._train_epi is not None and len(self._train_epi) > 0
        va_nonempty = self._val_epi is not None and len(self._val_epi) > 0
        if tr_nonempty != va_nonempty:
            raise ValueError(
                "train_episode_indices and val_episode_indices must both be "
                "non-empty lists or both omitted / null.")
        if use_episode_lists and self.preprocessing_profile == 'legacy_minimal':
            raise ValueError(
                "legacy_minimal cannot be combined with explicit episode indices; "
                "use 'minimal' or 'standard' for CV.")

        self.tasks_to_complete = (
            list(tasks_to_complete) if tasks_to_complete is not None else None)
        self.enforce_task_order = enforce_task_order

        if self.preprocessing_profile == 'legacy_minimal':
            cprint(
                "[KitchenDataset] preprocessing_profile=legacy_minimal: "
                "stride=horizon, val_ratio=0, no augmentation",
                "yellow",
            )
            val_ratio_effective = 0.0
            self.obs_noise_std = 0.0
            self.action_noise_std = 0.0
            sequence_stride = max(1, int(horizon))
        elif self.preprocessing_profile == 'raw':
            cprint(
                "[KitchenDataset] preprocessing_profile=raw: "
                "stride=horizon (no sliding overlap), no Gaussian augmentation; "
                "tanpa split acak val_ratio bila tidak pakai daftar episode eksplisit",
                "yellow",
            )
            val_ratio_effective = 0.0
            self.obs_noise_std = 0.0
            self.action_noise_std = 0.0
            sequence_stride = max(1, int(horizon))
        elif self.preprocessing_profile == 'minimal':
            cprint(
                "[KitchenDataset] preprocessing_profile=minimal: "
                "sliding stride 1, no Gaussian augmentation",
                "yellow",
            )
            val_ratio_effective = float(val_ratio)
            self.obs_noise_std = 0.0
            self.action_noise_std = 0.0
            sequence_stride = 1
        else:  # standard
            val_ratio_effective = float(val_ratio)
            self.obs_noise_std = float(obs_noise_std)
            self.action_noise_std = float(action_noise_std)
            sequence_stride = 1

        # Profil ``raw`` tetap stride=horizon meski ada daftar episode (CV).
        if use_episode_lists and self.preprocessing_profile != 'raw':
            sequence_stride = 1  # CV: sliding penuh kecuali ``raw``

        self.obs_noise_dim = obs_noise_dim
        self.normalizer_mode = normalizer_mode
        self._is_train_split = True  # val/test split flips this to False in copy

        # Sliding-window preprocessing config (opt-in).
        preprocess = dict(preprocess) if preprocess else {}
        self.preprocess_enabled = bool(preprocess.get('enabled', False))
        self.preprocess_window_ratio = float(preprocess.get('window_ratio', 0.25))
        self.preprocess_stride = int(preprocess.get('stride', 1))
        self.preprocess_train_ratio = float(preprocess.get('train_ratio', 0.70))
        self.preprocess_val_ratio = float(preprocess.get('val_ratio', 0.20))
        self.preprocess_test_ratio = float(preprocess.get('test_ratio', 0.10))
        self.preprocess_split_seed = int(preprocess.get('split_seed', seed))

        cprint(f"[KitchenDataset] loading minari dataset '{dataset_id}'", 'yellow')
        try:
            minari_ds = minari.load_dataset(dataset_id, download=download)
        except TypeError:
            # older Minari API without download kwarg
            try:
                minari_ds = minari.load_dataset(dataset_id)
            except FileNotFoundError:
                minari.download_dataset(dataset_id)
                minari_ds = minari.load_dataset(dataset_id)

        # Decide goal_keys ordering.
        # Priority: explicit goal_keys arg > tasks_to_complete > alphabetical.
        if goal_keys is not None:
            self._goal_keys = list(goal_keys)
        elif self.tasks_to_complete is not None:
            self._goal_keys = list(self.tasks_to_complete)
        else:
            self._goal_keys = None  # will default to alphabetical on first ep

        replay_buffer = ReplayBuffer.create_empty_numpy()
        n_episodes = 0
        n_skipped_order = 0

        for episode in minari_ds.iterate_episodes():
            actions = np.asarray(episode.actions, dtype=np.float32)
            T = actions.shape[0]
            if T == 0:
                continue

            obs = episode.observations
            assert isinstance(obs, dict), \
                f"expected dict observations, got {type(obs)}"
            observation = np.asarray(obs['observation'], dtype=np.float32)
            # Minari stores observations with length T+1 (including the terminal
            # observation). Align with actions of length T.
            if observation.shape[0] == T + 1:
                observation = observation[:-1]
            elif observation.shape[0] != T:
                observation = observation[:T]

            desired_goal = obs['desired_goal']
            assert isinstance(desired_goal, dict), \
                f"expected dict desired_goal, got {type(desired_goal)}"

            if self._goal_keys is None:
                self._goal_keys = _sorted_goal_keys(desired_goal)
                cprint(f"[KitchenDataset] goal keys order (fallback alphabetical): "
                       f"{self._goal_keys}", 'cyan')

            # Optional episode-order filter.
            if self.enforce_task_order and self.tasks_to_complete is not None:
                order = _extract_completion_order(episode, self.tasks_to_complete)
                if order is not None:
                    if order != list(self.tasks_to_complete):
                        n_skipped_order += 1
                        continue
                # If order couldn't be parsed we keep the episode (no change
                # in behaviour vs. previous versions).

            goal_flat = _flatten_goal_seq(
                desired_goal, self._goal_keys, length=T + 1 if any(
                    isinstance(v, np.ndarray) and np.asarray(v).ndim >= 2
                    and np.asarray(v).shape[0] == T + 1
                    for v in desired_goal.values()) else T,
            )
            if goal_flat.shape[0] == T + 1:
                goal_flat = goal_flat[:-1]
            elif goal_flat.shape[0] != T:
                goal_flat = goal_flat[:T]

            agent_pos = np.concatenate([observation, goal_flat], axis=-1).astype(
                np.float32)

            replay_buffer.add_episode({
                'state': agent_pos,
                'action': actions,
            })
            n_episodes += 1

        assert n_episodes > 0, \
            f"no episodes loaded from minari dataset '{dataset_id}'"
        cprint(f"[KitchenDataset] loaded {n_episodes} episodes, "
               f"{replay_buffer.n_steps} total steps", 'green')
        cprint(f"[KitchenDataset] goal keys order: {self._goal_keys}", 'cyan')
        if self.enforce_task_order and self.tasks_to_complete is not None:
            cprint(f"[KitchenDataset] order filter: tasks_to_complete="
                   f"{list(self.tasks_to_complete)}, skipped={n_skipped_order} "
                   f"episode(s) not matching order", 'cyan')

        # Optional sliding-window preprocessing rebuilds the replay buffer so
        # each window is an independent "episode" for the sampler, and the
        # shuffled windows are split into train/val/test.
        if self.preprocess_enabled:
            replay_buffer, split_masks = _build_windowed_buffer(
                orig_buffer=replay_buffer,
                window_ratio=self.preprocess_window_ratio,
                stride=self.preprocess_stride,
                train_ratio=self.preprocess_train_ratio,
                val_ratio=self.preprocess_val_ratio,
                test_ratio=self.preprocess_test_ratio,
                split_seed=self.preprocess_split_seed,
            )
            train_mask, val_mask, test_mask = split_masks
            cprint(
                f"[KitchenDataset] sliding-window preprocessing: "
                f"window_ratio={self.preprocess_window_ratio} "
                f"stride={self.preprocess_stride} "
                f"windows={replay_buffer.n_episodes} "
                f"train/val/test="
                f"{int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}",
                'cyan')
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed,
            )
            self._val_episode_mask = val_mask
        elif use_episode_lists:
            n_ep = replay_buffer.n_episodes
            train_set = set(self._train_epi)
            val_set = set(self._val_epi)
            if train_set & val_set:
                raise ValueError(
                    f"train/val episode overlap: {train_set & val_set}")
            for i in self._train_epi + self._val_epi:
                if i < 0 or i >= n_ep:
                    raise ValueError(
                        f"episode index {i} out of range for n_episodes={n_ep}")
            train_mask = np.zeros(n_ep, dtype=bool)
            val_mask = np.zeros(n_ep, dtype=bool)
            test_mask = np.zeros(n_ep, dtype=bool)
            for i in self._train_epi:
                train_mask[int(i)] = True
            for i in self._val_epi:
                val_mask[int(i)] = True
            self._val_episode_mask = val_mask
        else:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio_effective,
                seed=seed,
            )
            train_mask = ~val_mask
            test_mask = np.zeros_like(val_mask)
            cprint(
                f"[KitchenDataset] preprocessing disabled: "
                f"episodes train/val/test="
                f"{int(train_mask.sum())}/{int(val_mask.sum())}/0",
                'cyan')
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed,
            )
            has_val_holdout = bool(np.any(~train_mask))
            self._val_episode_mask = (
                (~train_mask) if has_val_holdout else train_mask)

        self._sequence_stride = int(sequence_stride)

        self.replay_buffer = replay_buffer
        self._agent_pos_dim = int(replay_buffer['state'].shape[-1])
        self._action_dim = int(replay_buffer['action'].shape[-1])
        # Infer the size of the `observation` (robot-state) prefix so that
        # obs-noise augmentation leaves the static goal suffix alone.
        # Derived from the flattened layout: total - observation dim.
        # Fallback: assume 59 for FrankaKitchen robot state.
        self._robot_obs_dim = (
            int(self.obs_noise_dim) if self.obs_noise_dim is not None
            else max(self._agent_pos_dim - 11, 1))  # 11 = kitchen-complete goal dim
        cprint(f"[KitchenDataset] agent_pos_dim={self._agent_pos_dim} "
               f"action_dim={self._action_dim} "
               f"robot_obs_dim(for noise)={self._robot_obs_dim}", 'cyan')

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            sequence_stride=self._sequence_stride,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    @property
    def goal_keys(self) -> List[str]:
        return list(self._goal_keys) if self._goal_keys is not None else []

    @property
    def agent_pos_dim(self) -> int:
        return self._agent_pos_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _make_split(self, mask: np.ndarray) -> 'KitchenDataset':
        split = copy.copy(self)
        split.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=mask,
            sequence_stride=self._sequence_stride,
        )
        split.train_mask = mask
        split._is_train_split = False
        return split

    def get_validation_dataset(self):
        if self.preprocess_enabled:
            return self._make_split(self.val_mask)
        return self._make_split(self._val_episode_mask)

    def get_test_dataset(self):
        """Return the held-out test split (only populated when
        `preprocess.enabled=true`)."""
        if not self.preprocess_enabled or not np.any(self.test_mask):
            return None
        return self._make_split(self.test_mask)

    def get_normalizer(self, mode=None, **kwargs):
        if mode is None:
            mode = self.normalizer_mode
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)

        # Augmentation: training split only, and only when std > 0.
        if self._is_train_split:
            if self.obs_noise_std > 0.0:
                noise = np.random.normal(
                    loc=0.0, scale=self.obs_noise_std,
                    size=(agent_pos.shape[0], self._robot_obs_dim),
                ).astype(np.float32)
                agent_pos = agent_pos.copy()
                agent_pos[:, :self._robot_obs_dim] += noise
            if self.action_noise_std > 0.0:
                action = action + np.random.normal(
                    loc=0.0, scale=self.action_noise_std,
                    size=action.shape,
                ).astype(np.float32)

        return {
            'obs': {
                'agent_pos': agent_pos,
            },
            'action': action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
