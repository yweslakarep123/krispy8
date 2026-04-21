"""Gymnasium-native multi-step wrapper (obs / action framestacks).

Ported from the original Gym 0.21 version to Gymnasium's 5-tuple step API
(`terminated`, `truncated`) and 2-tuple reset API (`obs, info`).
"""

from collections import defaultdict, deque

import dill
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    x = list(x)
    if len(x) == 0:
        return []
    n = min(len(x), n)
    sub = x[-n:]
    if isinstance(sub[0], torch.Tensor):
        try:
            return torch.stack(sub)
        except Exception:
            return sub
    try:
        return np.array(sub)
    except (ValueError, TypeError):
        # info entries like `tasks_to_complete` are lists of strings with
        # variable length; keep them as a plain python list in that case.
        return list(sub)


def dict_take_last_n(x, n):
    result = {}
    for key, value in x.items():
        try:
            result[key] = take_last_n(value, n)
        except Exception:
            result[key] = list(value)
    return result


def aggregate(data, method='max'):
    if len(data) == 0:
        return 0.0
    if isinstance(data[0], torch.Tensor):
        stacked = torch.stack(data)
        if method == 'max':
            return torch.max(stacked)
        elif method == 'min':
            return torch.min(stacked)
        elif method == 'mean':
            return torch.mean(stacked)
        elif method == 'sum':
            return torch.sum(stacked)
        raise NotImplementedError(method)
    else:
        arr = np.asarray(data)
        if method == 'max':
            return np.max(arr)
        elif method == 'min':
            return np.min(arr)
        elif method == 'mean':
            return np.mean(arr)
        elif method == 'sum':
            return np.sum(arr)
        raise NotImplementedError(method)


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
    else:
        # fallback: wrap scalars / lists as numpy
        arr = np.asarray(all_obs[-1])
        result = np.zeros((n_steps,) + arr.shape, dtype=arr.dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.asarray(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    """Gymnasium wrapper that frame-stacks observations and executes
    multi-step action sequences."""

    def __init__(self,
                 env,
                 n_obs_steps,
                 n_action_steps,
                 max_episode_steps=None,
                 reward_agg_method='max'):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method

        self.obs = deque(maxlen=n_obs_steps + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=n_obs_steps + 1))

    def reset(self, *, seed=None, options=None):
        reset_result = self.env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs, info = reset_result, {}

        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        self._add_info(info)

        return self._get_obs(self.n_obs_steps), dict(info)

    def step(self, action):
        """action: (n_action_steps,) + action_shape."""
        last_info = {}
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                break
            step_result = self.env.step(act)
            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = bool(terminated) or bool(truncated)
            elif len(step_result) == 4:
                observation, reward, done, info = step_result
                terminated, truncated = done, False
            else:
                raise RuntimeError(
                    f'Unexpected step result length {len(step_result)}')

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                    and (len(self.reward) >= self.max_episode_steps):
                done = True
                truncated = True
            self.done.append(done)
            self._add_info(info)
            last_info = info

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max') if len(self.done) > 0 else False
        info = dict_take_last_n(self.info, self.n_obs_steps)
        # preserve non-list raw info from last env.step (useful for `tasks_to_complete` etc.)
        info['raw_info'] = last_info

        terminated_any = bool(done)
        return observation, float(reward), terminated_any, False, info

    def _get_obs(self, n_steps=1):
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                value = self.obs[0][key]
                if isinstance(value, dict):
                    sub = dict()
                    for sub_key in value.keys():
                        sub[sub_key] = stack_last_n_obs(
                            [obs[key][sub_key] for obs in self.obs], n_steps)
                    result[key] = sub
                else:
                    result[key] = stack_last_n_obs(
                        [obs[key] for obs in self.obs], n_steps)
            return result
        raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        if not isinstance(info, dict):
            return
        for key, value in info.items():
            # Skip non-stackable info entries (lists of strings, nested dicts, ...)
            try:
                self.info[key].append(value)
            except Exception:
                continue

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

    def get_infos(self):
        return {k: list(v) for k, v in self.info.items()}
