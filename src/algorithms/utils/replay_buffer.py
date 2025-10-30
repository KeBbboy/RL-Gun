import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs, global_obs, act_all, rew, next_obs_all, next_global_obs, done, mask_all=None, next_mask_all=None):
        data = (obs, global_obs, act_all, rew, next_obs_all, next_global_obs, done, mask_all, next_mask_all)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, glob_obs_t, actions, rewards, next_obses_all, next_glob_obs_t, dones, masks, next_masks = [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            if len(data) == 9:  # 新格式，包含global_obs和mask
                obs, global_obs, act_all, rew, next_obs_all, next_global_obs, done, mask_all, next_mask_all = data
            elif len(data) == 7:  # 中间格式
                obs, global_obs, act_all, rew, next_obs_all, next_global_obs, done = data
                mask_all, next_mask_all = None, None
            else:  # 旧格式，向后兼容
                obs, act_all, rew, next_obs_all, done = data
                global_obs, next_global_obs, mask_all, next_mask_all = None, None, None, None

            obses_t.append(np.array(obs, copy=False))
            glob_obs_t.append(np.array(global_obs, copy=False) if global_obs is not None else None)
            actions.append(np.array(act_all, copy=False))
            rewards.append(rew)
            next_obses_all.append(np.array(next_obs_all, copy=False))
            next_glob_obs_t.append(np.array(next_global_obs, copy=False) if next_global_obs is not None else None)
            dones.append(done)
            masks.append(mask_all)
            next_masks.append(next_mask_all)

        # 只有当所有mask都不为None时才返回mask数组
        if any(m is not None for m in masks):
            masks = np.array(masks) if masks[0] is not None else None
            next_masks = np.array(next_masks) if next_masks[0] is not None else None
        else:
            masks = None
            next_masks = None

        return (np.array(obses_t), np.array(glob_obs_t), np.array(actions), np.array(rewards),
                np.array(next_obses_all), np.array(next_glob_obs_t), np.array(dones), masks, next_masks)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
