import numpy as np
import random

class SumTree:
    """
    二叉树结构，用于高效存储优先级和采样。
    树的叶子节点存储优先级，非叶子节点存储左右子树的优先级和。
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # 完整二叉树数组长度: 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # 数据存储区
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0  # 下一个写入位置
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        将优先级差值向上递归更新父节点
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        根据值 s 在树中向下搜索，找到对应的叶子节点
        """
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回所有优先级和"""
        return self.tree[0]

    def add(self, p, data):
        """添加优先级 p 和对应数据"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    # def update(self, idx, p):
    #     """更新已有数据的优先级"""
    #     change = p - self.tree[idx]
    #     self.tree[idx] = p
    #     self._propagate(idx, change)
    def update(self, idx, p):
        p = float(p)
        if not np.isfinite(p) or p <= 0.0:
            p = 1e-12

        old = float(self.tree[idx])
        if not np.isfinite(old):
            old = 0.0

        change = p - old
        if not np.isfinite(change):
            return  # 避免把 NaN 继续向上传播

        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            # 把父节点里的 NaN/Inf 清理掉再加
            self.tree[idx] = np.nan_to_num(self.tree[idx], nan=0.0, posinf=0.0, neginf=0.0) + change
            
    def get(self, s):
        """根据 s 找到叶子节点，并返回 (idx, p, data)"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def rebuild_totals(self):
        """从叶子区间重算父节点，并把非有限/非正的叶子消毒为极小正数"""
        cap = self.capacity
        leaf_start = cap - 1
        # 消毒叶子：未填充的叶子设 0，已填充但非有限或 <=0 的设为 1e-12
        for pos in range(cap):
            idx = leaf_start + pos
            if pos >= self.n_entries:
                self.tree[idx] = 0.0
            else:
                v = float(self.tree[idx])
                if not np.isfinite(v) or v <= 0.0:
                    self.tree[idx] = 1e-12
        # 自底向上重算父节点
        for i in range(cap - 2, -1, -1):
            left = 2 * i + 1
            right = left + 1
            self.tree[i] = self.tree[left] + self.tree[right]


class PrioritizedReplayBuffer:
    """
    完整的 Prioritized Experience Replay 实现：
     - 用 SumTree 存储 p_e^σ（论文公式 33,34）
     - sample() 返回 IS 权重 w_i，并做归一化（论文建议）
    """
    def __init__(self, capacity, mu=0.6, sigma=1.0, eta=1e-6, beta=0.4):
        """
        Args:
          capacity: int, replay buffer 大小
          mu:       float, 公式 (33) 中的 μ
          sigma:    float, 公式 (34) 中的 σ
          eta:      float, 公式 (33)优先级最小扰动项 η
          beta:     float, IS 权重指数 β (用于 w_i 计算)
        """
        self.tree = SumTree(capacity)
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.beta = beta


    def _get_priority(self, td_error):
        # 公式 (33)
        """计算经验的优先级"""
        return (abs(td_error) + self.eta) ** self.mu

    def add(self, obs, global_obs, action, reward, next_obs, next_global_obs, done,
            mask_all=None, next_mask_all=None):
        """新增经验，使用当前最大优先级初始化（并规范 dtype/shape）"""
        
        # 1. 处理单个智能体观测 - 确保是数组
        obs = np.asarray(obs, dtype=np.float32)

        def _peek2(x):
            try:
                return x[:2] if hasattr(x, "__len__") else x
            except:
                return x

        try:
            print(f"[PER.add] action dtype={type(action)}, shape={np.shape(action)}, peek={_peek2(action)}")
            if mask_all is not None:
                print(f"[PER.add] mask_all type={type(mask_all)}, shape={np.shape(mask_all)}")
            if next_mask_all is not None:
                print(f"[PER.add] next_mask_all type={type(next_mask_all)}, shape={np.shape(next_mask_all)}")
        except Exception as e:
            print(f"[PER.add] debug print failed: {e}")
    
        # 2. 处理全局观测
        global_obs = np.asarray(global_obs, dtype=np.float32)
        next_global_obs = np.asarray(next_global_obs, dtype=np.float32)
        
        # 3. 关键修复：处理next_obs - 确保是正确的2D数组形状
        if isinstance(next_obs, list):
            # 检查每个智能体的观测是否为数组
            processed_next_obs = []
            for agent_obs in next_obs:
                agent_obs_array = np.asarray(agent_obs, dtype=np.float32)
                # 确保每个智能体观测是1D数组
                if agent_obs_array.ndim == 0:
                    agent_obs_array = np.array([agent_obs_array])
                elif agent_obs_array.ndim > 1:
                    agent_obs_array = agent_obs_array.flatten()
                processed_next_obs.append(agent_obs_array)
            
            # 检查所有智能体观测维度是否一致
            obs_lengths = [len(obs_arr) for obs_arr in processed_next_obs]
            if len(set(obs_lengths)) > 1:
                print(f"Warning: Inconsistent observation lengths: {obs_lengths}")
                # 填充到最大长度
                max_len = max(obs_lengths)
                padded_obs = []
                for obs_arr in processed_next_obs:
                    if len(obs_arr) < max_len:
                        padded = np.zeros(max_len, dtype=np.float32)
                        padded[:len(obs_arr)] = obs_arr
                        padded_obs.append(padded)
                    else:
                        padded_obs.append(obs_arr)
                processed_next_obs = padded_obs
            
            next_obs = np.stack(processed_next_obs, axis=0)  # (n_agents, obs_dim)
        else:
            next_obs = np.asarray(next_obs, dtype=np.float32)
            # 确保是2D数组
            if next_obs.ndim == 1:
                next_obs = next_obs.reshape(1, -1)
        
        # 4. 处理其他字段
        reward = np.float32(reward)
        done = np.float32(done)

        # 5. 修复：统一动作维度 - 填充到最大动作头数量
        max_action_heads = 3  # 无人机有3个动作头，是最多的
        
        # 处理动作数组，确保所有智能体都有相同的动作头数量
        if isinstance(action, list):
            standardized_actions = []
            for agent_action in action:
                if isinstance(agent_action, (list, tuple)):
                    # 转换为numpy数组并填充到统一长度
                    agent_act = np.asarray(agent_action, dtype=np.int32)
                    if len(agent_act) < max_action_heads:
                        # 用0填充到最大长度
                        padded_act = np.zeros(max_action_heads, dtype=np.int32)
                        padded_act[:len(agent_act)] = agent_act
                        standardized_actions.append(padded_act)
                    else:
                        # 截断到最大长度
                        standardized_actions.append(agent_act[:max_action_heads])
                else:
                    # 单个数值，填充为长度为max_action_heads的数组
                    padded_act = np.zeros(max_action_heads, dtype=np.int32)
                    padded_act[0] = int(agent_action)
                    standardized_actions.append(padded_act)
            
            action = np.stack(standardized_actions, axis=0)  # (n_agents, max_action_heads)
        else:
            action = np.asarray(action, dtype=np.int32)
            # 确保是2D数组
            if action.ndim == 1:
                action = action.reshape(1, -1)
            # 确保所有agent都有max_action_heads个动作
            if action.shape[1] < max_action_heads:
                padded = np.zeros((action.shape[0], max_action_heads), dtype=np.int32)
                padded[:, :action.shape[1]] = action
                action = padded

        # 6. 规范化 mask 形状：都转为 np.bool_ 的 (n_agents, total_nodes)
        if mask_all is not None:
            mask_all = np.asarray(mask_all, dtype=np.bool_)
        if next_mask_all is not None:
            next_mask_all = np.asarray(next_mask_all, dtype=np.bool_)

        data = (obs, global_obs, action, reward, next_obs, next_global_obs, done,
                mask_all, next_mask_all)
        
        # 用当前最大优先级初始化；若无/非有限/<=0，退回 1.0
        p0 = self.tree.tree.max() if self.tree.n_entries > 0 else 1.0
        if not np.isfinite(p0) or p0 <= 0.0:
            p0 = 1.0
        self.tree.add(p0, data)

        # # 若树为空或最大优先级为 0，默认 1.0
        # try:
        #     max_p = float(np.max(self.tree.tree[-self.tree.capacity:]))
        # except Exception:
        #     max_p = 1.0
        # if not np.isfinite(max_p) or max_p <= 0.0:
        #     max_p = 1.0

        # # 存储时使用 p_e^sigma
        # self.tree.add(max_p ** self.sigma, data)

    def sample(self, batch_size):
        """
        按优先级采样，并计算 IS 权重 w_i。
        返回：
        obses, global_obses, actions, rewards, next_obses, next_global_obses, dones, idxs, weights, masks, next_masks
        若缓冲中未存掩码，则最后两个返回值为 None。
        """
        assert self.tree.n_entries > 0, "PER buffer is empty."

        batch, idxs, probs = [], [], []

        # 1) 总优先级为 0 时，先尝试重建；仍为 0 则走均匀兜底
        total_p = float(self.tree.total())
        if total_p <= 0:
            # 尝试修复树（如果实现了该方法）
            try:
                if hasattr(self.tree, "rebuild_totals"):
                    # rebuild_totals 内部应把叶子里的非有限/非正优先级消毒后自底向上重算
                    self.tree.rebuild_totals()
                    total_p = float(self.tree.total())
            except Exception:
                total_p = 0.0

        if total_p <= 0:
            # —— 均匀兜底：从已有有效叶子里等概率抽样，保证训练不中断 ——
            valid_pairs = []
            for leaf_pos in range(self.tree.n_entries):
                d = self.tree.data[leaf_pos]
                if isinstance(d, (tuple, list)) and len(d) in (7, 9):
                    leaf_idx = leaf_pos + self.tree.capacity - 1
                    valid_pairs.append((leaf_idx, d))

            if len(valid_pairs) < batch_size:
                raise RuntimeError(f"Valid transitions {len(valid_pairs)} < batch_size {batch_size}")

            extras = random.sample(valid_pairs, batch_size)
            for leaf_idx, d in extras:
                batch.append(d)
                idxs.append(leaf_idx)
            # 均匀概率（非零），IS 权重后续会被归一
            probs = np.full((batch_size,), 1.0 / max(1, self.tree.n_entries), dtype=np.float32)

        else:
            # 2) 正常的优先级分段采样
            segment = total_p / float(batch_size)

            attempts, max_attempts = 0, batch_size * 4
            while len(batch) < batch_size and attempts < max_attempts:
                i = len(batch)
                a, b = segment * i, segment * (i + 1)
                s = random.uniform(a, b)

                idx, p, data = self.tree.get(s)

                if not (isinstance(data, (tuple, list)) and len(data) in (7, 9)):
                    attempts += 1
                    continue

                batch.append(data)
                idxs.append(idx)
                probs.append(float(p) / float(total_p))
                attempts += 1

            # 3) 仍未凑满：从有效叶子补齐（携带真实 leaf_idx 和其 p）
            if len(batch) < batch_size:
                valid_pairs = []
                for leaf_pos in range(self.tree.n_entries):
                    d = self.tree.data[leaf_pos]
                    if isinstance(d, (tuple, list)) and len(d) in (7, 9):
                        leaf_idx = leaf_pos + self.tree.capacity - 1
                        p_leaf = float(self.tree.tree[leaf_idx])
                        if np.isfinite(p_leaf) and p_leaf > 0.0:
                            valid_pairs.append((leaf_idx, p_leaf, d))

                need = batch_size - len(batch)
                if len(valid_pairs) < need:
                    raise RuntimeError(f"Valid transitions {len(valid_pairs)} < need {need}")

                extras = random.sample(valid_pairs, need)
                for leaf_idx, p_leaf, d in extras:
                    batch.append(d)
                    idxs.append(leaf_idx)
                    probs.append(float(p_leaf) / float(total_p))

        # 4) 打包（兼容 7/9 元组）
        all_have_masks = all(len(x) == 9 for x in batch)

        if all_have_masks:
            (obses, global_obses, actions, rewards,
            next_obses, next_global_obses, dones,
            masks, next_masks) = zip(*batch)
        else:
            (obses, global_obses, actions, rewards,
            next_obses, next_global_obses, dones) = zip(*[x[:7] for x in batch])
            masks, next_masks = None, None

        obses              = np.stack([np.asarray(x, dtype=np.float32) for x in obses], axis=0)             # (B, obs_dim)
        global_obses       = np.stack([np.asarray(x, dtype=np.float32) for x in global_obses], axis=0)      # (B, glob_dim)
        actions            = np.stack([np.asarray(x, dtype=np.int32)   for x in actions], axis=0)           # (B, n_agents, n_heads)
        rewards            = np.asarray(rewards, dtype=np.float32)                                          # (B,)
        next_obses         = np.stack([np.asarray(x, dtype=np.float32) for x in next_obses], axis=0)        # (B, n_agents, obs_dim)
        next_global_obses  = np.stack([np.asarray(x, dtype=np.float32) for x in next_global_obses], axis=0) # (B, glob_dim)
        dones              = np.asarray(dones,   dtype=np.float32)                                          # (B,)

        if all_have_masks:
            masks_arr      = np.stack([np.asarray(x, dtype=np.bool_) for x in masks], axis=0)
            next_masks_arr = np.stack([np.asarray(x, dtype=np.bool_) for x in next_masks], axis=0)
        else:
            masks_arr, next_masks_arr = None, None

        # 5) IS 权重（裁剪 + 归一 + 非有限兜底）
        probs = np.asarray(probs, dtype=np.float32)
        probs = np.clip(probs, 1e-12, None)  # 下限，防 1/0
        N = max(1, self.tree.n_entries)
        weights = (1.0 / (N * probs)) ** self.beta
        if not np.all(np.isfinite(weights)):
            weights = np.ones_like(probs, dtype=np.float32)
        else:
            wmax = weights.max()
            weights = weights / wmax if (wmax > 0 and np.isfinite(wmax)) else np.ones_like(probs, dtype=np.float32)

        return (obses, global_obses, actions, rewards,
                next_obses, next_global_obses, dones,
                idxs, weights, masks_arr, next_masks_arr)


    def update_priorities(self, idxs, td_errors):
        abs_td = np.abs(np.asarray(td_errors, dtype=np.float32))
        # 原有优先级公式基础上加下限/上限
        p = (abs_td + self.eta) ** self.mu
        p = np.clip(p, 1e-12, 1e12)

        leaf_lo = self.tree.capacity - 1
        leaf_hi = 2 * self.tree.capacity - 2

        for idx, pi in zip(idxs, p):
            ii = int(idx)
            if not (leaf_lo <= ii <= leaf_hi):
                continue
            if not np.isfinite(pi) or pi <= 0.0:
                pi = 1e-12
            self.tree.update(ii, float(pi))

    def __len__(self):
        return self.tree.n_entries
