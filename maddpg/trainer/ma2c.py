import numpy as np
import tensorflow as tf
from maddpg.trainer.agent_trainer import AgentTrainer

def _to_scalar(x):
        # 统一把可能的 Tensor / ndarray / python 数转换成 float 标量
        if x is None:
            return None
        if isinstance(x, tf.Tensor):
            x = tf.reshape(x, [-1])
            return float(x[0].numpy())
        else:
            x = np.asarray(x).reshape(-1)
            return float(x[0])

# =========================
# On-policy 轨迹缓冲（A2C，K-步回报）
# =========================
class OnPolicyBuffer:
    """MA2C专用的on-policy经验缓冲区"""
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.reset()

    def __len__(self):
        return len(self.obs)
    
    def reset(self):
        self.obs = []
        self.global_obs = []  # ✅ 新增：存储全局观测
        self.acts = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []
    
    def add(self, obs, global_obs, acts, log_probs, value, reward, done, mask=None):
        """存储一步经验"""
        self.obs.append(obs)
        self.global_obs.append(global_obs)  # ✅ 保存全局观测
        self.acts.append(acts)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)
    
    def get_trajectory(self, final_value):
        """计算优势函数和回报"""
        obs = np.array(self.obs, dtype=np.float32)
        global_obs = np.array(self.global_obs, dtype=np.float32)  # ✅ 返回全局观测
        acts = np.array(self.acts, dtype=np.int32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        masks = self.masks
        
        # 计算回报
        returns = []
        advantages = []
        R = final_value
        
        for r, v, done in zip(reversed(self.rewards), 
                              reversed(self.values), 
                              reversed(self.dones)):
            R = r + self.gamma * R * (1 - done)
            adv = R - v
            returns.insert(0, R)
            advantages.insert(0, adv)
        
        returns = np.array(returns, dtype=np.float32)
        advantages = np.array(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return obs, global_obs, acts, log_probs, returns, advantages, masks


class MA2CPolicy(tf.keras.Model):
    """MA2C的策略网络 - Actor用局部观测，Critic用全局观测"""
    def __init__(self, obs_dim, act_space_list, global_obs_dim, n_lstm=64, name='ma2c_policy'):
        super().__init__(name=name)
        
        self.act_space_list = act_space_list
        self.num_heads = len(act_space_list)
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        
        # ===== Actor支路（局部观测） =====
        self.actor_fc1 = tf.keras.layers.Dense(128, activation='relu', name='actor_fc1')
        self.actor_fc2 = tf.keras.layers.Dense(128, activation='relu', name='actor_fc2')
        self.actor_lstm = tf.keras.layers.LSTM(
            n_lstm, 
            return_sequences=True, 
            return_state=True,
            name='actor_lstm'
        )
        
        # 多个Actor头 - 输出logits
        self.actor_heads = []
        for i, space in enumerate(act_space_list):
            head = tf.keras.layers.Dense(space.n, activation=None, name=f'actor_head_{i}')
            self.actor_heads.append(head)
        
        # ===== Critic支路（全局观测） =====
        self.critic_fc1 = tf.keras.layers.Dense(256, activation='relu', name='critic_fc1')
        self.critic_fc2 = tf.keras.layers.Dense(128, activation='relu', name='critic_fc2')
        self.critic_head = tf.keras.layers.Dense(1, name='critic')
        
        self.actor_lstm_state = None
    
    def reset_states(self):
        self.actor_lstm_state = None
    
    def call(self, local_obs, global_obs=None, training=False):
        """
        前向传播
        Args:
            local_obs: 局部观测 (batch, [seq_len,] obs_dim) - 用于Actor
            global_obs: 全局观测 (batch, [seq_len,] global_obs_dim) - 用于Critic
            training: 是否训练模式
        Returns:
            logits_list: Actor输出的logits列表
            value: Critic输出的状态价值
        """
        # ===== Actor前向（局部观测） =====
        if len(local_obs.shape) == 2:
            local_obs = tf.expand_dims(local_obs, axis=1)  # (batch, 1, obs_dim)
        
        x = self.actor_fc1(local_obs)
        x = self.actor_fc2(x)
        
        if self.actor_lstm_state is None:
            lstm_out, h, c = self.actor_lstm(x)              # (batch, seq, n_lstm)
        else:
            lstm_out, h, c = self.actor_lstm(x, initial_state=self.actor_lstm_state)

        
        if training:
            self.actor_lstm_state = [h, c]
        
        # 多头 logits，挤掉时间维=1（若是单步）
        logits_list = []
        for head in self.actor_heads:
            logits = head(lstm_out)                 # (batch, seq_len, n_i)
            if logits.shape[1] == 1:
                logits = tf.squeeze(logits, axis=1) # (batch, n_i)
            logits_list.append(logits)
        
        # ===== Critic：优先使用全局观测 =====
        if global_obs is not None:
            if len(global_obs.shape) == 2:
                global_obs = tf.expand_dims(global_obs, axis=1)   # (batch, 1, gdim)
            elif len(global_obs.shape) == 1:
                global_obs = tf.reshape(global_obs, [1, 1, -1])   # (1, 1, gdim)

            v = self.critic_fc1(global_obs)   # (batch, 1, 256)
            v = self.critic_fc2(v)            # (batch, 1, 128)
            v = self.critic_head(v)           # (batch, 1, 1)
            value = tf.squeeze(v, axis=-1)                              # ✅ (batch, seq)
        else:
            # 兜底：无全局观时用 lstm_out 估值
            v = self.critic_head(lstm_out)    # (batch, 1, 1)
            value = tf.squeeze(v, axis=-1)                              # ✅ (batch, seq)
        
        return logits_list, value


class MA2CAgentTrainer(AgentTrainer):
    """MA2C智能体训练器"""
    
    def __init__(self, name, obs_shape_n, act_space_list, agent_index, args, 
                global_obs_dim=None, num_trucks=None, num_drones=None, total_nodes=None):
        self.name = name
        self.agent_index = agent_index
        self.args = args
        
        # 获取环境参数
        self.num_trucks = num_trucks if num_trucks is not None else 3
        self.num_drones = num_drones if num_drones is not None else 5
        self.total_nodes = total_nodes if total_nodes is not None else 6
        
        # 确定智能体类型
        if agent_index < self.num_trucks:
            self.agent_type = 'truck'
            # 卡车的动作空间列表
            self.my_act_space_list = [
                type('Discrete', (), {'n': self.total_nodes})(),  # truck_target_node
                type('Discrete', (), {'n': 2})()                  # truck_wait
            ]
        else:
            self.agent_type = 'drone'
            # 无人机的动作空间列表
            self.my_act_space_list = [
                type('Discrete', (), {'n': self.total_nodes})(),     # drone_service_node
                type('Discrete', (), {'n': self.num_trucks})(),      # drone_rendezvous_truck
                type('Discrete', (), {'n': 2})()                     # drone_continue
            ]
        
        # 获取观测维度
        self.obs_dim = obs_shape_n[agent_index][0] if isinstance(
            obs_shape_n[agent_index], tuple
        ) else obs_shape_n[agent_index]
        
        print(f"[MA2C] Agent {agent_index} ({self.agent_type}): "
            f"obs_dim={self.obs_dim}, action_heads={len(self.my_act_space_list)}")
        
        # 保存全局观测维度，供本类其它地方使用
        self.global_obs_dim = (
            global_obs_dim
            if global_obs_dim is not None
            else getattr(args, 'global_obs_dim', None)
        )

        if self.global_obs_dim is None:
            raise ValueError(
                f"[MA2C] global_obs_dim 未提供。请在创建 MA2CAgentTrainer 时传入 "
                f"global_obs_dim，或在 args 里配置 global_obs_dim。"
            )
        
        # 创建策略网络 - 传入正确的动作空间列表
        self.policy = MA2CPolicy(
            obs_dim=self.obs_dim,
            act_space_list=self.my_act_space_list,  # 使用智能体特定的动作空间
            global_obs_dim=self.global_obs_dim,  # ← 用已校验的成员
            n_lstm=args.num_units,
            name=f'agent_{agent_index}'
        )
        
        
        # === 先做一次 dummy 前向，确保变量已创建 ===
        _dummy_local  = tf.zeros([1, 1, self.obs_dim], dtype=tf.float32)
        _dummy_global = tf.zeros([1, 1, self.global_obs_dim], dtype=tf.float32)
        _ = self.policy(_dummy_local, _dummy_global, training=False)

        # === 优化器（从 config/args 读取，若无则给默认） ===
        self.actor_opt  = tf.keras.optimizers.Adam(learning_rate=getattr(self.args, 'lr_actor', 3e-4))
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=getattr(self.args, 'lr_critic', 1e-3))

        # === 变量分组（按层名前缀） ===
        self.actor_vars  = [v for v in self.policy.trainable_variables if 'actor_'  in v.name]
        self.critic_vars = [v for v in self.policy.trainable_variables if 'critic_' in v.name]

        # 兜底：若由于任何原因变量分组为空，则此处重新按名字抓取一次
        if not self.actor_vars or not self.critic_vars:
            print(f"[MA2C][Warn] 变量分组为空，请检查层命名是否以 actor_/critic_ 开头。"
                f"actor_vars={len(self.actor_vars)}, critic_vars={len(self.critic_vars)}")

        # === 分段更新的 rollout 长度 ===
        self.rollout_len = getattr(self.args, 'rollout_len')

        # 用于 bootstrap 的“下一状态”缓存（分段更新时需要）
        self._bootstrap_obs = None
        self._bootstrap_global_obs = None

        # # 优化器
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_actor)
        
        # On-policy缓冲区
        self.buffer = OnPolicyBuffer(gamma=args.gamma)
        
        # 训练参数
        self.entropy_coef = getattr(args, 'entropy_coef', 0.01)
        self.value_coef = getattr(args, 'value_coef', 0.5)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 0.5)

        # ✅ 新增：缓存最近一次action()的输出
        self._last_log_probs = None
        self._last_value = None
        self._last_mask = None
        self._last_global_obs = None
    
    
    def action(self, obs, global_obs, mask=None):
        """
        选择动作（MA2C 专用，不再兼容 MADDPG 的无 global_obs 接口）
        - 采样动作时，Critic 就吃到全局观测
        - 缓存 log_probs / value / mask / global_obs，供 experience() 使用
        """
        obs_tensor         = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)         # (1, obs_dim)
        global_obs_tensor  = tf.convert_to_tensor(global_obs[None, :], dtype=tf.float32)  # (1, gdim)

        # 带上 global_obs 前向，Critic 用全局信息
        logits_list, value_vec = self.policy(
            obs_tensor,
            global_obs=global_obs_tensor,
            training=True
        )
        # value_vec 现在形状是 (batch,) -> 这里 batch=1
        self._last_value = _to_scalar(value_vec)
        
        # === IAM 掩码（只对第 0 个 head：节点类动作）===
        masked_logits_list = []
        if self.args.use_iam and mask is not None:
            for i, logits in enumerate(logits_list):
                if i == 0:
                    mask_tensor = tf.convert_to_tensor(mask[None, :], dtype=tf.bool)
                    masked_logits = tf.where(
                        mask_tensor, 
                        logits,
                        tf.fill(tf.shape(logits), -1e9)
                    )
                    masked_logits_list.append(masked_logits)
                else:
                    masked_logits_list.append(logits)
        else:
            masked_logits_list = logits_list
        
        # 采样动作 + 计算本次对数概率
        actions = []
        log_probs = []
        
        for masked_logits in masked_logits_list:
            # masked_logits: (1, n_i)
            a = tf.random.categorical(masked_logits, 1)[0, 0]       # Tensor()
            actions.append(int(a.numpy()))
            lp = tf.nn.log_softmax(masked_logits)[0, a]
            log_probs.append(float(lp.numpy()))
        
        # 缓存给 experience() / update() 用
        self._last_log_probs  = log_probs
        self._last_mask       = mask
        self._last_global_obs = global_obs
        
        return actions
    
    def experience(self, obs, global_obs, act_all, rew,
               next_obs, next_obs_all, next_global_obs, done):
        """
        存储一条 on-policy 经验
        - my_action: 当前智能体的多头离散动作
        - log_probs: 来自 action() 的缓存
        - value:     优先使用 action() 时缓存的 Critic 估值；若没有则现算
        """
        my_action = act_all[self.agent_index] if isinstance(act_all, list) else act_all

        # ✅ 使用缓存的log_probs和mask
        log_probs = self._last_log_probs
        mask = self._last_mask
        
        # 优先用 action() 里已经计算好的 value
        if self._last_value is not None:
            value_scalar = float(self._last_value)
        else:
            # 兜底再算一次（带上全局观）
            obs_tensor        = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            global_obs_tensor = tf.convert_to_tensor(global_obs[None, :], dtype=tf.float32)
            _, value_vec = self.policy(obs_tensor, global_obs_tensor, training=False)
            value_scalar = _to_scalar(value_vec)
        
        # 存入 buffer（包含全局观）
        self.buffer.add(obs, global_obs, my_action, log_probs, value_scalar, rew, done, mask)
 
        # ✅ 在这里追加这两行（用于分段回合的 bootstrap）
        self._bootstrap_obs = next_obs
        self._bootstrap_global_obs = next_global_obs

        # ✅ 回合末：只重置 LSTM，不清空 buffer（留给 update() 在回合末补一次更新）
        if done:
            self.policy.reset_states()
            # self.buffer.reset()
            self._bootstrap_obs = None
            self._bootstrap_global_obs = None

        # 清理缓存（可选）
        self._last_log_probs = None
        self._last_value = None
        self._last_mask = None

    def preupdate(self):
        """预更新 - MA2C不需要"""
        pass

    def update(self, agents, t, **kwargs):
        """更新策略 - Actor/ Critic 各自用独立的梯度带，且都在带里前向（training=True）"""
        """
        更新策略（MA2C/A2C）
        触发条件：到 K 步（rollout_len） 或 回合末（done=True）
        - 分段（到 K 步）：bootstrap 终值 V(s_T+1)
        - 回合末（done）：终值 = 0
        """
        # —— 判定更新触发条件 —— 
        ready_by_done = bool(self.buffer.dones) and bool(self.buffer.dones[-1])
        ready_by_k    = (self.rollout_len is not None) and (len(self.buffer) >= self.rollout_len)
        if not (ready_by_done or ready_by_k):
            return None
        
        # —— 训练时，不要使用在线 LSTM 状态，也不要改写它 —— 
        saved_state = self.policy.actor_lstm_state
        self.policy.actor_lstm_state = None

        try:
            # ====== 终值 ======
            if ready_by_done:
                # 回合末：目标不再 bootstrap，直接用 0（Python float）
                final_value = 0.0
            else:
                # 分段：用缓存的 next state 做 bootstrap；若无缓存就退化用最后一帧
                # ====== 终值估计（bootstrap）======
                if self._bootstrap_obs is not None and self._bootstrap_global_obs is not None:
                    # 分段回合：用缓存的 next state 做 bootstrap
                    final_obs = self._bootstrap_obs
                    final_global_obs = self._bootstrap_global_obs
                else:
                    # 完整回合：用 buffer 最后一个观测
                    final_obs = self.buffer.obs[-1]
                    final_global_obs = self.buffer.global_obs[-1]

                obs_tensor_f = tf.convert_to_tensor(final_obs[None, :], dtype=tf.float32)
                glo_tensor_f = tf.convert_to_tensor(final_global_obs[None, :], dtype=tf.float32)
                # critic 前向拿到标量 V(s_T)
                _, final_value_vec = self.policy(obs_tensor_f, glo_tensor_f, training=False)
                # 始终转成 Python 标量（Tensor/ndarray 都可）
                final_value = _to_scalar(final_value_vec)
                # final_value = tf.reshape(final_value_vec, [-1])[0]  # (,) 张量

            # ====== 打包 trajectory =======
            obs_batch, global_obs_batch, acts_batch, old_log_probs, returns, advantages, masks = \
                self.buffer.get_trajectory(float(final_value))

            T = len(obs_batch)
            obs_tensor  = tf.convert_to_tensor(obs_batch[None, :, :], dtype=tf.float32)       # (1, T, obs_dim)
            glo_tensor  = tf.convert_to_tensor(global_obs_batch[None, :, :], dtype=tf.float32)# (1, T, gdim)
            acts_tensor = tf.convert_to_tensor(acts_batch, dtype=tf.int32)                    # (T, num_heads)
            ret_tensor  = tf.convert_to_tensor(returns, dtype=tf.float32)                     # (T,)
            adv_tensor  = tf.convert_to_tensor(advantages, dtype=tf.float32)                  # (T,)

            # ====== Critic 更新：单独带 ======
            with tf.GradientTape() as critic_tape:
                critic_tape.watch(self.critic_vars)
                # 只用 Critic 分支（training=True 保证在图里）
                _, values = self.policy(obs_tensor, glo_tensor, training=False)  # (1, T) 或 (T,)
                if len(values.shape) > 1:
                    values = tf.squeeze(values, axis=0)  # -> (T,)
                critic_loss = tf.reduce_mean(tf.square(tf.stop_gradient(ret_tensor) - values))

            critic_grads = critic_tape.gradient(critic_loss, self.critic_vars)
            critic_grads = [
                tf.clip_by_norm(g, self.max_grad_norm) if g is not None else None
                for g in critic_grads
            ]
            self.critic_opt.apply_gradients(zip(critic_grads, self.critic_vars))

            # ====== Actor 更新：单独带 ======
            with tf.GradientTape() as actor_tape:
                actor_tape.watch(self.actor_vars)
                logits_list, _ = self.policy(obs_tensor, glo_tensor, training=False)  # 只用 Actor 分支
                # logits_list: 每个 head 是 (1, T, action_dim)
                actor_losses  = []
                entropy_terms = []

                for head_idx, logits in enumerate(logits_list):
                    # 统一 logits 形状为 (T, A)
                    if len(logits.shape) == 3:
                        # (B, T, A) -> (T, A)
                        logits = tf.squeeze(logits, axis=0)
                    elif len(logits.shape) == 2:
                        # (B, A) 说明 seq_len==1 被 squeeze 过：取 batch 的第 0 项，再补一个 T 维
                        logits = tf.expand_dims(logits[0], axis=0)  # -> (1, A)
                    else:
                        # 兜底：把任何奇怪形状攒成 (1, A)
                        logits = tf.reshape(logits, [1, -1])
                    # # (T, A)
                    # logits = tf.squeeze(logits, axis=0)

                    # 掩码（仅 head 0: 节点选择）
                    if self.args.use_iam and head_idx == 0:
                        masked_logits_per_t = []
                        # 计算“动作有被 mask 禁用”的步数，便于调试
                        invalid_pick_count = 0
                        for t_step in range(T):
                            step_mask = masks[t_step]
                            if step_mask is not None:
                                m = tf.convert_to_tensor(step_mask, dtype=tf.bool)
                                # 若全 False，则降级为全 True（避免梯度全常数）
                                if not tf.reduce_any(m):
                                    m = tf.ones_like(m, dtype=tf.bool)
                                # 若该步选择的动作被禁用，记录一下（用于 debug 打印）
                                if acts_tensor[t_step, head_idx] >= 0:
                                    if not bool(m[acts_tensor[t_step, head_idx]].numpy()):
                                        invalid_pick_count += 1

                                masked = tf.where(m, logits[t_step],
                                                tf.fill(tf.shape(logits[t_step]), tf.constant(-1e9, tf.float32)))
                            else:
                                masked = logits[t_step]
                            masked_logits_per_t.append(masked)
                        logits = tf.stack(masked_logits_per_t, axis=0)
                        # （可选）调试：看下动作是否经常落在 mask=False 上
                        if invalid_pick_count > 0:
                            print(f"[MA2C][agent {self.agent_index}] head0 masked-pick count: {invalid_pick_count}/{T}")

                    # log π(a|s) —— 现在 logits 一定是 (T, A)
                    logp_all = tf.nn.log_softmax(logits, axis=-1)            # (T, A)
                    idx = tf.stack([tf.range(T, dtype=tf.int32), acts_tensor[:, head_idx]], axis=1)
                    new_logp = tf.gather_nd(logp_all, idx)                   # (T,)
                    
                    # policy gradient loss（adv 视作常数权重）
                    actor_loss = -tf.reduce_mean(new_logp * tf.stop_gradient(adv_tensor))
                    actor_losses.append(actor_loss)

                    # 熵正则
                    probs = tf.nn.softmax(logits, axis=-1)
                    entropy = -tf.reduce_sum(probs * logp_all, axis=-1)      # (T,)
                    entropy_terms.append(tf.reduce_mean(entropy))

                total_actor_loss = tf.add_n(actor_losses) / float(len(actor_losses))
                total_entropy    = tf.add_n(entropy_terms) / float(len(entropy_terms))
                # 只把熵加进 actor 的 loss（避免把 critic 变成“正则化项”）
                actor_total_loss = total_actor_loss - self.entropy_coef * total_entropy

            actor_grads = actor_tape.gradient(actor_total_loss, self.actor_vars)

            # 若全部是 None，打印提示并跳过
            if all(g is None for g in actor_grads):
                print(f"⚠️ [MA2C] Agent {self.agent_index}: no actor gradients, skip actor update.")
            else:
                actor_grads = [
                    tf.clip_by_norm(g, self.max_grad_norm) if g is not None else None
                    for g in actor_grads
                ]
                self.actor_opt.apply_gradients(zip(actor_grads, self.actor_vars))

            # 清空 buffer / LSTM 状态 / bootstrap 缓存
            self.buffer.reset()
            # self.policy.reset_states()
            self._bootstrap_obs = None
            self._bootstrap_global_obs = None
            if ready_by_done:
                # 回合已经结束：在线推理用的 LSTM 在 experience 里已 reset，这里再次确保
                self.policy.reset_states()

            print(f"  [MA2C] Agent {self.agent_index} - Actor Loss: {float(total_actor_loss.numpy()):.6f}, "
                f"Critic Loss: {float(critic_loss.numpy()):.6f}, Entropy: {float(total_entropy.numpy()):.6f}")

            return critic_loss, total_actor_loss
        
        finally:
            # 恢复在线决策用的 LSTM 状态（保持回合内记忆连续）
            self.policy.actor_lstm_state = saved_state


    
    # def update(self, agents, t, **kwargs):
    #     """更新策略 - Critic使用全局观测"""
    #     if not self.buffer.dones or not self.buffer.dones[-1]:
    #         return None
        
    #     if len(self.buffer.obs) < 2:
    #         return None
        
    #     # 获取最终价值估计（使用全局观测）
    #     final_obs = self.buffer.obs[-1]
    #     final_global_obs = self.buffer.global_obs[-1]  # ✅ 获取最终全局观测
        
    #     obs_tensor = tf.convert_to_tensor(final_obs[None, :], dtype=tf.float32)
    #     global_obs_tensor = tf.convert_to_tensor(final_global_obs[None, :], dtype=tf.float32)
        
    #     _, final_value = self.policy(obs_tensor, global_obs_tensor, training=False)
    #     final_value = _to_scalar(final_value)   # 统一用工具函数，避免 shape 雷
        
    #     # ✅ 获取轨迹数据（包含全局观测）
    #     obs_batch, global_obs_batch, acts_batch, old_log_probs, returns, advantages, masks = \
    #         self.buffer.get_trajectory(final_value)
        
    #     T = len(obs_batch)
        
    #     # 转换为张量
    #     obs_tensor = tf.convert_to_tensor(obs_batch[None, :, :], dtype=tf.float32)  # (1, T, obs_dim)
    #     global_obs_tensor = tf.convert_to_tensor(global_obs_batch[None, :, :], dtype=tf.float32)  # (1, T, global_obs_dim)
    #     acts_tensor = tf.convert_to_tensor(acts_batch, dtype=tf.int32)
    #     returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    #     advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
    #     # 计算损失和更新
    #     with tf.GradientTape() as tape:
    #         # ✅ 前向传播 - Actor用局部，Critic用全局
    #         logits_list, values = self.policy(
    #             obs_tensor, 
    #             global_obs_tensor,  # ✅ 传入全局观测序列
    #             training=False
    #         )
            
    #         # 计算每个头的Actor损失
    #         actor_losses = []
    #         entropy_losses = []
            
    #         for head_idx, logits in enumerate(logits_list):
    #             logits = tf.squeeze(logits, axis=0)  # (T, action_dim)
    #             actions_head = acts_tensor[:, head_idx]
                
    #             # 应用掩码（如果存在）
    #             if self.args.use_iam and head_idx == 0:
    #                 masked_logits_list = []
    #                 for t_step in range(T):
    #                     step_mask = masks[t_step]
    #                     if step_mask is not None:
    #                         mask_tensor = tf.convert_to_tensor(step_mask, dtype=tf.bool)
    #                         masked_logits = tf.where(
    #                             mask_tensor,
    #                             logits[t_step],
    #                             tf.fill(tf.shape(logits[t_step]), -1e9)
    #                         )
    #                     else:
    #                         masked_logits = logits[t_step]
    #                     masked_logits_list.append(masked_logits)
    #                 logits = tf.stack(masked_logits_list, axis=0)
                
    #             # 计算新的log probabilities
    #             log_probs_all = tf.nn.log_softmax(logits, axis=-1)
    #             action_masks = tf.one_hot(actions_head, logits.shape[-1])
    #             new_log_probs = tf.reduce_sum(action_masks * log_probs_all, axis=-1)
                
    #             # Policy gradient loss
    #             actor_loss = -tf.reduce_mean(new_log_probs * advantages_tensor)
    #             actor_losses.append(actor_loss)
                
    #             # Entropy正则
    #             probs = tf.nn.softmax(logits, axis=-1)
    #             entropy = -tf.reduce_sum(probs * log_probs_all, axis=-1)
    #             entropy_loss = -tf.reduce_mean(entropy)
    #             entropy_losses.append(entropy_loss)
            
    #         total_actor_loss = tf.reduce_mean(actor_losses)
    #         total_entropy_loss = tf.reduce_mean(entropy_losses)
            
    #         # ✅ Critic损失（基于全局观测的价值）
    #         if len(values.shape) > 1:
    #             values = tf.squeeze(values, axis=0)  # (T,)
    #         critic_loss = tf.reduce_mean(tf.square(returns_tensor - values))
            
    #         # 总损失
    #         total_loss = (total_actor_loss + 
    #                     self.value_coef * critic_loss + 
    #                     self.entropy_coef * total_entropy_loss)
        
    #     # 计算梯度并更新
    #     gradients = tape.gradient(total_loss, self.policy.trainable_variables)
    #     gradients = [tf.clip_by_norm(g, self.max_grad_norm) if g is not None else g 
    #                 for g in gradients]
    #     self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        
    #     # 清空buffer和重置LSTM状态
    #     self.buffer.reset()
    #     self.policy.reset_states()
        
    #     print(f"  [MA2C] Agent {self.agent_index} - Actor Loss: {total_actor_loss:.6f}, "
    #         f"Critic Loss: {critic_loss:.6f}, Entropy: {-total_entropy_loss:.6f}")
        
    #     return critic_loss, total_actor_loss

