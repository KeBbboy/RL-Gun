import numpy as np
import tensorflow as tf
from maddpg.trainer.agent_trainer import AgentTrainer

def _to_scalar(x):
    """统一把可能的 Tensor / ndarray / python 数转换成 float 标量"""
    if x is None:
        return None
    if isinstance(x, tf.Tensor):
        x = tf.reshape(x, [-1])
        return float(x[0].numpy())
    else:
        x = np.asarray(x).reshape(-1)
        return float(x[0])


class RolloutBuffer:
    """
    MAPPO专用的on-policy经验缓冲区
    关键特性:
    1. 存储完整轨迹用于GAE计算
    2. 支持集中式价值函数(centralized critic)
    3. 按episode分段存储
    """
    def __init__(self, gamma=0.95, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def __len__(self):
        return len(self.obs)
    
    def reset(self):
        """重置buffer"""
        self.obs = []           # 局部观测 (for actor)
        self.global_obs = []    # 全局观测 (for critic)
        self.acts = []          # 多头动作
        self.log_probs = []     # 动作对数概率
        self.rewards = []       # 奖励
        self.values = []        # 价值估计
        self.dones = []         # 终止标志
        self.masks = []         # IAM掩码
    
    def add(self, obs, global_obs, acts, log_probs, value, reward, done, mask=None):
        """存储一步经验"""
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.acts.append(acts)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)
    
    def compute_returns_and_advantages(self, final_value):
        """
        使用GAE计算优势函数和回报
        公式来自文档1:
        - δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        - A_t^GAE = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
        - R_t = A_t + V(s_t)
        """
        returns = []
        advantages = []
        gae = 0
        
        # 从后往前计算
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = final_value
                next_done = self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            
            # TD误差: δ_t = r_t + γ*V(s_{t+1})*(1-done) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value * (1 - next_done) - self.values[t]
            
            # GAE累积: A_t = δ_t + γλ*A_{t+1}*(1-done)
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - next_done)
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        # 转换为numpy数组
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # 优势标准化(重要!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def get_trajectory(self):
        """获取完整轨迹数据"""
        obs = np.array(self.obs, dtype=np.float32)
        global_obs = np.array(self.global_obs, dtype=np.float32)
        acts = np.array(self.acts, dtype=np.int32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        masks = self.masks
        
        return obs, global_obs, acts, log_probs, masks


class MAPPOPolicy(tf.keras.Model):
    """
    MAPPO策略网络
    关键设计:
    1. Actor使用局部观测(decentralized execution)
    2. Critic使用全局观测(centralized training)
    3. 支持多头离散动作空间
    """
    def __init__(self, obs_dim, act_space_list, global_obs_dim, hidden_units_actor=[256, 128],hidden_units_critic=[1024, 512, 256], name='mappo_policy'):
        super().__init__(name=name)
        
        self.act_space_list = act_space_list
        self.num_heads = len(act_space_list)
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        
        # ===== Actor网络 (局部观测) =====
        self.actor_fc1 = tf.keras.layers.Dense(hidden_units_actor[0], activation='relu', name='actor_fc1')
        self.actor_fc2 = tf.keras.layers.Dense(hidden_units_actor[1], activation='relu', name='actor_fc2')

        # 多头输出层
        self.actor_heads = []
        for i, space in enumerate(act_space_list):
            head = tf.keras.layers.Dense(space.n, activation=None, name=f'actor_head_{i}')
            self.actor_heads.append(head)
        
        # ===== Critic网络 (全局观测，可配置多层) =====
        self.critic_fcs = []
        for i, h in enumerate(hidden_units_critic, start=1):
            self.critic_fcs.append(
                tf.keras.layers.Dense(h, activation='relu', name=f'critic_fc{i}')
            )
        self.critic_head = tf.keras.layers.Dense(1, name='critic')
    
    def call(self, local_obs, global_obs=None, training=False):
        """
        前向传播
        Args:
            local_obs: 局部观测 (batch, obs_dim) - 用于Actor
            global_obs: 全局观测 (batch, global_obs_dim) - 用于Critic
        Returns:
            logits_list: 各头的logits
            value: 状态价值估计
        """
        # ===== Actor前向 (局部观测) =====
        if len(local_obs.shape) == 1:
            local_obs = tf.expand_dims(local_obs, axis=0)
        
        x = self.actor_fc1(local_obs)
        x = self.actor_fc2(x)
        
        # 多头logits
        logits_list = []
        for head in self.actor_heads:
            logits = head(x)            # 形状始终保持 [B, A_h]
            logits_list.append(logits)
        
        # ===== Critic前向 (全局观测) =====
        if global_obs is not None:
            if len(global_obs.shape) == 1:
                global_obs = tf.expand_dims(global_obs, axis=0)

            v = global_obs
            for layer in self.critic_fcs:
                v = layer(v)
            v = self.critic_head(v)
            value = tf.squeeze(v, axis=-1)  # (batch,)
        else:
            # 兜底:无全局观测时用Actor特征估值（仅为防御用，训练应始终提供global_obs）
            v = self.critic_head(x)
            value = tf.squeeze(v, axis=-1)
        
        return logits_list, value


class MAPPOAgentTrainer(AgentTrainer):
    """
    MAPPO智能体训练器
    实现要点:
    1. PPO的clip目标函数 (文档1公式)
    2. GAE优势估计
    3. 集中式Critic + 分散式Actor
    4. 多头离散动作处理
    """
    
    def __init__(self, name, obs_shape_n, act_space_list, agent_index, args, 
                 global_obs_dim=None, num_trucks=None, num_drones=None, total_nodes=None):
        self.name = name
        self.agent_index = agent_index
        self.args = args
        
        # 环境参数
        self.num_trucks = num_trucks if num_trucks is not None else 3
        self.num_drones = num_drones if num_drones is not None else 5
        self.total_nodes = total_nodes if total_nodes is not None else 6
        
        # 确定智能体类型和动作空间
        if agent_index < self.num_trucks:
            self.agent_type = 'truck'
            self.my_act_space_list = [
                type('Discrete', (), {'n': self.total_nodes})(),  # truck_target_node
                type('Discrete', (), {'n': 2})()                  # truck_wait
            ]
        else:
            self.agent_type = 'drone'
            self.my_act_space_list = [
                type('Discrete', (), {'n': self.total_nodes})(),     # drone_service_node
                type('Discrete', (), {'n': self.num_trucks})(),      # drone_rendezvous_truck
                type('Discrete', (), {'n': 2})()                     # drone_continue
            ]
        
        # 获取观测维度
        self.obs_dim = obs_shape_n[agent_index][0] if isinstance(
            obs_shape_n[agent_index], tuple
        ) else obs_shape_n[agent_index]
        
        # 全局观测维度
        self.global_obs_dim = (
            global_obs_dim
            if global_obs_dim is not None
            else getattr(args, 'global_obs_dim', None)
        )
        
        if self.global_obs_dim is None:
            raise ValueError("global_obs_dim must be provided for MAPPO")
        
        print(f"[MAPPO] Agent {agent_index} ({self.agent_type}): "
              f"obs_dim={self.obs_dim}, global_obs_dim={self.global_obs_dim}, "
              f"action_heads={len(self.my_act_space_list)}")
        
        # 创建策略网络
        self.policy = MAPPOPolicy(
            obs_dim=self.obs_dim,
            act_space_list=self.my_act_space_list,
            global_obs_dim=self.global_obs_dim,
            hidden_units_actor=getattr(args, 'hidden_units_actor', [256, 128]),
            hidden_units_critic=getattr(args, 'hidden_units_critic', [1024, 512, 256]),
            name=f'agent_{agent_index}'
        )
        
        # Dummy前向初始化变量
        _dummy_local = tf.zeros([1, self.obs_dim], dtype=tf.float32)
        _dummy_global = tf.zeros([1, self.global_obs_dim], dtype=tf.float32)
        _ = self.policy(_dummy_local, _dummy_global, training=False)
        
        # 优化器 (从config读取)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=getattr(self.args, 'lr_actor', 3e-4))
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=getattr(self.args, 'lr_critic', 1e-3))
        
        # 变量分组
        self.actor_vars = [v for v in self.policy.trainable_variables if 'actor_' in v.name]
        self.critic_vars = [v for v in self.policy.trainable_variables if 'critic_' in v.name]
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            gamma=getattr(args, 'gamma', 0.95),
            gae_lambda=getattr(args, 'gae_lambda', 0.95)
        )
        
        # 训练参数
        self.clip_param = getattr(args, 'clip_param', 0.2)  # PPO的ε
        self.entropy_coef = getattr(args, 'entropy_coef', 0.01)
        self.value_coef = getattr(args, 'value_coef', 0.5)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 0.5)
        
        # 缓存最近一次action()的输出
        self._last_log_probs = None
        self._last_value = None
        self._last_mask = None
        self._last_global_obs = None
    
    def action(self, obs, global_obs, mask=None):
        """
        选择动作 (MAPPO专用,需要全局观测)
        - 采样动作时,Critic就吃到全局观测
        - 缓存log_probs/value/mask/global_obs供experience()使用
        """
        obs_tensor = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        global_obs_tensor = tf.convert_to_tensor(global_obs[None, :], dtype=tf.float32)
        
        # 前向传播 (带全局观测)
        logits_list, value_vec = self.policy(
            obs_tensor,
            global_obs=global_obs_tensor,
            training=True
        )
        self._last_value = _to_scalar(value_vec)
        
        # === IAM掩码 (只对第0个head:节点类动作) ===
        masked_logits_list = []
        if self.args.use_iam and mask is not None:
            for i, logits in enumerate(logits_list):
                if i == 0:  # 节点选择head
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
        
        # 采样动作 + 计算log概率
        actions = []
        log_probs = []
        
        for masked_logits in masked_logits_list:
            # masked_logits: (1, n_i)
            a = tf.random.categorical(masked_logits, 1)[0, 0]
            actions.append(int(a.numpy()))
            
            lp = tf.nn.log_softmax(masked_logits)[0, a]
            log_probs.append(float(lp.numpy()))
        
        # 缓存给experience()用
        self._last_log_probs = log_probs
        self._last_mask = mask
        self._last_global_obs = global_obs
        
        return actions
    
    def experience(self, obs, global_obs, act_all, rew, 
                   next_obs, next_obs_all, next_global_obs, done):
        """
        存储一条on-policy经验
        Args:
            obs: 当前智能体局部观测
            global_obs: 全局观测
            act_all: 所有智能体动作 (本智能体的多头动作)
            rew: 奖励
            next_obs: 下一步局部观测
            next_obs_all: 所有智能体下一步观测 (MAPPO不需要)
            next_global_obs: 下一步全局观测
            done: 终止标志
        """
        my_action = act_all[self.agent_index] if isinstance(act_all, list) else act_all
        
        # 使用缓存的log_probs和value
        log_probs = self._last_log_probs
        mask = self._last_mask
        
        # 优先用action()里已计算好的value
        if self._last_value is not None:
            value_scalar = float(self._last_value)
        else:
            # 兜底再算一次
            obs_tensor = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            global_obs_tensor = tf.convert_to_tensor(global_obs[None, :], dtype=tf.float32)
            _, value_vec = self.policy(obs_tensor, global_obs_tensor, training=False)
            value_scalar = _to_scalar(value_vec)
        
        # 存入buffer (包含全局观)
        self.buffer.add(obs, global_obs, my_action, log_probs, value_scalar, rew, done, mask)
        
        # 清理缓存
        self._last_log_probs = None
        self._last_value = None
        self._last_mask = None
    
    def preupdate(self):
        """预更新 - MAPPO不需要"""
        pass
    
    # ---- 多 epoch + 小批训练 ----
    def update(self, agents, t, **kwargs):
        """
        更新策略 - PPO的clip目标 + GAE
        新增: 多 epoch + 小批训练
        """
        # 触发条件：按K步（rollout_len）收集
        rollout_len_cfg = getattr(self.args, 'rollout_len', 256)
        rollout_len = 256 if (rollout_len_cfg is None or int(rollout_len_cfg) <= 0) else int(rollout_len_cfg)
        if len(self.buffer) < rollout_len:
            return None

        # ====== 终值估计 (bootstrap) ======
        final_obs = self.buffer.obs[-1]
        final_global_obs = self.buffer.global_obs[-1]
        obs_tensor_f = tf.convert_to_tensor(final_obs[None, :], dtype=tf.float32)
        glo_tensor_f = tf.convert_to_tensor(final_global_obs[None, :], dtype=tf.float32)
        _, final_value_vec = self.policy(obs_tensor_f, glo_tensor_f, training=False)
        final_value = float(final_value_vec.numpy()[0])

        # ====== 计算GAE和回报 ======
        returns, advantages = self.buffer.compute_returns_and_advantages(final_value)
        print("  · advantages: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
            float(np.mean(advantages)), float(np.std(advantages) + 1e-8),
            float(np.min(advantages)), float(np.max(advantages))
        ))
        print("  · returns:    mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
            float(np.mean(returns)), float(np.std(returns) + 1e-8),
            float(np.min(returns)), float(np.max(returns))
        ))

        # ====== 打包整段轨迹 ======
        obs_batch, global_obs_batch, acts_batch, old_log_probs, masks = self.buffer.get_trajectory()

        T = len(obs_batch)
        obs_tensor = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        glo_tensor = tf.convert_to_tensor(global_obs_batch, dtype=tf.float32)
        acts_tensor = tf.convert_to_tensor(acts_batch, dtype=tf.int32)
        ret_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        adv_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        old_lp_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)

        # ====== 多 epoch + 小批训练 ======
        ppo_epochs = getattr(self.args, 'ppo_epochs', 5)
        minibatch_size = getattr(self.args, 'minibatch_size', 128)

        num_minibatches_per_epoch = max(1, int(np.ceil(T / float(minibatch_size))))
        total_actor_loss_acc = 0.0
        total_critic_loss_acc = 0.0
        total_entropy_acc = 0.0
        nb_updates = 0

        idx_all = np.arange(T)
        for epoch in range(ppo_epochs):
            np.random.shuffle(idx_all)
            for mb_start in range(0, T, minibatch_size):
                mb_idx_np = idx_all[mb_start: mb_start + minibatch_size]
                if mb_idx_np.size == 0:
                    continue

                # 取小批
                obs_mb = tf.gather(obs_tensor, mb_idx_np)
                glo_mb = tf.gather(glo_tensor, mb_idx_np)
                acts_mb = tf.gather(acts_tensor, mb_idx_np)
                ret_mb = tf.gather(ret_tensor, mb_idx_np)
                adv_mb = tf.gather(adv_tensor, mb_idx_np)
                old_lp_mb = tf.gather(old_lp_tensor, mb_idx_np)
                masks_mb = [masks[i] for i in mb_idx_np]
                T_mb = obs_mb.shape[0]

                # ----- Critic 更新 -----
                with tf.GradientTape() as critic_tape:
                    critic_tape.watch(self.critic_vars)
                    _, values_mb = self.policy(obs_mb, glo_mb, training=True)   # V(s)
                    critic_loss = 0.5 * tf.reduce_mean(tf.square(ret_mb - values_mb))
                critic_grads = critic_tape.gradient(critic_loss, self.critic_vars)
                critic_grads = [tf.clip_by_norm(g, self.max_grad_norm) if g is not None else None
                                for g in critic_grads]
                self.critic_opt.apply_gradients(zip(critic_grads, self.critic_vars))

                # ----- Actor 更新 -----
                with tf.GradientTape() as actor_tape:
                    actor_tape.watch(self.actor_vars)
                    logits_list, _ = self.policy(obs_mb, glo_mb, training=True)

                    actor_losses = []
                    entropy_terms = []

                    for head_idx, logits in enumerate(logits_list):
                        # IAM掩码（仅 head0）
                        if self.args.use_iam and head_idx == 0:
                            masked_logits_per_t = []
                            for t_step in range(T_mb):
                                step_mask = masks_mb[t_step]
                                if step_mask is not None:
                                    m = tf.convert_to_tensor(step_mask, dtype=tf.bool)
                                    if not tf.reduce_any(m):
                                        m = tf.ones_like(m, dtype=tf.bool)
                                    masked = tf.where(m, logits[t_step],
                                                    tf.fill(tf.shape(logits[t_step]), tf.constant(-1e9, tf.float32)))
                                else:
                                    masked = logits[t_step]
                                masked_logits_per_t.append(masked)
                            logits = tf.stack(masked_logits_per_t, axis=0)

                        logp_all = tf.nn.log_softmax(logits, axis=-1)
                        gather_idx = tf.stack([tf.range(T_mb, dtype=tf.int32), acts_mb[:, head_idx]], axis=1)
                        new_logp = tf.gather_nd(logp_all, gather_idx)

                        ratio = tf.exp(new_logp - old_lp_mb[:, head_idx])
                        surr1 = ratio * adv_mb
                        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_mb
                        actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                        actor_losses.append(actor_loss)

                        # 熵正则
                        probs = tf.nn.softmax(logits, axis=-1)
                        entropy = -tf.reduce_sum(probs * logp_all, axis=-1)
                        entropy_terms.append(tf.reduce_mean(entropy))

                    total_actor_loss = tf.add_n(actor_losses) / float(len(actor_losses))
                    total_entropy = tf.add_n(entropy_terms) / float(len(entropy_terms))
                    actor_total_loss = total_actor_loss - self.entropy_coef * total_entropy

                actor_grads = actor_tape.gradient(actor_total_loss, self.actor_vars)
                if not all(g is None for g in actor_grads):
                    actor_grads = [tf.clip_by_norm(g, self.max_grad_norm) if g is not None else None
                                for g in actor_grads]
                    self.actor_opt.apply_gradients(zip(actor_grads, self.actor_vars))

                # 累计统计
                total_actor_loss_acc += float(total_actor_loss.numpy())
                total_critic_loss_acc += float(critic_loss.numpy())
                total_entropy_acc += float(total_entropy.numpy())
                nb_updates += 1

        # ===== 训练后评估一次 Critic EV（全量）=====
        _, values_full = self.policy(obs_tensor, glo_tensor, training=False)
        target_np = ret_tensor.numpy()
        pred_np = values_full.numpy()
        var_y = np.var(target_np)
        ev = 1.0 - (np.var(target_np - pred_np) / (var_y + 1e-8))
        print(f"  · critic EV (after epochs): {ev:.3f}")

        # 清空buffer
        self.buffer.reset()

        # 计算并打印本轮平均指标
        if nb_updates > 0:
            mean_actor_loss  = total_actor_loss_acc  / nb_updates
            mean_critic_loss = total_critic_loss_acc / nb_updates
            mean_entropy     = total_entropy_acc     / nb_updates

            print(f"  [MAPPO] Agent {self.agent_index} - "
                f"Actor Loss: {mean_actor_loss:.6f}, "
                f"Critic Loss: {mean_critic_loss:.6f}, "
                f"Entropy: {mean_entropy:.6f}")

            # ✅ 返回平均损失，兼容 train.py 的日志记录
            return mean_critic_loss, mean_actor_loss

        # 理论上不会走到这里（因为满足 rollout_len 才会更新），但加个兜底更安全
        return None, None



    # def update(self, agents, t, **kwargs):
    #     """
    #     更新策略 - PPO的clip目标 + GAE
    #     关键公式 (文档1):
    #     1. L_clip = E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)]
    #     2. r_t = π_θ(a|s) / π_old(a|s)
    #     3. L_total = L_clip + c_v*L_value - β*H
    #     """
    #     # # 支持 rollout_len 触发 + 末回合兜底
    #     # rollout_len = getattr(self.args, 'rollout_len', 256)

    #     # ready_by_k = (len(self.buffer) >= rollout_len)
    #     # ready_by_done = (self.buffer.dones and self.buffer.dones[-1])

    #     # if not (ready_by_k or ready_by_done):
    #     #     return None

    #     # ---- 只按 K 步触发（不要求回合末）----
    #     rollout_len_cfg = getattr(self.args, 'rollout_len', 256)
    #     # 强兜底，避免 None/0 覆盖默认值
    #     rollout_len = 256 if (rollout_len_cfg is None or int(rollout_len_cfg) <= 0) else int(rollout_len_cfg)

    #     # 只看 K 步
    #     if len(self.buffer) < rollout_len:
    #         return None
        
    #     # ====== 终值估计 (bootstrap) ======
    #     # 用buffer最后一个观测估计V(s_T)
    #     final_obs = self.buffer.obs[-1]
    #     final_global_obs = self.buffer.global_obs[-1]
        
    #     obs_tensor_f = tf.convert_to_tensor(final_obs[None, :], dtype=tf.float32)
    #     glo_tensor_f = tf.convert_to_tensor(final_global_obs[None, :], dtype=tf.float32)
    #     _, final_value_vec = self.policy(obs_tensor_f, glo_tensor_f, training=False)
    #     final_value = float(final_value_vec.numpy()[0])
        
    #     # ====== 计算GAE和回报 ======
    #     returns, advantages = self.buffer.compute_returns_and_advantages(final_value)
    #     print("  · advantages: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    #         float(np.mean(advantages)), float(np.std(advantages) + 1e-8),
    #         float(np.min(advantages)), float(np.max(advantages))
    #     ))
    #     print("  · returns:    mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    #         float(np.mean(returns)), float(np.std(returns) + 1e-8),
    #         float(np.min(returns)), float(np.max(returns))
    #     ))


    #     # ====== 打包轨迹数据 ======
    #     obs_batch, global_obs_batch, acts_batch, old_log_probs, masks = self.buffer.get_trajectory()
        
    #     T = len(obs_batch)
    #     obs_tensor = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
    #     glo_tensor = tf.convert_to_tensor(global_obs_batch, dtype=tf.float32)
    #     acts_tensor = tf.convert_to_tensor(acts_batch, dtype=tf.int32)
    #     ret_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    #     adv_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
    #     old_lp_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        
    #     # ====== Critic更新 (单独带) ======
    #     with tf.GradientTape() as critic_tape:
    #         critic_tape.watch(self.critic_vars)
    #         _, values = self.policy(obs_tensor, glo_tensor, training=True)
            
    #         # Value loss: MSE
    #         critic_loss = 0.5 * tf.reduce_mean(tf.square(ret_tensor - values))
        
        
    #     critic_grads = critic_tape.gradient(critic_loss, self.critic_vars)
    #     critic_grads = [
    #         tf.clip_by_norm(g, self.max_grad_norm) if g is not None else None
    #         for g in critic_grads
    #     ]
    #     self.critic_opt.apply_gradients(zip(critic_grads, self.critic_vars))
        
    #     # ===== critic 的 explained variance（评估 V 的拟合程度）=====
    #     # EV = 1 - Var(target - pred) / Var(target)；越接近 1 越好，<0 表示比常数基线还差
    #     target_np = ret_tensor.numpy()
    #     pred_np   = values.numpy()
    #     var_y  = np.var(target_np)
    #     ev     = 1.0 - (np.var(target_np - pred_np) / (var_y + 1e-8))
    #     print(f"  · critic EV: {ev:.3f}")
        
    #     # ====== Actor更新 (单独带) ======
    #     with tf.GradientTape() as actor_tape:
    #         actor_tape.watch(self.actor_vars)
    #         logits_list, _ = self.policy(obs_tensor, glo_tensor, training=True)
            
    #         actor_losses = []
    #         entropy_terms = []
            
    #         for head_idx, logits in enumerate(logits_list):
    #             # 应用IAM掩码
    #             if self.args.use_iam and head_idx == 0:
    #                 masked_logits_per_t = []
    #                 for t_step in range(T):
    #                     step_mask = masks[t_step]
    #                     if step_mask is not None:
    #                         m = tf.convert_to_tensor(step_mask, dtype=tf.bool)
    #                         if not tf.reduce_any(m):
    #                             m = tf.ones_like(m, dtype=tf.bool)
    #                         masked = tf.where(m, logits[t_step],
    #                                         tf.fill(tf.shape(logits[t_step]), tf.constant(-1e9, tf.float32)))
    #                     else:
    #                         masked = logits[t_step]
    #                     masked_logits_per_t.append(masked)
    #                 logits = tf.stack(masked_logits_per_t, axis=0)
                
    #             # 计算新的log π(a|s)
    #             logp_all = tf.nn.log_softmax(logits, axis=-1)
    #             idx = tf.stack([tf.range(T, dtype=tf.int32), acts_tensor[:, head_idx]], axis=1)
    #             new_logp = tf.gather_nd(logp_all, idx)
                
    #             # 计算比率 r_t = π_new / π_old
    #             ratio = tf.exp(new_logp - old_lp_tensor[:, head_idx])
                
    #             # PPO clip目标: L = min(r*A, clip(r, 1-ε, 1+ε)*A)
    #             surr1 = ratio * adv_tensor
    #             surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_tensor
    #             actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
    #             actor_losses.append(actor_loss)
                
    #             # —— 每个 head 的 PPO 指标
    #             clipped = tf.logical_or(ratio > 1.0 + self.clip_param, ratio < 1.0 - self.clip_param)
    #             clip_frac = float(tf.reduce_mean(tf.cast(clipped, tf.float32)).numpy())
    #             approx_kl = float(tf.reduce_mean(old_lp_tensor[:, head_idx] - new_logp).numpy())
    #             r_mean = float(tf.reduce_mean(ratio).numpy())
    #             r_std  = float(tf.math.reduce_std(ratio).numpy())

    #             print(f"  · head[{head_idx}] clip_frac={clip_frac:.3f}  approx_kl={approx_kl:.4f}  "
    #                 f"ratio(mean±std)={r_mean:.3f}±{r_std:.3f}")
                    
    #             # 熵正则
    #             probs = tf.nn.softmax(logits, axis=-1)
    #             entropy = -tf.reduce_sum(probs * logp_all, axis=-1)
    #             entropy_terms.append(tf.reduce_mean(entropy))
            
    #         total_actor_loss = tf.add_n(actor_losses) / float(len(actor_losses))
    #         total_entropy = tf.add_n(entropy_terms) / float(len(entropy_terms))
            
    #         # 只把熵加进actor的loss
    #         actor_total_loss = total_actor_loss - self.entropy_coef * total_entropy
        
    #     actor_grads = actor_tape.gradient(actor_total_loss, self.actor_vars)
        
    #     if all(g is None for g in actor_grads):
    #         print(f"⚠️ [MAPPO] Agent {self.agent_index}: no actor gradients, skip actor update.")
    #     else:
    #         actor_grads = [
    #             tf.clip_by_norm(g, self.max_grad_norm) if g is not None else None
    #             for g in actor_grads
    #         ]
    #         self.actor_opt.apply_gradients(zip(actor_grads, self.actor_vars))
        
    #     # 清空buffer
    #     self.buffer.reset()
        
    #     print(f"  [MAPPO] Agent {self.agent_index} - Actor Loss: {float(total_actor_loss.numpy()):.6f}, "
    #           f"Critic Loss: {float(critic_loss.numpy()):.6f}, Entropy: {float(total_entropy.numpy()):.6f}")
        
    #     return critic_loss, total_actor_loss