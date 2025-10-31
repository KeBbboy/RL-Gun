import numpy as np
import tensorflow as tf
from src.algorithms.base import AgentTrainer
from src.algorithms.utils.replay_buffer import ReplayBuffer
from src.algorithms.utils.per_buffer import PrioritizedReplayBuffer
from src.algorithms.utils.networks import ActorModel, CriticModel
from src.algorithms.utils.helper import update_target_weights



# 策略网络训练（包含 Softmax + ε-greedy + IAM）
# 卡车策略网络训练器
class TruckPolicyTrainer:
    def __init__(self, obs_dim, total_nodes, args, agent_index):
        # 卡车动作空间：truck_target_node (total_nodes) + truck_wait (2)
        self.act_space_list = [
            type('Discrete', (), {'n': total_nodes})(),  # truck_target_node
            type('Discrete', (), {'n': 2})()             # truck_wait
        ]
        
        self.act_dim = sum(sp.n for sp in self.act_space_list)
        self.model = ActorModel(obs_dim, self.act_space_list, 
                               hidden_units=args.hidden_units_actor, activation=args.activation)
        self.target_model = ActorModel(obs_dim, self.act_space_list, 
                                      hidden_units=args.hidden_units_actor, activation=args.activation)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(args.lr_actor)
        self.agent_index = agent_index
        self.use_iam = args.use_iam
        self.epsilon = args.epsilon
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)

    def act(self, obs, mask=None):
        print(f"\n[TruckPolicyTrainer.act] Agent {self.agent_index} called")
        
        if np.any(np.isnan(obs)):
            print(f"[ERROR] NaN values detected in observation: {obs}")
            obs = np.nan_to_num(obs)
            print(f"[INFO] Replaced NaN values with zeros: {obs}")

        print(f"[TruckPolicyTrainer] Agent {self.agent_index} obs: {obs.shape} -> {obs}")

        # 网络前向传播
        logits_heads = self.model(obs)  # list of 2 tensors: [(1,total_nodes),(1,2)]
        print(f"[TruckPolicyTrainer] Agent {self.agent_index} logits heads:")
        for i, l in enumerate(logits_heads):
            print(f"  Head {i}: logits shape = {l.shape}")

        # IAM 掩码 - 只对第一个head（truck_target_node）应用mask
        if self.use_iam and mask is not None:
            print(f"[TruckPolicyTrainer] Agent {self.agent_index} mask shape: {mask.shape} -> mask = {mask}")
            
            # 只对truck_target_node头应用mask
            before_mask = logits_heads[0].numpy()
            print(f"  [IAM Mask] Head 0 before: {before_mask}")
            
            # 确保mask长度匹配logits的动作维度
            if len(mask) != logits_heads[0].shape[1]:
                print(f"  [IAM Mask] Warning: mask length {len(mask)} != action dim {logits_heads[0].shape[1]}")
                if len(mask) < logits_heads[0].shape[1]:
                    extended_mask = np.concatenate([mask, np.zeros(logits_heads[0].shape[1] - len(mask), dtype=bool)])
                else:
                    extended_mask = mask[:logits_heads[0].shape[1]]
            else:
                extended_mask = mask
            
            # 对无效动作设置很大的负值
            mask_tensor = tf.convert_to_tensor(extended_mask[None, :], dtype=tf.bool)
            very_low = tf.fill(tf.shape(logits_heads[0]), -1e9)
            logits_heads[0] = tf.where(mask_tensor, logits_heads[0], very_low)
            
            print(f"  [IAM Mask] Head 0 after : {logits_heads[0].numpy()}")

        # softmax + ε-greedy + 采样
        actions = []
        for i, logits in enumerate(logits_heads):
            probs = tf.nn.softmax(logits)

            # 对于truck_target_node头，额外禁止选择depot（节点0）当有其他选择时
            if i == 0 and self.use_iam:
                valid_actions = tf.reduce_sum(tf.cast(probs[:, 1:] > 1e-8, tf.int32), axis=1)
                should_avoid_depot = valid_actions > 0

                if should_avoid_depot:
                    probs_no_depot = tf.concat([tf.zeros_like(probs[:, :1]), probs[:, 1:]], axis=1)
                    probs = probs_no_depot / tf.reduce_sum(probs_no_depot, axis=1, keepdims=True)
                    print(f"  [IAM Protection] Head {i} depot probability suppressed")

            if np.random.rand() < self.epsilon:
                # 探索：基于概率采样
                a = tf.random.categorical(tf.math.log(probs + 1e-8), 1)[:, 0]
                print(f"  [Action] Head {i} ε-exploration: {a.numpy()}")
            else:
                a = tf.argmax(probs, axis=1)
                print(f"  [Action] Head {i} greedy: {a.numpy()}")

            # 确保动作在有效范围内
            a = tf.clip_by_value(a, 0, self.act_space_list[i].n - 1)
            
            # 对truck_target_node头进行IAM回退检查
            if i == 0 and self.use_iam and mask is not None:
                if a.numpy()[0] < len(mask) and not mask[a.numpy()[0]]:
                    valid_indices = np.where(mask)[0]
                    if len(valid_indices) > 0:
                        fallback = np.random.choice(valid_indices)
                        print(f"  [IAM Fallback] Head {i} action {a.numpy()[0]} invalid, using {fallback}")
                        a = tf.constant([fallback], dtype=a.dtype)
            
            actions.append(a)
            print(f"  [Action] Head {i} ε-greedy sampled: {a.numpy()}")

        final_action = tf.stack(actions, axis=1)[0].numpy().tolist()
        print(f"[TruckPolicyTrainer] Agent {self.agent_index} Final action: {final_action}")
        return final_action

    def train(self, obs, all_acts, global_obs, critic_fn, mask=None):
        """训练方法"""
        with tf.GradientTape() as tape:
            logits_heads = self.model(obs)
            
            # === 训练时也应用 IAM 于 head0 ===
            if self.use_iam and (mask is not None):
                # mask 形状：(B, total_nodes)
                mask = tf.cast(mask, tf.bool)                   # (B, total_nodes)
                very_low = tf.fill(tf.shape(logits_heads[0]), -1e9)
                logits_heads[0] = tf.where(mask, logits_heads[0], very_low)

            temperature = 1.0
            sampled_actions = []

            for i, logits in enumerate(logits_heads):
                logits = tf.clip_by_value(logits, -20.0, 20.0)
                # 检查logits是否有异常值
                if tf.reduce_any(tf.math.is_nan(logits)) or tf.reduce_any(tf.math.is_inf(logits)):
                    print(f"Agent {self.agent_index}: Invalid logits detected in head {i}")
                    logits = tf.zeros_like(logits)
                gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1) + 1e-8) + 1e-8)
                soft_actions = tf.nn.softmax((logits + gumbel_noise) / temperature)
                hard_actions = tf.one_hot(tf.argmax(soft_actions, axis=1), self.act_space_list[i].n)
                sampled_actions.append(soft_actions + tf.stop_gradient(hard_actions - soft_actions))

            # 将多头动作转换为扁平化one-hot
            agent_action = tf.concat(sampled_actions, axis=1)

            # 构建critic输入
            all_acts_copy = all_acts.copy()
            all_acts_copy[self.agent_index] = agent_action
            critic_input = tf.concat([global_obs] + all_acts_copy, axis=1)

            # 计算Q值和损失
            q_val = critic_fn(critic_input)
            q_val = tf.clip_by_value(q_val, -100.0, 100.0)
            actor_loss = -tf.reduce_mean(q_val)
            
            # 熵正则
            ent_list = []
            for soft_act in sampled_actions:
                ent = -tf.reduce_sum(soft_act * tf.math.log(soft_act + 1e-8), axis=1)
                ent_list.append(tf.reduce_mean(ent))
            avg_entropy = tf.add_n(ent_list) / float(len(ent_list))
            
            total_loss = actor_loss - self.entropy_coef * avg_entropy
            
            if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                print(f"Invalid total loss: {total_loss}, using actor_loss only")
                total_loss = actor_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)

        if grads is None or all(g is None for g in grads):
            print(f"Agent {self.agent_index}: No gradients computed!")
            return actor_loss
        grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return actor_loss

    def update_target(self, tau):
        update_target_weights(self.model.trainable_variables,
                              self.target_model.trainable_variables, tau)


# 无人机策略网络训练器
class DronePolicyTrainer:
    def __init__(self, obs_dim, total_nodes, num_trucks, args, agent_index):
        # 无人机动作空间：drone_service_node (total_nodes) + drone_rendezvous_truck (num_trucks) + drone_continue (2)
        self.act_space_list = [
            type('Discrete', (), {'n': total_nodes})(),  # drone_service_node
            type('Discrete', (), {'n': num_trucks})(),   # drone_rendezvous_truck
            type('Discrete', (), {'n': 2})()             # drone_continue
        ]
        
        self.act_dim = sum(sp.n for sp in self.act_space_list)
        self.model = ActorModel(obs_dim, self.act_space_list, 
                               hidden_units=args.hidden_units_actor, activation=args.activation)
        self.target_model = ActorModel(obs_dim, self.act_space_list, 
                                      hidden_units=args.hidden_units_actor, activation=args.activation)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(args.lr_actor)
        self.agent_index = agent_index
        self.use_iam = args.use_iam
        self.epsilon = args.epsilon
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)

    def act(self, obs, mask=None):
        print(f"\n[DronePolicyTrainer.act] Agent {self.agent_index} called")
        
        if np.any(np.isnan(obs)):
            print(f"[ERROR] NaN values detected in observation: {obs}")
            obs = np.nan_to_num(obs)
            print(f"[INFO] Replaced NaN values with zeros: {obs}")

        print(f"[DronePolicyTrainer] Agent {self.agent_index} obs: {obs.shape} -> {obs}")

        # 网络前向传播
        logits_heads = self.model(obs)  # list of 3 tensors: [(1,total_nodes),(1,num_trucks),(1,2)]
        print(f"[DronePolicyTrainer] Agent {self.agent_index} logits heads:")
        for i, l in enumerate(logits_heads):
            print(f"  Head {i}: logits shape = {l.shape}")

        # IAM 掩码 - 对drone_service_node头应用mask
        if self.use_iam and mask is not None:
            print(f"[DronePolicyTrainer] Agent {self.agent_index} mask shape: {mask.shape} -> mask = {mask}")
            
            # 只对drone_service_node头（第0个）应用mask
            before_mask = logits_heads[0].numpy()
            print(f"  [IAM Mask] Head 0 before: {before_mask}")
            
            if len(mask) != logits_heads[0].shape[1]:
                print(f"  [IAM Mask] Warning: mask length {len(mask)} != action dim {logits_heads[0].shape[1]}")
                if len(mask) < logits_heads[0].shape[1]:
                    extended_mask = np.concatenate([mask, np.zeros(logits_heads[0].shape[1] - len(mask), dtype=bool)])
                else:
                    extended_mask = mask[:logits_heads[0].shape[1]]
            else:
                extended_mask = mask
            
            mask_tensor = tf.convert_to_tensor(extended_mask[None, :], dtype=tf.bool)
            very_low = tf.fill(tf.shape(logits_heads[0]), -1e9)
            logits_heads[0] = tf.where(mask_tensor, logits_heads[0], very_low)
            
            print(f"  [IAM Mask] Head 0 after : {logits_heads[0].numpy()}")

        # softmax + ε-greedy + 采样
        actions = []
        for i, logits in enumerate(logits_heads):
            probs = tf.nn.softmax(logits)

            # 对于drone_service_node头，额外禁止选择depot（节点0）当有其他选择时
            if i == 0 and self.use_iam:
                valid_actions = tf.reduce_sum(tf.cast(probs[:, 1:] > 1e-8, tf.int32), axis=1)
                should_avoid_depot = valid_actions > 0

                if should_avoid_depot:
                    probs_no_depot = tf.concat([tf.zeros_like(probs[:, :1]), probs[:, 1:]], axis=1)
                    probs = probs_no_depot / tf.reduce_sum(probs_no_depot, axis=1, keepdims=True)
                    print(f"  [IAM Protection] Head {i} depot probability suppressed")

            if np.random.rand() < self.epsilon:
                a = tf.random.categorical(tf.math.log(probs + 1e-8), 1)[:, 0]
                print(f"  [Action] Head {i} ε-exploration: {a.numpy()}")
            else:
                a = tf.argmax(probs, axis=1)
                print(f"  [Action] Head {i} greedy: {a.numpy()}")

            a = tf.clip_by_value(a, 0, self.act_space_list[i].n - 1)
            
            # 对drone_service_node头进行IAM回退检查
            if i == 0 and self.use_iam and mask is not None:
                if a.numpy()[0] < len(mask) and not mask[a.numpy()[0]]:
                    valid_indices = np.where(mask)[0]
                    if len(valid_indices) > 0:
                        fallback = np.random.choice(valid_indices)
                        print(f"  [IAM Fallback] Head {i} action {a.numpy()[0]} invalid, using {fallback}")
                        a = tf.constant([fallback], dtype=a.dtype)
            
            actions.append(a)
            print(f"  [Action] Head {i} ε-greedy sampled: {a.numpy()}")

        final_action = tf.stack(actions, axis=1)[0].numpy().tolist()
        print(f"[DronePolicyTrainer] Agent {self.agent_index} Final action: {final_action}")
        return final_action

    def train(self, obs, all_acts, global_obs, critic_fn, mask=None):
        """训练方法（新增：支持 IAM 掩码，作用于 head0：drone_service_node）"""
        with tf.GradientTape() as tape:
            logits_heads = self.model(obs)
            
            # === 训练时也应用 IAM 于 head0（服务节点选择）===
            if self.use_iam and (mask is not None):
                mask = tf.cast(mask, tf.bool)                   # (B, total_nodes)
                very_low = tf.fill(tf.shape(logits_heads[0]), -1e9)
                logits_heads[0] = tf.where(mask, logits_heads[0], very_low)

            temperature = 1.0
            sampled_actions = []

            for i, logits in enumerate(logits_heads):
                logits = tf.clip_by_value(logits, -20.0, 20.0)
                gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1) + 1e-8) + 1e-8)
                soft_actions = tf.nn.softmax((logits + gumbel_noise) / temperature)
                hard_actions = tf.one_hot(tf.argmax(soft_actions, axis=1), self.act_space_list[i].n)
                sampled_actions.append(soft_actions + tf.stop_gradient(hard_actions - soft_actions))

            agent_action = tf.concat(sampled_actions, axis=1)

            all_acts_copy = all_acts.copy()
            all_acts_copy[self.agent_index] = agent_action
            critic_input = tf.concat([global_obs] + all_acts_copy, axis=1)

            q_val = critic_fn(critic_input)
            q_val = tf.clip_by_value(q_val, -100.0, 100.0)
            actor_loss = -tf.reduce_mean(q_val)
            
            # 熵正则
            ent_list = []
            for soft_act in sampled_actions:
                ent = -tf.reduce_sum(soft_act * tf.math.log(soft_act + 1e-8), axis=1)
                ent_list.append(tf.reduce_mean(ent))
            avg_entropy = tf.add_n(ent_list) / float(len(ent_list))
            
            total_loss = actor_loss - self.entropy_coef * avg_entropy
            
            if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                print(f"Invalid total loss: {total_loss}, using actor_loss only")
                total_loss = actor_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)

        if grads is None or all(g is None for g in grads):
            print(f"Agent {self.agent_index}: No gradients computed!")
            return actor_loss
        grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return actor_loss

    def update_target(self, tau):
        update_target_weights(self.model.trainable_variables,
                              self.target_model.trainable_variables, tau)


# Q 网络训练器
class CriticTrainer:
    def __init__(self, input_dim, args):
        self.model = CriticModel(input_dim, hidden_units=args.hidden_units_critic, activation=args.activation)
        self.model.build(input_shape=(None, input_dim))
        self.target_model = CriticModel(input_dim, hidden_units=args.hidden_units_critic, activation=args.activation)
        self.target_model.build(input_shape=(None, input_dim))
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(args.lr_critic)  # ← 单独的 critic lr
        self.gamma = args.gamma

    def train(self, obs_act, target_q):
        print(f"\n🔍🔍🔍 [CriticTrainer.train] start critic update")
        with tf.GradientTape() as tape:
            pred_q = tf.squeeze(self.model(obs_act), axis=1)
            loss = tf.reduce_mean(tf.square(target_q - pred_q))
        print(f"   🎯🎯🎯 [CriticTrainer.train] computed critic loss = {loss.numpy():.6f}")
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        print(f"✅✅✅ [CriticTrainer.train] critic weights updated")
        return loss, pred_q

    def update_target(self, tau):
        update_target_weights(self.model.trainable_variables,
                              self.target_model.trainable_variables, tau)


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, obs_shape_n, act_space_list, agent_index, args, global_obs_dim=None, 
                 num_trucks=None, num_drones=None, total_nodes=None):
        self.name = name
        self.agent_index = agent_index
        self.n = len(obs_shape_n)
        self.args = args
        self.target_update_counter = 0
        
        # 获取环境参数
        self.num_trucks = num_trucks if num_trucks is not None else getattr(args, 'num_trucks', 3)
        self.num_drones = num_drones if num_drones is not None else getattr(args, 'num_drones', 5)
        self.total_nodes = total_nodes if total_nodes is not None else getattr(args, 'total_nodes', 6)
        
        print(f"[MADDPGAgentTrainer] Initializing agent {agent_index}")
        print(f"  Environment: {self.num_trucks} trucks, {self.num_drones} drones, {self.total_nodes} nodes")        

        # **关键修复：动态获取正确的观测维度**
        def get_agent_obs_dim(agent_idx, obs_shape_list):
            """动态获取指定智能体的观测维度"""
            if agent_idx < len(obs_shape_list):
                if isinstance(obs_shape_list[agent_idx], tuple):
                    return obs_shape_list[agent_idx][0]
                else:
                    return obs_shape_list[agent_idx]
            else:
                # 如果索引超出范围，根据智能体类型推断
                if agent_idx < self.num_trucks:
                    # 卡车：使用第一个卡车的维度
                    return get_agent_obs_dim(0, obs_shape_list)
                else:
                    # 无人机：使用第一个无人机的维度
                    drone_start_idx = self.num_trucks
                    if drone_start_idx < len(obs_shape_list):
                        return get_agent_obs_dim(drone_start_idx, obs_shape_list)
                    else:
                        # 后备方案：使用默认维度
                        return 73 if agent_idx < self.num_trucks else 77

        # 获取当前智能体的实际观测维度
        self.obs_dim = get_agent_obs_dim(agent_index, obs_shape_n)
        
        # 判断智能体类型
        if agent_index < self.num_trucks:
            self.agent_type = 'truck'
            print(f"  Agent {agent_index} is TRUCK with obs_dim: {self.obs_dim}")
        else:
            self.agent_type = 'drone'
            print(f"  Agent {agent_index} is DRONE with obs_dim: {self.obs_dim}")

        # 创建对应的Actor网络，使用正确的观测维度
        if agent_index < self.num_trucks:
            # 卡车智能体
            self.actor = TruckPolicyTrainer(self.obs_dim, self.total_nodes, args, agent_index)
            self.single_agent_act_dim = self.total_nodes + 2  # truck_target_node + truck_wait
            print(f"  Agent {agent_index} is TRUCK with {self.single_agent_act_dim} action dims")
        else:
            # 无人机智能体
            self.actor = DronePolicyTrainer(self.obs_dim, self.total_nodes, self.num_trucks, args, agent_index)
            self.single_agent_act_dim = self.total_nodes + self.num_trucks + 2  # drone_service_node + drone_rendezvous_truck + drone_continue
            print(f"  Agent {agent_index} is DRONE with {self.single_agent_act_dim} action dims")

        # 计算所有智能体的总动作维度
        truck_act_dim = self.total_nodes + 2  # 每个卡车的动作维度
        drone_act_dim = self.total_nodes + self.num_trucks + 2  # 每个无人机的动作维度
        total_act_dim = truck_act_dim * self.num_trucks + drone_act_dim * self.num_drones

        # 动态获取global_obs_dim，如果没有提供则使用默认值
        if global_obs_dim is None:
            print("Warning: global_obs_dim not provided, using default value 96")
            global_obs_dim = 96

        # critic输入维度
        self.critic_input_dim = global_obs_dim + total_act_dim
        print(f"  Agent {agent_index} dimensions:")
        print(f"   - Obs dim: {self.obs_dim}")
        print(f"   - Act dim (this agent): {self.single_agent_act_dim}")
        print(f"   - Global obs dim: {global_obs_dim}")
        print(f"   - Total act dim (all agents): {total_act_dim}")
        print(f"   - Critic input dim: {self.critic_input_dim}")

        # critic trainer（所有智能体共享相同的critic结构）
        self.critic = CriticTrainer(self.critic_input_dim, args)        

        # buffer
        buf_size = int(1e6)
        if args.use_per:
            self.buffer = PrioritizedReplayBuffer(
                buf_size,
                mu=args.per_mu,
                sigma=args.per_sigma,
                eta=args.per_eta,
                beta=args.per_beta
            )
        else:
            self.buffer = ReplayBuffer(buf_size)


    def action(self, obs, mask=None):
        """生成动作"""
        obs_tensor = tf.convert_to_tensor(obs[None, :], tf.float32)
        if self.args.use_iam and mask is not None:
            return self.actor.act(obs_tensor, mask)
        else:
            return self.actor.act(obs_tensor)

    # 把接口改成写“联合动作”和“所有智能体的 next_obs”
    def experience(self, obs, global_obs, act_all, rew, next_obs, next_obs_all, next_global_obs, done,
                    mask_all=None, next_mask_all=None):
        """
        存储经验到buffer
        
        Args:
            obs: 当前智能体的观测 (obs_dim,)
            global_obs: 全局观测 (global_obs_dim,)
            act_all: 所有智能体的动作 (n_agents, max_action_heads)
            rew: 当前智能体的奖励 (scalar)
            next_obs: 当前智能体的下一步观测 (obs_dim,) - 这个参数可能不需要
            next_obs_all: 所有智能体的下一步观测 (n_agents, obs_dim)
            next_global_obs: 下一步全局观测 (global_obs_dim,)
            done: 结束标志 (scalar)
        """
        
        # 数据预处理和验证
        obs = np.asarray(obs, dtype=np.float32)
        global_obs = np.asarray(global_obs, dtype=np.float32)
        next_global_obs = np.asarray(next_global_obs, dtype=np.float32)

        # 关键修复：处理act_all确保形状一致
        if isinstance(act_all, list):
            processed_actions = []
            max_action_heads = max(len(np.asarray(a).reshape(-1)) for a in act_all)

            for agent_act in act_all:
                act_array = np.asarray(agent_act, dtype=np.int32).reshape(-1)
                # 填充到最大动作头数
                if len(act_array) < max_action_heads:
                    padded_act = np.zeros(max_action_heads, dtype=np.int32)
                    padded_act[:len(act_array)] = act_array
                    processed_actions.append(padded_act)
                else:
                    processed_actions.append(act_array)

            act_all = np.stack(processed_actions, axis=0)  # (n_agents, max_action_heads)
        else:
            act_all = np.asarray(act_all, dtype=np.int32)
            if act_all.ndim == 1:
                act_all = act_all.reshape(1, -1)

        # 关键修复：处理next_obs_all确保格式正确
        if isinstance(next_obs_all, list):
            processed_next_obs = []
            for agent_obs in next_obs_all:
                agent_obs_array = np.asarray(agent_obs, dtype=np.float32)
                # 确保是1D数组
                if agent_obs_array.ndim == 0:
                    agent_obs_array = np.array([agent_obs_array])
                elif agent_obs_array.ndim > 1:
                    agent_obs_array = agent_obs_array.flatten()
                processed_next_obs.append(agent_obs_array)
            
            # 检查维度一致性
            obs_lengths = [len(obs_arr) for obs_arr in processed_next_obs]
            if len(set(obs_lengths)) > 1:
                print(f"[Buffer] Warning: Inconsistent next_obs lengths: {obs_lengths}")
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
            
            next_obs_all = np.stack(processed_next_obs, axis=0)
        else:
            next_obs_all = np.asarray(next_obs_all, dtype=np.float32)
            if next_obs_all.ndim == 1:
                next_obs_all = next_obs_all.reshape(1, -1)
        
        # 兼容两种buffer
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.add(obs, global_obs, act_all, rew, next_obs_all, next_global_obs, done,
                            mask_all=mask_all, next_mask_all=next_mask_all)
        else:
            self.buffer.add(obs, global_obs, act_all, rew, next_obs_all, next_global_obs, done,
                            mask_all=mask_all, next_mask_all=next_mask_all)
                            
    def preupdate(self):
        pass

    def _multihead_onehot(self, a_batch, agent_idx):
        """根据智能体类型将动作转换为one-hot编码"""
        if agent_idx < self.num_trucks:
            # 卡车动作：[truck_target_node, truck_wait]
            head_dims = [self.total_nodes, 2]
            max_heads = 2
        else:
            # 无人机动作：[drone_service_node, drone_rendezvous_truck, drone_continue]
            head_dims = [self.total_nodes, self.num_trucks, 2]
            max_heads = 3
        
        onehots = []
        for i, dim in enumerate(head_dims):
            if i < a_batch.shape[1]:  # 确保不越界
                onehots.append(tf.one_hot(a_batch[:, i], dim, dtype=tf.float32))
            else:
                # 如果动作维度不够，用默认值填充
                default_action = tf.zeros((a_batch.shape[0],), dtype=tf.int32)
                onehots.append(tf.one_hot(default_action, dim, dtype=tf.float32))
        
        return tf.concat(onehots, axis=1)

    def update(self, agents, t, **kwargs):
        print(f"\n📄📄📄 [MADDPGAgentTrainer.update] Agent {self.agent_index} called at step {t}")
        target_update_freq = self.args.target_update_freq

        # 1) 检查更新条件
        if len(self.buffer) < self.args.batch_size:
            print(f"   📄📄📄⏭ Buffer size {len(self.buffer)} < batch size {self.args.batch_size}, skipping")
            return None

        # 2) 采样（兼容 PER 与非 PER；带上 masks）
        if self.args.use_per:
            (obs_b, glob_b, act_b, rew_b,
            next_obs_all_b, next_glob_b, done_b,
            idxs, weights, masks_b, next_masks_b) = self.buffer.sample(self.args.batch_size)
        else:
            (obs_b, glob_b, act_b, rew_b,
            next_obs_all_b, next_glob_b, done_b,
            masks_b, next_masks_b) = self.buffer.sample(self.args.batch_size)
            idxs, weights = None, None
        print("   [debug] masks_b is None?", masks_b is None, 
                "/ next_masks_b is None?", next_masks_b is None)
        # 批次是否有掩码 & 形状
        print("   [debug] masks_b is None?", masks_b is None,
            "/ next_masks_b is None?", next_masks_b is None)
        if masks_b is not None:
            print("   [debug] masks_b shape:", np.shape(masks_b),
                "/ next_masks_b shape:", np.shape(next_masks_b))

            # 统计每个 agent 在该批次里的“禁用比例”（False 比例，越大说明可选节点越少）
            # masks_b: (B, n_agents, total_nodes)
            invalid_ratio = (~masks_b).mean(axis=(0, 2))  # → (n_agents,)
            for ag_idx, r in enumerate(invalid_ratio):
                print(f"   [debug] agent{ag_idx} invalid-node ratio: {r:.2f}")
                
        # 3) 数值检查
        if (np.any(np.isnan(obs_b)) or np.any(np.isnan(glob_b))
            or np.any(np.isnan(rew_b)) or np.any(np.isnan(next_glob_b))):
            print(f"❌ NaN detected in batch data, skipping update")
            return None

        # 4) 动作整理 (B, n_agents, max_heads) -> 各 agent 的 multi-head onehot
        # 确保动作数组的正确维度
        act_b_arr = np.array(act_b)
        if act_b_arr.dtype == object:
            # 防止被以object列表形式存储
            act_b_arr = np.stack(act_b_arr, axis=0)
        if act_b_arr.ndim != 3:
            raise ValueError(
                f"[BAD BATCH] act_b must be (B, n_agents, max_heads); got {act_b_arr.shape}. "
                f"请确认 experience() 往 buffer 存的是全体智能体的动作 act_all，而不是单个动作。"
            )
        print(f"   ✅ act_b ok: shape={act_b_arr.shape} (B, n_agents, max_heads)")

        # 修复：为不同类型智能体处理动作的one-hot编码
        all_agents_act_onehots = []
        for ag_idx in range(self.n):
            # 根据智能体类型确定动作头维度
            if ag_idx < self.num_trucks:
                # 卡车：使用前2个动作头
                agent_actions = act_b_arr[:, ag_idx, :2]  # (B, 2)
                head_dims = [self.total_nodes, 2]
            else:
                # 无人机：使用所有3个动作头
                agent_actions = act_b_arr[:, ag_idx, :3]  # (B, 3)
                head_dims = [self.total_nodes, self.num_trucks, 2]
            
            # 合法性修正（越界动作置0）
            for i, max_val in enumerate(head_dims):
                if i < agent_actions.shape[1]:
                    a = agent_actions[:, i]
                    invalid = (a < 0) | (a >= max_val)
                    if np.any(invalid):
                        print(f"   ⚠️  Invalid actions found for agent{ag_idx}-head{i}: fixing to 0, count={np.sum(invalid)}")
                        a = a.copy()
                        a[invalid] = 0
                        agent_actions[:, i] = a

            # 转换为one-hot
            head_ohs = []
            for i, dim in enumerate(head_dims):
                if i < agent_actions.shape[1]:
                    oh = tf.one_hot(agent_actions[:, i], dim, dtype=tf.float32)
                else:
                    # 填充默认动作的one-hot
                    default_action = tf.zeros((agent_actions.shape[0],), dtype=tf.int32)
                    oh = tf.one_hot(default_action, dim, dtype=tf.float32)
                head_ohs.append(oh)
            
            concat_oh = tf.concat(head_ohs, axis=1)  # (B, sum(head_dims))
            all_agents_act_onehots.append(concat_oh)
            print(f"   🔧 one-hot for agent{ag_idx}: {concat_oh.shape}")

        # 5) 转成张量
        obs_tensor        = tf.convert_to_tensor(obs_b,        tf.float32)
        glob_tensor       = tf.convert_to_tensor(glob_b,       tf.float32)
        next_glob_tensor  = tf.convert_to_tensor(next_glob_b,  tf.float32)
        next_obs_all      = tf.convert_to_tensor(next_obs_all_b, tf.float32)
        rew               = tf.convert_to_tensor(rew_b,        tf.float32)
        done              = tf.convert_to_tensor(done_b,       tf.float32)

        # ===== 下一步 target：用各自 target actor(next_obs) =====
        # 6) 目标动作（**关键修复：在 target 上应用 next_masks_b 的 IAM**）
        target_act_onehots = []
        for ag_idx, ag in enumerate(agents):
            # **关键修复：动态获取每个智能体的实际观测维度**
            agent_obs_dim = ag.obs_dim  # 使用智能体自己记录的观测维度
            
            # 根据智能体实际观测维度提取对应的观测数据
            agent_next_obs = next_obs_all[:, ag_idx, :agent_obs_dim]
            
            # 根据智能体类型确定动作头维度
            if ag_idx < self.num_trucks:
                head_dims = [self.total_nodes, 2]
            else:
                head_dims = [self.total_nodes, self.num_trucks, 2]
            
            head_logits = ag.actor.target_model(agent_next_obs)  # list of tensors
                
            # === 在第0个head（选节点）上加 next-mask ===
            if self.args.use_iam and (next_masks_b is not None):
                mask_next = tf.convert_to_tensor(next_masks_b[:, ag_idx, :], dtype=tf.bool)  # (B, total_nodes)
                very_low  = tf.fill(tf.shape(head_logits[0]), -1e9)
                head0 = tf.where(mask_next, head_logits[0], very_low)
                head_logits = [head0] + head_logits[1:]
                # 统计本批次被屏蔽（False）的位置数
                num_blocked = int(tf.reduce_sum(tf.cast(~mask_next, tf.int32)).numpy())
                print(f"   [debug] target-mask applied for agent{ag_idx}, blocked positions (batch x nodes): {num_blocked}")
                        
            head_ohs = []
            for i, (logits, dim) in enumerate(zip(head_logits, head_dims)):
                logits = tf.clip_by_value(logits, -10.0, 10.0)
                a = tf.argmax(logits, axis=1)
                oh = tf.one_hot(a, dim, dtype=tf.float32)
                head_ohs.append(oh)
            concat_oh = tf.concat(head_ohs, axis=1)
            target_act_onehots.append(concat_oh)
            print(f"   🎯 target action one-hot for agent{ag_idx}: {concat_oh.shape} (obs_dim: {agent_obs_dim})")

        # 7)===== Critic 训练 =====
        print(f"➡️ Start critic training")
        critic_in_next = tf.concat([next_glob_tensor] + target_act_onehots, axis=1)
        target_q = tf.squeeze(self.critic.target_model(critic_in_next), axis=1)
        target_q = tf.clip_by_value(target_q, -1000.0, 1000.0)
        y = rew + self.args.gamma * (1. - done) * target_q
        if tf.reduce_any(tf.math.is_nan(y)) or tf.reduce_any(tf.math.is_inf(y)):
            print(f"❌ Invalid target values detected, skipping update")
            return None
        print(f"   y stats: min={float(tf.reduce_min(y)):.4f}, max={float(tf.reduce_max(y)):.4f}, mean={float(tf.reduce_mean(y)):.4f}")

        print(f"➡️➡️➡️ [MADDPGAgentTrainer.update] start critic training")
        print(f"   🔍 Critic input dimensions check:")
        print(f"     - glob_tensor: {glob_tensor.shape}")
        print(f"     - all_agents_act_onehots shapes: {[a.shape for a in all_agents_act_onehots]}")

        with tf.GradientTape() as tape_c:
            critic_in = tf.concat([glob_tensor] + all_agents_act_onehots, axis=1)
            print(f"     - critic_in: {critic_in.shape}")
            pred_q = tf.squeeze(self.critic.model(critic_in), axis=1)
            pred_q = tf.clip_by_value(pred_q, -1000.0, 1000.0)
            td_errors = y - pred_q
            if 'weights' in locals() and (weights is not None):
                w = tf.convert_to_tensor(weights, tf.float32)
                critic_loss = tf.reduce_mean(w * tf.square(td_errors))
            else:
                critic_loss = tf.reduce_mean(tf.square(td_errors))
            if tf.math.is_nan(critic_loss) or tf.math.is_inf(critic_loss):
                print(f"❌ Invalid critic loss detected: {critic_loss}, skipping update")
                return None

        grads_c = tape_c.gradient(critic_loss, self.critic.model.trainable_variables)
        grads_c = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads_c]
        print(f"🧮🧮🧮 [MADDPGAgentTrainer.update] critic gradients computed")
        self.critic.optimizer.apply_gradients(zip(grads_c, self.critic.model.trainable_variables))

        # PER：基于 TD 误差更新优先级
        try:
            if getattr(self.args, 'use_per', False) and idxs is not None:
                self.buffer.update_priorities(idxs, tf.abs(td_errors).numpy())
        except Exception as e:
            print(f"   ⚠️ PER priority update failed: {e}")
        print(f"✅ Critic updated, loss={critic_loss.numpy():.6f}")

        # ===== Actor 训练（其他智能体固定为 buffer 动作） =====
        # 8) Actor 训练（**关键修复：把当前 masks 传入 actor.train**）
        print(f"➡️ Start actor training")
        all_acts_const = [tf.stop_gradient(a) for a in all_agents_act_onehots]
        mask_cur = None
        if self.args.use_iam and (masks_b is not None):
            mask_cur = tf.convert_to_tensor(masks_b[:, self.agent_index, :], dtype=tf.bool)  # (B, total_nodes)

        actor_loss = self.actor.train(obs_tensor, all_acts_const, glob_tensor, self.critic.model, mask=mask_cur)
        if tf.math.is_nan(actor_loss) or tf.math.is_inf(actor_loss):
            print(f"❌ Invalid actor loss: {actor_loss}, skipping actor update")
        else:
            print(f"✅ Actor updated, loss={actor_loss.numpy():.6f}")
            
        # ===== 软更新 =====
        self.target_update_counter = getattr(self, "target_update_counter", 0) + 1
        if t % target_update_freq == 0:
            print(f"🔄 Updating target networks (counter={self.target_update_counter}, env_step={t})")
            tau = getattr(self.args, 'tau')
            update_target_weights(self.critic.model.trainable_variables,
                                self.critic.target_model.trainable_variables, tau)
            update_target_weights(self.actor.model.trainable_variables,
                                self.actor.target_model.trainable_variables, tau)
            self.target_update_counter = 0
            print(f"✅ Target networks updated with tau={tau}")

        # print("   Press ENTER to continue to next step...")
        # input("⏸️ 本次 actor-critic 训练完成，按回车继续...") 
        return critic_loss, actor_loss