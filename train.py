import argparse
import os
import tensorflow as tf
import time
from datetime import datetime
from trucks_and_drones.build_env import BuildEnvironment
import maddpg.trainer.utils as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from tensorflow.keras import layers
from trucks_and_drones.config import cfg
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Multi-Agent RL experiments")

    # ========== 新增:算法选择参数 ==========
    parser.add_argument("--algorithm", type=str, default="maddpg",
                       choices=["maddpg", "iql", "ma2c", "coma", "cima","mappo"],
                       help="Algorithm to use: maddpg/iql/ma2c/coma/cima/mappo")

    # 在parse_args()中添加MAPPO参数
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter epsilon")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Value loss coefficient")

    # Environment parameters
    parser.add_argument("--env-config", type=str, default=None,
                        help="Path to Python script that configures BuildEnvironment")
    parser.add_argument("--max-episode-len", type=int, default=1000,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000,
                        help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4,
                        help="learning rate for actor")
    parser.add_argument("--lr_critic", type=float, default=3e-4,
                        help="learning rate for critic")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--batch-size", type=int,
                        help="batch size")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the MLP layers")
    parser.add_argument("--update-freq", type=int,
                        help="how often (in training steps) to update networks")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="experiment",
                        help="name of the experiment (used for saving)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/",
                        help="directory in which to save models")
    parser.add_argument("--save-rate", type=int, default=2000,
                        help="save model once every this many episodes")

    # TensorBoard logging
    parser.add_argument("--log-dir", type=str, default="./logs/",
                        help="directory for TensorBoard logs")

    # Flags for PER and IAM
    parser.add_argument("--use-per", action="store_true", default=False,
                        help="use prioritized experience replay")
    parser.add_argument("--use-iam", action="store_true", default=False,
                        help="use invalid action masking")

    # 添加可视化控制参数
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Enable visualization during training")
    parser.add_argument("--visualize-interval", type=int, default=1,
                        help="Visualize every N episodes (only when --visualize is enabled)")
    

    # PER parameters
    parser.add_argument("--per_mu", type=float, default=0.6,
                        help="Priority exponent mu for PER (Equation 33)")
    parser.add_argument("--per_sigma", type=float, default=1.0,
                        help="Sampling probability exponent sigma for PER (Equation 34)")
    parser.add_argument("--per_eta", type=float, default=1e-6,
                        help="Small constant eta to avoid zero priority")
    parser.add_argument("--per_beta", type=float, default=0.4,
                        help="Importance sampling exponent beta for PER")

    parser.add_argument("--hidden_units_actor", nargs='+', type=int, default=[512, 256],
                        help="Hidden layer sizes for actor network")
    parser.add_argument("--hidden_units_critic", nargs='+', type=int, default=[1024, 512, 256],
                        help="Hidden layer sizes for critic network")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function: relu or tanh")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon value for epsilon-greedy exploration")

    return parser.parse_args()


def mlp_actor(input, num_outputs, scope, reuse=False):
    x = layers.Dense(512, activation='relu')(input)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(num_outputs)(x)
    return out


def mlp_critic(input, num_outputs, scope, reuse=False):
    x = layers.Dense(1024, activation='relu')(input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(num_outputs)(x)
    return out

# 线性 ε 调度：warmup 步内为 1.0，之后在 decay_steps 内下降到 0.1
def get_epsilon(step, warmup_steps=1500, eps_max=1.0, eps_min=0.1, decay_steps=30000):
    if step < warmup_steps:
        return eps_max
    # ratio = min(1.0, max(0.0, (step - warmup_steps) / float(decay_steps)))
    # return eps_max + (eps_min - eps_max) * ratio
    # ⭐ 使用指数衰减替代线性衰减
    decay_ratio = (step - warmup_steps) / float(decay_steps)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    return eps_min + (eps_max - eps_min) * np.exp(-3 * decay_ratio)  # 指数衰减

def train(args):
    # ========== 动态导入算法 ==========
    if args.algorithm == "maddpg":
        from maddpg.trainer.maddpg import MADDPGAgentTrainer as TrainerClass
    elif args.algorithm == "ma2c":
        from maddpg.trainer.ma2c import MA2CAgentTrainer as TrainerClass
    elif args.algorithm == "mappo":
        from maddpg.trainer.mappo import MAPPOAgentTrainer as TrainerClass
    elif args.algorithm == "coma":
        from maddpg.trainer.coma import COMAAgentTrainer as TrainerClass
    elif args.algorithm == "cima":
        from maddpg.trainer.cima import CIMAAgentTrainer as TrainerClass

    print(f"🤖 Using algorithm: {args.algorithm.upper()}")
    
    # 设置TensorBoard日志记录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建TensorBoard writers
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    episode_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'episode'))
    
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"📈 Run 'tensorboard --logdir {log_dir}' to view training curves")
    
    # 覆盖命令行参数：读取 config 中的训练轮数
    args.num_episodes = cfg['training']['num_episodes']
    args.batch_size   = cfg['training']['batch_size']
    args.update_freq  = cfg['training']['update_freq']
    args.target_update_freq = cfg['training']['target_update_freq']
    args.min_buffer_size = cfg['training']['min_buffer_size']
    args.updates_per_call   = cfg['training'].get('updates_per_call')
    args.warmup_steps       = cfg['training'].get('warmup_steps')
    args.entropy_coef       = cfg['training'].get('entropy_coef')
    args.tau                = cfg['training'].get('tau')
    args.lr_actor   = cfg['training'].get('lr_actor')
    args.lr_critic  = cfg['training'].get('lr_critic')
    args.rollout_len = cfg['training'].get('rollout_len')

    # 1) 从 cfg 里读取环境配置
    env_cfg = cfg.get('environment', {})
    num_trucks    = env_cfg.get('num_trucks')
    num_drones    = env_cfg.get('num_drones')
    num_depots    = env_cfg.get('num_depots')
    num_customers = env_cfg.get('num_customers')


    # 2) 确定观测字段：用 temp_db 里实际有的 'n_items' 而不是 'demand'
    obs_contin   = [ 'TW', 'time', 'n_items', 'deadline']
    obs_discrete = ['ET', 'ED', 'NT', 'ND', 'truck_node', 'drone_node', 'delta']
    # 'delta'是“unassigned”的底层字段
    tandem_fields = obs_discrete + obs_contin + ['LT_time', 'LD_time']


    # —— 3) 一次性构造环境 ——
    builder = (
        BuildEnvironment(
            "VRPD",
            enable_visualization=args.visualize  # 传递可视化开关
        )
        .trucks(num=num_trucks)
        .drones(num=num_drones)
        .depots(num=num_depots)
        .customers(num=num_customers)
        .visuals()
        .observations(
            contin_inputs=obs_contin,
            discrete_inputs=obs_discrete,  # 用 NT/ND
            combine_per_index=[tandem_fields for _ in range(num_trucks)],
            combine_per_type=['vehicles', 'customers', 'depots'],  # 关键：全局观测来源
            flatten=True,  # 把每个 agent 的观测扁平化
            output_as_array=False, # 返回 list_of_agent_obs 而不是大向量
        )
        .actions()
        .rewards()
        .compile()

    )
    env = builder.build()
    # env.visualizer.visualize_step(episode=0, step=0)

#     from trucks_and_drones.simulation.temp_database import BaseTempDatabase  # 只是类型提示，不必一定导
#     from trucks_and_drones.simulation.temp_database import export_instance_coords  # 

#     # 导出“静态基线用”的坐标（只含 depot+静态客户）
#     export_instance_coords(env.temp_db, path="rl_instance_coords.json", include_dynamic=False)
#     print("✅ RL 实例坐标已导出到 rl_instance_coords.json")

    # 4) 多智能体训练准备
    n_agents = len(builder.trucks) + len(builder.drones)
    
    # obs_shape_n = [env.observation_space[i].shape for i in range(n_agents)]
    obs_n, global_obs = env.reset()
    obs_shape_n = [obs.shape for obs in obs_n]
    
    # 获取全局观测维度
    global_obs_dim = global_obs.shape[0] if hasattr(global_obs, 'shape') else len(global_obs)
    print(f"🌍 Global observation dimension: {global_obs_dim}")
    print(">>> obs_shape_n =", obs_shape_n)
    # 详细分析观测空间组成
    print(f"\n🔬 === 观测空间组成分析 ===")
    
    # 从temp_db获取详细信息进行调试
    temp_db = builder.temp_db
    print(f"卡车状态 (ET): {temp_db.status_dict['ET']}")
    print(f"无人机状态 (ED): {temp_db.status_dict['ED']}")
    print(f"卡车载重 (TW): {temp_db.status_dict['TW']}")
    print(f"无人机载重 (DW): {temp_db.status_dict['DW']}")
    print(f"节点需求 (n_items): {temp_db.status_dict['n_items']}")
    print(f"节点截止时间 (deadline): {temp_db.constants_dict.get('deadline', 'Not found')}")
    print(f"未分配状态 (delta): {temp_db.status_dict['delta']}")


    act_space_n = list(env.action_space.spaces)
    
    # print("Observation space:", env.observation_space)
    print("Type:", type(env.observation_space))
    # for i, space in enumerate(env.observation_space):
    #     print(f" - Obs {i}: shape = {space.shape}, low = {space.low}, high = {space.high}")

    raw_aspace = env.act_decoder.action_space()
    # 原来 raw_aspace 是 Tuple(Tuple(Discrete,…), …)，先 unpack成 act_space_list：
    act_space_list = raw_aspace.spaces[0].spaces  # 取第一个 agent 的 “tuple of Discrete”    

    trainers = []
    for i in range(n_agents):
        trainers.append(
            TrainerClass(
                name=f"agent_{i}",
                obs_shape_n=obs_shape_n,
                act_space_list=act_space_list,
                agent_index=i,
                args=args,
                global_obs_dim=global_obs_dim,
                # 添加缺失的环境参数
                num_trucks=num_trucks,
                num_drones=num_drones,
                total_nodes=num_depots + num_customers
            )
        )

    # 5) 创建保存目录并保存初始模型
    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    train_step = 0
    
    # 6) 训练循环
    episode_rewards = [0.0]
    episode_step = 0
    obs_n, global_obs = env.reset()
    
    # 新增：专用“更新计数器”，让loss曲线连续
    update_steps = 0

    # ========== 新增：初始化总步数计数器 ==========
    env.visualizer.total_training_steps = 0
    # ==========================================

    # 初始化可视化 - 显示初始状态
    if args.visualize:
        print("🎬 Starting visualization...")
        env.visualizer.visualize_step(
            episode=len(episode_rewards)-1,
            step=episode_step,
            slow_down_pls=True,
            last_actions=None,
            last_rewards=None
        )
    print(f"Starting training with PER={args.use_per}, IAM={args.use_iam}")
    # 存储最后的动作用于显示
    last_actions = []
    last_rewards = []
    
    # 用于记录训练指标
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    num_updates = 0

    while True:
        # # 生成动作掩码
        # mask_n = [env.get_mask() for _ in range(n_agents)] if args.use_iam else [None] * n_agents

        mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)] if args.use_iam else [None] * n_agents

        # === ε 探索（使用 warmup_steps）===
        ws = args.warmup_steps if args.warmup_steps is not None else 0
        cur_eps = get_epsilon(train_step, warmup_steps=ws, eps_max=1.0, eps_min=0.1, decay_steps=30000)
        # for tr in trainers:
        #     tr.actor.epsilon = cur_eps
        for tr in trainers:
            # ✅ 只有MADDPG有epsilon属性，MA2C没有就跳过
            if hasattr(tr, 'actor') and hasattr(tr.actor, 'epsilon'):
                tr.actor.epsilon = cur_eps

        # 智能体决策
        if args.algorithm.lower() in ['ma2c', 'mappo']:
            # MA2C 的 action(obs, global_obs, mask)
            action_n = [tr.action(o, global_obs, m) for tr, o, m in zip(trainers, obs_n, mask_n)]
        else:
            action_n = [tr.action(o, m) for tr, o, m in zip(trainers, obs_n, mask_n)]
            last_actions = action_n.copy()  # 保存用于显示

        print(f"\n🎯 Episode {len(episode_rewards)-1}, Step {episode_step}")
        print(f"   Generated actions: {action_n}")


        # (new_obs_n, new_global_obs), rew_n, done, _ = env.step(action_n)
        (new_obs_n, new_global_obs), rew_n, done, info = env.step(action_n)

        # 优先用 info 里的被执行动作；若上层没透传，则回退到 temp_db 缓存；再不行才退回原动作
        executed_action_n = None
        if isinstance(info, dict):
            executed_action_n = info.get('executed_actions_multihead', None)
        if executed_action_n is None:
            executed_action_n = getattr(env.temp_db, 'last_executed_actions_multihead', None)
        if executed_action_n is None:
            # 最后的兜底（不建议长期使用）
            executed_action_n = action_n

        last_actions = executed_action_n.copy()  # 你后面有可视化，这里也同步成“被执行动作”

        # 便于人看：把 list[list[int]] 压成 tuple
        def _summ(a): 
            try: return [tuple(x) for x in a]
            except: return a

        print(f"   Proposed actions: {_summ(action_n)}")
        print(f"   ▶︎ Executed actions (used for replay): {_summ(executed_action_n)}")

        # 差异统计（每个 agent 的多头是否一致）
        try:
            diffs = [int(tuple(pa) != tuple(ea)) for pa, ea in zip(action_n, executed_action_n)]
            print(f"   Diff agents (proposed≠executed): {sum(diffs)}/{len(diffs)}")
        except Exception as e:
            print(f"   [Warn] diff check failed: {e}")

        last_rewards = rew_n.copy()  # 保存用于显示

        # === 关键：把 next_obs（每个 agent 一条 1D 向量）填充到统一长度 ===
        def pad1d_list(xs, fill=0.0):
            arrs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in xs]
            maxlen = max(a.shape[0] for a in arrs)
            out = [np.pad(a, (0, maxlen - a.shape[0]), mode="constant", constant_values=fill) for a in arrs]
            return np.stack(out, axis=0), maxlen  # (n_agents, maxlen)

        next_obs_all_pad, _ = pad1d_list(new_obs_n)
        next_mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)] if args.use_iam else [None]*n_agents

        # # **关键：用新状态再取下一步的 mask**
        # next_mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)] if args.use_iam else [None] * n_agents


        # ✅ experience调用完全相同，MADDPG和MA2C都用这个接口
        for i, tr in enumerate(trainers):
            try:
                print(f"   [Replay] About to store act_all sample (agent0 view): {executed_action_n[0]}")
                print(f"   [Replay] act_all len = {len(executed_action_n)}")
                # 简单形状检查
                assert len(executed_action_n) == n_agents, \
                    f"act_all length {len(executed_action_n)} != n_agents {n_agents}"
            except Exception as e:
                print(f"   [Replay] Pre-store check warn: {e}")

            tr.experience(
                obs=obs_n[i],                       # 本智能体当前观测
                global_obs=global_obs,              # 全局观测
                # act_all=action_n,                   # 所有智能体本步动作（离散多头）
                act_all=executed_action_n,
                rew=rew_n[i],                       # 协作设定下相同，保留为标量
                next_obs=new_obs_n[i],              # 本智能体下一观测（给 actor 用）
                next_obs_all=next_obs_all_pad,          # 所有智能体下一观测（给 target Q 用）
                next_global_obs=new_global_obs,
                done=done,
                mask_all=mask_n,
                next_mask_all=next_mask_n
            )
            # tr.experience(
            #     obs_n[i], global_obs, action_n[i], rew_n[i],
            #     new_obs_n[i], new_global_obs, done
            # )

        obs_n = new_obs_n
        global_obs = new_global_obs
        episode_step += 1
        train_step += 1       

        episode_rewards[-1] += rew_n[0]  # 因为所有智能体的奖励相同，只取第一个
        
        # 可视化当前步骤 - 根据参数和间隔决定
        if args.visualize and len(episode_rewards) % args.visualize_interval == 0:
            env.visualizer.visualize_step(
                episode=len(episode_rewards)-1,
                step=episode_step,
                slow_down_pls=(episode_step < 10),
                last_actions=last_actions,
                last_rewards=last_rewards
            )

        # 每5步暂停让用户观察
        if episode_step % 5 == 0 and episode_step > 0:
            print(f"\n📊 Episode {len(episode_rewards)-1}, Step {episode_step} Summary:")
            print(f"   Total reward so far: {episode_rewards[-1]:.3f}")
            print(f"   Step rewards: {rew_n}")
            print(f"   Actions taken: {last_actions}")
            print(f"   Vehicle positions: {env.temp_db.status_dict.get('v_coord', 'N/A')}")
            print(f"   Vehicle status ET: {env.temp_db.status_dict.get('ET', 'N/A')}")
            print(f"   Vehicle status ED: {env.temp_db.status_dict.get('ED', 'N/A')}")

            # 检查是否有节点被访问
            visited = getattr(env.temp_db, 'visited_nodes', set())
            delta = env.temp_db.get_val('delta') if hasattr(env.temp_db, 'get_val') else None
            print(f"   Visited nodes: {visited}")
            print(f"   Delta status: {delta}")

            # if episode_step < 2000:  # 前20步手动控制
            #     input("   Press Enter to continue...")
        
        # 训练智能体 - 修改这部分
        # ===== 参数更新（预热期内跳过）=====
        if args.algorithm.lower() in ["ma2c", "mappo"]:
            # On-policy：每步都尝试更新，update() 内部会在回合结束时才真正优化
            for tr in trainers:
                losses = tr.update(trainers, train_step)
                if losses is not None and len(losses) == 2:
                    critic_loss, actor_loss = losses
                    with train_writer.as_default():
                        tf.summary.scalar(f'Agent_{tr.agent_index}/Critic_Loss', float(critic_loss), step=train_step)
                        tf.summary.scalar(f'Agent_{tr.agent_index}/Actor_Loss', float(actor_loss), step=train_step)
                        tf.summary.scalar(f'Agent_{tr.agent_index}/Abs_Actor_Loss', abs(float(actor_loss)), step=train_step)
            train_writer.flush()
        else:
            # Off-policy（如 MADDPG）保持原来的最小缓冲/更新频率判断
            if (len(trainers[0].buffer) >= args.min_buffer_size and
                train_step % args.update_freq == 0 and
                train_step >= (args.warmup_steps or 0)):  # ← 预热未结束时不更新

                for tr in trainers: 
                    tr.preupdate()
                
                # ⭐ 多次梯度步：默认 1 次；配置为 N 就做 N 次
                updates = int(args.updates_per_call or 1)
                for _ in range(updates):
                    for tr in trainers:
                        losses = tr.update(trainers, train_step, target_update_freq=args.target_update_freq)
                        if losses is None or len(losses) != 2:
                            # 本次没更新（例如缓冲不够或loss为NaN被跳过），直接下一个
                            continue

                        critic_loss, actor_loss = losses
                        with train_writer.as_default():
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Critic_Loss', float(critic_loss), step=train_step)
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Actor_Loss',  float(actor_loss),  step=train_step)
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Abs_Actor_Loss', abs(float(actor_loss)), step=train_step)

                        print(f"   📊 Agent {tr.agent_index} - Critic Loss: {critic_loss:.6f}, Actor Loss: {actor_loss:.6f}")

                    train_writer.flush()
                     
                
            else:
                print(f"   Buffer size {len(trainers[0].buffer)} < min required {args.min_buffer_size}, skipping training")


        if done:
            print(f"\n🏁 Episode {len(episode_rewards)-1} completed!")
            print(f"   Final reward: {episode_rewards[-1]:.2f}")
            print(f"   Episode length: {episode_step}")
            print(f"   Reason: {'Task completed' if done else 'Max steps reached'}")

            # 获取成本统计
            cost_stats = env.reward_calc.get_episode_statistics()
            print(f"\n📊 Episode Cost Statistics:")
            print(f"   Total Cost (TC): {cost_stats['total_cost']:.2f}")
            print(f"   Travel Cost (Tra-C): {cost_stats['travel_cost']:.2f}")
            print(f"   Delay Penalty: {cost_stats['delay_penalty']:.2f}")
            print(f"   Unserved Penalty: {cost_stats['unserved_penalty']:.2f}")
            print(f"   Unserved Nodes: {cost_stats['unserved_count']}")
            print(f"   Served Nodes: {cost_stats['served_count']}")
            print(f"   Truck Travel Time: {cost_stats['truck_travel_time']:.2f}")
            print(f"   Drone Travel Time: {cost_stats['drone_travel_time']:.2f}")

            # 记录episode统计到TensorBoard
            episode_reward = episode_rewards[-1]
            episode_num = len(episode_rewards)-1
            
            with episode_writer.as_default():
                # 记录episode总奖励
                tf.summary.scalar('Episode/Total_Reward', episode_reward, step=episode_num)
                tf.summary.scalar('Episode/Episode_Length', episode_step, step=episode_num)
                # 记录成本统计
                tf.summary.scalar('Costs/Total_Cost', cost_stats['total_cost'], step=episode_num)
                tf.summary.scalar('Costs/Travel_Cost', cost_stats['travel_cost'], step=episode_num)
                tf.summary.scalar('Costs/Delay_Penalty', cost_stats['delay_penalty'], step=episode_num)
                tf.summary.scalar('Costs/Unserved_Penalty', cost_stats['unserved_penalty'], step=episode_num)
                tf.summary.scalar('Costs/Unserved_Count', cost_stats['unserved_count'], step=episode_num)
                tf.summary.scalar('Costs/Served_Count', cost_stats['served_count'], step=episode_num)
                tf.summary.scalar('Time/Truck_Travel_Time', cost_stats['truck_travel_time'], step=episode_num)
                tf.summary.scalar('Time/Drone_Travel_Time', cost_stats['drone_travel_time'], step=episode_num)
                
                # 记录环境统计
                try:
                    visited_nodes = len(getattr(env.temp_db, 'visited_nodes', set()))
                    delta = env.temp_db.get_val('delta')
                    total_nodes = len(delta)
                    completion_rate = visited_nodes / max(total_nodes - 1, 1)  # 排除depot
                    
                    tf.summary.scalar('Episode/Completion_Rate', completion_rate, step=episode_num)
                    tf.summary.scalar('Episode/Visited_Nodes', visited_nodes, step=episode_num)
                    # tf.summary.scalar('Episode/Remaining_Nodes', total_nodes - visited_nodes - 1, step=episode_num)
                    remaining = int(np.sum(env.temp_db.get_val('delta')[1:] == 1))
                    tf.summary.scalar('Episode/Remaining_Nodes', remaining, step=episode_num)
                except Exception as e:
                    print(f"   ⚠️  Could not record environment stats: {e}")
            
            # 强制写入TensorBoard
            train_writer.flush()
            episode_writer.flush()

            # 显示最终状态 - 根据参数决定
            if args.visualize and len(episode_rewards) % args.visualize_interval == 0:
                env.visualizer.visualize_step(
                    episode=len(episode_rewards)-1,
                    step=episode_step,
                    slow_down_pls=True,
                    last_actions=last_actions,
                    last_rewards=last_rewards
                )

            # input("   Episode finished. Press Enter to start next episode...")
            obs_n, global_obs = env.reset()
            print(f"Episode {len(episode_rewards)-1} reward: {episode_rewards[-1]:.2f}")


            # 重置episode统计
            episode_step = 0
            episode_rewards.append(0.0)
            total_critic_loss = 0.0
            total_actor_loss = 0.0
            num_updates = 0

            # 每400个episode保存一次模型
            if len(episode_rewards) % args.save_rate == 0 and len(episode_rewards) > 1:
                for i, agent in enumerate(trainers):
                    if args.algorithm.lower() == "ma2c":
                        # MA2C：一个 policy 同时包含 actor+critic
                        agent.policy.save_weights(f"{save_path}/agent_{i}_policy_ep{len(episode_rewards)}.h5")
                    elif args.algorithm.lower() == "mappo":
                        # MAPPO：保存整个 policy（含 actor heads + critic）
                        agent.policy.save_weights(f"{save_path}/agent_{i}_mappo_policy_ep{len(episode_rewards)}.weights.h5")
                    else:
                        agent.actor.model.save_weights(f"{save_path}/agent_{i}_actor_weights_ep{len(episode_rewards)}.h5")
                        agent.critic.model.save_weights(f"{save_path}/agent_{i}_critic_weights_ep{len(episode_rewards)}.h5")
                        agent.actor.target_model.save_weights(f"{save_path}/agent_{i}_actor_target_weights_ep{len(episode_rewards)}.h5")
                        agent.critic.target_model.save_weights(f"{save_path}/agent_{i}_critic_target_weights_ep{len(episode_rewards)}.h5")
                print(f"Models saved at episode {len(episode_rewards)}")

            # 显示新episode的初始状态
            env.visualizer.visualize_step(
                episode=len(episode_rewards)-1,
                step=episode_step,
                slow_down_pls=True,
                last_actions=None,
                last_rewards=None
            )


        # 达到最大训练轮数，保存模型并退出
        if len(episode_rewards) > args.num_episodes:
            print(f"\n🎓 Training completed after {args.num_episodes} episodes!")
            for i, agent in enumerate(trainers):
                if args.algorithm.lower() == "ma2c":
                    # MA2C：只保存 policy
                    agent.policy.save_weights(f"{save_path}/agent_{i}_ma2c_policy_final.h5")
                elif args.algorithm.lower() == "mappo":
                    agent.policy.save_weights(f"{save_path}/agent_{i}_mappo_policy_final.weights.h5")
                else:
                    # MADDPG：保存 actor/critic 及其 target
                    agent.actor.model.save_weights(f"{save_path}/agent_{i}_actor_final_weights.h5")
                    agent.critic.model.save_weights(f"{save_path}/agent_{i}_critic_final_weights.h5")
                    agent.actor.target_model.save_weights(f"{save_path}/agent_{i}_actor_target_final_weights.h5")
                    agent.critic.target_model.save_weights(f"{save_path}/agent_{i}_critic_target_final_weights.h5")
            print("Final model weights saved.")

            # 关闭TensorBoard writers
            train_writer.close()
            episode_writer.close()
            print(f"📊 TensorBoard logs saved to: {log_dir}")
            break

if __name__ == '__main__':
    args = parse_args()
    train(args)
