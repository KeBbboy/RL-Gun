"""
训练脚本
支持多种算法和配置
"""

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ConfigLoader, load_config_from_args, TrainingLogger, print_training_header, print_episode_summary, print_training_complete
from src.environment import BuildEnvironment
from trucks_and_drones.config import cfg


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Multi-Agent RL Training")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--algorithm", type=str, default="maddpg",
                       choices=["maddpg", "ma2c", "coma", "cima", "mappo"],
                       help="Algorithm to use")
    parser.add_argument("--environment", type=str, default=None,
                       choices=["small", "medium", "large"],
                       help="Environment size preset")
    
    # 训练参数（可覆盖配置文件）
    parser.add_argument("--num-episodes", type=int, default=None,
                       help="Number of episodes")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--lr-actor", type=float, default=None,
                       help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=None,
                       help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=None,
                       help="Discount factor")
    
    # 特性开关
    parser.add_argument("--use-per", action="store_true", default=None,
                       help="Use Prioritized Experience Replay")
    parser.add_argument("--use-iam", action="store_true", default=None,
                       help="Use Invalid Action Masking")
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="Enable visualization")
    parser.add_argument("--visualize-interval", type=int, default=1,
                       help="Visualize every N episodes")
    
    # 日志和保存
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for TensorBoard logs")
    parser.add_argument("--save-rate", type=int, default=None,
                       help="Save model every N episodes")
    
    return parser.parse_args()


def build_environment(config):
    """
    根据配置构建环境
    
    Args:
        config: 配置字典
        
    Returns:
        构建好的环境
    """
    env_cfg = config['environment']
    features = config.get('features', {})
    
    # 观测和动作配置
    obs_contin = ['TW', 'time', 'n_items', 'deadline']
    obs_discrete = ['ET', 'ED', 'NT', 'ND', 'truck_node', 'drone_node', 'delta']
    tandem_fields = obs_discrete + obs_contin + ['LT_time', 'LD_time']
    
    # 构建环境
    builder = (
        BuildEnvironment(
            "VRPD",
            enable_visualization=features.get('visualize', False)
        )
        .trucks(num=env_cfg['num_trucks'])
        .drones(num=env_cfg['num_drones'])
        .depots(num=env_cfg['num_depots'])
        .customers(num=env_cfg['num_customers'])
        .visuals()
        .observations(
            contin_inputs=obs_contin,
            discrete_inputs=obs_discrete,
            combine_per_index=[tandem_fields for _ in range(env_cfg['num_trucks'])],
            combine_per_type=['vehicles', 'customers', 'depots'],
            flatten=True,
            output_as_array=False,
        )
        .actions()
        .rewards()
        .compile()
    )
    
    env = builder.build()
    return env, builder


def create_trainers(algorithm, n_agents, obs_shape_n, act_space_list, global_obs_dim, config, builder):
    """
    创建训练器
    
    Args:
        algorithm: 算法名称
        n_agents: 智能体数量
        obs_shape_n: 观测空间形状列表
        act_space_list: 动作空间列表
        global_obs_dim: 全局观测维度
        config: 配置字典
        builder: 环境构建器
        
    Returns:
        训练器列表
    """
    # 动态导入算法
    if algorithm == "maddpg":
        from src.algorithms.maddpg import MADDPGAgentTrainer as TrainerClass
    elif algorithm == "ma2c":
        from src.algorithms.ma2c import MA2CAgentTrainer as TrainerClass
    elif algorithm == "mappo":
        from src.algorithms.mappo import MAPPOAgentTrainer as TrainerClass
    elif algorithm == "coma":
        from src.algorithms.coma import COMAAgentTrainer as TrainerClass
    elif algorithm == "cima":
        from src.algorithms.cima import CIMAAgentTrainer as TrainerClass
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"🤖 Using algorithm: {algorithm.upper()}")
    
    # 创建参数对象（兼容旧接口）
    class Args:
        pass
    
    args = Args()
    train_cfg = config['training']
    env_cfg = config['environment']
    features = config.get('features', {})
    network_cfg = config.get('network', {})

    # 设置参数
    for key, value in train_cfg.items():
        setattr(args, key, value)

    # 设置epsilon（初始值使用epsilon_max）
    args.epsilon = train_cfg.get('epsilon_max', 1.0)

    # 设置网络参数
    if 'actor' in network_cfg:
        args.hidden_units_actor = network_cfg['actor'].get('hidden_units', [256, 128])
        args.activation = network_cfg['actor'].get('activation', 'relu')
    else:
        args.hidden_units_actor = [256, 128]
        args.activation = 'relu'

    if 'critic' in network_cfg:
        args.hidden_units_critic = network_cfg['critic'].get('hidden_units', [512, 256])
    else:
        args.hidden_units_critic = [512, 256]

    args.use_per = features.get('use_per', False)
    args.use_iam = features.get('use_iam', False)
    
    # 创建训练器
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
                num_trucks=env_cfg['num_trucks'],
                num_drones=env_cfg['num_drones'],
                total_nodes=env_cfg['num_depots'] + env_cfg['num_customers']
            )
        )
    
    return trainers


def get_epsilon(step, warmup_steps=1500, eps_max=1.0, eps_min=0.1, decay_steps=30000):
    """计算epsilon值（用于探索）"""
    if step < warmup_steps:
        return eps_max
    if decay_steps <= 0:
        return eps_min
    decay_ratio = (step - warmup_steps) / float(decay_steps)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    return eps_min + (eps_max - eps_min) * np.exp(-3 * decay_ratio)


def pad1d_list(xs, fill=0.0):
    """
    将不同长度的1D数组填充到相同长度

    Args:
        xs: 列表，包含不同长度的数组
        fill: 填充值

    Returns:
        填充后的numpy数组和最大长度
    """
    if not xs:
        return np.array([]), 0
    arrs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in xs]
    maxlen = max(a.shape[0] for a in arrs)
    out = [np.pad(a, (0, maxlen - a.shape[0]), mode="constant", constant_values=fill) for a in arrs]
    return np.stack(out, axis=0), maxlen


def train(args):
    """主训练函数"""
    # 加载配置
    config = load_config_from_args(args)
    
    # 打印训练开始信息
    print_training_header()
    print(f"📋 Configuration:")
    print(f"   Algorithm: {config['algorithm']['name']}")
    print(f"   Environment: {config['environment']['num_trucks']}T + {config['environment']['num_drones']}D, {config['environment']['num_customers']} customers")
    print(f"   Episodes: {config['training']['num_episodes']}")
    print(f"   Features: PER={config['features'].get('use_per')}, IAM={config['features'].get('use_iam')}")
    
    # 创建日志记录器
    logger = TrainingLogger(
        log_dir=config['logging']['log_dir'],
        exp_name=config['logging']['exp_name'],
        use_tensorboard=config['logging'].get('tensorboard', True)
    )
    
    # 保存训练配置到日志目录
    logger.save_config(config, args)
    
    # 构建环境
    env, builder = build_environment(config)
    n_agents = config['environment']['num_trucks'] + config['environment']['num_drones']
    
    # 获取观测和动作空间
    obs_n, global_obs = env.reset()
    obs_shape_n = [obs.shape for obs in obs_n]
    global_obs_dim = global_obs.shape[0] if hasattr(global_obs, 'shape') else len(global_obs)
    
    raw_aspace = env.act_decoder.action_space()
    act_space_list = raw_aspace.spaces[0].spaces
    
    print(f"\n🔍 Environment Info:")
    print(f"   Agents: {n_agents}")
    print(f"   Observation shapes: {obs_shape_n}")
    print(f"   Global observation dim: {global_obs_dim}")
    print(f"   Action spaces: {[space.n for space in act_space_list]}")
    
    # 创建训练器
    trainers = create_trainers(
        algorithm=config['algorithm']['name'],
        n_agents=n_agents,
        obs_shape_n=obs_shape_n,
        act_space_list=act_space_list,
        global_obs_dim=global_obs_dim,
        config=config,
        builder=builder
    )
    
    # 训练循环
    train_cfg = config['training']
    features = config['features']
    algorithm_name = config['algorithm']['name'].lower()

    episode_rewards = [0.0]
    episode_step = 0
    train_step = 0
    rollout_step = 0  # 用于On-policy算法的rollout计数
    start_time = time.time()
    
    obs_n, global_obs = env.reset()
    
    # 初始化可视化
    if features.get('visualize', False):
        env.visualizer.total_training_steps = 0
        env.visualizer.visualize_step(
            episode=0, step=0, slow_down_pls=True,
            last_actions=None, last_rewards=None
        )
    
    logger.log_text(f"Training started with {config['algorithm']['name'].upper()}")
    
    # 创建进度条
    pbar = tqdm(total=train_cfg['num_episodes'], desc="Training Progress", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    while len(episode_rewards) <= train_cfg['num_episodes']:
        # 生成动作掩码
        if features.get('use_iam', False):
            mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)]
        else:
            mask_n = [None] * n_agents
        
        # Epsilon探索
        ws = train_cfg.get('warmup_steps', 0)
        eps_max = train_cfg.get('epsilon_max', 1.0)
        eps_min = train_cfg.get('epsilon_min', 0.1)
        decay_steps = train_cfg.get('epsilon_decay_steps', 30000)
        cur_eps = get_epsilon(train_step, warmup_steps=ws, eps_max=eps_max, eps_min=eps_min, decay_steps=decay_steps)
        
        for tr in trainers:
            if hasattr(tr, 'actor') and hasattr(tr.actor, 'epsilon'):
                tr.actor.epsilon = cur_eps
        
        # 智能体决策
        if algorithm_name in ['ma2c', 'mappo']:
            action_n = [tr.action(o, global_obs, m) for tr, o, m in zip(trainers, obs_n, mask_n)]
        else:
            action_n = [tr.action(o, m) for tr, o, m in zip(trainers, obs_n, mask_n)]
        
        # 环境步进
        (new_obs_n, new_global_obs), rew_n, done, info = env.step(action_n)
        
        # 获取执行的动作
        executed_action_n = info.get('executed_actions_multihead', action_n)

        # 存储经验
        next_obs_all_pad, _ = pad1d_list(new_obs_n)
        next_mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)] if features.get('use_iam', False) else [None] * n_agents
        
        for i, tr in enumerate(trainers):
            tr.experience(
                obs=obs_n[i],
                global_obs=global_obs,
                act_all=executed_action_n,
                rew=rew_n[i],
                next_obs=new_obs_n[i],
                next_obs_all=next_obs_all_pad,
                next_global_obs=new_global_obs,
                done=done,
                mask_all=mask_n,
                next_mask_all=next_mask_n
            )
        
        obs_n = new_obs_n
        global_obs = new_global_obs
        episode_step += 1
        train_step += 1
        rollout_step += 1  # 增加rollout计数
        # 使用所有智能体的平均奖励
        episode_rewards[-1] += np.mean(rew_n)
        
        # 可视化
        if features.get('visualize', False) and (len(episode_rewards)-1) % features.get('visualize_interval', 1) == 0:
            env.visualizer.visualize_step(
                episode=len(episode_rewards)-1,
                step=episode_step,
                slow_down_pls=(episode_step < 10),
                last_actions=executed_action_n,
                last_rewards=rew_n
            )
        
        # 训练更新
        if algorithm_name in ["ma2c", "mappo"]:
            # On-policy算法 - 每rollout_len步更新一次
            rollout_len = train_cfg.get('rollout_len', 25)
            if rollout_step >= rollout_len:
                for tr in trainers:
                    losses = tr.update(trainers, train_step)
                    if losses is not None and len(losses) == 2:
                        critic_loss, actor_loss = losses
                        logger.log_train_step(
                            step=train_step,
                            metrics={
                                'Critic损失': float(critic_loss),
                                'Actor损失': float(actor_loss)
                            },
                            agent_index=tr.agent_index
                        )
                rollout_step = 0  # 重置rollout计数
        else:
            # Off-policy算法
            if (len(trainers[0].buffer) >= train_cfg.get('min_buffer_size', 5000) and
                train_step % train_cfg.get('update_freq', 10) == 0 and
                train_step >= train_cfg.get('warmup_steps', 0)):

                for tr in trainers:
                    tr.preupdate()

                updates = int(train_cfg.get('updates_per_call', 1))
                for _ in range(updates):
                    for tr in trainers:
                        losses = tr.update(trainers, train_step, target_update_freq=train_cfg.get('target_update_freq', 100))
                        if losses is not None and len(losses) == 2:
                            critic_loss, actor_loss = losses
                            logger.log_train_step(
                                step=train_step,
                                metrics={
                                    'Critic损失': float(critic_loss),
                                    'Actor损失': float(actor_loss)
                                },
                                agent_index=tr.agent_index
                            )

        # Episode结束
        if done:

            # 获取统计信息
            cost_stats = env.reward_calc.get_episode_statistics()

            # 记录到logger
            logger.log_episode(
                episode=len(episode_rewards)-1,
                metrics={
                    '总奖励': episode_rewards[-1],
                    '回合长度': episode_step,
                    '总成本': cost_stats['total_cost'],
                    '旅行成本': cost_stats['travel_cost'],
                    '延迟惩罚': cost_stats['delay_penalty'],
                    '未服务惩罚': cost_stats['unserved_penalty'],
                    '服务数量': cost_stats['served_count'],
                    '未服务数量': cost_stats['unserved_count'],
                }
            )

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'Reward': f"{episode_rewards[-1]:.2f}",
                'Cost': f"{cost_stats['total_cost']:.2f}",
                'Served': f"{cost_stats['served_count']}/{cost_stats['served_count'] + cost_stats['unserved_count']}",
                'Steps': episode_step
            })

            # 打印episode摘要（每N个episode）
            if (len(episode_rewards)-1) % 10 == 0:
                print_episode_summary(
                    episode=len(episode_rewards)-1,
                    total_episodes=train_cfg['num_episodes'],
                    reward=episode_rewards[-1],
                    length=episode_step,
                    metrics=cost_stats
                )

            # 保存模型
            current_episode = len(episode_rewards) - 1
            if current_episode % config['logging'].get('save_rate', 2000) == 0 and current_episode > 0:
                save_dir = config['logging']['save_dir']
                os.makedirs(save_dir, exist_ok=True)

                for i, agent in enumerate(trainers):
                    if algorithm_name == "ma2c":
                        agent.policy.save_weights(f"{save_dir}/agent_{i}_policy_ep{current_episode}.h5")
                    elif algorithm_name == "mappo":
                        agent.policy.save_weights(f"{save_dir}/agent_{i}_mappo_policy_ep{current_episode}.weights.h5")
                    else:
                        agent.actor.model.save_weights(f"{save_dir}/agent_{i}_actor_ep{current_episode}.h5")
                        agent.critic.model.save_weights(f"{save_dir}/agent_{i}_critic_ep{current_episode}.h5")

                logger.log_text(f"Model saved at episode {current_episode}")

            # 重置
            obs_n, global_obs = env.reset()
            episode_step = 0
            rollout_step = 0  # 重置rollout计数
            episode_rewards.append(0.0)
    
    # 关闭进度条
    pbar.close()
    
    # 训练完成
    total_time = time.time() - start_time
    print_training_complete(total_time, train_cfg['num_episodes'])
    
    # 保存最终模型
    save_dir = config['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    
    for i, agent in enumerate(trainers):
        if algorithm_name == "ma2c":
            agent.policy.save_weights(f"{save_dir}/agent_{i}_ma2c_policy_final.h5")
        elif algorithm_name == "mappo":
            agent.policy.save_weights(f"{save_dir}/agent_{i}_mappo_policy_final.weights.h5")
        else:
            agent.actor.model.save_weights(f"{save_dir}/agent_{i}_actor_final.h5")
            agent.critic.model.save_weights(f"{save_dir}/agent_{i}_critic_final.h5")
    
    logger.log_text("Final models saved")
    
    # 保存训练摘要
    summary_data = {
        'algorithm': config['algorithm']['name'],
        'total_episodes': train_cfg['num_episodes'],
        'total_time_seconds': total_time,
        'total_time_formatted': f"{int(total_time//3600)}h {int((total_time%3600)//60)}m {int(total_time%60)}s",
        'avg_time_per_episode': total_time / train_cfg['num_episodes'] if train_cfg['num_episodes'] > 0 else 0.0,
        'num_agents': n_agents,
        'num_trucks': config['environment']['num_trucks'],
        'num_drones': config['environment']['num_drones'],
        'num_customers': config['environment']['num_customers'],
        'batch_size': train_cfg['batch_size'],
        'update_freq': train_cfg['update_freq'],
        'updates_per_call': train_cfg.get('updates_per_call', 1),
        'lr_actor': train_cfg['lr_actor'],
        'lr_critic': train_cfg['lr_critic'],
        'use_per': features.get('use_per', False),
        'use_iam': features.get('use_iam', False),
        'final_reward': float(episode_rewards[-2]) if len(episode_rewards) > 1 else 0.0,
        'model_save_dir': save_dir,
    }
    logger.save_training_summary(summary_data)
    
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)

