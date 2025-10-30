"""
评估脚本
用于评估训练好的模型
"""

import argparse
import os
import sys
import numpy as np
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ConfigLoader, load_json
from src.environment import BuildEnvironment
from trucks_and_drones.config import cfg


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["maddpg", "iql", "ma2c", "coma", "cima", "mappo"],
                       help="Algorithm used for training")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file (should match training config)")
    parser.add_argument("--environment", type=str, default="medium",
                       help="Environment size")
    parser.add_argument("--num-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--visualize", action="store_true",
                       help="Enable visualization")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy")
    
    return parser.parse_args()


def load_models(trainers, checkpoint_dir, algorithm):
    """
    加载训练好的模型
    
    Args:
        trainers: 训练器列表
        checkpoint_dir: 检查点目录
        algorithm: 算法名称
    """
    print(f"📥 Loading models from {checkpoint_dir}")
    
    for i, agent in enumerate(trainers):
        if algorithm == "ma2c":
            weight_file = os.path.join(checkpoint_dir, f"agent_{i}_ma2c_policy_final.h5")
            if os.path.exists(weight_file):
                agent.policy.load_weights(weight_file)
                print(f"   ✓ Loaded agent {i} policy")
        elif algorithm == "mappo":
            weight_file = os.path.join(checkpoint_dir, f"agent_{i}_mappo_policy_final.weights.h5")
            if os.path.exists(weight_file):
                agent.policy.load_weights(weight_file)
                print(f"   ✓ Loaded agent {i} policy")
        else:
            actor_file = os.path.join(checkpoint_dir, f"agent_{i}_actor_final.h5")
            critic_file = os.path.join(checkpoint_dir, f"agent_{i}_critic_final.h5")
            if os.path.exists(actor_file):
                agent.actor.model.load_weights(actor_file)
                print(f"   ✓ Loaded agent {i} actor")
            if os.path.exists(critic_file):
                agent.critic.model.load_weights(critic_file)
                print(f"   ✓ Loaded agent {i} critic")


def evaluate(args):
    """主评估函数"""
    # 加载配置
    loader = ConfigLoader()
    config = loader.load_config(
        algorithm=args.algorithm,
        environment=args.environment,
        custom_config=args.config
    )
    
    # 覆盖可视化设置
    config['features']['visualize'] = args.visualize
    
    print(f"\n🎯 Evaluation Configuration:")
    print(f"   Algorithm: {args.algorithm}")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Deterministic: {args.deterministic}")
    print(f"   Visualize: {args.visualize}")
    
    # 构建环境
    env_cfg = config['environment']
    obs_contin = ['TW', 'time', 'n_items', 'deadline']
    obs_discrete = ['ET', 'ED', 'NT', 'ND', 'truck_node', 'drone_node', 'delta']
    tandem_fields = obs_discrete + obs_contin + ['LT_time', 'LD_time']
    
    builder = (
        BuildEnvironment(
            "VRPD",
            enable_visualization=args.visualize
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
    n_agents = env_cfg['num_trucks'] + env_cfg['num_drones']
    
    # 获取空间信息
    obs_n, global_obs = env.reset()
    obs_shape_n = [obs.shape for obs in obs_n]
    global_obs_dim = global_obs.shape[0] if hasattr(global_obs, 'shape') else len(global_obs)
    raw_aspace = env.act_decoder.action_space()
    act_space_list = raw_aspace.spaces[0].spaces
    
    # 创建训练器（用于加载模型）
    if args.algorithm == "maddpg":
        from src.algorithms.maddpg import MADDPGAgentTrainer as TrainerClass
    elif args.algorithm == "ma2c":
        from src.algorithms.ma2c import MA2CAgentTrainer as TrainerClass
    elif args.algorithm == "mappo":
        from src.algorithms.mappo import MAPPOAgentTrainer as TrainerClass
    elif args.algorithm == "coma":
        from src.algorithms.coma import COMAAgentTrainer as TrainerClass
    elif args.algorithm == "cima":
        from src.algorithms.cima import CIMAAgentTrainer as TrainerClass
    
    class Args:
        pass
    
    dummy_args = Args()
    for key, value in config['training'].items():
        setattr(dummy_args, key, value)
    dummy_args.use_per = False
    dummy_args.use_iam = config['features'].get('use_iam', False)
    
    trainers = []
    for i in range(n_agents):
        trainers.append(
            TrainerClass(
                name=f"agent_{i}",
                obs_shape_n=obs_shape_n,
                act_space_list=act_space_list,
                agent_index=i,
                args=dummy_args,
                global_obs_dim=global_obs_dim,
                num_trucks=env_cfg['num_trucks'],
                num_drones=env_cfg['num_drones'],
                total_nodes=env_cfg['num_depots'] + env_cfg['num_customers']
            )
        )
    
    # 加载模型
    load_models(trainers, args.checkpoint_dir, args.algorithm)
    
    # 评估
    print(f"\n🔍 Starting evaluation...")
    
    all_rewards = []
    all_costs = []
    all_served_counts = []
    all_episode_lengths = []
    
    for episode in range(args.num_episodes):
        obs_n, global_obs = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        
        while not done:
            # 生成动作（确定性或随机）
            if args.algorithm in ['ma2c', 'mappo']:
                if args.deterministic:
                    # 对于on-policy算法，直接使用网络输出
                    action_n = [tr.action(o, global_obs, None) for tr, o in zip(trainers, obs_n)]
                else:
                    action_n = [tr.action(o, global_obs, None) for tr, o in zip(trainers, obs_n)]
            else:
                # 关闭探索
                for tr in trainers:
                    if hasattr(tr, 'actor') and hasattr(tr.actor, 'epsilon'):
                        tr.actor.epsilon = 0.0
                
                action_n = [tr.action(o, None) for tr, o in zip(trainers, obs_n)]
            
            # 执行动作
            (obs_n, global_obs), rew_n, done, info = env.step(action_n)
            episode_reward += rew_n[0]
            episode_step += 1
            
            # 可视化
            if args.visualize:
                env.visualizer.visualize_step(
                    episode=episode,
                    step=episode_step,
                    slow_down_pls=True,
                    last_actions=action_n,
                    last_rewards=rew_n
                )
        
        # 收集统计信息
        cost_stats = env.reward_calc.get_episode_statistics()
        all_rewards.append(episode_reward)
        all_costs.append(cost_stats['total_cost'])
        all_served_counts.append(cost_stats['served_count'])
        all_episode_lengths.append(episode_step)
        
        print(f"Episode {episode+1}/{args.num_episodes}: Reward={episode_reward:.2f}, Cost={cost_stats['total_cost']:.2f}, Served={cost_stats['served_count']}")
    
    # 计算统计结果
    results = {
        'algorithm': args.algorithm,
        'num_episodes': args.num_episodes,
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_cost': float(np.mean(all_costs)),
        'std_cost': float(np.std(all_costs)),
        'mean_served': float(np.mean(all_served_counts)),
        'mean_episode_length': float(np.mean(all_episode_lengths)),
        'all_rewards': [float(r) for r in all_rewards],
        'all_costs': [float(c) for c in all_costs],
    }
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Cost: {results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
    print(f"Mean Served Nodes: {results['mean_served']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.2f}")
    print(f"{'='*80}\n")
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {args.output}")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)

