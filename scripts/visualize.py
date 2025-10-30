"""
可视化工具
用于可视化训练过程和结果
"""

import argparse
import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Visualization tool")
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=["training_curves", "comparison", "episode"],
                       help="Visualization mode")
    parser.add_argument("--input", type=str, required=True,
                       help="Input file or directory")
    parser.add_argument("--output", type=str, default="visualization.png",
                       help="Output image file")
    parser.add_argument("--smooth", type=int, default=100,
                       help="Smoothing window size")
    
    return parser.parse_args()


def smooth_curve(data, window=100):
    """平滑曲线"""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        if i < window:
            smoothed.append(np.mean(data[:i+1]))
        else:
            smoothed.append(np.mean(data[i-window+1:i+1]))
    return smoothed


def plot_training_curves(input_file, output_file, smooth_window=100):
    """绘制训练曲线"""
    # 加载数据
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    rewards = data.get('rewards', [])
    losses = data.get('losses', {})
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # 奖励曲线
    if rewards:
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, label='Raw')
        if len(rewards) > smooth_window:
            smoothed = smooth_curve(rewards, smooth_window)
            ax.plot(smoothed, label=f'Smoothed (window={smooth_window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Critic Loss
    if 'critic_loss' in losses:
        ax = axes[0, 1]
        critic_loss = losses['critic_loss']
        ax.plot(critic_loss, alpha=0.3, label='Raw')
        if len(critic_loss) > smooth_window:
            smoothed = smooth_curve(critic_loss, smooth_window)
            ax.plot(smoothed, label=f'Smoothed (window={smooth_window})')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Critic Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Actor Loss
    if 'actor_loss' in losses:
        ax = axes[1, 0]
        actor_loss = losses['actor_loss']
        ax.plot(actor_loss, alpha=0.3, label='Raw')
        if len(actor_loss) > smooth_window:
            smoothed = smooth_curve(actor_loss, smooth_window)
            ax.plot(smoothed, label=f'Smoothed (window={smooth_window})')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Actor Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 移动平均奖励
    if rewards and len(rewards) >= 100:
        ax = axes[1, 1]
        windows = [10, 50, 100, 200]
        for w in windows:
            if len(rewards) >= w:
                smoothed = smooth_curve(rewards, w)
                ax.plot(smoothed, label=f'Window={w}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Moving Average Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to {output_file}")


def plot_comparison(input_dir, output_file):
    """比较多个算法的性能"""
    # 查找所有评估结果文件
    result_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.json') and 'evaluation' in file:
            result_files.append(os.path.join(input_dir, file))
    
    if not result_files:
        print("No evaluation result files found")
        return
    
    # 加载所有结果
    results = {}
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            algo = data.get('algorithm', os.path.basename(file).split('_')[0])
            results[algo] = data
    
    # 创建比较图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Algorithm Comparison', fontsize=16)
    
    algorithms = list(results.keys())
    
    # 平均奖励
    ax = axes[0, 0]
    means = [results[algo]['mean_reward'] for algo in algorithms]
    stds = [results[algo]['std_reward'] for algo in algorithms]
    ax.bar(algorithms, means, yerr=stds, capsize=5)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Average Reward Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 平均成本
    ax = axes[0, 1]
    means = [results[algo]['mean_cost'] for algo in algorithms]
    stds = [results[algo]['std_cost'] for algo in algorithms]
    ax.bar(algorithms, means, yerr=stds, capsize=5, color='orange')
    ax.set_ylabel('Mean Cost')
    ax.set_title('Average Cost Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 服务节点数
    ax = axes[1, 0]
    means = [results[algo]['mean_served'] for algo in algorithms]
    ax.bar(algorithms, means, color='green')
    ax.set_ylabel('Mean Served Nodes')
    ax.set_title('Service Rate Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Episode长度
    ax = axes[1, 1]
    means = [results[algo]['mean_episode_length'] for algo in algorithms]
    ax.bar(algorithms, means, color='purple')
    ax.set_ylabel('Mean Episode Length')
    ax.set_title('Efficiency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to {output_file}")


def plot_episode_details(input_file, output_file):
    """绘制单个episode的详细信息"""
    # 这个功能需要更详细的episode数据记录
    # 暂时提供框架
    print("Episode details visualization not yet implemented")
    print("This would show:")
    print("  - Vehicle trajectories")
    print("  - Node visit sequence")
    print("  - Cost breakdown over time")


def main():
    args = parse_args()
    
    if args.mode == "training_curves":
        plot_training_curves(args.input, args.output, args.smooth)
    elif args.mode == "comparison":
        plot_comparison(args.input, args.output)
    elif args.mode == "episode":
        plot_episode_details(args.input, args.output)


if __name__ == '__main__':
    main()

