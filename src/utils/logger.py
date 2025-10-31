"""
日志工具模块
提供训练日志记录和TensorBoard集成
"""

import os
import json
import yaml
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(
        self,
        log_dir: str,
        exp_name: str,
        use_tensorboard: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            exp_name: 实验名称
            use_tensorboard: 是否使用TensorBoard
        """
        self.use_tensorboard = use_tensorboard
        
        # 创建带时间戳的日志目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{exp_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建TensorBoard writers
        if self.use_tensorboard:
            self.train_writer = tf.summary.create_file_writer(
                os.path.join(self.log_dir, 'train')
            )
            self.episode_writer = tf.summary.create_file_writer(
                os.path.join(self.log_dir, 'episode')
            )
            print(f"📊 TensorBoard logs: {self.log_dir}")
            print(f"📈 Run: tensorboard --logdir {self.log_dir}")
        
        # 文本日志文件
        self.log_file = os.path.join(self.log_dir, 'training.log')
    
    def save_config(self, config: Dict[str, Any], args: Optional[Any] = None):
        """
        保存训练配置到日志目录
        
        Args:
            config: 配置字典
            args: 命令行参数对象（可选）
        """
        # 保存完整配置为YAML格式
        config_file = os.path.join(self.log_dir, 'config.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存为JSON格式（方便程序读取）
        config_json_file = os.path.join(self.log_dir, 'config.json')
        with open(config_json_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 如果有命令行参数，也保存下来
        if args is not None:
            args_file = os.path.join(self.log_dir, 'args.txt')
            with open(args_file, 'w', encoding='utf-8') as f:
                f.write("命令行参数:\n")
                f.write("="*50 + "\n")
                for key, value in vars(args).items():
                    f.write(f"{key}: {value}\n")
        
        print(f"💾 Configuration saved to: {self.log_dir}")
        print(f"   - config.yaml")
        print(f"   - config.json")
        if args is not None:
            print(f"   - args.txt")
    
    def log_train_step(
        self,
        step: int,
        metrics: Dict[str, float],
        agent_index: Optional[int] = None
    ):
        """
        记录训练步骤的指标
        
        Args:
            step: 训练步数
            metrics: 指标字典，如 {'critic_loss': 0.5, 'actor_loss': -0.3}
            agent_index: 智能体索引（可选）
        """
        if not self.use_tensorboard:
            return
        
        with self.train_writer.as_default():
            for name, value in metrics.items():
                if agent_index is not None:
                    tag = f'Agent_{agent_index}/{name}'
                else:
                    tag = name
                tf.summary.scalar(tag, float(value), step=step)
        
        self.train_writer.flush()
    
    def log_episode(
        self,
        episode: int,
        metrics: Dict[str, float]
    ):
        """
        记录episode的指标
        
        Args:
            episode: episode编号
            metrics: 指标字典
        """
        if not self.use_tensorboard:
            return
        
        with self.episode_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, float(value), step=episode)
        
        self.episode_writer.flush()
    
    def log_text(self, message: str, print_console: bool = True):
        """
        记录文本消息
        
        Args:
            message: 消息内容
            print_console: 是否同时打印到控制台
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if print_console:
            print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def save_training_summary(self, summary_data: Dict[str, Any]):
        """
        保存训练摘要信息 
        
        Args:
            summary_data: 训练摘要数据，包括总时间、总episodes等
        """
        summary_file = os.path.join(self.log_dir, 'training_summary.json')
        
        # 添加时间戳
        summary_data['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存为可读的文本格式
        summary_txt_file = os.path.join(self.log_dir, 'training_summary.txt')
        with open(summary_txt_file, 'w', encoding='utf-8') as f:
            f.write("训练摘要\n")
            f.write("="*60 + "\n\n")
            for key, value in summary_data.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📝 Training summary saved to: {self.log_dir}")
    
    def close(self):
        """关闭日志记录器"""
        if self.use_tensorboard:
            self.train_writer.close()
            self.episode_writer.close()


class MetricsTracker:
    """指标追踪器，用于统计和计算移动平均"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化指标追踪器
        
        Args:
            window_size: 移动平均窗口大小
        """
        self.window_size = window_size
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """
        更新指标
        
        Args:
            name: 指标名称
            value: 指标值
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # 保持窗口大小
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)
    
    def get_mean(self, name: str) -> Optional[float]:
        """
        获取指标的平均值
        
        Args:
            name: 指标名称
            
        Returns:
            平均值，如果指标不存在则返回None
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_latest(self, name: str) -> Optional[float]:
        """
        获取指标的最新值
        
        Args:
            name: 指标名称
            
        Returns:
            最新值，如果指标不存在则返回None
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        return self.metrics[name][-1]
    
    def reset(self, name: Optional[str] = None):
        """
        重置指标
        
        Args:
            name: 指标名称，如果为None则重置所有指标
        """
        if name is None:
            self.metrics = {}
        elif name in self.metrics:
            self.metrics[name] = []
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有指标的统计摘要
        
        Returns:
            包含mean, min, max, latest的字典
        """
        summary = {}
        for name, values in self.metrics.items():
            if len(values) > 0:
                summary[name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1],
                    'count': len(values)
                }
        return summary


def print_training_header():
    """打印训练开始的标题"""
    print("\n" + "="*80)
    print(" " * 20 + "🚀 MULTI-AGENT TRAINING STARTED 🚀")
    print("="*80 + "\n")


def print_episode_summary(
    episode: int,
    total_episodes: int,
    reward: float,
    length: int,
    metrics: Dict[str, Any]
):
    """
    打印episode摘要
    
    Args:
        episode: episode编号
        total_episodes: 总episode数
        reward: episode总奖励
        length: episode长度
        metrics: 其他指标
    """
    print(f"\n{'='*80}")
    print(f"🏁 Episode {episode}/{total_episodes} Completed")
    print(f"{'='*80}")
    print(f"  📈 Total Reward: {reward:.2f}")
    print(f"  📏 Episode Length: {length}")
    
    if metrics:
        print(f"\n  📊 Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"     • {key}: {value:.4f}")
            else:
                print(f"     • {key}: {value}")
    print(f"{'='*80}\n")


def print_training_complete(total_time: float, total_episodes: int):
    """
    打印训练完成信息
    
    Args:
        total_time: 总训练时间（秒）
        total_episodes: 总episode数
    """
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*80)
    print(" " * 20 + "🎓 TRAINING COMPLETED 🎓")
    print("="*80)
    print(f"  ⏱️  Total Time: {hours}h {minutes}m {seconds}s")
    print(f"  📚 Total Episodes: {total_episodes}")
    print(f"  ⚡ Average Time/Episode: {total_time/total_episodes:.2f}s")
    print("="*80 + "\n")

