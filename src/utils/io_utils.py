"""
IO工具模块
提供文件读写、模型保存加载等功能
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_dir(directory: str):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], file_path: str):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: str):
    """
    保存数据到pickle文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """
    从pickle文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_numpy(data: np.ndarray, file_path: str):
    """
    保存numpy数组
    
    Args:
        data: numpy数组
        file_path: 文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    np.save(file_path, data)


def load_numpy(file_path: str) -> np.ndarray:
    """
    加载numpy数组
    
    Args:
        file_path: 文件路径
        
    Returns:
        numpy数组
    """
    return np.load(file_path)


class ModelCheckpoint:
    """模型检查点管理器"""
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_to_keep: 最多保留的检查点数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        ensure_dir(checkpoint_dir)
        
        # 记录检查点信息
        self.checkpoint_info_file = os.path.join(checkpoint_dir, 'checkpoints.json')
        self.checkpoints = self._load_checkpoint_info()
    
    def _load_checkpoint_info(self) -> List[Dict[str, Any]]:
        """加载检查点信息"""
        if os.path.exists(self.checkpoint_info_file):
            return load_json(self.checkpoint_info_file)
        return []
    
    def _save_checkpoint_info(self):
        """保存检查点信息"""
        save_json(self.checkpoints, self.checkpoint_info_file)
    
    def save_checkpoint(
        self,
        agents: List[Any],
        episode: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        保存检查点
        
        Args:
            agents: 智能体列表
            episode: 当前episode
            metrics: 性能指标
        """
        checkpoint_name = f"checkpoint_ep{episode}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        ensure_dir(checkpoint_path)
        
        # 保存每个智能体的模型
        for i, agent in enumerate(agents):
            agent_dir = os.path.join(checkpoint_path, f"agent_{i}")
            ensure_dir(agent_dir)
            
            # 根据不同算法保存不同的模型
            if hasattr(agent, 'policy'):
                # MA2C, MAPPO等
                agent.policy.save_weights(
                    os.path.join(agent_dir, 'policy.weights.h5')
                )
            elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                # MADDPG等
                agent.actor.model.save_weights(
                    os.path.join(agent_dir, 'actor.weights.h5')
                )
                agent.critic.model.save_weights(
                    os.path.join(agent_dir, 'critic.weights.h5')
                )
                if hasattr(agent.actor, 'target_model'):
                    agent.actor.target_model.save_weights(
                        os.path.join(agent_dir, 'actor_target.weights.h5')
                    )
                if hasattr(agent.critic, 'target_model'):
                    agent.critic.target_model.save_weights(
                        os.path.join(agent_dir, 'critic_target.weights.h5')
                    )
        
        # 记录检查点信息
        checkpoint_info = {
            'name': checkpoint_name,
            'episode': episode,
            'path': checkpoint_path,
            'metrics': metrics or {}
        }
        self.checkpoints.append(checkpoint_info)
        
        # 保持最大数量限制
        if len(self.checkpoints) > self.max_to_keep:
            # 删除最旧的检查点
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint['path']):
                import shutil
                shutil.rmtree(old_checkpoint['path'])
        
        self._save_checkpoint_info()
        print(f"💾 Checkpoint saved: {checkpoint_name}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]['path']
    
    def get_best_checkpoint(self, metric_name: str = 'total_reward') -> Optional[str]:
        """
        获取最佳检查点路径
        
        Args:
            metric_name: 用于比较的指标名称
            
        Returns:
            最佳检查点路径
        """
        if not self.checkpoints:
            return None
        
        best_checkpoint = max(
            self.checkpoints,
            key=lambda x: x['metrics'].get(metric_name, float('-inf'))
        )
        return best_checkpoint['path']


def export_training_curves(
    rewards: List[float],
    losses: Dict[str, List[float]],
    output_file: str
):
    """
    导出训练曲线数据
    
    Args:
        rewards: 奖励列表
        losses: 损失字典
        output_file: 输出文件路径
    """
    data = {
        'rewards': rewards,
        'losses': losses
    }
    save_json(data, output_file)
    print(f"📊 Training curves exported to: {output_file}")


def load_training_curves(input_file: str) -> Dict[str, Any]:
    """
    加载训练曲线数据
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        训练曲线数据
    """
    return load_json(input_file)

