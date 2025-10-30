"""
评估指标模块
提供各种评估指标的计算
"""

import numpy as np
from typing import List, Dict, Any, Optional


class PerformanceMetrics:
    """性能指标计算器"""
    
    @staticmethod
    def calculate_returns(rewards: List[float], gamma: float = 0.95) -> List[float]:
        """
        计算折扣回报
        
        Args:
            rewards: 奖励列表
            gamma: 折扣因子
            
        Returns:
            折扣回报列表
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    @staticmethod
    def calculate_gae(
        rewards: List[float],
        values: List[float],
        next_value: float,
        gamma: float = 0.95,
        lambda_: float = 0.95
    ) -> np.ndarray:
        """
        计算广义优势估计 (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励列表
            values: 价值估计列表
            next_value: 下一个状态的价值
            gamma: 折扣因子
            lambda_: GAE lambda参数
            
        Returns:
            优势估计数组
        """
        advantages = []
        gae = 0
        
        # 添加next_value到values末尾
        values = list(values) + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    @staticmethod
    def moving_average(data: List[float], window: int = 100) -> List[float]:
        """
        计算移动平均
        
        Args:
            data: 数据列表
            window: 窗口大小
            
        Returns:
            移动平均列表
        """
        if len(data) < window:
            return data
        
        ma = []
        for i in range(len(data)):
            if i < window:
                ma.append(np.mean(data[:i+1]))
            else:
                ma.append(np.mean(data[i-window+1:i+1]))
        return ma


class VRPDMetrics:
    """VRPD问题特定的指标"""
    
    @staticmethod
    def calculate_total_cost(
        travel_cost: float,
        delay_penalty: float,
        unserved_penalty: float
    ) -> float:
        """
        计算总成本
        
        Args:
            travel_cost: 旅行成本
            delay_penalty: 延误惩罚
            unserved_penalty: 未服务惩罚
            
        Returns:
            总成本
        """
        return travel_cost + delay_penalty + unserved_penalty
    
    @staticmethod
    def calculate_service_rate(served: int, total: int) -> float:
        """
        计算服务率
        
        Args:
            served: 已服务节点数
            total: 总节点数
            
        Returns:
            服务率 (0-1)
        """
        if total == 0:
            return 0.0
        return served / total
    
    @staticmethod
    def calculate_vehicle_utilization(
        active_vehicles: int,
        total_vehicles: int
    ) -> float:
        """
        计算车辆利用率
        
        Args:
            active_vehicles: 活动车辆数
            total_vehicles: 总车辆数
            
        Returns:
            利用率 (0-1)
        """
        if total_vehicles == 0:
            return 0.0
        return active_vehicles / total_vehicles
    
    @staticmethod
    def format_episode_stats(stats: Dict[str, Any]) -> str:
        """
        格式化episode统计信息为可读字符串
        
        Args:
            stats: 统计字典
            
        Returns:
            格式化的字符串
        """
        lines = [
            "="*60,
            "Episode Statistics",
            "="*60,
        ]
        
        if 'total_cost' in stats:
            lines.append(f"Total Cost (TC): {stats['total_cost']:.2f}")
        if 'travel_cost' in stats:
            lines.append(f"Travel Cost: {stats['travel_cost']:.2f}")
        if 'delay_penalty' in stats:
            lines.append(f"Delay Penalty: {stats['delay_penalty']:.2f}")
        if 'unserved_penalty' in stats:
            lines.append(f"Unserved Penalty: {stats['unserved_penalty']:.2f}")
        
        lines.append("-"*60)
        
        if 'served_count' in stats and 'unserved_count' in stats:
            total = stats['served_count'] + stats['unserved_count']
            service_rate = VRPDMetrics.calculate_service_rate(
                stats['served_count'], total
            )
            lines.append(f"Served Nodes: {stats['served_count']}/{total} ({service_rate*100:.1f}%)")
        
        if 'truck_travel_time' in stats:
            lines.append(f"Truck Travel Time: {stats['truck_travel_time']:.2f}")
        if 'drone_travel_time' in stats:
            lines.append(f"Drone Travel Time: {stats['drone_travel_time']:.2f}")
        
        lines.append("="*60)
        
        return "\n".join(lines)


def normalize_rewards(rewards: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    标准化奖励
    
    Args:
        rewards: 奖励数组
        epsilon: 防止除零的小常数
        
    Returns:
        标准化后的奖励
    """
    mean = np.mean(rewards)
    std = np.std(rewards)
    return (rewards - mean) / (std + epsilon)


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算解释方差 (explained variance)
    用于评估价值函数的拟合质量
    
    Args:
        y_pred: 预测值
        y_true: 真实值
        
    Returns:
        解释方差
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_y

