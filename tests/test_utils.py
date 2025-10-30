"""
工具模块测试
"""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import (
    PerformanceMetrics, 
    VRPDMetrics,
    save_json, 
    load_json
)


class TestPerformanceMetrics:
    """性能指标测试"""
    
    def test_calculate_returns(self):
        """测试折扣回报计算"""
        rewards = [1, 1, 1]
        gamma = 0.9
        
        returns = PerformanceMetrics.calculate_returns(rewards, gamma)
        
        assert len(returns) == 3
        assert returns[0] > returns[1] > returns[2]
    
    def test_moving_average(self):
        """测试移动平均"""
        data = list(range(100))
        window = 10
        
        ma = PerformanceMetrics.moving_average(data, window)
        
        assert len(ma) == len(data)


class TestVRPDMetrics:
    """VRPD指标测试"""
    
    def test_calculate_total_cost(self):
        """测试总成本计算"""
        travel_cost = 100
        delay_penalty = 20
        unserved_penalty = 30
        
        total = VRPDMetrics.calculate_total_cost(
            travel_cost, delay_penalty, unserved_penalty
        )
        
        assert total == 150
    
    def test_calculate_service_rate(self):
        """测试服务率计算"""
        served = 8
        total = 10
        
        rate = VRPDMetrics.calculate_service_rate(served, total)
        
        assert rate == 0.8


class TestIOUtils:
    """IO工具测试"""
    
    def test_save_load_json(self):
        """测试JSON保存和加载"""
        data = {'key': 'value', 'number': 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            save_json(data, temp_file)
            loaded = load_json(temp_file)
            
            assert loaded == data
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

