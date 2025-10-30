"""
配置系统测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ConfigLoader


class TestConfigLoader:
    """配置加载器测试"""
    
    def test_load_default(self):
        """测试加载默认配置"""
        loader = ConfigLoader()
        config = loader.load_config()
        
        assert 'algorithm' in config
        assert 'environment' in config
        assert 'training' in config
    
    def test_load_algorithm_config(self):
        """测试加载算法配置"""
        loader = ConfigLoader()
        config = loader.load_config(algorithm='maddpg')
        
        assert config['algorithm']['name'] == 'maddpg'
    
    def test_load_environment_config(self):
        """测试加载环境配置"""
        loader = ConfigLoader()
        config = loader.load_config(environment='small')
        
        # Small环境应该有更少的智能体和客户
        assert config['environment']['num_customers'] <= 10
    
    def test_config_merge(self):
        """测试配置合并"""
        loader = ConfigLoader()
        
        base = {'a': 1, 'b': {'c': 2}}
        override = {'b': {'d': 3}, 'e': 4}
        
        result = loader.merge_configs(base, override)
        
        assert result['a'] == 1
        assert result['b']['c'] == 2
        assert result['b']['d'] == 3
        assert result['e'] == 4
    
    def test_get_value(self):
        """测试通过路径获取值"""
        loader = ConfigLoader()
        config = {'training': {'batch_size': 256}}
        
        value = loader.get_value(config, 'training.batch_size')
        assert value == 256
        
        # 测试不存在的键
        value = loader.get_value(config, 'training.nonexistent', default=999)
        assert value == 999


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

