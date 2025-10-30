"""
环境测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import BuildEnvironment


class TestEnvironment:
    """环境基本功能测试"""
    
    def test_build_environment(self):
        """测试环境构建"""
        builder = (
            BuildEnvironment("VRPD")
            .trucks(num=2)
            .drones(num=2)
            .depots(num=1)
            .customers(num=5)
            .observations(
                contin_inputs=['TW', 'time'],
                discrete_inputs=['ET', 'ED'],
                flatten=True
            )
            .actions()
            .rewards()
            .compile()
        )
        
        env = builder.build()
        assert env is not None
    
    def test_reset(self):
        """测试重置功能"""
        builder = (
            BuildEnvironment("VRPD")
            .trucks(num=2)
            .drones(num=2)
            .depots(num=1)
            .customers(num=5)
            .observations(
                contin_inputs=['TW'],
                discrete_inputs=['ET'],
                flatten=True,
                output_as_array=False
            )
            .actions()
            .rewards()
            .compile()
        )
        
        env = builder.build()
        obs_n, global_obs = env.reset()
        
        assert obs_n is not None
        assert len(obs_n) == 4  # 2 trucks + 2 drones
    
    def test_step(self):
        """测试步进功能"""
        builder = (
            BuildEnvironment("VRPD")
            .trucks(num=1)
            .drones(num=1)
            .depots(num=1)
            .customers(num=3)
            .observations(
                contin_inputs=['TW'],
                discrete_inputs=['ET'],
                flatten=True,
                output_as_array=False
            )
            .actions()
            .rewards()
            .compile()
        )
        
        env = builder.build()
        obs_n, global_obs = env.reset()
        
        # 创建随机动作
        n_agents = 2
        action_n = []
        for i in range(n_agents):
            # 每个智能体有多个动作头
            action = [0, 0, 0]  # 示例动作
            action_n.append(action)
        
        # 执行步进
        (new_obs_n, new_global_obs), rewards, done, info = env.step(action_n)
        
        assert new_obs_n is not None
        assert len(rewards) == n_agents
        assert isinstance(done, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

