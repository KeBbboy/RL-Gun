"""
配置加载器
支持YAML格式配置文件的加载和合并
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """配置加载器，支持多层配置合并"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            project_root: 项目根目录，如果为None则自动检测
        """
        if project_root is None:
            # 自动检测项目根目录（向上查找直到找到configs目录）
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / "configs").exists():
                    project_root = str(current)
                    break
                current = current.parent
            else:
                raise RuntimeError("Cannot find project root directory")
        
        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / "configs"
        
        if not self.configs_dir.exists():
            raise RuntimeError(f"Configs directory not found: {self.configs_dir}")
    
    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """
        加载单个YAML文件
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            配置字典
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.configs_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config if config is not None else {}
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个配置字典
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self.merge_configs(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    def load_config(
        self,
        algorithm: Optional[str] = None,
        environment: Optional[str] = None,
        custom_config: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        加载配置，支持多层合并
        优先级：custom_config > algorithm_config > environment_config > default_config
        
        Args:
            algorithm: 算法名称（如 'maddpg', 'ma2c'）
            environment: 环境配置名称（如 'small', 'medium', 'large'）
            custom_config: 自定义配置文件路径
            
        Returns:
            合并后的完整配置
        """
        # 1. 加载默认配置
        config = self.load_yaml("default.yaml")
        
        # 2. 合并环境配置
        if environment:
            env_config_path = f"environments/{environment}.yaml"
            try:
                env_config = self.load_yaml(env_config_path)
                config = self.merge_configs(config, env_config)
            except FileNotFoundError:
                print(f"Warning: Environment config '{env_config_path}' not found, skipping")
        
        # 3. 合并算法配置
        if algorithm:
            algo_config_path = f"algorithms/{algorithm}.yaml"
            try:
                algo_config = self.load_yaml(algo_config_path)
                config = self.merge_configs(config, algo_config)
            except FileNotFoundError:
                print(f"Warning: Algorithm config '{algo_config_path}' not found, skipping")
        
        # 4. 合并自定义配置
        if custom_config:
            try:
                custom = self.load_yaml(custom_config)
                config = self.merge_configs(config, custom)
            except FileNotFoundError:
                print(f"Warning: Custom config '{custom_config}' not found, skipping")
        
        return config
    
    def save_config(self, config: Dict[str, Any], file_path: str):
        """
        保存配置到YAML文件
        
        Args:
            config: 配置字典
            file_path: 保存路径
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.configs_dir / path
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def get_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        通过点分隔的路径获取配置值
        
        Args:
            config: 配置字典
            key_path: 键路径，如 'training.batch_size'
            default: 默认值
            
        Returns:
            配置值
            
        Example:
            >>> loader.get_value(config, 'training.batch_size')
            256
        """
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


def load_config_from_args(args) -> Dict[str, Any]:
    """
    从命令行参数加载配置
    
    Args:
        args: argparse解析的参数对象
        
    Returns:
        配置字典
    """
    loader = ConfigLoader()
    
    # 获取算法、环境、自定义配置
    algorithm = getattr(args, 'algorithm', None)
    environment = getattr(args, 'environment', None)
    custom_config = getattr(args, 'config', None)
    
    # 加载配置
    config = loader.load_config(
        algorithm=algorithm,
        environment=environment,
        custom_config=custom_config
    )
    
    # 用命令行参数覆盖配置（如果提供）
    override_mapping = {
        'num_episodes': ('training', 'num_episodes'),
        'batch_size': ('training', 'batch_size'),
        'lr_actor': ('training', 'lr_actor'),
        'lr_critic': ('training', 'lr_critic'),
        'gamma': ('training', 'gamma'),
        'use_per': ('features', 'use_per'),
        'use_iam': ('features', 'use_iam'),
        'visualize': ('features', 'visualize'),
        'exp_name': ('logging', 'exp_name'),
        'save_dir': ('logging', 'save_dir'),
        'log_dir': ('logging', 'log_dir'),
    }
    
    for arg_name, (config_section, config_key) in override_mapping.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                if config_section not in config:
                    config[config_section] = {}
                config[config_section][config_key] = arg_value
    
    return config


# 便捷函数
def quick_load(algorithm: str = 'maddpg', environment: str = 'medium') -> Dict[str, Any]:
    """
    快速加载配置的便捷函数
    
    Args:
        algorithm: 算法名称
        environment: 环境配置名称
        
    Returns:
        配置字典
    """
    loader = ConfigLoader()
    return loader.load_config(algorithm=algorithm, environment=environment)


if __name__ == '__main__':
    # 测试配置加载器
    loader = ConfigLoader()
    
    # 测试1: 加载默认配置
    print("=== Test 1: Default Config ===")
    config = loader.load_config()
    print(f"Algorithm: {config['algorithm']['name']}")
    print(f"Num episodes: {config['training']['num_episodes']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # 测试2: 加载特定算法和环境
    print("\n=== Test 2: MAPPO + Large Environment ===")
    config = loader.load_config(algorithm='mappo', environment='large')
    print(f"Algorithm: {config['algorithm']['name']}")
    print(f"Num customers: {config['environment']['num_customers']}")
    print(f"PPO epochs: {config['training'].get('ppo_epochs', 'N/A')}")
    
    # 测试3: 使用点路径获取值
    print("\n=== Test 3: Get Value by Path ===")
    batch_size = loader.get_value(config, 'training.batch_size')
    print(f"Batch size: {batch_size}")

