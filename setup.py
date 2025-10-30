"""
Setup script for Trucks and Drones Multi-Agent RL
"""

from setuptools import setup, find_packages
import os


# 读取README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README_NEW.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


# 读取requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements_new.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements


setup(
    name='trucks-and-drones-marl',
    version='2.0.0',
    license='MIT',
    description='Multi-Agent Reinforcement Learning for Vehicle Routing Problem with Drones',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    author='Maik Schürmann (原作者), Contributors',
    author_email='maik.schuermann97@gmail.com',
    url='https://github.com/your-repo/trucks-and-drones-marl',
    
    # 包含src目录下的所有包
    packages=find_packages(where='.', include=['src', 'src.*', 'trucks_and_drones', 'trucks_and_drones.*']),
    package_dir={'': '.'},
    
    # 包含配置文件
    package_data={
        'configs': ['*.yaml', '**/*.yaml'],
    },
    
    include_package_data=True,
    
    # 依赖
    install_requires=read_requirements(),
    
    # 额外依赖
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ],
        'viz': [
            'plotly>=5.0.0',
            'jupyter>=1.0.0',
        ],
    },
    
    # 可执行脚本
    entry_points={
        'console_scripts': [
            'marl-train=scripts.train:main',
            'marl-eval=scripts.evaluate:main',
            'marl-viz=scripts.visualize:main',
        ],
    },
    
    # 分类
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    
    keywords=[
        'multi-agent', 'reinforcement learning', 'MARL',
        'vehicle routing problem', 'VRP', 'VRPD',
        'drones', 'logistics', 'optimization',
        'MADDPG', 'MAPPO', 'MA2C', 'COMA',
    ],
    
    python_requires='>=3.7',
    
    # 项目URL
    project_urls={
        'Documentation': 'https://trucks-and-drones-marl.rtfd.io/',
        'Source': 'https://github.com/your-repo/trucks-and-drones-marl',
        'Tracker': 'https://github.com/your-repo/trucks-and-drones-marl/issues',
    },
)

