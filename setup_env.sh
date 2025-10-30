#!/bin/bash

echo "创建 conda 环境 vrpo..."
conda env create -f environment_fast.yml

echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate vrpo

echo "使用清华镜像源安装 pip 包..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

echo "安装当前项目..."
pip install -e .

echo "完成！使用 'conda activate vrpo' 激活环境"

