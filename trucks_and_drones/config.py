cfg = {
    # 环境相关的参数，一并传给 BuildEnvironment，再由它传到 BaseTempDatabase
    'environment': {
        'num_trucks':    3,
        'num_drones':    3,
        'num_depots':    1,
        'num_customers': 15,
        'WT_max':        1000.0,   # 卡车最大载重
        'WD_max':        80.0,    # 无人机最大载重
        'ct_cost':       1.0,     # 卡车单位时间成本
        'cd_cost':       0.5,     # 无人机单位时间成本
        'max_charge': 50,
        'truck_speed':   0.5,      # 30km/h（千米/小时） 0.5km/min
        'drone_speed':   1,      # 60km/h（千米/小时）  1km/min
        'horizon':       200,     # 分钟
        'area_size':     15.0,    # 千米
        'road_damage_ratio': 0.3,  # 20%的节点只能由无人机访问
        # 新增动态节点配置
        'dynamic_nodes': {
            'dod': 0,  # Degree of Dynamism (0.2, 0.5, 0.7)
            'delta_t': 5,  # 检查频率 (分钟)
            'enable': True,  # 是否启用动态节点
        },
    },
    
    # 和节点属性相关的默认常量
    'node': {
        'customer': {
            'deadline': 100.0,   # 默认截止时间 D_i
            'alpha':    0.5,     # 延误惩罚权重 α_i
            'beta':     30.0,    # 服务收益 β_i
        },
        'depot': {
            'deadline': 480,  # 仓库一般没有强制截止
            'alpha':    0.0,           # 不产生延误惩罚
            'beta':     0.0,           # 不产生服务收益
        }
    },
    
    'training': {
        'num_episodes': 2000,    # 总共跑多少个 episode 后停止
        'batch_size':   256,     # 新增：每次训练抽样的大小
        'update_freq':  10,         # 新增：每隔多少步更新一次
        'target_update_freq': 100, # 新增：target网络更新频率（每20步更新一次）
        'min_buffer_size': 5000,    # 2000 → 5000 这里！！！！！
        'updates_per_call': 1,      # 新增：每次触发做几次梯度步
        'warmup_steps': 3000,       # 新增：高探索预热 这里！！！！
        'entropy_coef': 0.01,       # 新增/覆盖
        'tau': 0.01,                 # 新增/覆盖 目标网络软更新
        'rollout_len': 25,     # 新增：分段回合长度（32~128都可以试）
        'lr_actor': 1e-4,      # 新增：Actor 学习率
        'lr_critic': 3e-4,     # 新增：Critic 学习率（略大一点更快学好V）
        'ppo_epochs': 5,             # 新增：每个rollout重复训练的epoch数
        'minibatch_size': 128,       # 新增：小批大小
    }
}