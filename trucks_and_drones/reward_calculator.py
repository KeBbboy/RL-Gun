import numpy as np

def reward_parameter(
    reward_modes        = None,
    reward_type         = 'sum_vehicle',
    restriction_rewards = ['travel_cost', 'delay_penalty'],
    action_rewards      = ['service_reward'],
):
    """
    保持与 build_env.py 接口一致，但实际计算时我们
    只使用了 service_reward、travel_cost 和 delay_penalty。
    """
    return {
        'reward_modes'       : reward_modes,
        'reward_type'        : reward_type,
        'restriction_rewards': restriction_rewards,
        'action_rewards'     : action_rewards,
    }


class RewardFunctions:
    """
    按论文公式计算本步系统即时奖励：
      r_t = sum_{k ∈ U_t} [ e^k − (c^k + d^k) ]

    修复内容：
    1. 正确计算服务奖励和旅行成本
    2. 修复延迟惩罚计算
    3. 考虑实际到达时间而不是当前时间
    """

    def __init__(self, temp_db):
        self.db = temp_db
        # 假定这些参数预先保存在 temp_db
        self.ct = temp_db.ct_cost       # 卡车单位时间成本 c_t
        self.cd = temp_db.cd_cost       # 无人机单位时间成本 c_d
        self.vt = temp_db.truck_speed   # 卡车速度 v_t
        self.vd = temp_db.drone_speed   # 无人机速度 v_d

        # 新增：累积成本统计（每个episode重置）
        self.reset_episode_costs()
        print(f"   📊 Initialized episode cost tracking")

    def reset_episode_costs(self):
        """重置每个episode的成本统计"""
        self.episode_travel_cost = 0.0
        self.episode_delay_penalty = 0.0
        self.episode_truck_travel_time = 0.0
        self.episode_drone_travel_time = 0.0
        self.served_nodes = set()  # 已服务节点
    
    def get_episode_statistics(self):
        """获取当前episode的统计数据"""
        # 计算未服务节点数
        delta = self.db.get_val('delta')
        beta = self.db.get_val('beta')  # 获取每个节点的beta值
        
        # 找出所有未服务的客户节点
        unserved_nodes = []
        for i, d in enumerate(delta):
            if i > 0:  # 排除depot (index 0)
                if d == 1:  # delta=1 表示激活但未服务
                    unserved_nodes.append(i)
        
        unserved_count = len(unserved_nodes)

        # 未服务惩罚：每个未服务节点的惩罚等于其beta值
        unserved_penalty = sum(beta[node_idx] for node_idx in unserved_nodes)
     
        # 总成本
        total_cost = self.episode_travel_cost + self.episode_delay_penalty + unserved_penalty
    
        return {
            'total_cost': float(total_cost),
            'travel_cost': float(self.episode_travel_cost),
            'delay_penalty': float(self.episode_delay_penalty),
            'unserved_penalty': float(unserved_penalty),
            'unserved_count': int(unserved_count),
            'served_count': int(len(self.served_nodes)),
            'truck_travel_time': float(self.episode_truck_travel_time),
            'drone_travel_time': float(self.episode_drone_travel_time)
        }
    

    def system_reward(self, actions: dict) -> float:
        """
        修复后的系统奖励计算 - 支持独立卡车和无人机智能体
        """
        # 当前仿真时刻
        t = self.db.total_time
        print(f"   💰 Reward calculation at time {t:.3f}")

        # 2) 节点属性
        D     = self.db.get_val('deadline')  # 每个节点的 Di
        alpha = self.db.get_val('alpha')     # 每个节点的 α_i
        beta  = self.db.get_val('beta')      # 每个节点的 β_i

        total_r = 0.0
        
        # 获取智能体数量
        num_trucks = getattr(self.db, 'num_trucks', 0)
        num_drones = getattr(self.db, 'num_drones', 0)

        # 遍历所有动作
        for agent_idx, act in actions.items():
            if not act:  # 空动作
                continue
                
            print(f"   💰 Calculating reward for agent {agent_idx}: {act}")
            
            # **关键修复：区分卡车和无人机智能体**
            if agent_idx < num_trucks:
                # 卡车智能体
                truck_idx = agent_idx
                truck_pos = self.db.status_dict.get('v_coord', [0] * num_trucks)[truck_idx]
                
                # 1. 卡车服务动作
                if 'truck_target_node' in act and act['truck_target_node'] is not None:
                    target_node = act['truck_target_node']
                    
                    if target_node > 0:  # 只有customer节点才有服务奖励
                        dist = self.db.distance(truck_pos, target_node)
                        travel_time = dist / self.vt
                        arrival_time = t + travel_time
                        
                        # 服务奖励
                        service_reward = beta[target_node]
                        
                        # 旅行成本
                        travel_cost = self.ct * travel_time
                        self.episode_travel_cost += travel_cost  # 累积
                        print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
                        self.episode_truck_travel_time += travel_time  # 累积卡车旅行时间                        
                        
                        # 延迟惩罚
                        delay_penalty = alpha[target_node] * max(0.0, arrival_time - D[target_node])
                        self.episode_delay_penalty += delay_penalty  # 累积
                        # 记录已服务节点
                        self.served_nodes.add(target_node)
                        
                        reward = service_reward - travel_cost - delay_penalty
                        total_r += reward
                        
                        print(f"     🚛 Truck {truck_idx} service node {target_node}: service={service_reward:.2f}, cost={travel_cost:.2f}, delay={delay_penalty:.2f}, net={reward:.2f}")
                    
                    elif target_node == 0:
                        # 返回depot只有旅行成本
                        dist = self.db.distance(truck_pos, target_node)
                        travel_time = dist / self.vt
                        travel_cost = self.ct * travel_time
                        self.episode_travel_cost += travel_cost  # 累积
                        print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
                        self.episode_truck_travel_time += travel_time  # 累积
                        total_r -= travel_cost
                        print(f"     🚛 Truck {truck_idx} to depot: cost={travel_cost:.2f}")

                # 2. 卡车等待动作
                if act.get('truck_wait', 0) == 1:
                    # 等待动作有小的时间成本
                    wait_cost = self.ct * 0.1  # 小的等待成本
                    self.episode_travel_cost += wait_cost  # 累积到旅行成本中
                    total_r -= wait_cost
                    print(f"     ⏳ Truck {truck_idx} waiting: cost={wait_cost:.2f}")
                    
            else:
                # 无人机智能体
                drone_idx = agent_idx - num_trucks
                drone_pos = self._get_current_drone_pos(drone_idx)
                
                # 3. 无人机服务动作
                if 'drone_service_node' in act and act['drone_service_node'] is not None:
                    target_node = act['drone_service_node']
                    if target_node > 0:  # 只有customer节点才有服务奖励
                        dist = self.db.distance(drone_pos, target_node)
                        travel_time = dist / self.vd
                        arrival_time = t + travel_time
                        
                        # 服务奖励
                        service_reward = beta[target_node]
                        
                        # 旅行成本
                        travel_cost = self.cd * travel_time
                        self.episode_travel_cost += travel_cost  # 累积
                        print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
                        self.episode_drone_travel_time += travel_time  # 累积无人机旅行时间
                        
                        # 延迟惩罚
                        delay_penalty = alpha[target_node] * max(0.0, arrival_time - D[target_node])
                        self.episode_delay_penalty += delay_penalty  # 累积
                        
                        # 记录已服务节点
                        self.served_nodes.add(target_node)
                        
                        reward = service_reward - travel_cost - delay_penalty
                        total_r += reward
                        
                        print(f"     🚁 Drone {drone_idx} service node {target_node}: service={service_reward:.2f}, cost={travel_cost:.2f}, delay={delay_penalty:.2f}, net={reward:.2f}")

                # 4. 无人机汇合动作 - **关键修复：只有旅行成本**
                if 'drone_rendezvous_node' in act and act['drone_rendezvous_node'] is not None:
                    target_node = act['drone_rendezvous_node']
                    dist = self.db.distance(drone_pos, target_node)
                    travel_time = dist / self.vd
                    travel_cost = self.cd * travel_time
                    self.episode_travel_cost += travel_cost  # 累积
                    print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
                    self.episode_drone_travel_time += travel_time  # 累积
                    total_r -= travel_cost
                    print(f"     🤝 Drone {drone_idx} rendezvous at {target_node}: cost={travel_cost:.2f}")

                # 5. 无人机继续跟随动作（通常无成本）
                if act.get('drone_continue', 0) == 1:
                    # 继续跟随通常没有额外成本
                    print(f"     🔄 Drone {drone_idx} continues (no cost)")

        print(f"   💰 Total system reward: {total_r:.3f}")
        return total_r

    def _get_current_drone_pos(self, drone_idx):
        """获取无人机当前位置 - 修复独立模式支持"""
        try:
            # **关键修复：正确处理独立模式和搭载模式**
            ED = self.db.status_dict['ED'][drone_idx]
            attached_truck = self.db.status_dict.get('attached_truck', [-1] * len(self.db.status_dict['ED']))[drone_idx]
            
            if attached_truck == -1:
                # 独立模式：使用无人机自己的坐标
                if 'drone_coord' in self.db.status_dict:
                    return self.db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0  # 默认在depot
            elif ED == 3 and attached_truck >= 0:
                # 搭载模式且在卡车上：返回附着卡车的位置
                if attached_truck < len(self.db.status_dict.get('v_coord', [])):
                    return self.db.status_dict['v_coord'][attached_truck]
                else:
                    return 0
            else:
                # 其他状态（飞行中、等待等）：使用无人机独立坐标
                if 'drone_coord' in self.db.status_dict:
                    return self.db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0
        except (KeyError, IndexError) as e:
            print(f"     Warning: Error getting drone {drone_idx} position: {e}")
            return 0

class BaseRewardCalculator:
    """
    修复后的奖励计算器
    接口与 build_env.py 保持一致：
      __init__(reward_params, temp_db)
      reward_function(actions) -> float
    """

    def __init__(self, reward_params: dict, temp_db):
        self.db = temp_db
        self.reward_params = reward_params
        # 构造内部计算器
        self.funcs = RewardFunctions(temp_db)
        print(f"   🔧 Reward calculator initialized with params: {reward_params}")

    def reset(self):  # 注意：方法名是 reset，不是 reset_episode
        """重置episode统计"""
        self.funcs.reset_episode_costs()
        print(f"   📊 Reset episode cost tracking")

    def get_episode_statistics(self):
        """获取episode统计"""
        stats = self.funcs.get_episode_statistics()
        print(f"   📊 Getting episode stats: {stats}")  # 调试输出
        return stats

    def reward_function(self, actions: dict) -> float:
        """
        修复后的奖励函数
        在 Env.step() 中被调用，传入 joint actions，返回本步系统即时 reward。
        """
        if not actions:
            print(f"   💰 No actions provided, reward = 0.0")
            return 0.0
            
        reward = self.funcs.system_reward(actions)
        
        # # 额外的奖励塑形：完成任务的额外奖励
        # try:
        #     delta = self.db.get_val('delta')
        #     unserved = sum(1 for i, d in enumerate(delta) if i > 0 and d == 1)
        #     if unserved == 0:
        #         completion_bonus = 1000.0
        #         reward += completion_bonus
        #         print(f"   🎉 Task completion bonus: {completion_bonus}")
        # except:
        #     pass
            
        return reward