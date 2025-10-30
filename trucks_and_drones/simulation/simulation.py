import numpy as np
from trucks_and_drones.config import cfg

class BaseSimulator:
    """
    Dispatch + Event‑Completion 两阶段严格对应论文 Section (3)。
    同时维护 visited_nodes 用于 IAM 中的节点访问掩码。
    
    修复内容：
    1. 在任务分配时就预先标记节点，避免重复分配
    2. 改进冲突解决，失败载具设为空闲状态，完全移除冲突动作
    3. 修复状态管理，避免不必要的状态切换
    4. **关键修复**：完全清空失败者的动作，确保不执行任何新动作
    5. **新增修复**：汇合逻辑同步，无人机请求汇合时立即触发卡车汇合
    6. **新增修复**：修复奖励计算和终止条件
    7. **关键修复**：奖励计算时机修正，在时间推进前计算
    8. **关键修复**：完善汇合同步机制，确保卡车响应无人机汇合请求
    9. **新增修复**：预分配时立即更新delta状态防止重复分配
    10. **汇合修复**：区分卡车服务动作和汇合动作，汇合时不提供服务
    11. **关键修复**：修复载具级别冲突处理，只拒绝失败的具体载具，不影响同智能体的其他载具
    """
    def __init__(self, temp_db, trucks, drones, reward_calc):
        self.temp_db            = temp_db
        self.trucks = trucks  # 独立的卡车列表
        self.drones = drones  # 独立的无人机列表
        self.reward_calc   = reward_calc
        # 分别记录卡车和无人机的调度类型
        self.truck_dispatch_type = [None] * len(trucks)
        self.drone_dispatch_type = [None] * len(drones)
        # 新增：记录预分配的节点（在载具实际到达前就标记）
        self.pre_assigned_nodes = set()

    # 固定节点每个episode不变版本
    def reset(self):
        
        print("\n" + "="*80)
        print("🔄 RESETTING SIMULATION FOR NEW EPISODE")
        print("="*80)
        
        # 打印重置前的状态
        print("\n📊 State BEFORE reset:")
        print(f"   Total time: {self.temp_db.total_time:.1f}")
        print(f"   Current time: {self.temp_db.current_time:.1f}")
        print(f"   Next check time: {self.temp_db.next_check_time:.1f}")
        print(f"   Visited nodes: {self.temp_db.visited_nodes}")
        print(f"   Pre-assigned nodes: {self.pre_assigned_nodes}")
        
        if hasattr(self.temp_db, 'dynamic_nodes_pool'):
            print(f"   Dynamic pool size: {len(self.temp_db.dynamic_nodes_pool)}")
            print(f"   Active dynamic nodes: {self.temp_db.active_dynamic_nodes}")
        
        # 重置数据库状态并初始化 IAM 访问记录
        # 清空访问过的节点集合和预分配集合
        self.temp_db.visited_nodes.clear()
        self.pre_assigned_nodes.clear()
        print("\n✅ Cleared visited_nodes and pre_assigned_nodes")

        # 修复：正确重置delta标志，保护动态节点状态
        # 不能简单地fill(1)，需要区分静态和动态节点
        if hasattr(self.temp_db, 'dynamic_enabled') and self.temp_db.dynamic_enabled:
            print("\n🔧 Processing DYNAMIC nodes reset...")
        
            # 有动态节点的情况
            static_end = self.temp_db.num_depots + self.temp_db.num_static_customers
            dynamic_start = static_end
            print(f"   Node ranges:")
            print(f"   - Depots: 0-{self.temp_db.num_depots-1}")
            print(f"   - Static customers: {self.temp_db.num_depots}-{static_end-1}")
            print(f"   - Dynamic customers: {dynamic_start}-{self.temp_db.num_nodes-1}")
            
            # 打印旧的动态节点属性
            print(f"\n📋 OLD dynamic node attributes:")
            for i in range(self.temp_db.num_dynamic_nodes):
                node_idx = dynamic_start + i
                print(f"   Node {node_idx}:")
                print(f"      - Coord: {self.temp_db.status_dict['n_coord'][node_idx]}")
                print(f"      - Demand: {self.temp_db.status_dict['n_items'][node_idx]:.1f}")
                print(f"      - Deadline: {self.temp_db.constants_dict['deadline'][node_idx]:.1f}")
                print(f"      - Delta: {self.temp_db.status_dict['delta'][node_idx]}")
            
            # 重置静态节点为未访问
            self.temp_db.status_dict['delta'][:static_end] = 1

            # 保持动态节点为未激活（或重新设置为-1）
            self.temp_db.status_dict['delta'][dynamic_start:] = -1
            
            # depot设置为已访问
            self.temp_db.status_dict['delta'][0] = 0
            
            # 重置动态节点池和激活状态
            self.temp_db.active_dynamic_nodes.clear()
            self.temp_db.next_check_time = self.temp_db.delta_t
            print(f"\n✅ Cleared active_dynamic_nodes")
            print(f"✅ Reset next_check_time to {self.temp_db.delta_t}")        

            # **关键修复：重新生成动态节点的所有属性**
            self.temp_db.dynamic_nodes_pool.clear()
            
            # 获取已存在的静态节点坐标，避免重复
            existing_coords = set()
            for i in range(static_end):
                if i < len(self.temp_db.status_dict.get('n_coord', [])):
                    coord = tuple(self.temp_db.status_dict['n_coord'][i])
                    if coord != (0, 0):
                        existing_coords.add(coord)
            print(f"\n📍 Existing static node coordinates: {existing_coords}")
        
            print(f"\n🎲 Regenerating {self.temp_db.num_dynamic_nodes} dynamic nodes...")
                        
            for i in range(self.temp_db.num_dynamic_nodes):
                node_idx = dynamic_start + i

                print(f"\n   Regenerating node {node_idx}:")
            
                # 保存旧属性用于比较
                old_coord = self.temp_db.status_dict['n_coord'][node_idx].copy()
                old_demand = self.temp_db.status_dict['n_items'][node_idx]
                old_deadline = self.temp_db.constants_dict['deadline'][node_idx]
                
                
                # 重新生成release time
                release_time = np.random.uniform(0, self.temp_db.horizon * 0.8)
                print(f"      - New release_time: {release_time:.1f}")
                
                # 重新生成坐标（避免与现有节点重复）
                from trucks_and_drones.simulation.temp_database import random_coordinates
                coord = random_coordinates(self.temp_db.grid)
                attempts = 0
                while tuple(coord) in existing_coords:
                    coord = random_coordinates(self.temp_db.grid)
                    attempts += 1
                    if attempts > 100:
                        print(f"      ⚠️ Warning: Difficulty finding unique coordinate after {attempts} attempts")
                        break
                existing_coords.add(tuple(coord))
                print(f"      - Coord: {old_coord} -> {coord} (attempts: {attempts})")
                                    
                # 重新生成需求
                demand = np.random.uniform(10, 50)
                print(f"      - Demand: {old_demand:.1f} -> {demand:.1f}")
                
                # 重新生成截止时间（应该在release_time之后）
                deadline = np.random.uniform(release_time + 50, self.temp_db.horizon)
                print(f"      - Deadline: {old_deadline:.1f} -> {deadline:.1f}")
            
                # 更新节点属性
                self.temp_db.status_dict['n_coord'][node_idx] = coord
                self.temp_db.status_dict['n_items'][node_idx] = demand
                self.temp_db.constants_dict['deadline'][node_idx] = deadline
                
                # 保持其他属性不变（type, alpha, beta已在init_db中设置）
                
                self.temp_db.dynamic_nodes_pool.append({
                    'node_idx': node_idx,
                    'release_time': release_time
                })
                
                print(f"Dynamic node {node_idx} regenerated: "
                    f"release_time={release_time:.1f}, coord={coord}, "
                    f"demand={demand:.1f}, deadline={deadline:.1f}")
            
            # 按release time排序
            self.temp_db.dynamic_nodes_pool.sort(key=lambda x: x['release_time'])
            
            release_times = [f"{x['release_time']:.1f}" for x in self.temp_db.dynamic_nodes_pool]
            print(f"Reset episode: {len(self.temp_db.dynamic_nodes_pool)} dynamic nodes with new attributes")
            print(f"Release times: {release_times}")

        else:
            # 无动态节点的情况（原逻辑）
            print("\n📌 No dynamic nodes configured - resetting all nodes to unvisited")
            self.temp_db.status_dict['delta'].fill(1)
            self.temp_db.status_dict['delta'][0] = 0
        
        self.temp_db.visited_nodes.add(0)

        # **重要：重置时间变量**
        self.temp_db.total_time = 0.0
        self.temp_db.current_time = 0.0

        print("\n" + "="*80)
        print("🎬 RESET COMPLETE - Ready for new episode")
        print("="*80 + "\n")
        
        # 重置卡车状态
        num_trucks = len(self.trucks)
        for k in range(num_trucks):
            self.temp_db.status_dict['ET'][k] = 3  # 空闲状态
            self.temp_db.status_dict['LT'][k] = 0.0
            self.temp_db.status_dict['NT'][k] = 0
            self.truck_dispatch_type[k] = None
            
            # 重置卡车载重
            if 'TW' in self.temp_db.status_dict:
                truck = self.trucks[k]
                if hasattr(truck, 'capacity'):
                    self.temp_db.status_dict['TW'][k] = truck.capacity
                elif hasattr(truck, 'max_weight'):
                    self.temp_db.status_dict['TW'][k] = truck.max_weight
                    
            # 初始化卡车坐标（从depot开始）
            if 'v_coord' not in self.temp_db.status_dict:
                self.temp_db.status_dict['v_coord'] = np.zeros(num_trucks, dtype=int)
            else:
                self.temp_db.status_dict['v_coord'][k] = 0
        
        # 重置无人机状态
        num_drones = len(self.drones)
        for k in range(num_drones):
            # 修复：区分搭载模式和独立模式
            # 可以通过配置或无人机属性来决定初始模式
            if hasattr(self.drones[k], 'independent_mode') and self.drones[k].independent_mode:
                # 独立模式：不附着任何卡车
                self.temp_db.status_dict['ED'][k] = 2  # 设为刚完成服务状态，可以自由行动
                if 'attached_truck' not in self.temp_db.status_dict:
                    self.temp_db.status_dict['attached_truck'] = np.full(num_drones, -1, dtype=int)
                else:
                    self.temp_db.status_dict['attached_truck'][k] = -1  # -1表示独立模式
            else:
                # 搭载模式：附着在卡车上
                self.temp_db.status_dict['ED'][k] = 3  # 在卡车上
                if 'attached_truck' not in self.temp_db.status_dict:
                    self.temp_db.status_dict['attached_truck'] = np.zeros(num_drones, dtype=int)            
                if num_trucks > 0:
                    # 按照无人机编号对卡车数量取模，进行循环分配
                    self.temp_db.status_dict['attached_truck'][k] = k % num_trucks
                else:
                    # 如果没有卡车，设为独立模式
                    self.temp_db.status_dict['attached_truck'][k] = -1
            
            self.temp_db.status_dict['LD'][k] = 0.0
            self.temp_db.status_dict['ND'][k] = 0
            self.drone_dispatch_type[k] = None
            
            # 重置无人机载重
            if 'DW' in self.temp_db.status_dict:
                drone = self.drones[k]
                if hasattr(drone, 'capacity'):
                    self.temp_db.status_dict['DW'][k] = drone.capacity
                elif hasattr(drone, 'max_weight'):
                    self.temp_db.status_dict['DW'][k] = drone.max_weight

            # 【新增】重置无人机电量到最大值
            if 'drone_battery' not in self.temp_db.status_dict:
                self.temp_db.status_dict['drone_battery'] = np.full(num_drones, self.temp_db.drone_battery)
            else:
                # 重置每个无人机的电量到最大值
                self.temp_db.status_dict['drone_battery'][k] = self.temp_db.drone_battery
                print(f"     🔋 Drone {k} battery reset to full: {self.temp_db.drone_battery}")
            
            
            # 初始化无人机位置
            if 'drone_coord' not in self.temp_db.status_dict:
                self.temp_db.status_dict['drone_coord'] = np.zeros(num_drones, dtype=int)
            else:
                self.temp_db.status_dict['drone_coord'][k] = 0

        self.temp_db.total_time = 0.0


    def reset_simulation(self):
        """
        gym 环境里 CustomEnv.reset() 会调用这个方法，
        我们直接委托到 reset() 即可。
        """
        return self.reset()

    def _is_node_available(self, node_id):
        """检查节点是否可用 - 排除未激活的动态节点"""
        # 获取节点状态
        delta = self.temp_db.status_dict['delta'][node_id]
        
        # delta=-1表示未激活的动态节点，不可用
        if delta == -1:
            return False
        
        # delta=0表示已访问，不可用
        if delta == 0:
            return False
        
        # 检查是否在已访问集合中
        if node_id in self.temp_db.visited_nodes:
            return False
        
        # 检查是否已被预分配
        if node_id in self.pre_assigned_nodes:
            return False
        
        # delta=1且满足其他条件，节点可用
        return True
    
    def _pre_assign_node(self, node_id, vehicle_type, vehicle_id):
        """预分配节点并立即更新状态"""
        self.pre_assigned_nodes.add(node_id)
        # **关键修复：立即更新delta状态，防止重复分配**
        self.temp_db.status_dict['delta'][node_id] = 0
        print(f"   📌 Node {node_id} pre-assigned to {vehicle_type} {vehicle_id} and marked as delta=0")


    def _get_truck_position(self, truck_idx):
        """获取卡车当前位置"""
        return self.temp_db.status_dict['v_coord'][truck_idx]

    def _get_drone_position(self, drone_idx):
        """获取无人机当前位置 - 修复独立模式支持"""
        try:
            ED = self.temp_db.status_dict['ED'][drone_idx]
            attached_truck = self.temp_db.status_dict.get('attached_truck', [-1] * len(self.drones))[drone_idx]
            
            if attached_truck == -1:
                # 独立模式：使用无人机自己的坐标
                if 'drone_coord' in self.temp_db.status_dict:
                    return self.temp_db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0  # 默认在depot
            elif ED == 3 and attached_truck >= 0:
                # 搭载模式且在卡车上：返回附着卡车的位置
                if attached_truck < len(self.trucks):
                    return self.temp_db.status_dict['v_coord'][attached_truck]
                else:
                    return 0
            else:
                # 其他状态（飞行中、等待等）：使用无人机独立坐标
                if 'drone_coord' in self.temp_db.status_dict:
                    return self.temp_db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0
        except (KeyError, IndexError) as e:
            print(f"     Warning: Error getting drone {drone_idx} position: {e}")
            return 0

    def _get_rendezvous_node_for_truck(self, truck_idx):
        """获取指定卡车的汇合节点"""
        try:
            ET = self.temp_db.status_dict['ET'][truck_idx]
            current_pos = self._get_truck_position(truck_idx)
            
            if ET == 0:  # 卡车正在移动
                target_node = self.temp_db.status_dict['NT'][truck_idx]
                print(f"    Truck {truck_idx} is moving to {target_node}, rendezvous there")
                return target_node
            else:  # 卡车空闲或等待
                print(f"    Truck {truck_idx} at position {current_pos}, rendezvous there")
                return current_pos
        except Exception as e:
            print(f"    Error getting rendezvous node for truck {truck_idx}: {e}")
            return 0
    
    def step(self, actions):
        """修复step方法，正确处理动态节点激活"""
        t_prev = self.temp_db.total_time
        num_trucks = len(self.trucks)
        num_drones = len(self.drones)
        
        # 确保current_time与total_time同步
        self.temp_db.current_time = self.temp_db.total_time
        
        

        print(f"🚛 Simulation step with {len(actions)} agent actions")
        print(f"   Current vehicle coordinates: {self.temp_db.status_dict.get('v_coord', 'Not initialized')}")
        current_drone_positions = [self._get_current_drone_pos(k) for k in range(num_drones)]
        print(f"   Current drone coordinates: {current_drone_positions}")
        print(f"   Pre-assigned nodes: {self.pre_assigned_nodes}")
        print(f"   Visited nodes: {self.temp_db.visited_nodes}")

        # —— 1) Dispatch 阶段 - 改进服务冲突检测和解决 ——
        
        # 第一步：收集所有拟执行的服务动作
        service_requests = {}  # node_id -> [(vehicle_type, vehicle_id, travel_time, action_dict)]
        
        # 修复service_requests收集部分的变量k问题
        for truck_idx in range(num_trucks):
            if truck_idx in actions:
                act = actions[truck_idx]
                print(f"   Analyzing truck {truck_idx} requests: {act}")
                
                current_pos = self._get_truck_position(truck_idx)
                tgt = act.get('truck_target_node')

                if tgt is not None and tgt != 0:
                    if (tgt >= 0 and tgt < len(self.temp_db.get_val('demand')) and 
                        self._is_node_available(tgt)):
                        d = self.temp_db.distance(current_pos, tgt)
                        truck_speed = getattr(self.trucks[truck_idx], 'speed', self.temp_db.truck_speed)
                        travel_time = d / truck_speed
                        
                        if tgt not in service_requests:
                            service_requests[tgt] = []
                        service_requests[tgt].append(('truck', truck_idx, travel_time, act))
                        print(f"     Truck {truck_idx} requests service node {tgt}, travel_time: {travel_time:.3f}")

        # 修复无人机服务请求收集
        for drone_idx in range(num_drones):
            agent_idx = num_trucks + drone_idx
            if agent_idx in actions:
                act = actions[agent_idx]
                print(f"   Analyzing drone {drone_idx} requests: {act}")
                
                current_pos = self._get_drone_position(drone_idx)        
                svc = act.get('drone_service_node')

                if svc is not None and svc != 0:
                    if (svc >= 0 and svc < len(self.temp_db.get_val('demand')) and 
                        self._is_node_available(svc)):
                        d = self.temp_db.distance(current_pos, svc)
                        drone_speed = getattr(self.drones[drone_idx], 'speed', self.temp_db.drone_speed)
                        travel_time = d / drone_speed 
                        
                        if svc not in service_requests:
                            service_requests[svc] = []
                        service_requests[svc].append(('drone', drone_idx, travel_time, act))
                        print(f"     Drone {drone_idx} requests service node {svc}, travel_time: {travel_time:.3f}")

        # 第二步：解决服务冲突并预分配节点
        approved_actions = {}  # vehicle_id -> approved_action_dict
        rejected_vehicle_actions = set()  # 记录被拒绝的具体载具动作 (vehicle_type, vehicle_id, action_key)
        newly_assigned_nodes = set()  # 本轮新分配的节点
        
        for node_id, requests in service_requests.items():
            if len(requests) > 1:
                print(f"   🔥 Service conflict at node {node_id}: {len(requests)} vehicles competing")
                # 按到达时间排序，最快的获胜
                requests.sort(key=lambda x: x[2])  # 按travel_time排序
                winner = requests[0]
                print(f"   🏆 Winner: {winner[0]} {winner[1]} with travel_time {winner[2]:.3f}")
                
                # **关键修复：使用新的预分配方法**
                vehicle_type, vehicle_id, travel_time, action_dict = winner
                self._pre_assign_node(node_id, vehicle_type, vehicle_id)
                newly_assigned_nodes.add(node_id)

                # 处理获胜者的动作
                if vehicle_type == 'truck':
                    action_key = vehicle_id
                else:  # drone
                    action_key = num_trucks + vehicle_id
                
                # 批准获胜者的动作 - 保留原有动作，只添加获胜的服务动作
                if action_key not in approved_actions:
                    approved_actions[action_key] = action_dict.copy()
                else:
                    # 如果已有动作，合并获胜的服务动作
                    if vehicle_type == 'truck':
                        approved_actions[action_key]['truck_target_node'] = action_dict.get('truck_target_node')
                    else:  # drone
                        approved_actions[action_key]['drone_service_node'] = action_dict.get('drone_service_node')
                
                # **关键修复：只拒绝失败载具的具体动作，不影响同智能体的其他载具**
                for loser in requests[1:]:
                    loser_type, loser_id, loser_time, loser_action = loser
                    print(f"   ❌ Rejected: {loser_type} {loser_id} (travel_time: {loser_time:.3f})")
                    
                    # 记录被拒绝的具体载具动作
                    if loser_type == 'truck':
                        rejected_vehicle_actions.add(('truck', loser_id, 'truck_target_node'))
                        loser_key = loser_id
                    else:  # drone
                        rejected_vehicle_actions.add(('drone', loser_id, 'drone_service_node'))
                        loser_key = num_trucks + loser_id
                    
                    # **修复：只移除失败的具体动作，保留其他动作**
                    if loser_key not in approved_actions:
                        approved_actions[loser_key] = loser_action.copy()
                    else:
                        # 合并动作但排除失败的服务动作
                        for key, value in loser_action.items():
                            if key not in approved_actions[loser_key]:
                                approved_actions[loser_key][key] = value
                    
                    # 移除失败的服务动作
                    if loser_type == 'truck' and 'truck_target_node' in approved_actions[loser_key]:
                        del approved_actions[loser_key]['truck_target_node']
                        print(f"     🔧 Removed truck service action for vehicle {loser_key}")
                    elif loser_type == 'drone' and 'drone_service_node' in approved_actions[loser_key]:
                        del approved_actions[loser_key]['drone_service_node'] 
                        print(f"     🔧 Removed drone service action for vehicle {loser_key}")

            else:
                # 无冲突，预分配节点并批准动作
                vehicle_type, vehicle_id, travel_time, action_dict = requests[0]
                self._pre_assign_node(node_id, vehicle_type, vehicle_id)
                newly_assigned_nodes.add(node_id)
                print(f"   📌 Node {node_id} pre-assigned to {vehicle_type} {vehicle_id} (no conflict)")
                
                if vehicle_type == 'truck':
                    action_key = vehicle_id
                else:
                    action_key = num_trucks + vehicle_id
                
                if action_key not in approved_actions:
                    approved_actions[action_key] = action_dict.copy()
                else:
                    for key, value in action_dict.items():
                        approved_actions[action_key][key] = value
                print(f"   ✅ No conflict: {vehicle_type} {vehicle_id} approved for node {node_id}")

        # **第三步：为没有参与冲突的载具添加其他动作**
        # 为没有参与冲突的载具添加其他动作
        for action_key, original_action in actions.items():
            if action_key not in approved_actions:
                approved_actions[action_key] = original_action.copy()
            else:
                for key, value in original_action.items():
                    if key not in approved_actions[action_key]:
                        # 检查是否被拒绝
                        is_truck = action_key < num_trucks
                        vehicle_id = action_key if is_truck else action_key - num_trucks
                        vehicle_type = 'truck' if is_truck else 'drone'
                        
                        is_rejected = (vehicle_type, vehicle_id, key) in rejected_vehicle_actions
                        if not is_rejected:
                            approved_actions[action_key][key] = value

        # **修复：按顺序打印最终批准的动作**
        def format_actions_by_order(actions_dict, num_trucks):
            """按卡车-无人机顺序格式化动作字典"""
            truck_actions = {}
            drone_actions = {}
            
            for key, action in actions_dict.items():
                if key < num_trucks:
                    truck_actions[key] = action
                else:
                    drone_actions[key] = action
            
            # 按key排序
            sorted_truck = dict(sorted(truck_actions.items()))
            sorted_drone = dict(sorted(drone_actions.items()))
            
            # 合并并保持顺序
            ordered_actions = {}
            ordered_actions.update(sorted_truck)
            ordered_actions.update(sorted_drone)
            
            return ordered_actions

        # 应用排序到两个打印语句
        sorted_approved_actions = format_actions_by_order(approved_actions, num_trucks)
        print(f"   Final approved actions: {sorted_approved_actions}")

        # 处理汇合逻辑 - 支持无人机选择任意卡车
        for action_key, act in approved_actions.items():
            if action_key >= num_trucks:  # 无人机动作
                drone_idx = action_key - num_trucks
                target_truck_idx = act.get('drone_rendezvous_truck')
                
                if target_truck_idx is not None and self.temp_db.status_dict['ED'][drone_idx] == 2:
                    # 验证卡车索引有效性
                    if target_truck_idx >= num_trucks:
                        target_truck_idx = 0
                    
                    # 获取目标卡车的汇合节点
                    rendezvous_node = self._get_rendezvous_node_for_truck(target_truck_idx)
                    approved_actions[action_key]['drone_rendezvous_node'] = rendezvous_node
                    approved_actions[action_key]['_target_truck'] = target_truck_idx
                    print(f"   Drone {drone_idx} will rendezvous with truck {target_truck_idx} at node {rendezvous_node}")
                    
                    # 更新无人机的附着关系
                    self.temp_db.status_dict['attached_truck'][drone_idx] = target_truck_idx


        # 执行批准的动作
        self._execute_actions(approved_actions, num_trucks, num_drones)

        # === NEW: after actions have been applied ===
        executed_multi = self._pack_executed_actions_multihead(approved_actions if 'approved_actions' in locals() else {})
        # 缓存到 temp_db（上层不可见时也能读到）
        self.temp_db.last_executed_actions_multihead = executed_multi
        print(f"   ✅ Executed multi-head (packed for replay): {executed_multi}")

        # 计算奖励
        print(f"   Reward calculation at time {self.temp_db.total_time:.3f}")
        r = self.reward_calc.reward_function(approved_actions)

        # 检查汇合完成
        self._check_rendezvous_completion(num_trucks, num_drones)

        # **关键修复：在推进时间前记录当前时间**
        time_before_advance = self.temp_db.total_time

        # 2) Event-Completion阶段
        self._advance_time(num_trucks, num_drones)

        # — 3) 计算 reward & 构造输出 —
        obs = None

        # **关键修复：推进时间后同步current_time**
        time_after_advance = self.temp_db.total_time
        self.temp_db.current_time = self.temp_db.total_time
        
        if time_after_advance > time_before_advance:
            print(f"   Time advanced from {time_before_advance:.3f} to {time_after_advance:.3f}")

            # === 动态节点激活检查 ===
            print(f"🔍 Checking dynamic nodes at time {self.temp_db.current_time:.1f}")
            activated_nodes = self.temp_db.check_dynamic_nodes_activation()
            if activated_nodes:
                print(f"✨ Activated {len(activated_nodes)} dynamic nodes: {activated_nodes}")
                # 节点激活不改变观测空间大小，只改变delta值
                for node_idx in activated_nodes:
                    print(f"   Node {node_idx}: delta changed from -1 to 1")
            
            # 打印当前状态
            delta = self.temp_db.get_val('delta')
            print(f"📊 Current node states: active={np.sum(delta==1)}, "
                f"visited={np.sum(delta==0)}, inactive={np.sum(delta==-1)}")
        
        else:
            print(f"   No time advancement (no active movements)")
        
        # **修复终止条件检查**
        done = self._check_terminal_state()
        print(f"   🎯 Simulation result: reward={r:.3f}, done={done}")
        print(f"   Final state - Pre-assigned: {self.pre_assigned_nodes}, Visited: {self.temp_db.visited_nodes}")
        # return obs, r, done, {}
        info = {'executed_actions_multihead': executed_multi}
        return obs, r, done, info

    # === NEW: pack approved/applied actions to multi-head discrete indices ===
    def _pack_executed_actions_multihead(self, approved_actions):
        """
        把“最终被执行”的动作打包成 multi-head 离散索引（按 agent 顺序：先卡车再无人机）
        Trucks: [truck_target_node, truck_wait]
        Drones: [drone_service_node, drone_rendezvous_truck, drone_continue]
        """
        n_trucks = len(self.trucks)
        n_drones = len(self.drones)
        total_nodes = len(self.temp_db.get_val('delta'))

        exec_actions = []

        # ---------- Trucks ----------
        for k in range(n_trucks):
            act = approved_actions.get(k, {})  # 顶层用整型索引拿动作字典
            # 缺省：等待，目标=当前位置
            cur_pos = int(self.temp_db.status_dict.get('v_coord', [0]*max(1, n_trucks))[k]) if 'v_coord' in self.temp_db.status_dict else 0
            t_target = cur_pos
            t_wait = 1

            # 如果明确给了目标点，就认为本步执行了去该点（除非显式 truck_wait=1）
            if 'truck_target_node' in act and int(act.get('truck_wait', 0)) != 1:
                t_target = int(act['truck_target_node'])
                t_wait = 0
            else:
                # 如果只给了等待
                if int(act.get('truck_wait', 0)) == 1:
                    t_wait = 1
                    t_target = cur_pos  # 目标保持当前位置
                # 否则保持默认 no-op（等待）

            # 裁剪到合法范围
            t_target = max(0, min(int(t_target), total_nodes - 1))
            t_wait = 1 if int(t_wait) == 1 else 0

            exec_actions.append([t_target, t_wait])

        # ---------- Drones ----------
        for k in range(n_drones):
            agent_idx = n_trucks + k
            act = approved_actions.get(agent_idx, {})

            d_service = 0            # 0 表示“无服务”
            d_rendez = 0             # 0..n_trucks-1（若无卡车则保持 0）
            d_continue = 1           # 缺省继续

            if 'drone_service_node' in act:
                d_service = int(act['drone_service_node'])
                d_continue = 0
            elif ('drone_rendezvous_truck' in act) or ('_target_truck' in act):
                d_service = 0
                d_rendez = int(act.get('drone_rendezvous_truck', act.get('_target_truck', 0)))
                d_continue = 0
            elif int(act.get('drone_continue', 0)) == 1:
                d_service = 0
                d_continue = 1
            # 否则保持缺省 no-op（continue=1）

            # 合法化
            d_service = max(0, min(int(d_service), total_nodes - 1))
            max_truck_idx = max(0, n_trucks - 1)
            d_rendez = max(0, min(int(d_rendez), max_truck_idx))
            d_continue = 1 if int(d_continue) == 1 else 0

            exec_actions.append([d_service, d_rendez, d_continue])

        return exec_actions
    
    def _execute_actions(self, approved_actions, num_trucks, num_drones):
        """执行批准的动作 - 修复汇合逻辑时序问题"""
        
        # **关键修复1：预先保存所有载具的初始状态**
        initial_truck_states = {}
        for truck_idx in range(num_trucks):
            initial_truck_states[truck_idx] = {
                'ET': self.temp_db.status_dict['ET'][truck_idx],
                'LT': self.temp_db.status_dict['LT'][truck_idx],
                'pos': self._get_truck_position(truck_idx),
                'TW': self.temp_db.status_dict['TW'][truck_idx]  # 新增：保存初始载重
            }
        
        # **关键修复2：预先处理所有汇合请求，更新附着关系**
        rendezvous_requests = {}  # drone_idx -> target_truck_idx
        for action_key, act in approved_actions.items():
            if action_key >= num_trucks:  # 无人机动作
                drone_idx = action_key - num_trucks
                target_truck_idx = act.get('_target_truck')
                
                if target_truck_idx is not None and self.temp_db.status_dict['ED'][drone_idx] == 2:
                    # 验证卡车索引有效性
                    if target_truck_idx >= num_trucks:
                        target_truck_idx = 0
                    
                    # 记录汇合请求
                    rendezvous_requests[drone_idx] = target_truck_idx
                    
                    # **立即更新附着关系，避免时序问题**
                    old_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                    self.temp_db.status_dict['attached_truck'][drone_idx] = target_truck_idx
                    print(f"   🔄 Drone {drone_idx} attachment changed: truck {old_truck} -> truck {target_truck_idx}")
                
                    # **使用初始状态计算汇合节点**
                    if target_truck_idx in initial_truck_states:
                        initial_ET = initial_truck_states[target_truck_idx]['ET']
                        initial_pos = initial_truck_states[target_truck_idx]['pos']
                        
                        # 基于初始状态确定汇合节点
                        if initial_ET == 0:  # 卡车初始就在移动
                            rendezvous_node = self.temp_db.status_dict['NT'][target_truck_idx]
                        else:  # 卡车初始空闲/等待
                            rendezvous_node = initial_pos
                            
                        approved_actions[action_key]['drone_rendezvous_node'] = rendezvous_node
                        print(f"   Pre-calculated rendezvous: drone {drone_idx} -> truck {target_truck_idx} at node {rendezvous_node} (truck initial ET={initial_ET}, pos={initial_pos})")
        
        # 执行卡车动作
        for truck_idx in range(num_trucks):
            if truck_idx not in approved_actions:
                continue
                
            act = approved_actions[truck_idx]
            current_pos = self._get_truck_position(truck_idx)
            current_status = self.temp_db.status_dict['ET'][truck_idx]
            current_load = self.temp_db.status_dict['TW'][truck_idx]
            
            print(f"   🚛 Executing truck {truck_idx}: {act} (current_load: {current_load:.1f})")

            # 处理卡车动作
            tgt = act.get('truck_target_node')
            wait = act.get('truck_wait', 0) == 1
            
            if tgt is not None and not wait and current_status in [2, 3]:
                if tgt == current_pos:
                    # 立即服务：卡车已经在目标位置
                    if tgt != 0:
                        # 立即提供服务
                        self.temp_db.status_dict['ET'][truck_idx] = 2
                        if tgt in self.pre_assigned_nodes:
                            self.pre_assigned_nodes.remove(tgt)
                        self.temp_db.visited_nodes.add(tgt)
                        self.temp_db.mark_visited(tgt)

                        # **关键修复：立即服务时才减少载重**
                        served = self.trucks[truck_idx].serve(tgt)
                        old_load = self.temp_db.status_dict['TW'][truck_idx]
                        self.temp_db.status_dict['TW'][truck_idx] -= served
                        new_load = self.temp_db.status_dict['TW'][truck_idx]
                        print(f"     📦 Truck {truck_idx} immediately served node {tgt}: delivered {served:.1f}, load {old_load:.1f} -> {new_load:.1f}")
                else:
                    # **关键修复：移动到目标节点时不减载重，只记录移动信息**
                    d = self.temp_db.distance(current_pos, tgt)
                    truck_speed = getattr(self.trucks[truck_idx], 'speed', self.temp_db.truck_speed)
                    travel_time = d / truck_speed
                        
                    # 预先检查载重是否足够，但不减少
                    served = self.trucks[truck_idx].serve(tgt)  # 只计算需要的服务量，不实际提供
                    current_load = self.temp_db.status_dict['TW'][truck_idx]
                    
                    if current_load >= served:
                        self.temp_db.status_dict['ET'][truck_idx] = 0
                        self.temp_db.status_dict['LT'][truck_idx] = travel_time
                        self.temp_db.status_dict['NT'][truck_idx] = tgt
                        self.truck_dispatch_type[truck_idx] = 'truck_service'
                        
                        # **重要：不在这里减载重，载重在到达时减少**
                        print(f"     Truck {truck_idx} dispatched to {tgt}, travel time: {travel_time:.3f}")
                        print(f"     📋 Will deliver {served:.1f} upon arrival (current load: {current_load:.1f})")
                    else:
                        print(f"     ❌ Truck {truck_idx} insufficient load for node {tgt}: has {current_load:.1f}, needs {served:.1f}")
                        
            elif wait and current_status in [2, 3]:
                # 设置等待状态
                self.temp_db.status_dict['ET'][truck_idx] = 1  # 设置为等待状态
                self.temp_db.status_dict['LT'][truck_idx] = 0.0  # 清除剩余时间
                self.truck_dispatch_type[truck_idx] = 'truck_wait'
                print(f"     Truck {truck_idx} set to waiting state (ET=1)")


        # 执行无人机动作
        for drone_idx in range(num_drones):
            action_key = num_trucks + drone_idx
            if action_key not in approved_actions:
                continue
                
            act = approved_actions[action_key]
            current_pos = self._get_drone_position(drone_idx)
            current_status = self.temp_db.status_dict['ED'][drone_idx]
            current_load = self.temp_db.status_dict['DW'][drone_idx]
            current_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
            
            print(f"   🚁 Executing drone {drone_idx}: {act} (load: {current_load:.1f}, battery: {current_battery:.1f})")
            
            
            # 处理汇合请求 - 使用预计算的汇合节点
            rvz_node = act.get('drone_rendezvous_node')
            if rvz_node is not None and current_status == 2:
                attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                if current_pos == rvz_node:
                    self.temp_db.status_dict['ED'][drone_idx] = 1
                    # 载重归还：无人机返回卡车时归还剩余载重
                    if attached_truck >= 0 and attached_truck < num_trucks and current_load > 0:
                        old_truck_load = self.temp_db.status_dict['TW'][attached_truck]
                        self.temp_db.status_dict['TW'][attached_truck] += current_load
                        new_truck_load = self.temp_db.status_dict['TW'][attached_truck]
                        self.temp_db.status_dict['DW'][drone_idx] = 0.0
                        print(f"     📦 Drone {drone_idx} returned {current_load:.1f} load to truck {attached_truck}: truck load {old_truck_load:.1f} -> {new_truck_load:.1f}")
                    
                    print(f"     🤝 Drone {drone_idx} waiting for rendezvous at {rvz_node}")
                else:
                    d = self.temp_db.distance(current_pos, rvz_node)
                    drone_speed = getattr(self.drones[drone_idx], 'speed', self.temp_db.drone_speed)
                    travel_time = d / drone_speed

                    # 电量消耗：飞行时消耗电量
                    battery_consumption = d
                    old_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
                    self.temp_db.status_dict['drone_battery'][drone_idx] -= battery_consumption
                    new_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
                    
                    self.temp_db.status_dict['ED'][drone_idx] = 0
                    self.temp_db.status_dict['LD'][drone_idx] = travel_time
                    self.temp_db.status_dict['ND'][drone_idx] = rvz_node
                    self.drone_dispatch_type[drone_idx] = 'drone_rendezvous'
                    print(f"     🎯 Drone {drone_idx} flying to rendezvous at {rvz_node}, travel_time: {travel_time:.3f}, battery: {old_battery:.1f} -> {new_battery:.1f}")
                continue
            
            # 处理服务请求
            svc = act.get('drone_service_node')
            cont = act.get('drone_continue', 0) == 1

            if svc is not None and not cont and current_status in [2, 3]:  # **修复：包含ED=2状态**
                # 使用初始卡车状态进行判断
                attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                
                if attached_truck != -1 and current_status == 3:  # **修复：只有ED=3时才检查卡车状态**
                    # 搭载模式：使用保存的初始状态检查卡车可用性
                    if attached_truck in initial_truck_states:
                        initial_truck_status = initial_truck_states[attached_truck]['ET']
                        if initial_truck_status not in [1, 2, 3]:
                            print(f"     Drone {drone_idx} cannot service - truck {attached_truck} was not available (initial_ET={initial_truck_status})")
                            continue
                    else:
                        print(f"     Drone {drone_idx} cannot service - truck {attached_truck} state unknown")
                        continue
                        
                if svc == current_pos and svc != 0:
                    # 立即提供服务（已在目标位置）
                    self.temp_db.status_dict['ED'][drone_idx] = 2
                    if svc in self.pre_assigned_nodes:
                        self.pre_assigned_nodes.remove(svc)
                    self.temp_db.visited_nodes.add(svc)
                    self.temp_db.mark_visited(svc)

                    # **载重管理：服务时减少无人机载重**
                    served = self.drones[drone_idx].serve(svc)
                    old_load = self.temp_db.status_dict['DW'][drone_idx]
                    self.temp_db.status_dict['DW'][drone_idx] -= served
                    new_load = self.temp_db.status_dict['DW'][drone_idx]
                    
                    print(f"     📦 Drone {drone_idx} served node {svc}: delivered {served:.1f}, load {old_load:.1f} -> {new_load:.1f}")
                        
                    if attached_truck == -1:
                        print(f"     Independent drone {drone_idx} served node {svc}")
                    else:
                        print(f"     Drone {drone_idx} immediately served node {svc}")
                        
                elif svc != 0:
                    # **关键修复：无人机需要移动到服务节点时，状态必须设为ED=0**
                    # 移动到服务节点- 载重和电量管理
                    if attached_truck != -1 and current_status == 3:
                        # 搭载模式：从初始卡车位置出发，先装载货物
                        departure_pos = initial_truck_states[attached_truck]['pos']

                        # 【新增】从卡车起飞时确保满电
                        if self.temp_db.status_dict['drone_battery'][drone_idx] < self.temp_db.drone_battery:
                            print(f"     🔋 Charging drone {drone_idx} before departure")
                            self.temp_db.status_dict['drone_battery'][drone_idx] = self.temp_db.drone_battery
                                          
                        # **载重转移：无人机从卡车获得满载重**
                        drone_capacity = getattr(self.drones[drone_idx], 'capacity', self.temp_db.WD_max)
                        truck_current_load = self.temp_db.status_dict['TW'][attached_truck]
                        
                        # 计算无人机应该携带的载重
                        target_demand = self.temp_db.get_val('demand')[svc]
                        drone_load = min(drone_capacity, truck_current_load)
                        
                        if drone_load > 0:
                            # 从卡车转移载重到无人机
                            old_truck_load = self.temp_db.status_dict['TW'][attached_truck]
                            old_drone_load = self.temp_db.status_dict['DW'][drone_idx]
                            
                            self.temp_db.status_dict['TW'][attached_truck] -= drone_load
                            self.temp_db.status_dict['DW'][drone_idx] = drone_load
                            
                            new_truck_load = self.temp_db.status_dict['TW'][attached_truck]
                            new_drone_load = self.temp_db.status_dict['DW'][drone_idx]
                            
                            print(f"     📦 Load transfer: truck {attached_truck}: {old_truck_load:.1f} -> {new_truck_load:.1f}, drone {drone_idx}: {old_drone_load:.1f} -> {new_drone_load:.1f}")
                        else:
                            print(f"     ⚠️ Warning: No load available for drone {drone_idx} (truck_load: {truck_current_load:.1f}, target_demand: {target_demand:.1f})")
                    elif current_status == 2:  # **修复：ED=2状态的无人机从当前位置出发**
                        # 独立模式或刚完成服务：从当前位置出发
                        departure_pos = current_pos
                    else:
                        # 独立模式：从当前位置出发
                        departure_pos = current_pos
                        
                    self.temp_db.status_dict['drone_coord'][drone_idx] = departure_pos
                    
                    d = self.temp_db.distance(departure_pos, svc)
                    drone_speed = getattr(self.drones[drone_idx], 'speed', self.temp_db.drone_speed)
                    travel_time = d / drone_speed
                    
                    # **电量消耗：飞行时消耗电量**
                    if 'drone_battery' not in self.temp_db.status_dict:
                        self.temp_db.status_dict['drone_battery'] = np.full(num_drones, 1000.0)
                    
                    # 电量消耗：飞行时消耗电量
                    battery_consumption = d
                    old_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
                    self.temp_db.status_dict['drone_battery'][drone_idx] -= battery_consumption
                    new_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
                    
                    # **关键修复：设置无人机状态为在途(ED=0)**
                    self.temp_db.status_dict['ED'][drone_idx] = 0  # 在途状态
                    self.temp_db.status_dict['LD'][drone_idx] = travel_time
                    self.temp_db.status_dict['ND'][drone_idx] = svc
                    self.drone_dispatch_type[drone_idx] = 'drone_service'
                    
                    print(f"     🎯 Drone {drone_idx} flying to service {svc}, travel_time: {travel_time:.3f}, battery: {old_battery:.1f} -> {new_battery:.1f}")
                        
                    
                    if attached_truck == -1:
                        print(f"     Independent drone {drone_idx} dispatched to service {svc}, travel time: {travel_time:.3f}")
                    else:
                        print(f"     Drone {drone_idx} dispatched to service {svc}, travel time: {travel_time:.3f}")
                        
            elif cont and current_status == 3:
                # 继续跟随卡车
                attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                if attached_truck != -1:
                    truck_pos = self._get_truck_position(attached_truck)
                    self.temp_db.status_dict['drone_coord'][drone_idx] = truck_pos
                    print(f"     Drone {drone_idx} continues on truck {attached_truck}")
                    
    def _check_rendezvous_completion(self, num_trucks, num_drones):
        """检查汇合完成 - 修复等待逻辑"""
        print(f"   Checking rendezvous completion")
        
        # 收集每个卡车当前等待的无人机
        currently_waiting_drones = {}  # truck_idx -> set of drone_idx
        
        for drone_idx in range(num_drones):
            ED = self.temp_db.status_dict['ED'][drone_idx]
            attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
            
            # 只统计真正需要汇合的无人机
            if ED == 1 and attached_truck != -1:  # 等待汇合且不是独立模式
                if attached_truck not in currently_waiting_drones:
                    currently_waiting_drones[attached_truck] = set()
                currently_waiting_drones[attached_truck].add(drone_idx)
            elif ED == 0 and attached_truck != -1:  # 正在前往汇合的无人机
                dispatch_type = self.drone_dispatch_type[drone_idx]
                if dispatch_type == 'drone_rendezvous':
                    if attached_truck not in currently_waiting_drones:
                        currently_waiting_drones[attached_truck] = set()
                    currently_waiting_drones[attached_truck].add(drone_idx)
        
        print(f"   Current waiting drones per truck: {currently_waiting_drones}")
        
        # 处理已到达汇合点的无人机
        completed_rendezvous = []
        for drone_idx in range(num_drones):
            ED = self.temp_db.status_dict['ED'][drone_idx]
            if ED == 1:  # 无人机在等待汇合
                drone_pos = self.temp_db.status_dict['drone_coord'][drone_idx]
                attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                drone_load = self.temp_db.status_dict['DW'][drone_idx]
                
                if attached_truck != -1 and attached_truck < num_trucks:
                    truck_pos = self._get_truck_position(attached_truck)
                    truck_status = self.temp_db.status_dict['ET'][attached_truck]
                    
                    # 汇合条件：位置相同且卡车可接收
                    if drone_pos == truck_pos and truck_status in [1, 2, 3]:
                        # 载重归还：汇合完成时归还剩余载重
                        if drone_load > 0:
                            old_truck_load = self.temp_db.status_dict['TW'][attached_truck]
                            self.temp_db.status_dict['TW'][attached_truck] += drone_load
                            new_truck_load = self.temp_db.status_dict['TW'][attached_truck]
                            self.temp_db.status_dict['DW'][drone_idx] = 0.0
                            print(f"     📦 Rendezvous load return: drone {drone_idx} returned {drone_load:.1f} to truck {attached_truck}: {old_truck_load:.1f} -> {new_truck_load:.1f}")
                        
                        # 【新增】汇合时为无人机充满电
                        old_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
                        self.temp_db.status_dict['drone_battery'][drone_idx] = self.temp_db.drone_battery  # 充满电
                        new_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
                        print(f"     🔋 Drone {drone_idx} recharged: {old_battery:.1f} -> {new_battery:.1f} (FULL)")
                        
                        # 汇合完成
                        self.temp_db.status_dict['ED'][drone_idx] = 3
                        completed_rendezvous.append((drone_idx, attached_truck))
                        print(f"     Rendezvous completed: drone {drone_idx} with truck {attached_truck} at {drone_pos}")
        
        # 移除已完成汇合的无人机
        for drone_idx, truck_idx in completed_rendezvous:
            if truck_idx in currently_waiting_drones:
                currently_waiting_drones[truck_idx].discard(drone_idx)
        
        # 检查卡车是否可以结束等待状态
        for truck_idx in range(num_trucks):
            ET = self.temp_db.status_dict['ET'][truck_idx]
            if ET == 1:  # 卡车在等待
                waiting_drones = currently_waiting_drones.get(truck_idx, set())
                truck_load = self.temp_db.status_dict['TW'][truck_idx]
                
                # 关键修复：只有当没有无人机需要汇合时才结束等待
                if len(waiting_drones) == 0:
                    self.temp_db.status_dict['ET'][truck_idx] = 3
                    print(f"     Truck {truck_idx} finished waiting (load: {truck_load:.1f}), no more drones to rendezvous with")
                else:
                    print(f"     Truck {truck_idx} still waiting (load: {truck_load:.1f}) for {len(waiting_drones)} drones: {waiting_drones}")

    
    def _advance_time(self, num_trucks, num_drones):
        """推进时间"""
        times = []
        
        # 收集所有活动时间
        for truck_idx in range(num_trucks):
            lt = self.temp_db.status_dict['LT'][truck_idx]
            if lt > 1e-9:
                times.append(lt)
                
        for drone_idx in range(num_drones):
            ld = self.temp_db.status_dict['LD'][drone_idx]
            if ld > 1e-9:
                times.append(ld)

        # **关键修复：智能推进到动态节点激活时间**
        if not times:
            # 检查是否还有未激活的动态节点
            has_pending_nodes = False
            next_activation_time = float('inf')
            
            if self.temp_db.dynamic_enabled:
                # 检查动态节点池，找到最早的激活时间
                if self.temp_db.dynamic_nodes_pool and len(self.temp_db.dynamic_nodes_pool) > 0:
                    has_pending_nodes = True
                    # 节点池已按release_time排序，第一个就是最早的
                    if self.temp_db.dynamic_nodes_pool:
                        next_activation_time = self.temp_db.dynamic_nodes_pool[0]['release_time']
                        print(f"   Next node activation at time {next_activation_time:.1f}")
                
                # 也检查是否有未激活节点（通过delta=-1）
                delta = self.temp_db.get_val('delta')
                if np.any(delta == -1):
                    has_pending_nodes = True
            
            if has_pending_nodes:
                current_time = self.temp_db.total_time
                next_check = self.temp_db.next_check_time
                
                # 智能选择推进目标
                # 情况1：下一个检查点内有节点要激活
                if next_activation_time <= next_check:
                    # 推进到检查点（这样会触发激活检查）
                    if next_check > current_time and next_check <= self.temp_db.horizon:
                        delta_time = next_check - current_time
                        print(f"   No active movements, advancing {delta_time:.3f} to check point {next_check:.1f} (will activate nodes)")
                        self.temp_db.total_time = next_check
                        return
                # 情况2：下一个检查点内没有节点激活，直接跳到激活时间
                else:
                    if next_activation_time < float('inf') and next_activation_time > current_time:
                        if next_activation_time <= self.temp_db.horizon:
                            delta_time = next_activation_time - current_time
                            print(f"   No active movements, jumping {delta_time:.3f} directly to activation time {next_activation_time:.1f}")
                            self.temp_db.total_time = next_activation_time
                            # 更新下一个检查时间为激活时间后的下一个检查点
                            import math
                            next_check_multiplier = math.floor(next_activation_time / self.temp_db.delta_t) + 1
                            self.temp_db.next_check_time = next_check_multiplier * self.temp_db.delta_t
                            print(f"   Updated next_check_time to {self.temp_db.next_check_time:.1f}")
                            return
                    # 如果没有明确的激活时间，按原逻辑推进到检查点
                    elif next_check > current_time and next_check <= self.temp_db.horizon:
                        delta_time = next_check - current_time
                        print(f"   No active movements, advancing {delta_time:.3f} to check point {next_check:.1f}")
                        self.temp_db.total_time = next_check
                        return
                
                # 边界情况：如果都超过horizon，推进到horizon
                if current_time < self.temp_db.horizon:
                    delta_time = self.temp_db.horizon - current_time
                    print(f"   Advancing to horizon: {delta_time:.3f}")
                    self.temp_db.total_time = self.temp_db.horizon
                    return
            
            print("   No active movements and no pending dynamic nodes, time stays the same")
            return
            
        delta = min(times)
        print(f"   Time advancing by {delta:.3f}")
        self.temp_db.total_time += delta

        # 推进卡车时间
        for truck_idx in range(num_trucks):
            if self.temp_db.status_dict['LT'][truck_idx] > 1e-9:
                self.temp_db.status_dict['LT'][truck_idx] -= delta
                if abs(self.temp_db.status_dict['LT'][truck_idx]) <= 1e-9:
                    self.temp_db.status_dict['LT'][truck_idx] = 0
                    self._complete_truck_action(truck_idx)

        # 推进无人机时间
        for drone_idx in range(num_drones):
            if self.temp_db.status_dict['LD'][drone_idx] > 1e-9:
                self.temp_db.status_dict['LD'][drone_idx] -= delta
                if abs(self.temp_db.status_dict['LD'][drone_idx]) <= 1e-9:
                    self.temp_db.status_dict['LD'][drone_idx] = 0
                    self._complete_drone_action(drone_idx)

    def _complete_truck_action(self, truck_idx):
        """完成卡车动作 - 正确处理载重减少时机"""
        dispatch_type = self.truck_dispatch_type[truck_idx]
        target_node = self.temp_db.status_dict['NT'][truck_idx]
        current_load = self.temp_db.status_dict['TW'][truck_idx]
        
        self.temp_db.status_dict['v_coord'][truck_idx] = target_node
        print(f"     🚛 Truck {truck_idx} arrived at node {target_node} (load: {current_load:.1f})")
        
        if dispatch_type == 'truck_service' and target_node != 0:
            # **关键修复：只在到达时进行服务和载重减少**
            # 1. 完成服务逻辑
            if target_node in self.pre_assigned_nodes:
                self.pre_assigned_nodes.remove(target_node)
            self.temp_db.visited_nodes.add(target_node)
            self.temp_db.mark_visited(target_node)

            # **载重减少只在这里进行一次**
            served = self.trucks[truck_idx].serve(target_node)
            old_load = self.temp_db.status_dict['TW'][truck_idx]
            self.temp_db.status_dict['TW'][truck_idx] -= served
            new_load = self.temp_db.status_dict['TW'][truck_idx]    
            print(f"     📦 Truck {truck_idx} served customer node {target_node}: delivered {served:.1f}, load {old_load:.1f} -> {new_load:.1f}")
        
            # 2. 检查是否需要等待无人机汇合
            if self._should_truck_wait_for_rendezvous(truck_idx):
                self.temp_db.status_dict['ET'][truck_idx] = 1  # 等待状态
                print(f"     Truck {truck_idx} waiting for drone rendezvous after service")
            else:
                self.temp_db.status_dict['ET'][truck_idx] = 2  # 刚完成服务状态
        else:
            # 到达depot或其他节点后的状态处理
            if self._should_truck_wait_for_rendezvous(truck_idx):
                self.temp_db.status_dict['ET'][truck_idx] = 1  # 等待状态
                print(f"     Truck {truck_idx} waiting for drone rendezvous at node {target_node}")
            else:
                self.temp_db.status_dict['ET'][truck_idx] = 3  # 空闲状态
                print(f"     Truck {truck_idx} idle at node {target_node}")
                    
        self.truck_dispatch_type[truck_idx] = None

    def _should_truck_wait_for_rendezvous(self, truck_idx):
        """检查卡车是否应该等待无人机汇合 - 修复汇合检测逻辑"""
        try:
            num_drones = self.temp_db.num_drones
            for drone_idx in range(num_drones):
                ED = self.temp_db.status_dict['ED'][drone_idx]
                attached_truck = self.temp_db.status_dict.get('attached_truck', [0] * num_drones)[drone_idx]
                
                # **关键修复：只有当无人机明确需要与此卡车汇合时才等待**
                if attached_truck == truck_idx:
                    # 检查无人机是否正在前往汇合或刚完成服务需要汇合
                    if ED == 2:  # 无人机刚完成服务，需要汇合
                        print(f"     Truck {truck_idx} should wait: drone {drone_idx} needs rendezvous (ED=2)")
                        return True
                    elif ED == 0:  # 无人机在途中，检查是否是去汇合的
                        drone_dispatch_type = self.drone_dispatch_type[drone_idx]
                        if drone_dispatch_type == 'drone_rendezvous':
                            print(f"     Truck {truck_idx} should wait: drone {drone_idx} en route for rendezvous")
                            return True
                    elif ED == 1:  # 无人机已在等待汇合
                        print(f"     Truck {truck_idx} should wait: drone {drone_idx} waiting for rendezvous (ED=1)")
                        return True
                        
            print(f"     Truck {truck_idx} no drones need rendezvous, no waiting required")
            return False
        except Exception as e:
            print(f"     Error checking rendezvous requirement: {e}")
            return False

    def _complete_drone_action(self, drone_idx):
        """完成无人机动作 - 修复独立模式状态管理"""
        dispatch_type = self.drone_dispatch_type[drone_idx]
        target_node = self.temp_db.status_dict['ND'][drone_idx]
        attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
        current_load = self.temp_db.status_dict['DW'][drone_idx]
        current_battery = self.temp_db.status_dict['drone_battery'][drone_idx]
        
        self.temp_db.status_dict['drone_coord'][drone_idx] = target_node
        print(f"     🚁 Drone {drone_idx} arrived at node {target_node} (load: {current_load:.1f}, battery: {current_battery:.1f})")
        
        if dispatch_type == 'drone_service' and target_node != 0:
            # 无人机完成服务
            if target_node in self.pre_assigned_nodes:
                self.pre_assigned_nodes.remove(target_node)
            self.temp_db.visited_nodes.add(target_node)
            self.temp_db.mark_visited(target_node)
            
            # 载重和电量管理：服务时减少无人机载重和电量
            served = self.drones[drone_idx].serve(target_node)
            old_load = self.temp_db.status_dict['DW'][drone_idx]
            if 'DW' in self.temp_db.status_dict:
                self.temp_db.status_dict['DW'][drone_idx] -= served
            new_load = self.temp_db.status_dict['DW'][drone_idx]
            
            print(f"     📦 Drone {drone_idx} served node {target_node}: delivered {served:.1f}, load {old_load:.1f} -> {new_load:.1f}, battery: {current_battery:.1f}")
            
            # 设置为刚完成服务状态，让Actor网络决定下一步动作
            self.temp_db.status_dict['ED'][drone_idx] = 2
                
            # 关键修复：独立任务完成后保持独立状态
            if attached_truck == -1:
                print(f"     Independent drone {drone_idx} served customer node {target_node}")
                # 保持独立模式标记
                self.temp_db.status_dict['attached_truck'][drone_idx] = -1
            else:
                print(f"     Drone {drone_idx} served customer node {target_node}")
                
        elif dispatch_type == 'drone_rendezvous':
            # 无人机完成汇合移动，等待卡车
            self.temp_db.status_dict['ED'][drone_idx] = 1  # 等待汇合状态
            print(f"     Drone {drone_idx} waiting for rendezvous at node {target_node}")
        else:
            # 其他情况，根据是否独立决定状态
            if attached_truck == -1:
                self.temp_db.status_dict['ED'][drone_idx] = 2  # 独立模式设为可行动状态
            else:
                self.temp_db.status_dict['ED'][drone_idx] = 1  # 搭载模式设为等待状态
                
        self.drone_dispatch_type[drone_idx] = None

    def _get_current_drone_pos(self, drone_idx):
        """获取无人机当前位置"""
        try:
            ED = self.temp_db.status_dict['ED'][drone_idx]
            if ED == 3:
                # 无人机在卡车上，返回附着卡车的位置
                attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                if attached_truck < len(self.trucks):
                    return self.temp_db.status_dict['v_coord'][attached_truck]
                else:
                    # 如果附着卡车索引无效，返回第一个卡车位置或depot
                    return self.temp_db.status_dict['v_coord'][0] if len(self.trucks) > 0 else 0
            elif ED in [0, 1, 2]:
                # 无人机独立飞行、等待或刚完成服务
                if 'drone_coord' in self.temp_db.status_dict:
                    return self.temp_db.status_dict['drone_coord'][drone_idx]
                else:
                    # 如果没有独立坐标记录，返回附着卡车位置作为后备
                    attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                    if attached_truck < len(self.trucks):
                        return self.temp_db.status_dict['v_coord'][attached_truck]
                    else:
                        return 0
            else:
                # 未知状态，返回附着卡车位置
                attached_truck = self.temp_db.status_dict['attached_truck'][drone_idx]
                if attached_truck < len(self.trucks):
                    return self.temp_db.status_dict['v_coord'][attached_truck]
                else:
                    return 0
        except (KeyError, IndexError) as e:
            print(f"     Warning: Error getting drone {drone_idx} position: {e}")
            # 异常情况下返回depot位置
            return 0

    def _check_terminal_state(self):
        """检查终止条件 - 考虑动态节点"""
        try:
            delta = self.temp_db.get_val('delta')
            
            # 统计不同状态的节点
            inactive_dynamic = np.sum(delta == -1)  # 未激活的动态节点
            visited = np.sum(delta == 0)           # 已访问的节点
            active_unvisited = np.sum(delta == 1)  # 激活但未访问的节点
            
            # **新增：检查是否超过时间限制**
            if self.temp_db.total_time >= self.temp_db.horizon:
                print(f"   🎯 Time horizon reached: {self.temp_db.total_time:.1f} >= {self.temp_db.horizon:.1f}")
                return True

            # 检查是否还有未来会激活的动态节点
            future_activations = 0
            if self.temp_db.dynamic_enabled and hasattr(self.temp_db, 'dynamic_nodes_pool'):
                future_activations = len(self.temp_db.dynamic_nodes_pool)
            
            # 如果还有未激活的动态节点或未来会激活的节点，不能终止
            if inactive_dynamic > 0 or future_activations > 0:
                # 时间已用完则终止
                time_remaining = self.temp_db.horizon - self.temp_db.total_time
                if time_remaining <= 0:
                    print(f"   🎯 Time limit reached, terminating episode")
                    return True
                
                print(f"   📊 Terminal check: {active_unvisited} active unvisited, "
                    f"{inactive_dynamic} inactive dynamic, {future_activations} pending")
                return False
            
            # 检查是否所有激活的customer节点都被服务
            if active_unvisited > 0:
                print(f"   📊 Terminal check: {active_unvisited} customers still unserved")
                return False
            
            # 检查所有卡车是否都回到depot
            num_trucks = len(self.trucks)
            for truck_idx in range(num_trucks):
                truck_pos = self.temp_db.status_dict['v_coord'][truck_idx]
                ET = self.temp_db.status_dict['ET'][truck_idx]
                
                # 卡车必须在depot且空闲
                if truck_pos != 0 or ET not in [2, 3]:
                    print(f"   📊 Truck {truck_idx} not at depot: pos={truck_pos}, ET={ET}")
                    return False
            
            # 检查所有无人机是否都在卡车上
            num_drones = len(self.drones)
            for drone_idx in range(num_drones):
                ED = self.temp_db.status_dict['ED'][drone_idx]
                
                # 无人机必须在卡车上
                if ED != 3:
                    print(f"   📊 Drone {drone_idx} not on truck: ED={ED}")
                    return False
            
            print(f"   🎯 Terminal condition met: All active customers served, vehicles at depot")
            print(f"      Statistics: {visited} visited, {inactive_dynamic} never activated")
            return True
            
        except Exception as e:
            print(f"   ❌ Error in terminal state check: {e}")
            return False

    def _check_episode_statistics(self):
        """
        在episode结束时统计性能指标（可选的辅助方法）
        """
        try:
            delta = self.temp_db.get_val('delta')
            current_time = self.temp_db.current_time
            horizon = self.temp_db.horizon
            
            # 统计各种节点状态
            served_nodes = sum(1 for i, d in enumerate(delta) if i > 0 and d == 0)
            unserved_active = sum(1 for i, d in enumerate(delta) if i > 0 and d == 1)
            inactive_dynamic = sum(1 for i, d in enumerate(delta) if i > 0 and d == -1)
            
            # 统计未来还会激活的节点
            future_activations = 0
            if self.temp_db.dynamic_enabled:
                future_activations = len([
                    node for node in self.temp_db.dynamic_nodes_pool
                    if node['release_time'] > current_time
                ])
            
            print(f"\n📈 Episode Statistics:")
            print(f"   Time used: {current_time:.1f}/{horizon:.1f} ({current_time/horizon*100:.1f}%)")
            print(f"   Served customers: {served_nodes}")
            print(f"   Unserved active customers: {unserved_active}")
            print(f"   Inactive dynamic customers: {inactive_dynamic}")
            print(f"   Future activations (missed): {future_activations}")
            print(f"   Service rate: {served_nodes/(served_nodes + unserved_active + future_activations)*100:.1f}%")
            
            return {
                'served': served_nodes,
                'unserved_active': unserved_active,
                'inactive_dynamic': inactive_dynamic,
                'future_activations': future_activations,
                'time_used': current_time,
                'time_utilization': current_time / horizon,
                'service_rate': served_nodes / (served_nodes + unserved_active + future_activations)
            }
            
        except Exception as e:
            print(f"   ❌ Error in episode statistics: {e}")
            return {}