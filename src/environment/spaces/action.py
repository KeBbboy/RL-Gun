import numpy as np
from gym import spaces

def None_to_empty_list(var):
    """
    将 None 转为 [], 将单值（int/str）转为 [var]，
    list/tuple/ndarray 保持不变，方便统一处理。
    """
    if isinstance(var, (list, tuple, np.ndarray)):
        return var
    if var is None:
        return []
    return [var]

class BaseActDecoder:
    def __init__(self, act_params, temp_db, simulator):
        """
        act_params 必须包含：
        truck_discrete_outputs = ['truck_target_node']      # XTv_k,i
        truck_binary_discrete = ['truck_wait']            # XTw_k
        drone_discrete_outputs = ['drone_service_node', 'drone_rendezvous_truck']      # XDv_k,i, XDr_k,i  
        drone_binary_discrete = ['drone_continue']         # XDc_k
        """
        self.temp_db   = temp_db
        self.simulator = simulator
        
        # Define truck-specific actions
        self.truck_discrete_outputs = act_params.get('truck_discrete_outputs', [
            'truck_target_node'  # XTv_k,i
        ])
        self.truck_binary_discrete = act_params.get('truck_binary_discrete', [
            'truck_wait'  # XTw_k
        ])
        
        # Define drone-specific actions - 修正：增加第三个动作头
        self.drone_discrete_outputs = act_params.get('drone_discrete_outputs', [
            'drone_service_node',     # XDv_k,i
            'drone_rendezvous_truck'   # XDr_k,i
        ])
        self.drone_binary_discrete = act_params.get('drone_binary_discrete', [
            'drone_continue',   # XDc_k
        ])

        # Validate allowed actions
        allowed_truck_d = {'truck_target_node'}
        allowed_truck_b = {'truck_wait'}
        allowed_drone_d = {'drone_service_node', 'drone_rendezvous_truck',}
        allowed_drone_b = {'drone_continue'}
        
        if not set(self.truck_discrete_outputs).issubset(allowed_truck_d):
            raise ValueError(f"Unsupported truck discrete outputs: {self.truck_discrete_outputs}")
        if not set(self.truck_binary_discrete).issubset(allowed_truck_b):
            raise ValueError(f"Unsupported truck binary outputs: {self.truck_binary_discrete}")
        if not set(self.drone_discrete_outputs).issubset(allowed_drone_d):
            raise ValueError(f"Unsupported drone discrete outputs: {self.drone_discrete_outputs}")  
        if not set(self.drone_binary_discrete).issubset(allowed_drone_b):
            raise ValueError(f"Unsupported drone binary outputs: {self.drone_binary_discrete}")

        self.truck_act_spaces = []
        self.drone_act_spaces = []
        self.truck_func_dict = {}
        self.drone_func_dict = {}


    def reset(self):
        """
        在每个 episode 开始时由 Environment 调用，
        如果 decoder 内部有状态，需要在这里清理；否则 pass 即可。
        """
        pass

    def finish_init(self):
        """构造固定大小的动作空间，包含所有节点（动态节点预留空间）"""
        # 使用总节点数（包括未激活的动态节点）
        total_nodes = self.temp_db.num_nodes  # 这是固定的值
        num_trucks = self.temp_db.num_trucks
        
        print(f"ActionDecoder.finish_init: total_nodes={total_nodes} (fixed), trucks={num_trucks}")
        
        # 构建卡车动作空间 - 使用固定的总节点数
        for key in self.truck_discrete_outputs:
            self.truck_act_spaces.append(spaces.Discrete(total_nodes))
            self.truck_func_dict[key] = lambda a, k=key: (np.argmax(a) if hasattr(a, '__len__') else int(a))
        
        for key in self.truck_binary_discrete:
            self.truck_act_spaces.append(spaces.Discrete(2))
            self.truck_func_dict[key] = lambda a, k=key: int(a)
        
        # 构建无人机动作空间 - 使用固定的总节点数
        for key in self.drone_discrete_outputs:
            if key == 'drone_rendezvous_truck':
                self.drone_act_spaces.append(spaces.Discrete(num_trucks))
            else:
                self.drone_act_spaces.append(spaces.Discrete(total_nodes))
            self.drone_func_dict[key] = lambda a, k=key: (np.argmax(a) if hasattr(a, '__len__') else int(a))
        
        for key in self.drone_binary_discrete:
            self.drone_act_spaces.append(spaces.Discrete(2))
            self.drone_func_dict[key] = lambda a, k=key: int(a)
        
        print(f"Truck action spaces: {len(self.truck_act_spaces)} heads, node dim: {total_nodes}")
        print(f"Drone action spaces: {len(self.drone_act_spaces)} heads, node dim: {total_nodes}")


    def action_space(self):
        """Return action spaces for all agents (trucks first, then drones)"""
        num_trucks = self.temp_db.num_trucks
        num_drones = self.temp_db.num_drones

        action_spaces = []
        
        # 添加卡车动作空间
        for _ in range(num_trucks):
            action_spaces.append(spaces.Tuple(self.truck_act_spaces))
            
        # 添加无人机动作空间    
        for _ in range(num_drones):
            action_spaces.append(spaces.Tuple(self.drone_act_spaces))
            
        return spaces.Tuple(action_spaces)

    def _get_agent_type(self, agent_idx):
        """Determine if agent is truck or drone based on index"""
        num_trucks = self.temp_db.num_trucks
        if agent_idx < num_trucks:
            return 'truck', agent_idx  # truck index
        else:
            return 'drone', agent_idx - num_trucks  # drone index

    
    def _get_valid_nodes_for_truck(self, k):
        """获取卡车可以访问的有效节点 - 排除未激活的动态节点"""
        try:
            delta = self.temp_db.get_val('delta')
            TW = self.temp_db.get_val('TW')[k] if 'TW' in self.temp_db.status_dict else float('inf')
            demand = self.temp_db.get_val('demand') if hasattr(self.temp_db, 'get_val') else None
            
            # 关键修复：只考虑激活的节点（delta >= 0）
            # delta = -1: 未激活的动态节点（不可选）
            # delta = 0: 已访问的节点（不可选）
            # delta = 1: 激活且未访问的节点（可选）
            
            active_unassigned = {i for i, d in enumerate(delta) if d == 1}  # 只选择delta=1的节点
            active_customers = {i for i in active_unassigned if i > 0}  # 排除depot
            
            # 获取受损道路节点
            if hasattr(self.temp_db, 'get_road_damaged_nodes'):
                damaged = set(self.temp_db.get_road_damaged_nodes())
            else:
                damaged = set()
            
            # 卡车只能访问：激活的customer且未受损的节点
            truck_valid = active_customers - damaged
            
            # 根据载重限制进一步过滤
            if demand is not None and TW != float('inf'):
                truck_valid = {i for i in truck_valid if i < len(demand) and demand[i] <= TW}
            
            # 确保depot总是可达的
            truck_valid.add(0)
            
            print(f"    [Decoder] Truck {k} valid nodes: {truck_valid} (excluded inactive dynamic nodes)")
            return truck_valid
            
        except Exception as e:
            print(f"    [Decoder] Error getting truck valid nodes for agent {k}: {e}")
            return {0}


    def _get_valid_nodes_for_drone(self, k):
        """获取无人机可以访问的有效节点 - 排除未激活的动态节点"""
        try:
            delta = self.temp_db.get_val('delta')
            WD_max = self.temp_db.get_val('WD_max')[k] if 'WD_max' in self.temp_db.status_dict else float('inf')
            demand = self.temp_db.get_val('demand') if hasattr(self.temp_db, 'get_val') else None
            
            # 关键修复：只考虑激活的节点（delta >= 0）
            active_unassigned = {i for i, d in enumerate(delta) if d == 1}
            
            # 无人机可以访问所有激活未分配的节点（包括受损道路）
            drone_valid = active_unassigned.copy()
            
            # 根据无人机载重限制过滤
            if demand is not None and WD_max != float('inf'):
                drone_valid = {i for i in drone_valid if i < len(demand) and demand[i] <= WD_max}
            
            # 确保depot总是可达的
            drone_valid.add(0)
            
            print(f"    [Decoder] Drone {k} valid nodes: {drone_valid} (excluded inactive dynamic nodes)")
            return drone_valid
            
        except Exception as e:
            print(f"    [Decoder] Error getting drone valid nodes for agent {k}: {e}")
            return {0}
    

    def _get_truck_target_node(self, truck_idx, raw_action):
        """获取卡车选择的目标服务节点"""
        try:
            # 获取卡车的目标节点动作
            decoded_truck = self._decode_truck_action(raw_action, truck_idx)
            return decoded_truck.get('truck_target_node', 0)
        except:
            return 0

    def _get_rendezvous_node_for_truck(self, target_truck_idx):
        """获取指定卡车的汇合节点（基于其目标服务节点或当前位置）"""
        try:
            # 获取目标卡车的状态
            ET = self.temp_db.get_val('ET')[target_truck_idx]
            current_pos = self.temp_db.status_dict.get('v_coord', [0] * len(self.temp_db.get_val('ET')))[target_truck_idx]
            
            # 如果卡车正在移动中，汇合点是其目标节点
            if ET == 0:
                target_node = self.temp_db.get_val('NT')[target_truck_idx]
                print(f"    [Decoder] Truck {target_truck_idx} is moving to {target_node}, rendezvous there")
                return target_node
            # 如果卡车空闲或刚完成服务，汇合点是其当前位置
            elif ET in [2, 3]:
                print(f"    [Decoder] Truck {target_truck_idx} at position {current_pos}, rendezvous there")
                return current_pos
            # 如果卡车在等待，汇合点是其当前位置
            elif ET == 1:
                print(f"    [Decoder] Truck {target_truck_idx} waiting at {current_pos}, rendezvous there")
                return current_pos
            else:
                return current_pos
                
        except Exception as e:
            print(f"    [Decoder] Error getting rendezvous node for truck {target_truck_idx}: {e}")
            return 0

    def decode(self, raw_actions):
        """
        修改后的解码方法，修复汇合逻辑错误
        
        动作逻辑说明：
        1. 卡车智能体：
        - 在途时(ET=0, LT>0)：不执行新动作
        - 空闲时(ET=3)：可以执行truck_target_node或truck_wait
        - 刚完成服务时(ET=2)：可以执行下一个动作
        - 等待时(ET=1)：只能继续等待
        
        2. 无人机智能体：
        - 在途时(ED=0, LD>0)：不执行新动作
        - 在卡车上时(ED=3)：可以执行drone_service_node或drone_continue
        - 刚完成服务时(ED=2)：必须执行drone_rendezvous_truck选择汇合卡车或继续独立服务
        - 等待时(ED=1)：只能等待卡车到达
        
        3. 汇合逻辑：
        - 无人机选择汇合卡车后，汇合节点基于目标卡车的状态：
            * 如果卡车在移动，汇合点是卡车的目标节点
            * 如果卡车空闲/完成服务，汇合点是卡车当前位置
        """
        decoded = self._base_decode(raw_actions)
    
        num_trucks = self.temp_db.num_trucks
        num_drones = self.temp_db.num_drones
        final_actions = {}
        
        print(f"    [Decoder] Processing {num_trucks} trucks and {num_drones} drones")
        
        # **关键修复：先处理无人机动作，确定实际的汇合请求**
        actual_rendezvous_requests = {}  # drone_idx -> target_truck_idx
        
        # 第一步：处理无人机动作，确定实际选择的动作类型
        for drone_idx in range(num_drones):
            agent_idx = num_trucks + drone_idx
            if agent_idx in decoded:
                drone_action = decoded[agent_idx]
                ED = self.temp_db.status_dict['ED'][drone_idx]
                
                # 只有刚完成服务的无人机(ED=2)才能选择汇合或继续服务
                if ED == 2:
                    service_node = drone_action.get('drone_service_node')
                    target_truck_idx = drone_action.get('drone_rendezvous_truck')
                    
                    # **关键修复：根据Actor网络的实际选择确定动作优先级**
                    if service_node is not None and service_node != 0:
                        # 检查无人机是否有能力服务该节点
                        if self._can_drone_service_node_with_resources(drone_idx, service_node):
                            # 无人机选择继续服务，不进行汇合
                            print(f"    [Decoder] Drone {drone_idx} chooses to service node {service_node}, no rendezvous")
                            # 设置为独立模式
                            self.temp_db.status_dict['attached_truck'][drone_idx] = -1
                        else:
                            # 无人机资源不足，必须汇合
                            if target_truck_idx is not None and target_truck_idx < num_trucks:
                                actual_rendezvous_requests[drone_idx] = target_truck_idx
                                print(f"    [Decoder] Drone {drone_idx} insufficient resources, must rendezvous with truck {target_truck_idx}")
                            else:
                                # 没有有效汇合目标，选择最近的卡车
                                actual_rendezvous_requests[drone_idx] = 0
                                print(f"    [Decoder] Drone {drone_idx} insufficient resources, default rendezvous with truck 0")
                                
                    elif target_truck_idx is not None:
                        # 无人机明确选择汇合
                        if target_truck_idx >= num_trucks:
                            target_truck_idx = 0
                        actual_rendezvous_requests[drone_idx] = target_truck_idx
                        print(f"    [Decoder] Drone {drone_idx} explicitly chooses rendezvous with truck {target_truck_idx}")
                    else:
                        # 没有明确动作，默认保持当前模式
                        print(f"    [Decoder] Drone {drone_idx} no clear action, maintaining current state")
        
        # 第二步：基于实际汇合请求处理卡车智能体
        for truck_idx in range(num_trucks):
            if truck_idx in decoded:
                truck_action = decoded[truck_idx]
                
                # **关键修复：基于实际汇合请求检查是否需要等待**
                should_wait_for_rendezvous = any(
                    target_truck == truck_idx for target_truck in actual_rendezvous_requests.values()
                )
                
                if should_wait_for_rendezvous:
                    print(f"    [Decoder] Truck {truck_idx} will wait due to actual rendezvous requests")
                    final_action = {'truck_wait': 1}
                else:
                    final_action = self._process_truck_action(truck_idx, truck_action)
                    
                if final_action:
                    final_actions[truck_idx] = final_action
        
        # 第三步：处理无人机智能体
        for drone_idx in range(num_drones):
            agent_idx = num_trucks + drone_idx
            if agent_idx in decoded:
                drone_action = decoded[agent_idx]
                
                # **关键修复：如果是实际汇合请求，创建完整的汇合动作**
                if drone_idx in actual_rendezvous_requests:
                    target_truck_idx = actual_rendezvous_requests[drone_idx]
                    # 获取目标卡车的汇合节点
                    rendezvous_node = self._get_rendezvous_node_for_truck(target_truck_idx)
                    
                    # **关键修复：创建完整的汇合动作**
                    final_action = {
                        'drone_rendezvous_node': rendezvous_node,
                        '_target_truck': target_truck_idx
                    }
                    final_actions[agent_idx] = final_action
                    
                    # 更新附着关系
                    self.temp_db.status_dict['attached_truck'][drone_idx] = target_truck_idx
                    print(f"    [Decoder] Drone {drone_idx} will rendezvous with truck {target_truck_idx} at node {rendezvous_node}")
                else:
                    # 处理非汇合动作
                    final_action = self._process_drone_action(drone_idx, drone_action)
                    if final_action:
                        final_actions[agent_idx] = final_action
        
        print(f"    [Decoder] Final decoded actions: {final_actions}")
        return final_actions

    def _decode_truck_action(self, raw_action, truck_idx):
        """Decode truck action - needs implementation"""
        decoded = {}
        action_idx = 0
        
        # Decode truck discrete outputs
        for key in self.truck_discrete_outputs:
            if action_idx < len(raw_action):
                decoded[key] = self.truck_func_dict[key](raw_action[action_idx])
                action_idx += 1
        
        # Decode truck binary discrete
        for key in self.truck_binary_discrete:
            if action_idx < len(raw_action):
                decoded[key] = self.truck_func_dict[key](raw_action[action_idx])
                action_idx += 1
        
        return decoded

    def _decode_drone_action(self, raw_action, drone_idx):
        """Decode drone action - needs implementation"""
        decoded = {}
        action_idx = 0
        
        # Decode drone discrete outputs
        for key in self.drone_discrete_outputs:
            if action_idx < len(raw_action):
                decoded[key] = self.drone_func_dict[key](raw_action[action_idx])
                action_idx += 1
        
        # Decode drone binary discrete
        for key in self.drone_binary_discrete:
            if action_idx < len(raw_action):
                decoded[key] = self.drone_func_dict[key](raw_action[action_idx])
                action_idx += 1
        
        return decoded
    

    def _process_truck_action(self, truck_idx, raw_action):
        """处理卡车智能体的动作 - 修复汇合逻辑错误"""
        try:
            # 如果传入的是列表，先解码为字典
            if isinstance(raw_action, (list, tuple)):
                truck_action = self._decode_truck_action(raw_action, truck_idx)
            else:
                truck_action = raw_action
                
            ET = self.temp_db.get_val('ET')[truck_idx]
            LT = self.temp_db.get_val('LT')[truck_idx]
            current_pos = self.temp_db.status_dict.get('v_coord', [0] * len(self.temp_db.get_val('ET')))[truck_idx]
            print(f"    [Decoder] Processing truck {truck_idx}: ET={ET}, LT={LT}, pos={current_pos}")
        
            # 检查卡车是否在途
            if ET == 0 and LT > 0:
                print(f"    [Decoder] Truck {truck_idx} is in transit, no new actions allowed")
                return {}

            # **新增：检查任务完成状态**
            if ET in [2, 3] and current_pos == 0:  # 卡车在depot且空闲/刚完成服务
                delta = self.temp_db.get_val('delta')
                unserved_customers = sum(1 for i, d in enumerate(delta) if i > 0 and d == 1)
                
                if unserved_customers == 0:  # 所有customer节点都已分配/服务
                    print(f"    [Decoder] Truck {truck_idx} at depot, all customers served, staying idle")
                    return {}  # 不执行任何动作，保持当前状态
            
            # **关键修复：移除错误的汇合检测，在decode()方法中统一处理**
            # 这里不再检查汇合请求，因为汇合逻辑已经在decode()方法中处理
            
            # 获取有效节点
            valid_nodes = self._get_valid_nodes_for_truck(truck_idx)
            
            final_action = {}
            
            # 处理等待状态
            if ET == 1:
                print(f"    [Decoder] Truck {truck_idx} is waiting, can only continue waiting")
                final_action['truck_wait'] = 1
                return final_action
            
            # 处理正常动作（空闲或刚完成服务）
            if ET in [2, 3]:
                target_node = truck_action.get('truck_target_node')
                wait = truck_action.get('truck_wait', 0)
                
                if target_node is not None and target_node in valid_nodes:
                    final_action['truck_target_node'] = target_node
                    print(f"    [Decoder] Truck {truck_idx} targets node {target_node}")
                # elif wait == 1:
                #     final_action['truck_wait'] = 1
                #     print(f"    [Decoder] Truck {truck_idx} chooses to wait")
                
                else:
                    # 默认行为：选择最近的未服务节点或返回depot
                    if valid_nodes and len(valid_nodes) > 1:
                        fallback = min(valid_nodes - {0})
                        final_action['truck_target_node'] = fallback
                        print(f"    [Decoder] Truck {truck_idx} using fallback target: {fallback}")
                    else:
                        final_action['truck_target_node'] = 0
                        print(f"    [Decoder] Truck {truck_idx} returning to depot")
            
            return final_action
            
        except Exception as e:
            print(f"    [Decoder] Error processing truck {truck_idx}: {e}")
            return {}
    
    def _is_truck_selected_for_rendezvous(self, truck_idx):
        """检查卡车是否被无人机选为汇合目标 - 修复检测逻辑"""
        try:
            num_drones = self.temp_db.num_drones
            
            # **关键修复：只检查真正需要与此卡车汇合的无人机**
            for drone_idx in range(num_drones):
                ED = self.temp_db.status_dict['ED'][drone_idx]
                
                # 只有刚完成服务的无人机(ED=2)才可能请求汇合
                if ED == 2:
                    # **修复：检查当前动作中的汇合请求，而不是依赖附着关系**
                    # 这里需要访问当前正在处理的动作，但由于架构限制，
                    # 我们改为检查无人机是否真的需要汇合
                    
                    current_attached = self.temp_db.status_dict.get('attached_truck', [0] * num_drones)[drone_idx]
                    
                    # 检查无人机是否可能选择与此卡车汇合
                    # 这里可以根据具体的汇合策略来判断
                    
                    # 简化处理：如果无人机当前附着在此卡车上且刚完成服务，可能需要汇合
                    if current_attached == truck_idx:
                        print(f"     Truck {truck_idx} may need to wait: drone {drone_idx} attached and just completed service (ED=2)")
                        return True
                        
                # **新增：检查正在前往汇合的无人机**
                elif ED == 0:  # 无人机在途中
                    attached_truck = self.temp_db.status_dict.get('attached_truck', [0] * num_drones)[drone_idx]
                    if attached_truck == truck_idx:
                        # 检查调度类型是否是汇合
                        drone_dispatch_type = getattr(self.simulator, 'drone_dispatch_type', [None] * num_drones)
                        if (drone_idx < len(drone_dispatch_type) and 
                            drone_dispatch_type[drone_idx] == 'drone_rendezvous'):
                            print(f"     Truck {truck_idx} should wait: drone {drone_idx} en route for rendezvous")
                            return True
                            
                elif ED == 1:  # 无人机在等待汇合
                    attached_truck = self.temp_db.status_dict.get('attached_truck', [0] * num_drones)[drone_idx]
                    if attached_truck == truck_idx:
                        print(f"     Truck {truck_idx} should wait: drone {drone_idx} waiting for rendezvous (ED=1)")
                        return True
            
            print(f"     Truck {truck_idx} no drones need rendezvous")
            return False
            
        except Exception as e:
            print(f"     Error checking rendezvous selection: {e}")
            return False

    def _process_drone_action(self, drone_idx, raw_action):
        """处理无人机智能体的动作 - 修复独立模式判断 - 基于Actor网络决策的连续服务"""
        try:
            # 如果传入的是列表，先解码为字典
            if isinstance(raw_action, (list, tuple)):
                drone_action = self._decode_drone_action(raw_action, drone_idx)
            else:
                drone_action = raw_action
                
            # 获取无人机状态
            ED = self.temp_db.get_val('ED')[drone_idx]
            LD = self.temp_db.get_val('LD')[drone_idx]
            attached_truck = self.temp_db.status_dict.get('attached_truck', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
            current_load = self.temp_db.status_dict.get('DW', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
            current_battery = self.temp_db.status_dict.get('drone_battery', [1000.0] * len(self.temp_db.get_val('ED')))[drone_idx]
        
            # 关键修复：正确判断独立模式
            is_independent = (attached_truck == -1)
            
            print(f"    [Decoder] Processing drone {drone_idx}: ED={ED}, LD={LD}, attached_truck={attached_truck}, independent={is_independent}")
            print(f"    [Decoder] Drone {drone_idx} resources: load={current_load:.1f}, battery={current_battery:.1f}")
        
            # 检查无人机是否在途
            if ED == 0 and LD > 0:
                print(f"    [Decoder] Drone {drone_idx} is in transit, no new actions allowed")
                return {}
            
            # 搭载模式：检查卡车状态
            if not is_independent and ED == 3 and attached_truck >= 0 and attached_truck < len(self.temp_db.get_val('ET')):
                truck_ET = self.temp_db.get_val('ET')[attached_truck]
                truck_LT = self.temp_db.get_val('LT')[attached_truck]
                
                # 如果卡车正在移动中，搭载的无人机不能执行新动作
                if truck_ET == 0 and truck_LT > 0:
                    print(f"    [Decoder] Drone {drone_idx} cannot act - truck {attached_truck} is in transit (ET={truck_ET}, LT={truck_LT})")
                    return {}
            
            final_action = {}
            
            # **关键修复：无人机刚完成服务(ED=2)的处理逻辑**
            if ED == 2:
                print(f"    [Decoder] Drone {drone_idx} just completed service")
                
                # 检查Actor是否选择继续服务其他节点
                service_node = drone_action.get('drone_service_node')
                if service_node is not None and service_node != 0:
                    # 验证无人机是否有能力服务该节点
                    if self._can_drone_service_node_with_resources(drone_idx, service_node):
                        # 明确设置为独立模式（如果之前不是）
                        self.temp_db.status_dict['attached_truck'][drone_idx] = -1
                        final_action['drone_service_node'] = service_node
                        print(f"    [Decoder] Drone {drone_idx} Actor chose to continue servicing node {service_node}")
                        return final_action
                    else:
                        print(f"    [Decoder] Drone {drone_idx} insufficient resources for node {service_node}, should choose rendezvous")
                        return {}
                else:        
                    # 检查是否要选择汇合模式
                    target_truck = drone_action.get('drone_rendezvous_truck')
                    rendezvous_node = drone_action.get('drone_rendezvous_node')
                    
                    if target_truck is not None and rendezvous_node is not None:
                        # 执行汇合动作
                        final_action['drone_rendezvous_node'] = rendezvous_node
                        final_action['_target_truck'] = target_truck
                        print(f"    [Decoder] Drone {drone_idx} will rendezvous with truck {target_truck} at node {rendezvous_node}")
                        return final_action
                    else:
                        # 关键修复：明确设置为独立模式
                        print(f"    [Decoder] Drone {drone_idx} remains in independent mode")
                        self.temp_db.status_dict['attached_truck'][drone_idx] = -1
                        return {}
            
            # 其余处理逻辑保持不变...
            elif ED == 1:
                print(f"    [Decoder] Drone {drone_idx} is waiting for truck rendezvous")
                return {}
            
            elif ED == 3 or is_independent:
                service_node = drone_action.get('drone_service_node')
                continue_flag = drone_action.get('drone_continue', 0) == 1
                
                if is_independent:
                    # 独立模式：不需检查卡车状态，可以自由执行任务
                    if service_node is not None:
                        valid_nodes = self._get_valid_nodes_for_drone(drone_idx)
                        if service_node in valid_nodes and service_node != 0:
                            if self._can_drone_service_node_with_resources(drone_idx, service_node):
                                final_action['drone_service_node'] = service_node
                                print(f"    [Decoder] Independent drone {drone_idx} will service node {service_node}")
                            else:
                                print(f"    [Decoder] Independent drone {drone_idx} cannot service node {service_node}")
                        else:
                            print(f"    [Decoder] Invalid service node {service_node} for independent drone {drone_idx}")
                    # 独立无人机不支持continue动作
                    return final_action
                else:
                    # 搭载模式：检查卡车状态，但更宽松的限制
                    if attached_truck >= 0 and attached_truck < len(self.temp_db.get_val('ET')):
                        truck_ET = self.temp_db.get_val('ET')[attached_truck]
                        truck_LT = self.temp_db.get_val('LT')[attached_truck]
                        
                        # 修复：只有当卡车正在移动时才强制无人机继续
                        if truck_ET == 0 and truck_LT > 0:
                            print(f"    [Decoder] Drone {drone_idx} must continue - truck {attached_truck} is in transit (ET={truck_ET}, LT={truck_LT})")
                            final_action['drone_continue'] = 1
                            return final_action
                    
                    # 处理无人机动作选择
                    if continue_flag == 1:
                        # final_action['drone_continue'] = 1
                        # print(f"    [Decoder] Drone {drone_idx} continues on truck")
                        # ========= DeadlockGuard：在以下条件下覆盖 continue 为“强制服务” =========
                        # 条件：仍有未服务客户 + 所有卡车无可达客户(除仓库) + 当前无人机存在可服务目标
                        try:
                            # a) 仍有未服务客户
                            delta = self.temp_db.get_val('delta')
                            unserved_nodes = {i for i, d in enumerate(delta) if i > 0 and d == 1}

                            # b) 所有卡车是否均无可达客户（只剩 depot=0）
                            num_trucks = self.temp_db.num_trucks
                            trucks_can_reach = any(
                                len(self._get_valid_nodes_for_truck(k) - {0}) > 0
                                for k in range(num_trucks)
                            )

                            # c) 该无人机的可达客户集合（去掉 depot=0）
                            valid_nodes = self._get_valid_nodes_for_drone(drone_idx) - {0}

                            if unserved_nodes and not trucks_can_reach and valid_nodes:
                                # 优先使用 actor 第1头给的 service_node；若不可行则从可行集合里找一个
                                candidate = None
                                if (service_node is not None and
                                    service_node in valid_nodes and
                                    self._can_drone_service_node_with_resources(drone_idx, service_node)):
                                    candidate = service_node
                                else:
                                    # 按“可服务”验证挑第一个可行点（已包含载重/电量/动态节点等校验）
                                    for n in sorted(valid_nodes):
                                        if self._can_drone_service_node_with_resources(drone_idx, n):
                                            candidate = n
                                            break

                                if candidate is not None:
                                    final_action['drone_service_node'] = candidate
                                    print(f"    [DeadlockGuard] Override: drone {drone_idx} forced to service node {candidate} "
                                        f"(trucks blocked & customers remain)")
                                else:
                                    # 没有真正可服务的点，仍然继续
                                    final_action['drone_continue'] = 1
                                    print(f"    [DeadlockGuard] No feasible node for drone {drone_idx}, continue on truck")
                            else:
                                # 不满足触发条件，按原逻辑继续
                                final_action['drone_continue'] = 1
                                print(f"    [Decoder] Drone {drone_idx} continues on truck")
                        except Exception as e:
                            # 防御性：任何异常都回退为原逻辑的 continue，不影响训练
                            final_action['drone_continue'] = 1
                            print(f"    [DeadlockGuard] Fallback continue due to error: {e}")
                            
                    elif service_node is not None:
                        valid_nodes = self._get_valid_nodes_for_drone(drone_idx)
                        if service_node in valid_nodes and service_node != 0:
                            if self._can_drone_service_node_with_resources(drone_idx, service_node):
                                final_action['drone_service_node'] = service_node
                                print(f"    [Decoder] Drone {drone_idx} will service node {service_node}")
                            else:
                                final_action['drone_continue'] = 1
                                print(f"    [Decoder] Drone {drone_idx} cannot service node {service_node}, continue on truck")
                        else:
                            # 修复：无效服务节点时，检查是否有其他可服务节点
                            if len(valid_nodes) > 1:  # 除了depot还有其他节点
                                print(f"    [Decoder] Invalid service node {service_node}, but other nodes available, continue on truck")
                            else:
                                print(f"    [Decoder] No valid service nodes available, drone continues on truck")
                            final_action['drone_continue'] = 1
                    
                    else:
                        # 修复：没有明确动作时，检查是否有可服务节点
                        valid_nodes = self._get_valid_nodes_for_drone(drone_idx)
                        if len(valid_nodes) > 1:  # 有除depot之外的可服务节点
                            # 如果有可服务节点但无人机没有选择服务，默认继续待在卡车上
                            final_action['drone_continue'] = 1
                            print(f"    [Decoder] Drone {drone_idx} has valid nodes {valid_nodes} but no service request, continue on truck")
                        else:
                            # 没有可服务节点，继续待在卡车上
                            final_action['drone_continue'] = 1
                            print(f"    [Decoder] No valid service nodes for drone {drone_idx}, continue on truck")

            
            return final_action


            
        except Exception as e:
            print(f"    [Decoder] Error processing drone {drone_idx}: {e}")
            return {}
    
    def _can_drone_service_node_with_resources(self, drone_idx, target_node):
        """检查无人机是否有足够的载重和电量服务目标节点 - 排除未激活节点"""
        try:
            # 关键修复：首先检查节点是否激活
            delta = self.temp_db.get_val('delta')
            
            # 检查节点是否存在且激活
            if target_node >= len(delta):
                print(f"    [Decoder] Node {target_node} out of range")
                return False
            
            # delta=-1表示未激活的动态节点，不能服务
            if delta[target_node] == -1:
                print(f"    [Decoder] Node {target_node} is inactive dynamic node (delta=-1), cannot service")
                return False
            
            # delta=0表示已访问，不能再次服务
            if delta[target_node] == 0:
                print(f"    [Decoder] Node {target_node} already visited (delta=0)")
                return False
            
            # 检查是否在已访问节点中
            if target_node in self.temp_db.visited_nodes:
                print(f"    [Decoder] Node {target_node} already in visited_nodes")
                return False
            
            # 检查是否已被预分配
            if hasattr(self.simulator, 'pre_assigned_nodes') and target_node in self.simulator.pre_assigned_nodes:
                print(f"    [Decoder] Node {target_node} already pre-assigned")
                return False
            
            # # 获取无人机当前状态
            # current_pos = self._get_drone_position(drone_idx)
            current_load = self.temp_db.status_dict.get('DW', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
            # current_battery = self.temp_db.status_dict.get('drone_battery', [1000.0] * len(self.temp_db.get_val('ED')))[drone_idx]
            
            # 获取无人机当前状态
            current_pos = self._get_drone_position(drone_idx)
            attached_truck = self.temp_db.status_dict.get('attached_truck', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
            current_battery = self.temp_db.status_dict.get('drone_battery', [1000.0] * len(self.temp_db.get_val('ED')))[drone_idx]
            

            # # 获取目标节点需求
            # target_demand = self.temp_db.get_val('demand')[target_node]
            
            # # 检查载重是否足够
            # if current_load < target_demand:
            #     print(f"    [Decoder] Drone {drone_idx} insufficient load: has {current_load:.1f}, needs {target_demand:.1f}")
            #     return False
            # **关键修复：根据无人机模式确定可用载重**

            if attached_truck == -1:
                # 独立模式：使用当前载重
                available_load = self.temp_db.status_dict.get('DW', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
                print(f"    [Decoder] Independent drone {drone_idx} current load: {available_load:.1f}")
            else:
                # 搭载模式：计算可从卡车获取的载重
                ED = self.temp_db.status_dict['ED'][drone_idx]
                
                if ED == 3:  # 无人机在卡车上，可以从卡车获取载重
                    if attached_truck < len(self.temp_db.get_val('TW')):
                        truck_current_load = self.temp_db.status_dict['TW'][attached_truck]
                        drone_capacity = getattr(self.drones[drone_idx], 'capacity', self.temp_db.WD_max) if hasattr(self, 'drones') else self.temp_db.WD_max
                        
                        # 可获取的载重 = min(卡车当前载重, 无人机容量)
                        available_load = min(truck_current_load, drone_capacity)
                        print(f"    [Decoder] Drone {drone_idx} can get load from truck {attached_truck}: truck_load={truck_current_load:.1f}, drone_capacity={drone_capacity:.1f}, available={available_load:.1f}")
                    else:
                        available_load = 0
                        print(f"    [Decoder] Invalid attached_truck {attached_truck} for drone {drone_idx}")
                else:
                    # 无人机不在卡车上，使用当前载重
                    available_load = self.temp_db.status_dict.get('DW', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
                    print(f"    [Decoder] Drone {drone_idx} not on truck (ED={ED}), current load: {available_load:.1f}")
            
            # 获取目标节点需求
            target_demand = self.temp_db.get_val('demand')[target_node]
            
            # **关键修复：检查可用载重是否足够（而不是当前载重）**
            if available_load < target_demand:
                print(f"    [Decoder] Drone {drone_idx} insufficient available load: has {available_load:.1f}, needs {target_demand:.1f}")
                return False
            
            # 计算到目标节点的距离
            distance_to_target = self.temp_db.distance(current_pos, target_node)
            
            # 估算返回位置的距离
            attached_truck = self.temp_db.status_dict.get('attached_truck', [0] * len(self.temp_db.get_val('ED')))[drone_idx]
            if attached_truck >= 0 and attached_truck < len(self.temp_db.get_val('ET')):
                truck_pos = self.temp_db.status_dict['v_coord'][attached_truck]
                distance_back = self.temp_db.distance(target_node, truck_pos)
            else:
                distance_back = self.temp_db.distance(target_node, 0)
            
            # 计算总电量需求
            total_battery_needed = distance_to_target + distance_back
            
            # 检查电量是否足够
            if current_battery < total_battery_needed:
                print(f"    [Decoder] Drone {drone_idx} insufficient battery: has {current_battery:.1f}, needs {total_battery_needed:.1f}")
                return False
            
            print(f"    [Decoder] Drone {drone_idx} can service active node {target_node}: "
                f"delta={delta[target_node]}, load {current_load:.1f}>={target_demand:.1f}, "
                f"battery {current_battery:.1f}>={total_battery_needed:.1f}")
            return True
            
        except Exception as e:
            print(f"    [Decoder] Error checking drone {drone_idx} service capability: {e}")
            return False

    def _base_decode(self, raw_actions):
        """基础解码方法"""
        decoded = {}
        num_trucks = self.temp_db.num_trucks
        num_drones = self.temp_db.num_drones
        
        # 解码卡车动作
        for truck_idx in range(num_trucks):
            if truck_idx < len(raw_actions):
                decoded[truck_idx] = self._decode_truck_action(raw_actions[truck_idx], truck_idx)
        
        # 解码无人机动作
        for drone_idx in range(num_drones):
            agent_idx = num_trucks + drone_idx
            if agent_idx < len(raw_actions):
                decoded[agent_idx] = self._decode_drone_action(raw_actions[agent_idx], drone_idx)
        
        return decoded

    def _can_drone_service_and_return(self, drone_idx, target_node):
        """检查无人机是否能够服务目标节点并返回卡车"""
        try:
            # 修复：使用正确的方法获取无人机当前位置
            current_pos = self._get_drone_position(drone_idx)
            
            # 获取无人机当前电量
            battery = self.temp_db.status_dict.get('drone_battery', np.full(self.temp_db.num_drones, 1000.0))[drone_idx]
            
            # 计算往返距离
            if hasattr(self.temp_db, 'distance'):
                distance_to_target = self.temp_db.distance(current_pos, target_node)
                distance_back = self.temp_db.distance(target_node, current_pos)
                total_distance = distance_to_target + distance_back              
            # else:
            #     # 简化计算：假设所有节点都在合理距离内
            #     total_distance = 2.0  # 默认往返距离
            
            # 获取无人机速度 - 修复：从simulator获取无人机对象
            drone_speed = self.temp_db.drone_speed  # 使用temp_db中的默认速度
            if hasattr(self.simulator, 'drones') and drone_idx < len(self.simulator.drones):
                # 如果能访问到simulator中的无人机对象，使用其速度
                drone = self.simulator.drones[drone_idx]
                if hasattr(drone, 'speed'):
                    drone_speed = drone.speed
            
            # total_time = total_distance / drone_speed
            
            # can_service = total_time <= battery
            # print(f"    [Decoder] Drone {drone_idx} service check: pos={current_pos}, distance={total_distance:.2f}, time={total_time:.2f}, battery={battery:.2f}, can_service={can_service}")
            
            # return can_service

            # 【修复】：电量消耗应该基于距离，不是时间
            # 假设电量消耗与距离成正比
            battery_consumption = total_distance  # 简化：1单位距离消耗1单位电量
            
            can_service = battery >= battery_consumption
            print(f"    [Decoder] Drone {drone_idx} service check: pos={current_pos}, "
                f"distance={total_distance:.2f}, battery_needed={battery_consumption:.2f}, "
                f"battery_available={battery:.2f}, can_service={can_service}")
            
            return can_service
            
        except Exception as e:
            print(f"    [Decoder] Error checking drone service capability: {e}")
            # 保守估计：允许服务
            return True

    def _get_drone_position(self, drone_idx):
        """获取无人机当前位置 - 新增方法，复用simulation.py的逻辑"""
        try:
            ED = self.temp_db.status_dict['ED'][drone_idx]
            attached_truck = self.temp_db.status_dict.get('attached_truck', [-1] * len(self.temp_db.get_val('ED')))[drone_idx]
            
            if attached_truck == -1:
                # 独立模式：使用无人机自己的坐标
                if 'drone_coord' in self.temp_db.status_dict:
                    return self.temp_db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0  # 默认在depot
            elif ED == 3 and attached_truck >= 0:
                # 搭载模式且在卡车上：返回附着卡车的位置
                if attached_truck < len(self.temp_db.get_val('ET')):
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

    def step(self, raw_actions):
        """执行动作"""
        actions = self.decode(raw_actions)
        if hasattr(self.simulator, 'apply_actions'):
            self.simulator.apply_actions(actions)
        return actions