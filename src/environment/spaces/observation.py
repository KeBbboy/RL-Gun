import numpy as np
from gym import spaces

def None_to_empty_list(variable):
    if isinstance(variable, (list, tuple, np.ndarray)):
        return variable
    elif variable is None:
        return []
    elif isinstance(variable, str):
        return [variable]
    else:
        return variable

def flatten_list(list_of_lists):
    result = []
    for sub in list_of_lists:
        # 如果是字符串，就当一个整体
        if isinstance(sub, str):
            result.append(sub)
        # 如果是一维 numpy 数组或 list/tuple，也按元素展开
        elif isinstance(sub, (list, tuple, np.ndarray)):
            result.extend(sub)
        else:
            result.append(sub)
    return result


class BaseObsEncoder:

    def __init__(self, obs_params, temp_db, visualizor):
        # Init parameters, avoiding mutable defaults
        self.obs_params = obs_params
        self.temp_db = temp_db
        self.visualizor = visualizor
        self.contin_dict = {}
        self.discrete_dict = {}
        
        # 数据有效性检查
        if not hasattr(temp_db, 'horizon') or temp_db.horizon <= 0:
            raise ValueError(f"Invalid horizon value: {temp_db.horizon}. Must be > 0.")

        # Define truck-specific inputs
        self.truck_contin_inputs = [
            'TW',           # Current load
            'LT_time',      # Remaining time for truck
            'time',         # Global time
        ]
        
        self.truck_discrete_inputs = [
            'ET',           # Truck status (0-3)
            'NT',           # Truck target node
            'truck_node',   # Current truck node when ET ∈ {2,3}
        ]
        
        # Define drone-specific inputs
        self.drone_contin_inputs = [
            'DW',           # Current drone load
            'LD_time',      # Remaining time for drone
            'time',         # Global time
        ]
        
        self.drone_discrete_inputs = [
            'ED',           # Drone status (0-3)
            'ND',           # Drone target node
            'drone_node',   # Current drone node when ED ∈ {2,3}
            'attached_truck',
        ]
        
        # Node-level inputs (shared by all agents)
        self.node_contin_inputs = [
            'demand',       # Node demand
            'deadline',     # Node deadline
        ]
        
        self.node_discrete_inputs = [
            'unassigned',   # Node unassigned flag
            'road_damaged', # 🚧 新增：道路损坏标志
        ]

        # Set other parameters
        setattr(self, 'discrete_bins', obs_params.get('discrete_bins', 4))
        setattr(self, 'flatten', obs_params.get('flatten', True))
        setattr(self, 'flatten_images', obs_params.get('flatten_images', False))
        setattr(self, 'output_as_array', obs_params.get('output_as_array', True))
        setattr(self, 'image_input', obs_params.get('image_input') or [])

        # Expand grouped keys
        self.truck_contin_inputs = self._expand_grouped_keys(self.truck_contin_inputs)
        self.truck_discrete_inputs = self._expand_grouped_keys(self.truck_discrete_inputs)
        self.drone_contin_inputs = self._expand_grouped_keys(self.drone_contin_inputs)
        self.drone_discrete_inputs = self._expand_grouped_keys(self.drone_discrete_inputs)
        self.node_contin_inputs = self._expand_grouped_keys(self.node_contin_inputs)
        self.node_discrete_inputs = self._expand_grouped_keys(self.node_discrete_inputs)

    def _expand_grouped_keys(self, keys):
        """Expand grouped keys using temp_db.key_groups_dict"""
        expanded = []
        for key in keys:
            if key in self.temp_db.key_groups_dict:
                expanded.extend(self.temp_db.key_groups_dict[key])
            else:
                expanded.append(key)
        return expanded

    def reset(self):
        pass

    def coord_to_contin(self, key):
        ''' Normalizes list of Coordinates'''
        coord_list = self.temp_db.get_val(key)
        # 用真实区域边长 area_size（15 km）来归一化
        area = self.temp_db.area_size  # 15.0
        array_x = np.array([elem[0] / area for elem in coord_list])
        array_y = np.array([elem[1] / area for elem in coord_list])
        return np.nan_to_num(np.append(array_x, array_y))

    def value_to_contin(self, key):
        """改进的归一化方法，正确处理动态节点"""
        value_list = np.array(self.temp_db.get_val(key), dtype=float)
        delta = self.temp_db.get_val('delta')
        
        # 处理无效值
        if np.any(np.isnan(value_list)) or np.any(np.isinf(value_list)):
            print(f"⚠️ Warning: Invalid values in {key}: {value_list}")
            value_list = np.nan_to_num(value_list, nan=0.0, posinf=self.temp_db.horizon, neginf=0.0)
        
        if key in ('demand', 'n_items'):
            normalized_values = np.zeros_like(value_list)
            # 对于未激活节点(delta=-1)，观测值设为0
            active_mask = delta >= 0  
            if np.any(active_mask):
                normalized_values[active_mask] = np.clip(value_list[active_mask] / 50.0, 0, 1)
            # 未激活节点保持为0
            inactive_mask = delta == -1
            normalized_values[inactive_mask] = 0
            
        elif key == 'deadline':
            horizon = self.temp_db.horizon if self.temp_db.horizon > 0 else 100.0
            normalized_values = np.zeros_like(value_list)
            active_mask = delta >= 0
            if np.any(active_mask):
                normalized_values[active_mask] = np.clip(value_list[active_mask] / horizon, 0, 1)
            # 未激活节点保持为0
            inactive_mask = delta == -1
            normalized_values[inactive_mask] = 0
            
        else:
            # 其他key的处理保持不变
            min_val, max_val = self.temp_db.min_max_dict.get(key, (None, None))
            if max_val is None or min_val is None:
                print(f"⚠️ Warning: Invalid min/max for {key}: [{min_val}, {max_val}]")
                return np.zeros((len(value_list), 1), dtype=float)
            
            if abs(max_val - min_val) < 1e-8:
                print(f"⚠️ Warning: Min-max range too small for {key}: [{min_val}, {max_val}]")
                return np.zeros((len(value_list), 1), dtype=float)
            
            normalized_values = (value_list - min_val) / (max_val - min_val)
            normalized_values = np.clip(normalized_values, 0, 1)
        
        # 最终清理
        normalized_values = np.nan_to_num(normalized_values, nan=0.0, posinf=1.0, neginf=0.0)
        return normalized_values.reshape(-1, 1)

    def discrete_to_onehot(self, key, values):
        """
        通用 one-hot 编码 - 修复attached_truck的独立模式支持
        """
        if key in ['ET', 'ED']:
            bins = 4
        elif key in ['NT', 'ND']:
           bins = self.temp_db.num_nodes
        elif key in ['unassigned', 'delta']:
            bins = 3
        elif key == 'road_damaged':  # 🚧 新增
            bins = 2  # 0: 正常, 1: 损坏
        elif key == 'attached_truck':
            # 修复：为独立模式增加一个额外的维度
            bins = self.temp_db.num_trucks + 1  # +1 用于表示独立模式
        else:
            raise ValueError(f"Key {key} is not a discrete field")

        arr = np.zeros((len(values), bins), dtype=float)
        idx = np.array(values, dtype=int)
        
        # 处理attached_truck的特殊情况
        if key in ['unassigned', 'delta']:
            for i, val in enumerate(idx):
                if val == -1:  # 未激活动态节点
                    arr[i, 0] = 1.0
                elif val == 0:  # 已访问节点
                    arr[i, 1] = 1.0
                elif val == 1:  # 激活未访问节点
                    arr[i, 2] = 1.0
                # 其他值默认为全零（异常情况）
        elif key == 'road_damaged':  # 🚧 新增
            # 简单的二进制编码
            valid_mask = (idx >= 0) & (idx < bins)
            valid_indices = np.arange(len(values))[valid_mask]
            valid_idx = idx[valid_mask]
            arr[valid_indices, valid_idx] = 1.0
        elif key == 'attached_truck':
            # 处理attached_truck的特殊情况
            for i, val in enumerate(idx):
                if val == -1:
                    # 独立模式映射到最后一个索引
                    arr[i, -1] = 1.0
                elif 0 <= val < self.temp_db.num_trucks:
                    # 正常的卡车索引
                    arr[i, val] = 1.0
                # 其他无效值保持全零
        else:
            # 其他key的正常处理
            valid_mask = (idx >= 0) & (idx < bins)
            valid_indices = np.arange(len(values))[valid_mask]
            valid_idx = idx[valid_mask]
            arr[valid_indices, valid_idx] = 1.0
            
        return arr

    def _get_truck_obs(self, truck_idx):
        """Get observation for a specific truck"""
        truck_obs = []
        
        # Process truck-specific continuous inputs
        for key in self.truck_contin_inputs:
            if key == 'time':
                val = np.array([self.temp_db.current_time], dtype=float).reshape(-1, 1)
            elif key in self.contin_dict:
                val = self.contin_dict[key][truck_idx:truck_idx+1]
            else:
                val = self.value_to_contin(key)[truck_idx:truck_idx+1]
            truck_obs.append(val)
        
        # Process truck-specific discrete inputs
        for key in self.truck_discrete_inputs:
            if key in self.discrete_dict:
                val = self.discrete_dict[key][truck_idx:truck_idx+1]
            else:
                raw_values = self.temp_db.get_val(key)
                val = self.discrete_to_onehot(key, [raw_values[truck_idx]])
            truck_obs.append(val)
        
        return np.concatenate([arr.flatten() for arr in truck_obs])

    def _get_drone_obs(self, drone_idx):
        """Get observation for a specific drone - 修复独立模式支持"""
        drone_obs = []
        
        # Process drone-specific continuous inputs
        for key in self.drone_contin_inputs:
            if key == 'time':
                val = np.array([self.temp_db.current_time], dtype=float).reshape(-1, 1)
            elif key == 'DW':
                # Fixed: Proper drone weight handling
                if 'DW' in self.temp_db.status_dict and drone_idx < len(self.temp_db.status_dict['DW']):
                    val = np.array([self.temp_db.status_dict['DW'][drone_idx]], dtype=float).reshape(-1, 1)
                else:
                    val = np.zeros((1, 1), dtype=float)
            elif key in self.contin_dict:
                if drone_idx < len(self.contin_dict[key]):
                    val = self.contin_dict[key][drone_idx:drone_idx+1]
                else:
                    val = np.zeros((1, 1), dtype=float)
            else:
                raw_values = self.temp_db.get_val(key)
                if drone_idx < len(raw_values):
                    val = self.value_to_contin(key)[drone_idx:drone_idx+1]
                else:
                    val = np.zeros((1, 1), dtype=float)
            drone_obs.append(val)
        
        # Process drone-specific discrete inputs
        for key in self.drone_discrete_inputs:
            if key == 'attached_truck':
                # 修复：正确处理独立模式的attached_truck=-1
                if ('attached_truck' in self.temp_db.status_dict and 
                    drone_idx < len(self.temp_db.status_dict['attached_truck'])):
                    attached = self.temp_db.status_dict['attached_truck'][drone_idx]
                    
                    # 处理独立模式（attached_truck=-1）
                    if attached == -1:
                        # 为独立模式创建特殊的one-hot编码，可以使用额外的维度
                        # 或者将-1映射到特殊索引（比如最后一个索引）
                        special_bins = self.temp_db.num_trucks + 1  # 增加一个维度表示独立模式
                        arr = np.zeros((1, special_bins), dtype=float)
                        arr[0, -1] = 1.0  # 最后一个位置表示独立模式
                        val = arr
                    else:
                        # 正常的卡车附着模式
                        if attached >= self.temp_db.num_trucks:
                            attached = 0  # 防止索引越界
                        val = self.discrete_to_onehot('attached_truck', [attached])
                else:
                    # 默认值处理
                    val = self.discrete_to_onehot('attached_truck', [0])
            elif key in self.discrete_dict:
                if drone_idx < len(self.discrete_dict[key]):
                    val = self.discrete_dict[key][drone_idx:drone_idx+1]
                else:
                    val = np.zeros((1, self.discrete_dict[key].shape[1]), dtype=float)
            else:
                raw_values = self.temp_db.get_val(key)
                if drone_idx < len(raw_values):
                    val = self.discrete_to_onehot(key, [raw_values[drone_idx]])
                else:
                    # Default value based on key type
                    if key in ['ED', 'ND']:
                        val = self.discrete_to_onehot(key, [0])
                    else:
                        val = np.zeros((1, 1), dtype=float)
            drone_obs.append(val)

        return np.concatenate([arr.flatten() for arr in drone_obs])


    def _get_node_obs(self):
        """获取节点级观测，包含动态节点状态信息"""
        node_obs = []
        
        # Process node-level continuous inputs
        for key in self.node_contin_inputs:
            if key == 'demand':
                val = self.value_to_contin('n_items')
            else:
                val = self.value_to_contin(key)
            node_obs.append(val)
        
        # Process node-level discrete inputs with dynamic node support
        for key in self.node_discrete_inputs:
            if key == 'unassigned':
                # **关键修复：处理动态节点的三种状态**
                unassigned_vals = self.temp_db.get_val('delta')
                # delta值含义：
                # -1: 动态节点未激活
                #  0: 已访问/已服务
                #  1: 激活且未访问
                
                # 为了保持网络兼容性，将三种状态映射到更丰富的编码
                # 方案1：扩展编码 (推荐)
                val = self.discrete_to_onehot_with_dynamic('delta', unassigned_vals)
            else:
                raw_values = self.temp_db.get_val(key)
                val = self.discrete_to_onehot(key, raw_values)
            node_obs.append(val)
        
        # Flatten all node observations
        return np.concatenate([arr.flatten() for arr in node_obs])

    def discrete_to_onehot_with_dynamic(self, key, values):
        """
        支持动态节点的三状态one-hot编码 - 重定向到统一方法
        -1: 未激活动态节点 -> [1,0,0]
         0: 已访问节点 -> [0,1,0]
         1: 激活未访问节点 -> [0,0,1]
        """
        # 直接调用修复后的discrete_to_onehot方法
        return self.discrete_to_onehot(key, values)

    def observe_state(self):
        """生成固定维度的观测，包含所有节点（激活和未激活）"""
        print("Processing state observation with fixed node space (including inactive dynamic nodes)")
        
        # 初始化基础状态字典
        self._process_basic_states()
        
        # 关键修复：强制使用总节点数，不管激活状态如何
        total_nodes = self.temp_db.num_nodes
        print(f"🔧 Using TOTAL nodes for observation: {total_nodes} (not just active nodes)")
        
        # 获取节点级观测 - 包含所有节点（静态 + 动态空间）
        node_obs = self._get_node_obs_fixed()    
        
        # 收集所有智能体的观测
        agent_obs = []
        
        # 添加卡车智能体观测
        num_trucks = self.temp_db.num_trucks
        for truck_idx in range(num_trucks):
            truck_obs = self._get_truck_obs(truck_idx)
            combined_obs = np.concatenate([truck_obs, node_obs])
            agent_obs.append(combined_obs.astype(np.float32))
            print(f"Truck {truck_idx} obs length: {len(combined_obs)} (fixed total: {total_nodes} nodes)")
        
        # 添加无人机智能体观测
        num_drones = self.temp_db.num_drones
        for drone_idx in range(num_drones):
            drone_obs = self._get_drone_obs(drone_idx)
            combined_obs = np.concatenate([drone_obs, node_obs])
            agent_obs.append(combined_obs.astype(np.float32))
            print(f"Drone {drone_idx} obs length: {len(combined_obs)} (fixed total: {total_nodes} nodes)")
        
        # 全局观测包含所有卡车 + 无人机 + 节点观测
        all_truck_obs = []
        all_drone_obs = []
        
        for truck_idx in range(num_trucks):
            all_truck_obs.append(self._get_truck_obs(truck_idx))
        for drone_idx in range(num_drones):
            all_drone_obs.append(self._get_drone_obs(drone_idx))
        
        global_obs = np.concatenate([
            np.concatenate(all_truck_obs) if all_truck_obs else np.array([]),
            np.concatenate(all_drone_obs) if all_drone_obs else np.array([]),
            node_obs
        ]).astype(np.float32)
        
        # **关键修复：直接从status_dict获取原始delta数组**
        delta = self.temp_db.status_dict['delta']  # 直接访问原始数组
        
        # 验证数组长度
        if len(delta) != total_nodes:
            print(f"⚠️ Critical: delta array length {len(delta)} != expected {total_nodes}")
            print(f"⚠️ This indicates a serious data corruption issue!")
            # 尝试从其他来源获取
            try:
                backup_delta = self.temp_db.get_val('delta')
                print(f"⚠️ Backup delta from get_val: {backup_delta}")
            except:
                print(f"⚠️ get_val('delta') also failed")
            
        # 统计各种节点状态
        inactive_nodes = np.sum(delta == -1)  # 未激活的动态节点
        visited_nodes = np.sum(delta == 0)    # 已访问的节点
        active_nodes = np.sum(delta == 1)     # 激活且未访问的节点
        
        # **添加详细的调试信息，确认计数正确**
        print(f"🔍 Raw delta array from status_dict: {delta}")
        print(f"🔍 Array details - shape: {delta.shape}, dtype: {delta.dtype}")
        
        # 分别打印各种状态的节点索引
        inactive_indices = np.where(delta == -1)[0]
        visited_indices = np.where(delta == 0)[0] 
        active_indices = np.where(delta == 1)[0]
        
        print(f"🔍 Node indices by state:")
        print(f"    Inactive (delta=-1): {inactive_indices}")
        print(f"    Visited (delta=0): {visited_indices}")
        print(f"    Active (delta=1): {active_indices}")
        
        print(f"🔍 Delta counts - inactive: {inactive_nodes}, visited: {visited_nodes}, active: {active_nodes}")
        print(f"🔍 Total verification: {inactive_nodes + visited_nodes + active_nodes} == {total_nodes}")
        
        # 验证计数逻辑
        if inactive_nodes + visited_nodes + active_nodes != total_nodes:
            print(f"🚨 COUNTING ERROR: Sum {inactive_nodes + visited_nodes + active_nodes} != total {total_nodes}")
            print(f"🚨 Please check delta array integrity!")

        # 添加调试信息确认计数正确
        print(f"🔍 Delta array: {delta}")
        print(f"🔍 Delta counts - inactive: {inactive_nodes}, visited: {visited_nodes}, active: {active_nodes}")
        print(f"🔍 Total should equal {total_nodes}: {inactive_nodes + visited_nodes + active_nodes}")
                
        print(f"Global obs length: {len(global_obs)} (total nodes: {total_nodes}, "
            f"active: {active_nodes}, visited: {visited_nodes}, inactive: {inactive_nodes})")
        print(f"Total agents: {len(agent_obs)} (Trucks: {num_trucks}, Drones: {num_drones})")
        
        return agent_obs, global_obs

    def _get_node_obs_fixed(self):
        """获取固定大小的节点级观测，包含所有节点（激活和未激活）"""
        node_obs = []
        
        # 关键修复：强制使用总节点数
        total_nodes = self.temp_db.num_nodes
        
        print(f"🔧 Creating node observation for {total_nodes} nodes")
        
        # 处理节点级连续输入
        for key in self.node_contin_inputs:
            if key == 'demand':
                # 使用n_items作为demand
                val = self.temp_db.get_val('n_items')
            else:
                val = self.temp_db.get_val(key)
            
            # 获取delta数组 - 必须在截断val之前获取
            delta = self.temp_db.get_val('delta')
            
            # 确保数组长度等于总节点数
            if len(val) != total_nodes:
                print(f"⚠️ Fixing array length: {key} length {len(val)} -> {total_nodes}")
                if len(val) < total_nodes:
                    # 用零填充到总节点数
                    padded_val = np.zeros(total_nodes, dtype=float)
                    padded_val[:len(val)] = val
                    val = padded_val
                else:
                    # 截断到总节点数
                    val = val[:total_nodes]
            
            # 确保delta长度也正确
            if len(delta) != total_nodes:
                print(f"⚠️ Fixing delta length: {len(delta)} -> {total_nodes}")
                padded_delta = np.full(total_nodes, -1, dtype=int)
                copy_length = min(len(delta), total_nodes)
                padded_delta[:copy_length] = delta[:copy_length]
                delta = padded_delta
            
            # 归一化处理 - 现在val和delta长度相同
            if key == 'demand' or key == 'n_items':
                # 对于demand：只有激活节点（delta >= 0）才进行归一化
                normalized_val = np.zeros_like(val)
                active_mask = delta >= 0
                if np.any(active_mask):
                    normalized_val[active_mask] = np.clip(val[active_mask] / 50.0, 0, 1)
                # 未激活节点（delta = -1）保持为0
            elif key == 'deadline':
                horizon = self.temp_db.horizon if self.temp_db.horizon > 0 else 100.0
                normalized_val = np.zeros_like(val)
                active_mask = delta >= 0
                if np.any(active_mask):
                    normalized_val[active_mask] = np.clip(val[active_mask] / horizon, 0, 1)
            else:
                normalized_val = val
            
            node_obs.append(normalized_val.reshape(-1, 1))
        
        # 处理节点级离散输入
        for key in self.node_discrete_inputs:
            if key == 'unassigned':
                # 关键修复：获取完整的delta数组并进行三状态编码
                delta_vals = self.temp_db.get_val('delta')
                
                # 确保长度正确
                if len(delta_vals) != total_nodes:
                    print(f"⚠️ Fixing delta array: length {len(delta_vals)} -> {total_nodes}")
                    padded_delta = np.full(total_nodes, -1, dtype=int)  # 默认为未激活
                    copy_length = min(len(delta_vals), total_nodes)
                    padded_delta[:copy_length] = delta_vals[:copy_length]
                    delta_vals = padded_delta
                
                # 使用修复后的三状态编码
                val = self.discrete_to_onehot('delta', delta_vals)
                
                print(f"🔧 Delta states: inactive={np.sum(delta_vals==-1)}, "
                    f"visited={np.sum(delta_vals==0)}, active={np.sum(delta_vals==1)}")
                print(f"🔧 Unassigned encoding shape: {val.shape} (should be {total_nodes}x3)")
            elif key == 'road_damaged':  # 🚧 新增
                # 获取道路损坏状态
                if 'road_damaged' in self.temp_db.status_dict:
                    road_damaged_vals = self.temp_db.status_dict['road_damaged']
                else:
                    # 如果没有初始化，创建全零数组
                    road_damaged_vals = np.zeros(total_nodes, dtype=int)
                
                # 确保长度正确
                if len(road_damaged_vals) != total_nodes:
                    padded_damaged = np.zeros(total_nodes, dtype=int)
                    copy_length = min(len(road_damaged_vals), total_nodes)
                    padded_damaged[:copy_length] = road_damaged_vals[:copy_length]
                    road_damaged_vals = padded_damaged
                
                val = self.discrete_to_onehot('road_damaged', road_damaged_vals)
                print(f"🚧 Road damaged encoding: {np.sum(road_damaged_vals==1)} damaged nodes")
            else:
                raw_values = self.temp_db.get_val(key)
                # 确保长度匹配
                if len(raw_values) != total_nodes:
                    padded_values = np.zeros(total_nodes, dtype=int)
                    copy_length = min(len(raw_values), total_nodes)
                    padded_values[:copy_length] = raw_values[:copy_length]
                    raw_values = padded_values
                val = self.discrete_to_onehot(key, raw_values)
            
            node_obs.append(val)
        
        # 展平所有节点观测
        flattened_obs = np.concatenate([arr.flatten() for arr in node_obs])
        print(f"🔧 Node observation dimensions: {[arr.shape for arr in node_obs]} -> flattened: {len(flattened_obs)}")
        
        return flattened_obs

    def _process_basic_states(self):
        """Process basic states and populate contin_dict and discrete_dict"""
        # Process basic truck/drone states
        raw_ET = self.temp_db.status_dict['ET']
        raw_ED = self.temp_db.status_dict['ED'] 
        raw_LT = self.temp_db.status_dict['LT']
        raw_LD = self.temp_db.status_dict['LD']
        raw_NT = self.temp_db.status_dict['NT']
        raw_ND = self.temp_db.status_dict['ND']

        # Process one-hot encodings for discrete states
        self.discrete_dict['ET'] = self.discrete_to_onehot('ET', raw_ET)
        self.discrete_dict['ED'] = self.discrete_to_onehot('ED', raw_ED)
        self.discrete_dict['NT'] = self.discrete_to_onehot('NT', raw_NT)
        self.discrete_dict['ND'] = self.discrete_to_onehot('ND', raw_ND)

        # Process time fields
        horizon = self.temp_db.horizon if self.temp_db.horizon > 0 else 100.0
        
        LT_time = np.where(np.isin(raw_ET, [0, 1]), raw_LT, 0.0)
        LD_time = np.where(np.isin(raw_ED, [0, 1]), raw_LD, 0.0)
        
        self.contin_dict['LT_time'] = np.clip(LT_time.reshape(-1, 1) / horizon, 0, 1)
        self.contin_dict['LD_time'] = np.clip(LD_time.reshape(-1, 1) / horizon, 0, 1)

        # Process current position indices
        truck_node = np.where(np.isin(raw_ET, [2, 3]), raw_LT.astype(int), 0)
        drone_node = np.where(np.isin(raw_ED, [2, 3]), raw_LD.astype(int), 0)
        
        self.discrete_dict['truck_node'] = self.discrete_to_onehot('NT', truck_node)
        self.discrete_dict['drone_node'] = self.discrete_to_onehot('ND', drone_node)

        # Process truck weight
        self.contin_dict['TW'] = self.value_to_contin('TW')
        # Process drone weight (DW)
        self.contin_dict['DW'] = self.value_to_contin('DW')

    def obs_space(self):
        """Define observation space for separated truck/drone agents"""
        agent_obs, global_obs = self.observe_state()
        
        # Create individual observation spaces for each agent type
        agent_obs_spaces = []
        
        for i, obs in enumerate(agent_obs):
            agent_obs_spaces.append(
                spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
            )
        
        # Add global observation space
        global_obs_space = spaces.Box(-np.inf, np.inf, shape=global_obs.shape, dtype=np.float32)
        
        # Return tuple of all agent spaces plus global space
        return spaces.Tuple(tuple(agent_obs_spaces) + (global_obs_space,))