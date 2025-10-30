import numpy as np

from src.environment.core.restrictions import RestrValueObject, is_None, is_not_None, none_add, none_subtract
from src.environment.core.common_sim_func import param_interpret, random_coordinates
class Truck:
    """
    只处理位置、速度、载重，不跟电池打交道。
    """
    def __init__(self, temp_db, k_index, capacity, speed):
        self.db      = temp_db
        self.k       = k_index
        self.capacity= capacity   # TWtk 初始载重
        self.speed   = speed      # v_t
        # 初始化状态字段
        self.db.status_dict['ET'][k_index] = 3    # say: available
        self.db.status_dict['TW'][k_index] = capacity

    def travel_time(self, from_node, to_node):
        dist = self.db.distance_node_to_node(from_node, to_node)
        return dist / self.speed

    def serve(self, node_i):
        """
        计算服务节点i需要的载重量，但不实际减少载重。
        载重减少应该由simulation.py在适当时机处理。
        """
        demand = self.db.get_val('demand')[node_i]
        current_load = self.db.status_dict['TW'][self.k]
        
        # # 计算能够提供的服务量（不超过当前载重和节点需求）
        # served = min(demand, current_load)
        
        # # **关键修复：不在这里减少载重，只返回服务量**
        # # **不调用 mark_visited，这应该由 simulation.py 处理**
        
        # print(f"     Truck {self.k} calculated service for node {node_i}: demand={demand:.1f}, current_load={current_load:.1f}, can_serve={served:.1f}")

        # 【关键修复】：必须有足够载重才能服务
        if current_load < demand:
            print(f"     ⚠️ Truck {self.k} CANNOT serve node {node_i}: "
                  f"insufficient load ({current_load:.1f} < {demand:.1f})")
            return 0  # 不允许部分服务
        
        # 只有载重充足时才服务
        served = demand  # 完全满足需求
        
        print(f"     ✅ Truck {self.k} serving node {node_i}: "
              f"demand={demand:.1f}, current_load={current_load:.1f}, serving={served:.1f}")
        
        
        return served

    # def serve(self, node_i):
    #     """在节点 i 服务，直接将 TW 减去需求 demand[i]。"""
    #     w = self.db.get_val('demand')[node_i]
    #     # 简单判断是否有足够载重
    #     served = min(w, self.db.status_dict['TW'][self.k])
    #     self.db.status_dict['TW'][self.k] -= served
    #     self.db.mark_visited(node_i)
    #     return served

class Drone:
    """
    处理位置、电池、电量返航判断、载重。
    """
    def __init__(self, temp_db, k_index, max_battery, speed, max_payload):
        self.db        = temp_db
        self.k         = k_index
        self.battery   = max_battery
        self.speed     = speed   # v_d
        self.payload   = max_payload  # WD_max

        # 初始化
        self.db.status_dict['ED'][k_index] = 3
        self.db.constants_dict['WD_max'][k_index] = max_payload
        self.db.constants_dict['max_charge'][k_index] = max_battery

    def max_flight_time(self):
        return self.battery / 1.0  # 假设 1 单位电量=1 时间

    def travel_time(self, from_node, to_node):
        dist = self.db.distance_node_to_node(from_node, to_node)
        return dist / self.speed

    def can_service_and_return(self, from_node, service_node, return_node):
        """
        检查无人机能否飞 from_node→service_node→return_node，按电量判断。
        """
        t1 = self.travel_time(from_node, service_node)
        t2 = self.travel_time(service_node, return_node)
        return (t1 + t2) <= self.max_flight_time()

    # def serve(self, node_i):
    #     w = self.db.get_val('demand')[node_i]
    #     served = min(w, self.payload)
    #     self.db.mark_visited(node_i)
    #     return served
    def serve(self, node_i):
        """
        计算服务节点i需要的载重量，但不实际减少载重。
        载重减少应该由simulation.py在适当时机处理。
        """
        demand = self.db.get_val('demand')[node_i]
        current_load = self.db.status_dict['DW'][self.k]
        
        # 【关键修复】：必须有足够载重才能服务
        if current_load < demand:
            print(f"     ⚠️ Drone {self.k} CANNOT serve node {node_i}: "
                  f"insufficient load ({current_load:.1f} < {demand:.1f})")
            return 0  # 不允许部分服务
        
        # 只有载重充足时才服务
        served = demand  # 完全满足需求
        
        print(f"     ✅ Drone {self.k} serving node {node_i}: "
              f"demand={demand:.1f}, current_load={current_load:.1f}, serving={served:.1f}")
        
        return served

# 新的工厂函数，生成独立的卡车和无人机列表
def create_independent_vehicles(K_trucks, K_drones, temp_db,
                               truck_capacity, truck_speed,
                               drone_battery, drone_speed, drone_payload):
    """
    创建独立的卡车和无人机列表，支持不同数量
    
    Args:
        K_trucks: 卡车数量
        K_drones: 无人机数量
        temp_db: 临时数据库
        truck_capacity: 卡车载重
        truck_speed: 卡车速度
        drone_battery: 无人机电池容量
        drone_speed: 无人机速度
        drone_payload: 无人机载重
        
    Returns:
        trucks: 卡车列表
        drones: 无人机列表
    """
    trucks = [Truck(temp_db, k, truck_capacity, truck_speed) for k in range(K_trucks)]
    drones = [Drone(temp_db, k, drone_battery, drone_speed, drone_payload) for k in range(K_drones)]
    
    return trucks, drones


