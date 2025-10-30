#要改参数给配置
import numpy as np
import gym

from trucks_and_drones.config import cfg
from src.environment.core.database import BaseTempDatabase
from src.environment.core.vehicles import create_independent_vehicles
from src.environment.core.nodes import BaseNodeCreator
from src.environment.core.auto_agent import BaseAutoAgent
from src.environment.core.simulation import BaseSimulator

from src.environment.rendering.visualizer import BaseVisualizer
from src.environment.spaces.observation import BaseObsEncoder
from src.environment.spaces.action import BaseActDecoder
from src.environment.rewards.calculator import BaseRewardCalculator

from src.environment.vrpd_env import CustomEnv
from src.environment.core.common_func import param_interpret


class BuildEnvironment:

    def __init__(
        self,
        name: str,
        grid: (list, tuple, np.ndarray) = [10, 10],
        reward_signals: (list, tuple, np.ndarray) = [1, 1, -1],
        max_steps_per_episode: int = 1000,
        debug_mode: bool = False,
        dynamic_nodes_enabled: bool = None,
        dod: float = None,
        delta_t: int = None,
        enable_visualization: bool = True,  # 新增参数
    ):

        self.name                = name
        self.grid                = grid
        self.reward_signals      = reward_signals
        self.max_steps_per_episode = max_steps_per_episode
        self.debug_mode          = debug_mode

        # **关键修复：动态节点配置**
        self.dynamic_nodes_enabled = dynamic_nodes_enabled
        self.dod = dod
        self.delta_t = delta_t
        self.enable_visualization = enable_visualization  # 新增

        self.vehicle_params = []
        self.node_params    = []

        self.visual_params = None
        self.obs_params    = None
        self.act_params    = None
        self.reward_params = None


    def trucks(
        self,
        num: (int, list, tuple, np.ndarray) = 1,
        loadable: bool = True,
        weight=0,
        range_type='simple',
        max_range=None,
        max_charge=None,
        init_charge=None,
        charge_rate=None,
        travel_type='street',
        speed: float = None,
        cargo_type='standard+extra',
        # max_cargo=100,
        cargo_rate=None,
        init_cargo='max',
        max_v_cap=1,
        v_rate=None,
        symbol='circle',
        color='purple',
    ):
        """Add truck params."""
        env_cfg = cfg.get('environment', {})  # 先定义
        if speed is None:
            env_cfg = cfg.get('environment', {})
            speed = env_cfg.get('truck_speed')  # 实际赋值
            # 打印调试信息
            print(f"Debug - Environment config: {env_cfg}")
        self.vehicle_params.append({
            'v_name'     : 'truck',
            'num'        : num,
            'loadable'   : loadable,
            'weight'     : weight,
            'range_type' : range_type,
            'max_range'  : max_range,
            'max_charge' : max_charge,
            'init_charge': init_charge,
            'charge_rate': charge_rate,
            'travel_type': travel_type,
            'speed'      : speed,
            'cargo_type' : cargo_type,
            # 'max_cargo'  : max_cargo,
            'init_cargo' : init_cargo,
            'cargo_rate' : cargo_rate,
            'max_v_cap'  : max_v_cap,
            'v_rate'     : v_rate,
            'symbol'     : symbol,
            'color'      : color,
        })
        return self


    def drones(
        self,
        num: (int, list, tuple, np.ndarray) = 1,
        loadable: bool = True,
        weight=0,
        range_type='battery',
        max_range=None,
        max_charge=None,
        init_charge='max',
        charge_rate=None,
        travel_type='arial',
        speed: float = None,
        cargo_type='standard',
        max_cargo=None,
        cargo_rate=None,
        init_cargo='max',
        max_v_cap=0,
        v_rate=0,
        symbol='triangle-up',
        color='blue',
    ):
        """Add drone params."""
        env_cfg = cfg.get('environment', {})
        print(f"Debug - Environment config for drones: {env_cfg}")
        if speed is None:
            speed = env_cfg.get('drone_speed')  # 使用60.0作为最终后备值
            print(f"Debug - Resolved drone speed: {speed}")
            
        if max_charge is None:
            max_charge = env_cfg.get('max_charge')
            print(f"Debug - Resolved drone max_charge: {max_charge}")
            
        if max_cargo is None:
            max_cargo = env_cfg.get('WD_max')
            print(f"Debug - Resolved drone max_cargo: {max_cargo}")

        self.vehicle_params.append({
            'v_name'     : 'drone',
            'num'        : num,
            'loadable'   : loadable,
            'weight'     : weight,
            'range_type' : range_type,
            'max_range'  : max_range,
            'max_charge' : max_charge,
            'init_charge': init_charge,
            'charge_rate': charge_rate,
            'travel_type': travel_type,
            'speed'      : speed,
            'cargo_type' : cargo_type,
            'max_cargo'  : max_cargo,
            'init_cargo' : init_cargo,
            'cargo_rate' : cargo_rate,
            'max_v_cap'  : max_v_cap,
            'v_rate'     : v_rate,
            'symbol'     : symbol,
            'color'      : color,
        })
        return self


    def depots(
        self,
        num=1,
        max_items=None,
        init_items=None,
        item_rate=0,
        item_recharge=0,
        init_items_at_step=0,
        symbol='rectangle',
        color='orange',
    ):
        """
                添加 depot 节点，并从 cfg 中加载 deadline, alpha, beta。
                """
        depot_conf = cfg['node']['depot']
        self.node_params.append({
            'num'                : num,
            'n_name'             : 'depot',
            'max_items'          : max_items,
            'init_items'         : init_items,
            'item_rate'          : item_rate,
            'item_recharge'      : item_recharge,
            'init_items_at_step' : init_items_at_step,
            'symbol'             : symbol,
            'color'              : color,
            # 'deadline': depot_conf['deadline'],
            'alpha': depot_conf['alpha'],
            'beta': depot_conf['beta'],
        })
        return self

    def customers(
            self,
            num=1,
            max_items=1,
            init_items='max',
            item_rate=0,
            item_recharge=0,
            init_items_at_step=0,
            symbol='rectangle',
            color='light-grey',
    ):
        """
        添加 customer 节点，并从 cfg 中加载 deadline, alpha, beta。
        """

        cust_conf = cfg['node']['customer']
        env_cfg = cfg.get('environment', {})
        dynamic_cfg = env_cfg.get('dynamic_nodes', {})
        print(f"🔍 Reading dynamic config from current cfg: {dynamic_cfg}")

        # 使用传入的参数或配置中的值
        dynamic_enabled = dynamic_cfg.get('enable', False) if self.dynamic_nodes_enabled is None else self.dynamic_nodes_enabled
        dod = dynamic_cfg.get('dod') if self.dod is None else self.dod
        print(f"🎯 Final dynamic configuration:")
        print(f"   dynamic_enabled: {dynamic_enabled}")
        print(f"   dod: {dod}")
        print(f"   delta_t: {dynamic_cfg.get('delta_t', 50.0)}")

        if dynamic_enabled:
            static_customers = int(num * (1 - dod))
            dynamic_customers = num - static_customers
            
            print(f"Dynamic nodes configuration:")
            print(f"  Total customers requested: {num}")
            print(f"  DoD: {dod}")
            print(f"  Static customers: {static_customers}")
            print(f"  Dynamic customers: {dynamic_customers}")
            
            # 更新配置，确保temp_db能访问正确信息
            cfg['environment']['num_customers'] = num  # 总客户数
            cfg['environment']['num_static_customers'] = static_customers
            cfg['environment']['num_dynamic_customers'] = dynamic_customers
            cfg['environment']['dynamic_nodes'] = {
                'enable': True,
                'dod': dod,
                'delta_t': self.delta_t if self.delta_t is not None else dynamic_cfg.get('delta_t', 50.0)
            }
            
            # 只为静态节点创建node_params条目
            actual_num = static_customers
        else:
            actual_num = num
            print(f"Dynamic nodes disabled: creating {actual_num} static customers")       

        # **关键修复：只为静态节点创建node_params条目**
        self.node_params.append({
            'num': actual_num,  # 只包含静态客户节点数量
            'n_name': 'customer',
            'max_items': max_items,
            'init_items': init_items,
            'item_rate': item_rate,
            'item_recharge': item_recharge,
            'init_items_at_step': init_items_at_step,
            'symbol': symbol,
            'color': color,
            'alpha': cust_conf['alpha'],
            'beta': cust_conf['beta'],
        })
        return self


    def visuals(
        self,
        grid_surface_dim=(750, 750),
        grid_padding=10,
        info_surface_height=200,
        marker_size=12
    ):
        self.visual_params = {
            'grid_surface_dim'   : grid_surface_dim,
            'grid_padding'       : grid_padding,
            'info_surface_height': info_surface_height,
            'marker_size'        : marker_size,
        }
        return self


    def observations(
        self,
        image_input=None,
        contin_inputs=None,
        discrete_inputs=None,
        discrete_bins=4,
        combine_per_index=None,
        combine_per_type=None,
        flatten=True,
        flatten_images=False,
        output_as_array=False,
    ):
        # defaults…
        if image_input       is None: image_input       = []
        if contin_inputs     is None: contin_inputs     = ['LT','LD','TW','time','demand','deadline']
        if discrete_inputs   is None: discrete_inputs   = ['ET','ED','NT','ND','unassigned']
        if combine_per_index is None: combine_per_index = ['per_vehicle']
        if combine_per_type  is None: combine_per_type  = []

        self.obs_params = {
            'image_input'      : image_input,
            'contin_inputs'    : ['LT_time','LD_time','TW','time','n_items','deadline'],
            'discrete_inputs'  : ['ET','ED','NT','ND','truck_node','drone_node','unassigned'],
            'combine_per_index': ['per_vehicle'],
            'combine_per_type': [ 'customers', 'depots'],  # 构造 global_obs
            'flatten'          : True,  # 保持每个 agent 的观测已经扁平化好
            'output_as_array'  : False, # 返回 ([obs0,obs1], global_obs)
        }
        return self


    def actions(
        self,
        mode='single_vehicle',
        flattened='per_output',
        contin_outputs=None,
        discrete_outputs=None,
        binary_discrete=None,
        binary_contin=None,
        num_discrete_bins=None,
        combine='all',
        multiple_action_spaces=False,
    ):
        if contin_outputs     is None: contin_outputs     = []
        if discrete_outputs   is None: discrete_outputs   = ['truck_target_node','drone_rendezvous_node','drone_service_node']
        if binary_discrete    is None: binary_discrete    = ['truck_wait','drone_continue']
        if binary_contin      is None: binary_contin      = []
        if num_discrete_bins  is None: num_discrete_bins  = 300

        self.act_params = {
            'mode'                 : mode,
            'flattened'            : flattened,
            'contin_outputs'       : contin_outputs,
            'discrete_outputs'     : discrete_outputs,
            'binary_discrete'      : binary_discrete,
            'binary_contin'        : binary_contin,
            'num_discrete_bins'    : num_discrete_bins,
            'combine'              : combine,
            'multiple_action_spaces': multiple_action_spaces,
        }
        return self


    def rewards(
        self,
        reward_modes=None,
        reward_type='sum_vehicle',
        restriction_rewards=['travel_cost','delay_penalty'],
        action_rewards=['service_reward']
    ):
        self.reward_params = {
            'reward_modes'       : reward_modes,
            'reward_type'        : reward_type,
            'restriction_rewards': restriction_rewards,
            'action_rewards'     : action_rewards,
        }
        return self


    def compile(
        self,
        temp_database_cls: BaseTempDatabase = BaseTempDatabase,
        node_creator_cls: BaseNodeCreator = BaseNodeCreator,
        auto_agent_cls: BaseAutoAgent = BaseAutoAgent,
        simulator_cls: BaseSimulator = BaseSimulator,
        visualizer_cls: BaseVisualizer = BaseVisualizer,
        obs_encoder_cls: BaseObsEncoder = BaseObsEncoder,
        act_decoder_cls: BaseActDecoder = BaseActDecoder,
        reward_calculator_cls: BaseRewardCalculator = BaseRewardCalculator,
    ):
        """编译环境，依次构造各个组件"""

        # 1) 保证所有默认配置已填
        if not any(p['v_name']=='truck' for p in self.vehicle_params):
            self.trucks()
        if not any(p['v_name']=='drone' for p in self.vehicle_params):
            self.drones()
        if not self.node_params:
            self.depots(); self.customers()
        if self.visual_params is None:
            self.visuals()
        if self.obs_params is None:
            self.observations()
        if self.act_params is None:
            self.actions()
        if self.reward_params is None:
            self.rewards()

        # 先填充所有 node_params 里丢失的 init_items
        for p in self.node_params:
            if p.get('init_items') is None:
                p['init_items'] = p.get('max_items', 0) or 0

        # **关键修复：在创建temp_db前确保动态节点配置已设置**
        if self.dynamic_nodes_enabled is not None:
            if 'dynamic_nodes' not in cfg['environment']:
                cfg['environment']['dynamic_nodes'] = {}
            cfg['environment']['dynamic_nodes']['enable'] = self.dynamic_nodes_enabled
            if self.dod is not None:
                cfg['environment']['dynamic_nodes']['dod'] = self.dod
            if self.delta_t is not None:
                cfg['environment']['dynamic_nodes']['delta_t'] = self.delta_t

        # 2) 创建 DB 对象
        self.temp_db = temp_database_cls(self.name, self.grid, self.reward_signals, self.debug_mode)

        # 3) 获取卡车和无人机配置并计算数量
        truck_cfg = next(p for p in self.vehicle_params if p['v_name'] == 'truck')
        drone_cfg = next(p for p in self.vehicle_params if p['v_name'] == 'drone')
        
        K_trucks = param_interpret(truck_cfg['num'])
        K_drones = param_interpret(drone_cfg['num'])
        
        print(f"Creating {K_trucks} trucks and {K_drones} drones")
        
        # 设置 temp_db 的车辆数量信息
        self.temp_db.num_trucks = K_trucks
        self.temp_db.num_drones = K_drones
        self.temp_db.num_vehicles = K_trucks + K_drones

        # **关键修复：确保在访问status_dict之前先调用init_db()**
        # 4) 初始化所有 status_dict / constants_dict / signals_dict
        self.temp_db.init_db()
        

        # **现在可以安全地访问status_dict相关方法**
        print(f"📊 Database initialized with:")
        print(f"   Total nodes: {self.temp_db.num_nodes} (includes all static + dynamic space)")
        print(f"   Static customers: {self.temp_db.num_static_customers}")
        print(f"   Dynamic customers: {self.temp_db.num_dynamic_nodes}")
        print(f"   Initial active nodes: {self.temp_db.get_current_node_count()}")

        # 5) 用 NodeCreator.__init__ 设置 temp_db 的节点数据
        # **关键说明：NodeCreator只会创建静态节点，动态节点通过temp_db内部机制管理**
        self.node_creator = node_creator_cls(self.node_params, self.temp_db)


        # 6) 创建静态节点数据
        self.node_creator.create()

        print(f"📊 After node creation:")
        print(f"   Static nodes created: {len(self.temp_db.c_indices)} customers + {len(self.temp_db.d_indices)} depots")
        print(f"   Dynamic nodes prepared: {len(self.temp_db.dynamic_nodes_pool)} in pool")
        
        # 安全地获取delta状态
        try:
            delta_states = self.temp_db.get_val('delta')
            delta_dict = dict(zip(range(len(delta_states)), delta_states))
            print(f"   Node delta states: {delta_dict}")
            print(f"   Final verification: total_nodes={self.temp_db.num_nodes}, delta_length={len(delta_states)}")
        except Exception as e:
            print(f"   Warning: Could not get delta states: {e}")

        # 7) 创建独立的卡车和无人机列表
        # self.trucks, self.drones = create_independent_vehicles(
        #     K_trucks, K_drones,
        #     self.temp_db,
        #     truck_capacity=self.temp_db.WT_max,
        #     truck_speed=truck_cfg['speed'],
        #     drone_battery=drone_cfg['max_charge'],
        #     drone_speed=drone_cfg['speed'],
        #     drone_payload=drone_cfg['max_cargo'],
        # )
        truck_cfg = next(p for p in self.vehicle_params if p['v_name'] == 'truck')
        drone_cfg = next(p for p in self.vehicle_params if p['v_name'] == 'drone')

        # **关键修复：确保使用正确的速度值**
        # 优先使用配置文件中的值，如果车辆配置中没有或为None，则使用配置文件的值
        env_cfg = cfg.get('environment', {})
        final_truck_speed = truck_cfg.get('speed')
        # if final_truck_speed is None or final_truck_speed <= 0:
        #     final_truck_speed = env_cfg.get('truck_speed', 30.0)

        final_drone_speed = drone_cfg.get('speed')  
        # if final_drone_speed is None or final_drone_speed <= 0:
        #     final_drone_speed = env_cfg.get('drone_speed', 60.0)

        # 同样处理其他可能需要从配置文件获取的参数
        final_drone_battery = drone_cfg.get('max_charge')
        # if final_drone_battery is None or final_drone_battery <= 0:
        #     final_drone_battery = env_cfg.get('max_charge', 100)

        final_drone_payload = drone_cfg.get('max_cargo')
        # if final_drone_payload is None or final_drone_payload <= 0:
        #     final_drone_payload = env_cfg.get('WD_max', 80.0)

        print(f"Creating vehicles with resolved parameters:")
        print(f"  Truck speed: {final_truck_speed} km/h (from {'config' if truck_cfg.get('speed') is None else 'vehicle_params'})")
        print(f"  Drone speed: {final_drone_speed} km/h (from {'config' if drone_cfg.get('speed') is None else 'vehicle_params'})")
        print(f"  Drone battery: {final_drone_battery}")
        print(f"  Drone payload: {final_drone_payload}")

        self.trucks, self.drones = create_independent_vehicles(
            K_trucks, K_drones,
            self.temp_db,
            truck_capacity=self.temp_db.WT_max,
            truck_speed=final_truck_speed,          # 使用解析后的正确速度
            drone_battery=final_drone_battery,      # 使用解析后的正确电池容量
            drone_speed=final_drone_speed,          # 使用解析后的正确速度
            drone_payload=final_drone_payload,      # 使用解析后的正确载重
        )

        # 验证创建的车辆速度是否正确
        print(f"Verification - Created vehicle speeds:")
        for i, truck in enumerate(self.trucks):
            print(f"  Truck {i}: speed = {getattr(truck, 'speed', 'NO SPEED ATTR')}")
        for i, drone in enumerate(self.drones):
            print(f"  Drone {i}: speed = {getattr(drone, 'speed', 'NO SPEED ATTR')}")

        # 8) 关键修复：先创建一个reward_calc实例，然后复用
        self.reward_calc = reward_calculator_cls(self.reward_params, self.temp_db)
        
        # 9) 创建simulation时使用同一个reward_calc实例
        self.simulation = simulator_cls(self.temp_db, self.trucks, self.drones, self.reward_calc)  # 使用同一个实例
        
        # 8) 剩下的组件照常创建
        # self.simulation = simulator_cls(self.temp_db, self.trucks, self.drones,
        #                                 reward_calculator_cls(self.reward_params, self.temp_db))
        # self.visualizer = visualizer_cls(self.name, self.visual_params, self.temp_db)
        self.visualizer = visualizer_cls(
            self.name, 
            self.visual_params, 
            self.temp_db,
            enabled=self.enable_visualization  # 传递启用标志
        )
        self.obs_encoder = obs_encoder_cls(self.obs_params, self.temp_db, self.visualizer)
        self.act_decoder = act_decoder_cls(self.act_params, self.temp_db, self.simulation)
        # self.reward_calc = reward_calculator_cls(self.reward_params, self.temp_db)

        return self


    def build(self) -> gym.Env:
        """After compile(), produce a Gym Env."""
        return CustomEnv(
            self.name,
            self.simulation,
            self.visualizer,
            self.obs_encoder,
            self.act_decoder,
            self.reward_calc
        )