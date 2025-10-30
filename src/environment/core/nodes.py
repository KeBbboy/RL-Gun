import numpy as np
from src.environment.core.restrictions import RestrValueObject
from trucks_and_drones.simulation.common_sim_func import param_interpret, random_coordinates, max_param_val

class BaseNodeClass:
    """
    代表一个节点（depot 或 customer），但所有状态保存在 temp_db 中。
    该类在初始化时将 deadline、alpha、beta、init_items 写入 temp_db。
    """
    def __init__(self, temp_db, n_index, n_type, n_params):
        self.temp_db = temp_db
        self.n_index = n_index
        self.n_type  = n_type

        # 这里 init_items 已经是数字，不会再是 'max'
        self.init_demand = param_interpret(n_params['init_items'])

        # 写入节点常量
        # self.temp_db.constants_dict['deadline'][n_index] = n_params['deadline']
        self.temp_db.constants_dict['alpha'][n_index]    = n_params['alpha']
        self.temp_db.constants_dict['beta'][n_index]     = n_params['beta']

        # 写入初始需求/库存
        if self.temp_db.status_dict['n_items'][n_index] == 0:
             self.temp_db.status_dict['n_items'][n_index] = self.init_demand
        self.n_name = n_params['n_name']


class BaseNodeCreator:
    """
    根据 node_params_list 在 temp_db 中创建所有节点：
      1. 统计 num_nodes/num_depots/num_customers
      2. create() 依次 init_db→add_node→实例化 BaseNodeClass
    """
    def __init__(self, n_params_list, temp_db, NodeClass=BaseNodeClass):
        self.temp_db       = temp_db
        self.n_params_list = n_params_list
        self.NodeClass     = NodeClass

        # # 预先统计节点数量
        # self.temp_db.num_nodes     = sum(param_interpret(p['num']) for p in n_params_list)
        # self.temp_db.num_depots    = sum(param_interpret(p['num']) for p in n_params_list if p['n_name']=='depot')
        # self.temp_db.num_customers = sum(param_interpret(p['num']) for p in n_params_list if p['n_name']=='customer')

    def create(self):
        # 1) 分配所有 status_dict/constants_dict/signals_dict
        # self.temp_db.init_db()

        n_index = 0
        n_type  = 0
        for params in self.n_params_list:
            cnt = param_interpret(params['num'])
            # 预先算好 max_items 的数值
            max_it = max_param_val(params['max_items']) if params['max_items'] is not None else 0
            # 如果 init_items 不是数字，就用 max_it
            raw_init = params.get('init_items')
            if raw_init is None or raw_init == 'max':
                init_it = max_it
            else:
                init_it = raw_init

            for _ in range(cnt):
                # （a）先调用 add_node 分配坐标、n_type、n_items(0) 等
                self.temp_db.add_node(None, n_index, n_type)

                # （b）再用已经确定的 init_it 调 BaseNodeClass 真正写入常量和初始需求
                self.NodeClass(
                    self.temp_db,
                    n_index,
                    n_type,
                    {
                        'n_name'    : params['n_name'],
                        # 'deadline'  : params['deadline'],
                        'alpha'     : params['alpha'],
                        'beta'      : params['beta'],
                        'init_items': init_it,       # 这里用 init_it
                    }
                )
                n_index += 1
            n_type += 1

        # 2) 可视化配置
        for params in self.n_params_list:
            self.temp_db.node_visuals.append([
                params.get('symbol', 'rectangle'),
                params.get('color', 'grey'),
            ])

        # 3) 更新 n_type 范围
        self.temp_db.min_max_dict['n_type'] = np.array([0, len(self.n_params_list) - 1])
