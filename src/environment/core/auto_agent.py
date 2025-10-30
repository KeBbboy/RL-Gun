'''

'''

import numpy as np


class BaseAutoAgent:

    def __init__(self, temp_db):

        self.temp_db = temp_db

    def find_destination(self):
        return None

    def find_v_to_unload(self):
        return None

    def find_v_to_load(self):
        return None

    def find_customer(self):
        return None

    def find_depot(self):
        return None


    # def find_destination(self):
    #
    #     # 情形 1：某辆卡车“卡住”了，就找另一辆最近的卡车过来帮它。
    #     if (np.sum(self.temp_db.status_dict['v_stuck']) != 0
    #         and bool(self.temp_db.constants_dict['v_is_truck'][self.temp_db.cur_v_index])
    #         ):
    #
    #         v_index = self.temp_db.nearest_neighbour(self.temp_db.vehicles(
    #             self.temp_db.status_dict['v_coord'],
    #             include=[[self.temp_db.status_dict['v_stuck'], 1]]
    #             )
    #         )
    #
    #         if v_index is None:
    #             return None
    #         return self.temp_db.status_dict['v_coord'][v_index]
    #
    #     # 情形 2/3：如果有货且有客户在等就去客户；否则去最近仓库。
    #     else:
    #         if (self.temp_db.base_groups['vehicles'][self.temp_db.cur_v_index].v_items.cur_value() > 0
    #             and 0 in self.temp_db.customers(self.temp_db.status_dict['n_waiting'])[0]
    #             and np.sum(self.temp_db.customers(self.temp_db.status_dict['n_items'])[0]) != 0
    #             ):
    #
    #             n_index = self.temp_db.nearest_neighbour(self.temp_db.customers(
    #                 self.temp_db.status_dict['n_coord'],
    #                 include=[[self.temp_db.status_dict['n_waiting'], 0]],
    #                 exclude=[[self.temp_db.status_dict['n_items'], 0]]
    #                 )
    #             )
    #
    #         else:
    #             n_index = self.find_depot()
    #             self.temp_db.status_dict['v_to_n'][self.temp_db.cur_v_index] = n_index
    #
    #         self.temp_db.status_dict['v_to_n'][self.temp_db.cur_v_index] = n_index
    #
    #         if n_index is None:
    #             return None
    #         return self.temp_db.status_dict['n_coord'][n_index]
    #
    #
    # # 找到一辆要卸货的车；
    # def find_v_to_unload(self):
    #
    #     if any(self.temp_db.v_transporting_v[self.temp_db.cur_v_index]):
    #         return self.temp_db.v_transporting_v[self.temp_db.cur_v_index][0]
    #     return None
    #
    #
    # # 找到一辆可装载货物的卡车；
    # def find_v_to_load(self):
    #
    #     if self.temp_db.constants_dict['v_is_truck'][self.temp_db.cur_v_index]:
    #         return self.temp_db.nearest_neighbour(self.temp_db.vehicles(
    #                 self.temp_db.status_dict['v_coord'],
    #                 include=[[self.temp_db.constants_dict['v_loadable'], 1], [self.temp_db.status_dict['v_free'], 1]]
    #             )
    #         )
    #     else:
    #         return None
    #
    # # 找到最近的、还有需求的客户；
    # def find_customer(self):
    #     return self.temp_db.nearest_neighbour(self.temp_db.customers(
    #             self.temp_db.status_dict['n_coord'],
    #             exclude=[[self.temp_db.status_dict['n_items'], 0]]
    #         )
    #     )
    #
    # # 找到最近的仓库。
    # def find_depot(self):
    #     return self.temp_db.nearest_neighbour(self.temp_db.depots(
    #             self.temp_db.status_dict['n_coord'],
    #             # exclude=[[self.temp_db.status_dict['n_items'], 0]]
    #         )
    #     )

