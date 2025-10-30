'''

'''
import numpy as np
# Used for obj creation:
# ----------------------------------------------------------------------------------------------------------------

# def param_interpret(var):
#     if isinstance(var, (list, tuple, np.ndarray)):
#         if len(var) == 2:
#             return np.random.randint(var[0],var[1]+1)
#     return var
def param_interpret(var):
    """
    将 num 参数统一解析为一个整数：
      - 如果是单个 int/float，直接返回 int(var)。
      - 如果是 list/tuple/np.ndarray 且 len==1，返回 int(var[0])。
      - 如果是 list/tuple/np.ndarray 且 len==2，返回区间 [var[0],var[1]] 内的随机整数。
      - 如果是 list/tuple/np.ndarray 且 len>2，返回 int(var[0])（可根据需求修改逻辑）。
    """
    # 标量
    if isinstance(var, (int, float, np.integer, np.floating)):
        return int(var)

    # numpy 数组先转列表
    if isinstance(var, np.ndarray):
        var = var.tolist()

    # 序列
    if isinstance(var, (list, tuple)):
        length = len(var)
        if length == 0:
            raise ValueError("param_interpret: empty sequence")
        if length == 1:
            return int(var[0])
        if length == 2:
            low, high = var
            return int(np.random.randint(int(low), int(high) + 1))
        # 更多元素时默认取第一个
        return int(var[0])

    # 其他类型，尝试转 int
    try:
        return int(var)
    except:
        raise ValueError(f"param_interpret: cannot interpret parameter {var!r} as integer")


def max_param_val(var):
    """
        返回 var 的最大值。支持标量、list、tuple、np.ndarray。
    """
    if isinstance(var, np.ndarray):
        return np.max(var)
    if isinstance(var, (list, tuple)):
        return max(var)
    return var

def random_coordinates(grid):
    return (np.random.randint(0,grid[0]+1), np.random.randint(0,grid[1]+1))

def return_indices_of_a(list_a, list_b):
    return [i for i, v in enumerate(list_a) if v in set(list_b)]

def l_ignore_none(l):
	return [i for i in l if i is not None]

def clip_pos(value):
	if value < 0:
		value = 0
	return value

#def compare_coordinates
