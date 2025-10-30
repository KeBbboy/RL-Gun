from graphviz import Digraph

# Create a flowchart using graphviz
dot = Digraph(comment='UTM Flowchart', format='png')
dot.attr(rankdir='TB', size='8,10')

# Capacity Management Subgraph
with dot.subgraph(name='cluster_capacity') as c:
    c.attr(label='容量管理', color='blue')
    c.node('A', '多源数据输入\n(历史航迹、气象、地形、地面交通)')
    c.node('B', 'LSTM/Transformer预测\n(航流密度、速度、航向)')
    c.node('C', '多源实时监测\n(GNSS、雷达、5G)')
    c.node('D', '六边形网格动态细化\n(加密高密度区域)')
    c.node('E', '扩展卡尔曼+粒子滤波\n(状态估计)')
    c.node('F', '路径生成\n(RRT*初始)')
    c.node('G', 'MPC二次优化\n(平滑、时长、容量)')
    c.node('H', '容量管理输出\n(最优航路)')
    c.edges([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H')])

# Conflict Detection Subgraph
with dot.subgraph(name='cluster_detection') as d:
    d.attr(label='冲突探测', color='green')
    d.node('I', '轨迹预测\n(EKF+LSTM)')
    d.node('J', '空间索引检索\n(八叉树/KD-树)')
    d.node('K', '蒙特卡洛采样\n(概率碰撞评估)')
    d.node('L_decision', '碰撞概率 > 阈值?')
    d.edges([('H', 'I'), ('I', 'J'), ('J', 'K'), ('K', 'L_decision')])

# Conflict Resolution Subgraph
with dot.subgraph(name='cluster_resolution') as r:
    r.attr(label='冲突解决', color='red')
    r.node('M', '规则避让\n(几何计算)')
    r.node('N', 'NLP优化\n(SQP+增广拉格朗日)')
    r.node('O', 'DRL+GNN策略')
    r.node('P', '控制指令下发')
    r.node('Q', '无人机执行')
    r.edges([('L_decision', 'M'), 
             ('L_decision', 'N'), 
             ('L_decision', 'O'),
             ('M', 'P'), ('N', 'P'), ('O', 'P'),
             ('P', 'Q'), ('Q', 'E')])

# Add labels to decision edges
dot.edge('L_decision', 'M', label='否')
dot.edge('L_decision', 'N', label='复杂')
dot.edge('L_decision', 'O', label='极端')

# Render and display
dot.render(filename='/mnt/data/utm_flowchart', cleanup=True)
dot