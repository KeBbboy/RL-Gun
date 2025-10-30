import sys
import os
import numpy as np
import time

# 将 pygame 导入延迟到需要时再进行（可选依赖）
try:
    # ─── 在导入 pygame 之前设置，保证窗口水平+垂直居中 ─────────────────────────────────
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    os.environ['SDL_VIDEO_WINDOW_POS']    = 'center'
    import pygame
    from pygame import Surface
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None
    Surface = None
    print("⚠️  Warning: pygame not installed. Visualization will be disabled.")



class BaseVisualizer:
    def __init__(self, name, visual_params, temp_db, enabled=True):  # 添加enabled参数
        self.enabled = enabled
        self.name = name
        self.temp_db = temp_db
        self.grid = temp_db.grid
        
        # 如果未启用可视化或pygame不可用，提前返回
        if not self.enabled or not PYGAME_AVAILABLE:
            self.no_open_window = True
            self.paused = False
            if not self.enabled:
                print("📊 Visualization disabled for this session")
            return

        # Initialize pygame
        pygame.init()

        self.no_open_window = True
        self.paused = False
        
        # Define parameter:
        self.name = name
        self.temp_db = temp_db
        self.grid = temp_db.grid
        [setattr(self, k, v) for k, v in visual_params.items()]

        # ===================== 新增：路径追踪相关 =====================
        # 存储每个episode的完整路径
        self.episode_paths = {}  # {episode: {'trucks': [], 'drones': []}}
        self.current_episode = 0
        self.current_step = 0
        
        # 当前episode的路径缓存
        self.truck_paths = []  # 每个truck的路径 [[step0_pos, step1_pos, ...], ...]
        self.drone_paths = []  # 每个drone的路径 [[step0_pos, step1_pos, ...], ...]
        
        # 初始化路径记录
        num_vehicles = len(self.temp_db.status_dict.get('v_coord', []))
        for i in range(num_vehicles):
            self.truck_paths.append([])
            self.drone_paths.append([])
        # ================================================================

        # 大幅缩小网格尺寸 - 缩放到60%
        ow, oh = visual_params['grid_surface_dim']
        visual_params['grid_surface_dim'] = [int(ow * 0.6), int(oh * 0.6)]
        self.grid_surface_dim_original = [int(ow * 0.6), int(oh * 0.6)]

        # Define some colors
        self.color_dict = {
            'no_items_white': (240, 240, 240),
            'white': (255, 255, 255),
            'light-grey': (195, 195, 195),
            'grey': (128, 128, 128),
            'black': (0, 0, 0),
            'half_transp': (255, 255, 255, 125),
            'full_transp': (255, 255, 255, 0),
            'red': (165, 36, 36),
            'green': (67, 149, 64),
            'blue': (81, 73, 186),
            'purple': (151, 69, 176),
            'light-blue': (65, 163, 212),
            'orange': (239, 179, 110),
            'yellow': (239, 203, 24),
            'dark_red': (139, 0, 0),
            'dark_green': (0, 100, 0),
            'pink': (255, 192, 203),
            'inactive_dynamic': (100, 100, 100),      # 灰色 - 未激活的动态节点
            'active_dynamic': (0, 200, 100),          # 青绿色 - 激活的动态节点
            'static_customer': (67, 149, 64),         # 原绿色 - 静态客户节点
            'served_static': (128, 128, 128),         # 深灰色 - 已服务的静态节点
            'served_dynamic': (150, 150, 150),        # 浅灰色 - 已服务的动态节点
            'dynamic_border': (255, 165, 0),          # 橙色边框 - 动态节点标识
            'time_critical': (255, 100, 100),         # 红色 - 接近deadline的节点
        }
        # ========== 新增：动态生成载具颜色 ==========
        num_vehicles = len(self.temp_db.status_dict.get('v_coord', []))
        self.truck_colors = self._generate_vehicle_colors(num_vehicles, 'truck')
        self.drone_colors = self._generate_vehicle_colors(num_vehicles, 'drone')
        print(f"🎨 Generated colors for {num_vehicles} vehicles:")
        print(f"   Truck colors: {self.truck_colors}")
        print(f"   Drone colors: {self.drone_colors}")
        # ============================================

        # 缩小marker_size
        self.marker_size = max(4, int(self.marker_size * 0.7))  # 缩小但不低于4

        # Grid calculations
        self.x_mulipl = int(round(self.grid_surface_dim_original[0] / (self.grid[0])))
        self.y_mulipl = int(round(self.grid_surface_dim_original[1] / (self.grid[1])))

        self.axis_size = 15  # 缩小坐标轴空间
        self.inner_grid_padding = self.marker_size * 2

        self.grid_surface_dim = [
            self.grid_surface_dim_original[0] + (self.marker_size * 4),
            self.grid_surface_dim_original[1] + (self.marker_size * 4)
        ]

        # 缩小网格填充
        self.grid_padding = max(10, int(getattr(self, 'grid_padding', 20) * 0.6))
        
        # 缩小信息面板高度
        self.info_surface_height = max(60, int(getattr(self, 'info_surface_height', 100) * 0.7))

        # Initialize surfaces
        self.grid_surface = Surface(self.grid_surface_dim, pygame.SRCALPHA)
        self.grid_info_surface = Surface([
            self.grid_surface_dim[0] + self.axis_size,
            self.grid_surface_dim[1] + self.axis_size
        ], pygame.SRCALPHA)
        self.travel_surface = Surface(self.grid_surface_dim, pygame.SRCALPHA)
        self.info_surface = Surface([self.grid_surface_dim[0], self.info_surface_height], pygame.SRCALPHA)
        
        # 缩小状态面板
        status_width  = 250   # 进一步缩小
        status_height = (
            self.grid_surface_dim[1]
            + self.info_surface_height
            + self.axis_size
        )
        self.status_surface = Surface([status_width, status_height], pygame.SRCALPHA)

        # 计算窗口尺寸 - 确保不超过常见屏幕尺寸
        content_width = (
            self.grid_surface_dim[0]   # 网格区
            + self.axis_size            # 坐标轴区
            + status_width              # 状态面板区
            + 30                        # 间隔
        )
        content_height = (
            self.grid_surface_dim[1]
            + self.axis_size
            + self.info_surface_height
        )
        
        # 添加边距让内容居中，但限制最大尺寸
        margin_x = 40
        margin_y = 40
        
        self.window_width = min(1200, content_width + 2 * margin_x)  # 限制最大宽度
        self.window_height = min(800, content_height + 2 * margin_y)  # 限制最大高度
        
        # 计算内容在窗口中的起始位置（居中）
        self.content_start_x = (self.window_width - content_width) // 2
        self.content_start_y = (self.window_height - content_height) // 2
        
        # 确保起始位置不为负数
        self.content_start_x = max(10, self.content_start_x)
        self.content_start_y = max(10, self.content_start_y)

        # 缩小字体
        self.big_font = pygame.font.SysFont('Arial', 24, bold=True)      # 24->18
        self.medium_font = pygame.font.SysFont('Arial', 18, bold=False)  # 18->14
        self.small_font = pygame.font.SysFont('Arial', 14, bold=False)   # 14->11
        self.tiny_font = pygame.font.SysFont('Arial', 12, bold=False)     # 12->9

        # Create window immediately
        self.create_window()
        
        # ─── Initial environment snapshot ────────────────────────────────────────
        # Draw everything once, then wait for the user to confirm before training.
        self.reset_surfaces()
        self.draw_nodes()             # draw depot & customer nodes with coords & attributes
        self.draw_vehicles()          # draw current truck/drone positions
        # draw_status_info requires episode & step; set both to zero for this "pre‑training" view
        try:
            self.draw_status_info(episode=0, step=0, last_actions=None, last_rewards=None)
        except TypeError:
            # if your draw_status_info signature differs, just omit last_actions/last_rewards
            self.draw_status_info(0, 0)

        # Blit all layers and flip - 使用新的居中位置
        grid_x = self.content_start_x
        grid_y = self.content_start_y
        
        self.screen.blit(self.grid_info_surface, (grid_x, grid_y))
        self.screen.blit(self.grid_surface, (grid_x, grid_y))
        self.screen.blit(self.travel_surface, (grid_x, grid_y))
        
        info_y = grid_y + self.grid_surface_dim[1] + self.axis_size
        self.screen.blit(self.info_surface, (grid_x, info_y))
        
        status_x = grid_x + self.grid_surface_dim[0] + self.axis_size + 15
        self.screen.blit(self.status_surface, (status_x, grid_y))
        
        pygame.display.flip()

        # Pause here until the user is happy with the setup
        # print("\n🔍 Environment initialized. Press ENTER to begin training…")
        # input()

    # 2. 新增方法：动态生成载具颜色
    def _generate_vehicle_colors(self, num_vehicles, vehicle_type):
        """为不同数量的载具生成区分度高的颜色"""
        colors = []
        
        if vehicle_type == 'truck':
            # 卡车使用暖色调（红、橙、紫红、深红等）
            base_colors = [
                (200, 50, 50),   # 明亮红
                (255, 100, 0),   # 橙红
                (180, 0, 100),   # 紫红
                (139, 0, 0),     # 深红
                (255, 140, 0),   # 深橙
                (160, 32, 120),  # 紫色
                (205, 92, 92),   # 印第安红
                (220, 20, 60),   # 猩红
            ]
        else:  # drone
            # 无人机使用冷色调（蓝、青、蓝绿、深蓝等）
            base_colors = [
                (0, 100, 200),   # 明亮蓝
                (0, 191, 255),   # 深天蓝
                (70, 130, 180),  # 钢蓝
                (0, 0, 139),     # 深蓝
                (0, 139, 139),   # 深青
                (32, 178, 170),  # 浅海蓝绿
                (106, 90, 205),  # 石板蓝
                (65, 105, 225),  # 皇家蓝
            ]
        # 根据载具数量调整颜色
        for i in range(num_vehicles):
            if i < len(base_colors):
                colors.append(base_colors[i])
            else:
                # 超出预设颜色时，基于HSV生成新颜色
                if vehicle_type == 'truck':
                    # 卡车：红色系 (H: 0-60度)
                    h = (i * 37) % 60  # 使用质数37避免重复
                    s = 0.8 + (i % 3) * 0.1  # 饱和度 0.8-1.0
                    v = 0.7 + (i % 4) * 0.075  # 明度 0.7-0.925
                else:
                    # 无人机：蓝色系 (H: 180-280度)
                    h = 180 + (i * 41) % 100  # 蓝色到紫色范围
                    s = 0.7 + (i % 4) * 0.075
                    v = 0.6 + (i % 5) * 0.08
                
                # HSV转RGB
                import colorsys
                r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
                colors.append((int(r*255), int(g*255), int(b*255)))
        
        return colors
    
    def create_window(self):
        """创建 pygame 窗口"""
        if self.no_open_window:
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption(f"VRPD Simulation - {self.name}")
            self.no_open_window = False
            print("🖥️  Visualization window created")

    def reset_surfaces(self):
        """重置所有表面"""
        self.grid_surface.fill(self.color_dict['full_transp'])
        self.grid_info_surface.fill(self.color_dict['full_transp'])
        self.travel_surface.fill(self.color_dict['full_transp'])
        self.info_surface.fill(self.color_dict['full_transp'])
        self.status_surface.fill(self.color_dict['full_transp'])
        self.screen.fill(self.color_dict['white'])

        # Draw grid background
        for i in range(self.grid[0] + 1):
            start_x = self.axis_size + self.marker_size * 2 + i * self.x_mulipl
            pygame.draw.line(self.grid_info_surface, self.color_dict['light-grey'],
                             (start_x, self.axis_size + self.marker_size * 2),
                             (start_x, self.axis_size + self.marker_size * 2 + self.grid[1] * self.y_mulipl))

        for i in range(self.grid[1] + 1):
            start_y = self.axis_size + self.marker_size * 2 + i * self.y_mulipl
            pygame.draw.line(self.grid_info_surface, self.color_dict['light-grey'],
                             (self.axis_size + self.marker_size * 2, start_y),
                             (self.axis_size + self.marker_size * 2 + self.grid[0] * self.x_mulipl, start_y))

        # 绘制坐标轴标签 - 共用零点设计
    
        # —— X 轴标签 —— #
        # 横坐标放在网格下方，预留足够空间
        x_label_y = (
            self.axis_size +                    # 顶部预留空间
            self.marker_size * 2 +              # 网格边距
            self.grid[1] * self.y_mulipl +      # 网格高度
            5                                   # 与网格的间距
        )
        
        # 从0开始标注X轴，确保所有数字都显示
        for i in range(self.grid[0] + 1):
            x_label = self.small_font.render(str(i), True, self.color_dict['black'])
            x_pos = self.axis_size + self.marker_size * 2 + i * self.x_mulipl
            
            # 居中对齐，但处理边界情况
            if i == 0:
                # 最左边的数字稍微右移
                offset_x = -2
            elif i == self.grid[0]:
                # 最右边的数字左移，确保不被裁切
                offset_x = x_label.get_width() - 2
            else:
                # 中间的数字居中
                offset_x = x_label.get_width() // 2
                
            self.grid_info_surface.blit(x_label, (x_pos - offset_x, x_label_y))

        # —— Y 轴标签 —— #
        # 纵坐标放在网格左侧
        y_label_x = 2  # 靠左边对齐
        
        # 从网格最高点开始标注Y轴，包括零点
        for i in range(self.grid[1] + 1):
            y_value = self.grid[1] - i  # 从上到下：grid[1], grid[1]-1, ..., 1, 0
            y_label = self.small_font.render(str(y_value), True, self.color_dict['black'])
            y_pos = self.axis_size + self.marker_size * 2 + i * self.y_mulipl
            
            # 垂直居中对齐
            self.grid_info_surface.blit(y_label, (
                y_label_x, 
                y_pos - y_label.get_height() // 2
            ))       


    def draw_nodes(self):
        """绘制节点 - 只显示激活的节点，跳过未激活的动态节点"""
        try:
            coords = self.temp_db.get_val('n_coord')  # shape: (num_nodes, 2)
            demands = self.temp_db.get_val('n_items')  # shape: (num_nodes,)
            delta = self.temp_db.get_val('delta')  # 访问状态
            deadlines = self.temp_db.get_val('deadline')  # 截止时间
            current_time = getattr(self.temp_db, 'current_time', 0.0)
            
            # 🚧 获取道路破损信息
            road_damaged = self.temp_db.get_val('road_damaged')  # 获取道路破损状态数组
            damaged_nodes = set(i for i, damaged in enumerate(road_damaged) if damaged == 1)
            
            # 统计节点状态
            active_nodes = np.where(delta == 1)[0]
            visited_nodes = np.where(delta == 0)[0]
            inactive_nodes = np.where(delta == -1)[0]
            
            print(f"🎨 Drawing nodes (total: {len(coords)}):")
            print(f"   Active unvisited (delta=1): {active_nodes}")
            print(f"   Visited (delta=0): {visited_nodes}")
            print(f"   Inactive dynamic (delta=-1): {inactive_nodes}")
            print(f"   🚧 Road damaged nodes: {sorted(damaged_nodes)}")
            
            # 确定各种节点的分类
            static_customer_start = self.temp_db.num_depots
            static_customer_end = static_customer_start + getattr(self.temp_db, 'num_static_customers', self.temp_db.num_customers)
            
            for i, (x, y) in enumerate(coords):
                # 关键修复：跳过未激活的动态节点（delta=-1）
                if delta[i] == -1:
                    # 不绘制未激活的动态节点
                    continue
                
                # 对于激活的节点，正常绘制
                screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)
                
                # 选择颜色和标签
                if i in self.temp_db.depot_indices:  # Depot
                    color = self.color_dict['orange']
                    node_type = f"D{i}"
                    marker_size = self.marker_size + 3
                    border_color = self.color_dict['black']
                    border_width = 2
                    
                elif i < static_customer_end:  # 静态客户节点
                    if delta[i] == 0:  # 已访问
                        color = self.color_dict['served_static']
                        node_type = f"S{i}✓"
                    else:  # 未访问 (delta[i] == 1)
                        # 检查是否临近deadline
                        time_remaining = deadlines[i] - current_time
                        if time_remaining < 50:
                            color = self.color_dict['time_critical']
                        else:
                            color = self.color_dict['static_customer']
                        node_type = f"S{i}"
                    marker_size = self.marker_size
                    border_color = self.color_dict['black']
                    border_width = 2
                    
                else:  # 动态客户节点（已激活的）
                    if delta[i] == 0:  # 已访问
                        color = self.color_dict['served_dynamic']
                        node_type = f"D{i}✓"
                        marker_size = self.marker_size
                        border_color = self.color_dict['dynamic_border']
                        border_width = 3
                    else:  # 激活未访问 (delta[i] == 1)
                        # 检查是否临近deadline
                        time_remaining = deadlines[i] - current_time
                        if time_remaining < 50:
                            color = self.color_dict['time_critical']
                        else:
                            color = self.color_dict['active_dynamic']
                        node_type = f"D{i}★"  # 星号表示动态激活
                        marker_size = self.marker_size + 1
                        border_color = self.color_dict['dynamic_border']
                        border_width = 3

                # 🚧 检查是否是道路破损节点
                if i in damaged_nodes and i != 0:  # depot不会道路破损
                    # 修改边框颜色和宽度以表示道路破损
                    border_color = self.color_dict['red']  # 红色边框表示道路破损
                    border_width = 4  # 加粗边框
                    # 添加特殊标记到节点类型
                    if i not in self.temp_db.depot_indices:
                        node_type += "🚫"  # 添加禁止符号        
                
                # 绘制节点圆圈
                pygame.draw.circle(self.grid_surface, color, (screen_x, screen_y), marker_size)
                pygame.draw.circle(self.grid_surface, border_color, (screen_x, screen_y), marker_size, border_width)
                
                # 🚧 如果是道路破损节点，绘制额外的警告标记（X形）
                if i in damaged_nodes and i != 0:
                    # 绘制红色X
                    cross_size = marker_size - 2
                    pygame.draw.line(self.grid_surface, self.color_dict['red'],
                                (screen_x - cross_size, screen_y - cross_size),
                                (screen_x + cross_size, screen_y + cross_size), 3)
                    pygame.draw.line(self.grid_surface, self.color_dict['red'],
                                (screen_x - cross_size, screen_y + cross_size),
                                (screen_x + cross_size, screen_y - cross_size), 3)

                # 绘制节点标签
                label_color = self.color_dict['white'] if delta[i] != 0 else self.color_dict['black']
                label = self.tiny_font.render(node_type, True, label_color)
                label_rect = label.get_rect(center=(screen_x, screen_y))
                self.grid_surface.blit(label, label_rect)
                
                # 绘制需求信息（只对customer节点）
                if i not in self.temp_db.depot_indices:
                    # 需求文本
                    demand_text = f"{demands[i]:.1f}" if demands[i] > 0 else "0"
                    demand_label = self.small_font.render(demand_text, True, self.color_dict['black'])
                    self.grid_surface.blit(demand_label, (screen_x + marker_size + 5, screen_y - 10))
                    
                    # Deadline信息
                    time_remaining = deadlines[i] - current_time
                    if time_remaining < 0:
                        deadline_color = self.color_dict['red']  # 已过期
                        deadline_text = f"⚠{deadlines[i]:.0f}"
                    elif time_remaining < 50:
                        deadline_color = self.color_dict['dark_red']  # 紧急
                        deadline_text = f"!{deadlines[i]:.0f}"
                    else:
                        deadline_color = self.color_dict['dark_green']  # 正常
                        deadline_text = f"D:{deadlines[i]:.0f}"
                    
                    deadline_label = self.tiny_font.render(deadline_text, True, deadline_color)
                    self.grid_surface.blit(deadline_label, (screen_x + marker_size + 5, screen_y + 5))

                    # 🚧 如果是道路破损节点，添加额外警告文本
                    if i in damaged_nodes:
                        damaged_label = self.tiny_font.render("Road Damaged!", True, self.color_dict['red'])
                        self.grid_surface.blit(damaged_label, (screen_x - 30, screen_y + 20))
                       
                print(f"   Drew node {i}: pos=({x},{y}), type={node_type}, delta={delta[i]}, demand={demands[i]:.1f}")
            
            # 绘制动态节点激活时间线
            self._draw_dynamic_timeline(current_time)
            # 🚧 绘制道路破损统计信息
            self._draw_road_damage_legend()
            
        except Exception as e:
            print(f"❌ Error drawing nodes: {e}")
            import traceback
            traceback.print_exc()

    def _draw_road_damage_legend(self):
        """🚧 绘制道路破损图例和统计信息"""
        try:
            # 获取道路破损信息
            road_damaged = self.temp_db.get_val('road_damaged')
            damaged_nodes = [i for i, damaged in enumerate(road_damaged) if damaged == 1]
            
            if not damaged_nodes:
                return
            
            # 在信息面板绘制图例
            legend_x = 10
            legend_y = 10
            
            # 标题
            title = self.medium_font.render("🚧 Road Damage Info", True, self.color_dict['red'])
            self.info_surface.blit(title, (legend_x, legend_y))
            
            # 统计信息
            legend_y += 25
            info_text = f"Damaged nodes: {len(damaged_nodes)} ({damaged_nodes})"
            info_label = self.small_font.render(info_text, True, self.color_dict['black'])
            self.info_surface.blit(info_label, (legend_x, legend_y))
            
            # 图例说明
            legend_y += 20
            legend_items = [
                ("Red border + X", "Road damaged (truck cannot access)"),
                ("Normal border", "Accessible by all vehicles")
            ]
            
            for symbol, description in legend_items:
                symbol_label = self.small_font.render(symbol + ":", True, self.color_dict['dark_red'])
                self.info_surface.blit(symbol_label, (legend_x, legend_y))
                
                desc_label = self.small_font.render(description, True, self.color_dict['black'])
                self.info_surface.blit(desc_label, (legend_x + 100, legend_y))
                legend_y += 18
                
        except Exception as e:
            print(f"Warning: Could not draw road damage legend: {e}")

    def _draw_dynamic_timeline(self, current_time):
        """绘制动态节点的激活时间线 - 显示未来将激活的节点"""
        try:
            if not getattr(self.temp_db, 'dynamic_enabled', False):
                return
            
            # 在状态面板的一个区域绘制时间线
            timeline_y = 300
            timeline_width = 200
            timeline_height = 20
            
            # 绘制时间线背景
            timeline_rect = pygame.Rect(10, timeline_y, timeline_width, timeline_height)
            pygame.draw.rect(self.status_surface, self.color_dict['light-grey'], timeline_rect)
            pygame.draw.rect(self.status_surface, self.color_dict['black'], timeline_rect, 2)
            
            # 绘制当前时间指示器
            horizon = self.temp_db.horizon
            if horizon > 0:
                current_pos = int((current_time / horizon) * timeline_width)
                current_line = pygame.Rect(10 + current_pos, timeline_y, 2, timeline_height)
                pygame.draw.rect(self.status_surface, self.color_dict['red'], current_line)
            
            # 绘制未来将激活的动态节点
            if hasattr(self.temp_db, 'dynamic_nodes_pool'):
                for node in self.temp_db.dynamic_nodes_pool:
                    if horizon > 0:
                        release_pos = int((node['release_time'] / horizon) * timeline_width)
                        if 0 <= release_pos <= timeline_width:
                            # 绘制激活点
                            release_point = pygame.Rect(10 + release_pos - 1, timeline_y - 2, 2, timeline_height + 4)
                            pygame.draw.rect(self.status_surface, self.color_dict['dynamic_border'], release_point)
                            
                            # 显示节点编号
                            node_label = self.tiny_font.render(f"{node['node_idx']}", True, self.color_dict['black'])
                            self.status_surface.blit(node_label, (10 + release_pos - 5, timeline_y - 15))
            
            # 显示已激活的动态节点
            delta = self.temp_db.get_val('delta')
            active_dynamic = []
            for i in range(self.temp_db.num_depots + self.temp_db.num_static_customers, self.temp_db.num_nodes):
                if i < len(delta) and delta[i] >= 0:
                    active_dynamic.append(i)
            
            # 标签
            timeline_label = self.small_font.render("Dynamic Nodes Timeline", True, self.color_dict['black'])
            self.status_surface.blit(timeline_label, (10, timeline_y - 25))
            
            time_label = self.tiny_font.render(f"Time: {current_time:.1f}/{horizon:.1f}", True, self.color_dict['black'])
            self.status_surface.blit(time_label, (10, timeline_y + timeline_height + 5))
            
            if active_dynamic:
                active_label = self.tiny_font.render(f"Active: {active_dynamic}", True, self.color_dict['dark_green'])
                self.status_surface.blit(active_label, (10, timeline_y + timeline_height + 20))
            
            pending = len(self.temp_db.dynamic_nodes_pool) if hasattr(self.temp_db, 'dynamic_nodes_pool') else 0
            if pending > 0:
                pending_label = self.tiny_font.render(f"Pending: {pending} nodes", True, self.color_dict['orange'])
                self.status_surface.blit(pending_label, (10, timeline_y + timeline_height + 35))
            
        except Exception as e:
            print(f"Warning: Could not draw timeline: {e}")

    # # 3. 修改 _record_vehicle_positions 方法（完全替换）
    # def _record_vehicle_positions(self):
    #     """记录当前步骤的车辆位置用于路径追踪- 修复位置获取逻辑"""
    #     try:
    #         v_coords = self.temp_db.status_dict.get('v_coord', [])
    #         coords = self.temp_db.get_val('n_coord')

    #         num_trucks = self.temp_db.num_trucks
    #         num_drones = self.temp_db.num_drones
            
    #         # 记录truck位置（前 num_trucks 个车辆）
    #         for k in range(num_trucks):
    #             if k < len(v_coords) and k < len(self.truck_paths):
    #                 pos_index = v_coords[k]
    #                 if pos_index < len(coords):
    #                     x, y = coords[pos_index]
    #                     self.truck_paths[k].append((x, y))
            
    #         # 记录drone位置 - 修复逻辑
    #         ED = self.temp_db.status_dict.get('ED', [])
    #         ND = self.temp_db.status_dict.get('ND', [])
            
    #         for k in range(num_drones):
    #             vehicle_idx = num_trucks + k  # 在 v_coords 中的实际索引
                
    #             if k < len(self.drone_paths) and k < len(ED):
    #                 if ED[k] == 3:  # Drone on truck
    #                     # 无人机在卡车上，位置与对应卡车相同
    #                     # 需要找到这个无人机附属的卡车
    #                     if vehicle_idx < len(v_coords):
    #                         pos_index = v_coords[vehicle_idx]
    #                         if pos_index < len(coords):
    #                             x, y = coords[pos_index]
    #                             self.drone_paths[k].append((x, y))
    #                 elif ED[k] == 0:  # Drone in transit
    #                     # 无人机在运输中，使用目标位置
    #                     if k < len(ND) and ND[k] < len(coords):
    #                         # 获取无人机当前实际位置
    #                         if 'drone_coord' in self.temp_db.status_dict and k < len(self.temp_db.status_dict['drone_coord']):
    #                             drone_pos = self.temp_db.status_dict['drone_coord'][k]
    #                             if drone_pos < len(coords):
    #                                 x, y = coords[drone_pos]
    #                                 self.drone_paths[k].append((x, y))
    #                         else:
    #                             # 备用方案：使用目标位置
    #                             x, y = coords[ND[k]]
    #                             self.drone_paths[k].append((x, y))
    #                 elif ED[k] in [1, 2]:  # Drone waiting or serving
    #                     # 无人机等待或服务中，使用其独立坐标
    #                     if 'drone_coord' in self.temp_db.status_dict and k < len(self.temp_db.status_dict['drone_coord']):
    #                         drone_pos = self.temp_db.status_dict['drone_coord'][k]
    #                         if drone_pos < len(coords):
    #                             x, y = coords[drone_pos]
    #                             self.drone_paths[k].append((x, y))
    #                     elif k < len(ND) and ND[k] < len(coords):
    #                         # 备用方案
    #                         x, y = coords[ND[k]]
    #                         self.drone_paths[k].append((x, y))
                            
    #     except Exception as e:
    #         print(f"❌ Error recording vehicle positions: {e}")
    #         import traceback
    #         traceback.print_exc()


    def _record_vehicle_positions(self):
        """记录当前步骤的车辆位置用于路径追踪 - 修复位置获取逻辑"""
        try:
            v_coords = self.temp_db.status_dict.get('v_coord', [])
            coords = self.temp_db.get_val('n_coord')

            num_trucks = self.temp_db.num_trucks
            num_drones = self.temp_db.num_drones
            
            # 记录truck位置（前 num_trucks 个车辆）
            for k in range(num_trucks):
                if k < len(v_coords) and k < len(self.truck_paths):
                    pos_index = v_coords[k]
                    if pos_index < len(coords):
                        x, y = coords[pos_index]
                        self.truck_paths[k].append((x, y))
            
            # 记录drone位置 - 关键修复：使用正确的位置获取逻辑
            ED = self.temp_db.status_dict.get('ED', [])
            ND = self.temp_db.status_dict.get('ND', [])
            
            for k in range(num_drones):
                if k < len(self.drone_paths) and k < len(ED):
                    # 获取无人机当前实际位置
                    current_pos = self._get_accurate_drone_position(k)
                    
                    if current_pos < len(coords):
                        x, y = coords[current_pos]
                        self.drone_paths[k].append((x, y))
                        
        except Exception as e:
            print(f"⚠ Error recording vehicle positions: {e}")
            import traceback
            traceback.print_exc()

    def _get_accurate_drone_position(self, drone_idx):
        """获取无人机的准确当前位置"""
        try:
            ED = self.temp_db.status_dict.get('ED', [])
            attached_truck = self.temp_db.status_dict.get('attached_truck', [])
            
            if drone_idx >= len(ED):
                return 0
                
            # 根据无人机状态确定位置
            if ED[drone_idx] == 3:  # 在卡车上
                if (drone_idx < len(attached_truck) and 
                    attached_truck[drone_idx] >= 0 and 
                    attached_truck[drone_idx] < self.temp_db.num_trucks):
                    return self.temp_db.status_dict['v_coord'][attached_truck[drone_idx]]
                else:
                    return 0
                    
            elif ED[drone_idx] in [0, 1, 2]:  # 在途中、等待、或刚完成服务
                # 优先使用独立坐标
                if 'drone_coord' in self.temp_db.status_dict:
                    if drone_idx < len(self.temp_db.status_dict['drone_coord']):
                        return self.temp_db.status_dict['drone_coord'][drone_idx]
                
                # 备用方案：使用v_coord中的位置
                vehicle_idx = self.temp_db.num_trucks + drone_idx
                if vehicle_idx < len(self.temp_db.status_dict.get('v_coord', [])):
                    return self.temp_db.status_dict['v_coord'][vehicle_idx]
                
                return 0
            else:
                return 0
                
        except Exception as e:
            print(f"Warning: Error getting drone {drone_idx} position: {e}")
            return 0

    # 4. 修改 _draw_path_with_arrows 方法（完全替换）
    def _draw_path_with_arrows(self, path, color, line_width=3, arrow_color=None):
        """绘制带箭头的路径 - 修复为直接使用坐标"""
        if len(path) < 2:
            return
            
        if arrow_color is None:
            arrow_color = color
            
        try:
            for i in range(len(path) - 1):
                start_pos = path[i]
                end_pos = path[i + 1]
                
                # 检查坐标格式
                if not (isinstance(start_pos, (tuple, list)) and len(start_pos) == 2):
                    continue
                if not (isinstance(end_pos, (tuple, list)) and len(end_pos) == 2):
                    continue
                    
                # 转换为屏幕坐标
                start_x, start_y = start_pos
                end_x, end_y = end_pos
                
                start_screen_x = int(self.axis_size + self.marker_size * 2 + start_x * self.x_mulipl)
                start_screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - start_y) * self.y_mulipl)
                end_screen_x = int(self.axis_size + self.marker_size * 2 + end_x * self.x_mulipl)
                end_screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - end_y) * self.y_mulipl)
                
                # 绘制路径线段
                if start_screen_x != end_screen_x or start_screen_y != end_screen_y:
                    pygame.draw.line(self.travel_surface, color,
                                (start_screen_x, start_screen_y), 
                                (end_screen_x, end_screen_y), line_width)
                    
                    # 绘制箭头（在线段中点）
                    self._draw_arrow(start_screen_x, start_screen_y, 
                                end_screen_x, end_screen_y, arrow_color)
                    
        except Exception as e:
            print(f"❌ Error drawing path: {e}")

    # 5. 修改 draw_paths 方法（完全替换）
    def draw_paths(self):
        """绘制所有车辆的完整路径 - 使用动态颜色系统"""
        # 绘制truck路径
        for k, path in enumerate(self.truck_paths):
            if len(path) >= 2:
                truck_color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
                self._draw_path_with_arrows(path, truck_color, line_width=4)
        
        # 绘制drone路径
        for k, path in enumerate(self.drone_paths):
            if len(path) >= 2:
                drone_color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
                self._draw_path_with_arrows(path, drone_color, line_width=2)


    def draw_vehicles(self):
        """绘制车辆位置和状态"""
        try:
            # 卡车/无人机分开
            v_coords = self.temp_db.status_dict.get('v_coord')
            ET = self.temp_db.status_dict.get('ET')
            ED = self.temp_db.status_dict.get('ED')
            TW = self.temp_db.status_dict.get('TW')
            print("▶ TW in viz:", TW)
            DW = self.temp_db.status_dict.get('DW', [])  # 无人机负载
            LT = self.temp_db.status_dict.get('LT')
            LD = self.temp_db.status_dict.get('LD')
            NT = self.temp_db.status_dict.get('NT')
            ND = self.temp_db.status_dict.get('ND')

            # 修正：使用正确的坐标键名
            coords = self.temp_db.get_val('n_coord')

            print(f"🚛 Drawing {len(v_coords)} vehicles:")
            print(f"   Vehicle positions (node indices): {v_coords}")
            print(f"   ET (truck status): {ET}")
            print(f"   ED (drone status): {ED}")

            # 分别处理卡车和无人机
            num_trucks = self.temp_db.num_trucks
            num_drones = self.temp_db.num_drones
            
            # 绘制卡车 (前 num_trucks 个车辆)
            for k in range(num_trucks):
                if k >= len(v_coords):
                    continue
                    
                # Get vehicle position from node coordinates
                pos_index = int(v_coords[k])

                if pos_index < len(coords):
                    x, y = coords[pos_index]
                    # 修正坐标转换，与节点绘制保持一致
                    screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                    screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)

                    # ========== 修改：使用动态卡车颜色系统 ==========
                    base_truck_color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
                
                    if k < len(ET):  # 确保索引有效
                        if ET[k] == 0:  # In transit - 保持原色
                            truck_color = base_truck_color
                        elif ET[k] == 1:  # Waiting - 加入黄色调
                            r, g, b = base_truck_color
                            truck_color = (min(255, r + 50), min(255, g + 100), b)
                        elif ET[k] == 2:  # Serving - 加入绿色调
                            r, g, b = base_truck_color
                            truck_color = (r, min(255, g + 80), b)
                        else:  # Idle - 降低亮度
                            r, g, b = base_truck_color
                            truck_color = (max(50, r - 50), max(50, g - 50), max(50, b - 50))
                    else:
                        truck_color = base_truck_color
                    # ============================================

                    # Draw truck as rectangle
                    truck_rect = pygame.Rect(screen_x - 10, screen_y - 20, 20, 15)
                    pygame.draw.rect(self.grid_surface, truck_color, truck_rect)
                    pygame.draw.rect(self.grid_surface, self.color_dict['black'], truck_rect, 2)

                    # Draw vehicle ID
                    id_label = self.tiny_font.render(f"T{k}", True, self.color_dict['white'])
                    id_rect = id_label.get_rect(center=(screen_x, screen_y - 12))
                    self.grid_surface.blit(id_label, id_rect)

                    # Draw load info
                    if k < len(TW):
                        load_text = f"{TW[k]:.1f}"
                        load_label = self.small_font.render(load_text, True, self.color_dict['black'])
                        self.grid_surface.blit(load_label, (screen_x - 20, screen_y + 10))

                    print(f"   Truck {k}: at node {pos_index}, pos=({x},{y}), load={TW[k] if k < len(TW) else 'N/A'}")
            
            # 绘制无人机 (从 num_trucks 开始的后续车辆)
            for k in range(num_drones):
                vehicle_idx = num_trucks + k  # 在 v_coords 中的实际索引
                
                if vehicle_idx >= len(v_coords):
                    continue
                    
                pos_index = int(v_coords[vehicle_idx])
                
                if pos_index < len(coords):
                    x, y = coords[pos_index]
                    screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                    screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)

                    # 使用动态无人机颜色系统
                    base_drone_color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
                    
                    # Draw drone status
                    if k < len(ED):  # 确保索引有效
                        if ED[k] == 3:  # Drone on truck
                            pygame.draw.circle(self.grid_surface, base_drone_color,
                                            (screen_x, screen_y - 25), 5)
                            pygame.draw.circle(self.grid_surface, self.color_dict['black'],
                                            (screen_x, screen_y - 25), 5, 1)
                        elif ED[k] == 0:  # Drone in transit
                            # Draw drone separately if in transit to different node
                            if k < len(ND):
                                drone_target = int(ND[k])
                                if drone_target < len(coords) and drone_target != pos_index:
                                    dx, dy = coords[drone_target]
                                    drone_screen_x = int(self.axis_size + self.marker_size * 2 + dx * self.x_mulipl)
                                    drone_screen_y = int(
                                        self.axis_size + self.marker_size * 2 + (self.grid[1] - dy) * self.y_mulipl)

                                    # # Draw line showing drone path
                                    # pygame.draw.line(self.grid_surface, base_drone_color,
                                    #                 (screen_x, screen_y), (drone_screen_x, drone_screen_y), 2)

                                    # # Draw drone at target
                                    # pygame.draw.circle(self.grid_surface, base_drone_color,
                                    #                 (drone_screen_x, drone_screen_y + 15), 4)
                        elif ED[k] in [1, 2]:  # Drone waiting or serving
                            # 无人机独立位置时的绘制
                            if 'drone_coord' in self.temp_db.status_dict:
                                if k < len(self.temp_db.status_dict['drone_coord']):
                                    drone_pos_index = self.temp_db.status_dict['drone_coord'][k]
                                    if drone_pos_index < len(coords):
                                        dx, dy = coords[drone_pos_index]
                                        drone_screen_x = int(self.axis_size + self.marker_size * 2 + dx * self.x_mulipl)
                                        drone_screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - dy) * self.y_mulipl)
                                        
                                        # 等待状态：稍微透明
                                        if ED[k] == 1:
                                            r, g, b = base_drone_color
                                            drone_color = (r, g, b, 180)  # 透明度
                                        else:  # 服务状态：高亮
                                            r, g, b = base_drone_color
                                            drone_color = (min(255, r + 30), min(255, g + 30), min(255, b + 30))
                                        
                                        pygame.draw.circle(self.grid_surface, drone_color[:3],
                                                        (drone_screen_x, drone_screen_y), 6)
                                        pygame.draw.circle(self.grid_surface, self.color_dict['black'],
                                                        (drone_screen_x, drone_screen_y), 6, 1)
                            else:
                                # 备用方案：在当前位置绘制
                                pygame.draw.circle(self.grid_surface, base_drone_color,
                                                (screen_x, screen_y + 15), 6)
                                pygame.draw.circle(self.grid_surface, self.color_dict['black'],
                                                (screen_x, screen_y + 15), 6, 1)

                    # Draw drone ID
                    id_label = self.tiny_font.render(f"D{k}", True, self.color_dict['white'])
                    id_rect = id_label.get_rect(center=(screen_x, screen_y + 20))
                    self.grid_surface.blit(id_label, id_rect)

                    print(f"   Drone {k}: at node {pos_index}, pos=({x},{y})")
                    
        except Exception as e:
            print(f"❌ Error drawing vehicles: {e}")
            import traceback
            traceback.print_exc()

    
    def _draw_arrow(self, start_x, start_y, end_x, end_y, color, size=8):
        """在两点间绘制箭头"""
        # 计算方向向量
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return
            
        # 单位方向向量
        ux = dx / length
        uy = dy / length
        
        # 箭头位置（在线段的70%处）
        arrow_pos_x = int(start_x + dx * 0.7)
        arrow_pos_y = int(start_y + dy * 0.7)
        
        # 箭头的两个翅膀点
        wing_length = size
        wing_angle = 0.6  # 约35度
        
        # 旋转向量得到箭头翅膀
        wing1_x = arrow_pos_x - int(wing_length * (ux * np.cos(wing_angle) - uy * np.sin(wing_angle)))
        wing1_y = arrow_pos_y - int(wing_length * (ux * np.sin(wing_angle) + uy * np.cos(wing_angle)))
        
        wing2_x = arrow_pos_x - int(wing_length * (ux * np.cos(-wing_angle) - uy * np.sin(-wing_angle)))
        wing2_y = arrow_pos_y - int(wing_length * (ux * np.sin(-wing_angle) + uy * np.cos(-wing_angle)))
        
        # 绘制箭头
        pygame.draw.polygon(self.travel_surface, color, [
            (arrow_pos_x, arrow_pos_y),
            (wing1_x, wing1_y),
            (wing2_x, wing2_y)
        ])

    

    def draw_status_info(self, episode, step, last_actions=None, last_rewards=None):
        """绘制状态信息面板"""
        y_offset = 10
        line_height = 22

        # Title
        title = self.big_font.render("VRPD Status", True, self.color_dict['black'])
        self.status_surface.blit(title, (10, y_offset))
        y_offset += 35

        # Episode and step info
        episode_text = self.medium_font.render(f"Episode: {episode}", True, self.color_dict['black'])
        self.status_surface.blit(episode_text, (10, y_offset))
        y_offset += line_height

        step_text = self.medium_font.render(f"Step: {step}", True, self.color_dict['black'])
        self.status_surface.blit(step_text, (10, y_offset))
        y_offset += line_height

        # ========== 新增：显示总训练步数 ==========
        total_steps = getattr(self, 'total_training_steps', 0)
        total_step_text = self.medium_font.render(f"Total Steps: {total_steps}", True, self.color_dict['dark_green'])
        self.status_surface.blit(total_step_text, (10, y_offset))
        y_offset += line_height + 5
        # =====================================

        # Time info
        try:
            total_time = getattr(self.temp_db, 'total_time', 0.0)
            time_text = self.medium_font.render(f"Time: {total_time:.2f}", True, self.color_dict['black'])
            self.status_surface.blit(time_text, (10, y_offset))
            y_offset += line_height + 5
        except:
            pass

        # ========== 修改：增强路径统计信息显示 ==========
        try:
            path_title = self.medium_font.render("Path Info:", True, self.color_dict['purple'])
            self.status_surface.blit(path_title, (10, y_offset))
            y_offset += line_height
            
            for k, truck_path in enumerate(self.truck_paths):
                if truck_path:
                    path_length = len(truck_path)
                    # 使用载具颜色显示
                    color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
                    path_text = self.small_font.render(f"  T{k}: {path_length} steps", True, color)
                    self.status_surface.blit(path_text, (20, y_offset))
                    y_offset += 16
            
            for k, drone_path in enumerate(self.drone_paths):
                if drone_path:
                    path_length = len(drone_path)
                    # 使用载具颜色显示
                    color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
                    path_text = self.small_font.render(f"  D{k}: {path_length} steps", True, color)
                    self.status_surface.blit(path_text, (20, y_offset))
                    y_offset += 16
            y_offset += 5
        except Exception as e:
            error_text = self.small_font.render(f"Path Info Error: {str(e)[:20]}", True, self.color_dict['red'])
            self.status_surface.blit(error_text, (10, y_offset))
            y_offset += line_height
        # =============================================

        # Node status summary
        try:
            demands = self.temp_db.get_val('n_items')
            delta = self.temp_db.get_val('delta')
            visited_count = np.sum(delta == 0)
            total_demand = np.sum(demands[self.temp_db.customer_indices])

            # 🚧 获取道路破损信息
            road_damaged = self.temp_db.get_val('road_damaged')
            damaged_count = np.sum(road_damaged == 1)
            damaged_nodes = [i for i, d in enumerate(road_damaged) if d == 1]


            nodes_title = self.medium_font.render("Node Status:", True, self.color_dict['dark_green'])
            self.status_surface.blit(nodes_title, (10, y_offset))
            y_offset += line_height

            visited_text = self.small_font.render(f"  Visited: {visited_count}/{len(delta)}", True,
                                                  self.color_dict['black'])
            self.status_surface.blit(visited_text, (20, y_offset))
            y_offset += 18

            demand_text = self.small_font.render(f"  Total demand: {total_demand:.1f}", True, self.color_dict['black'])
            self.status_surface.blit(demand_text, (20, y_offset))
            y_offset += 25

            # 🚧 添加道路破损统计
            if damaged_count > 0:
                damaged_text = self.small_font.render(f"  🚧 Damaged: {damaged_count} nodes", True, 
                                                    self.color_dict['red'])
                self.status_surface.blit(damaged_text, (20, y_offset))
                y_offset += 18
                
                # 显示具体的破损节点
                if len(damaged_nodes) <= 5:  # 如果节点不多，显示具体编号
                    nodes_str = str(damaged_nodes)
                else:  # 节点太多时显示部分
                    nodes_str = f"{damaged_nodes[:3]}...({damaged_count} total)"
                damaged_detail = self.tiny_font.render(f"    Nodes: {nodes_str}", True, 
                                                    self.color_dict['dark_red'])
                self.status_surface.blit(damaged_detail, (20, y_offset))
                y_offset += 16
            
            y_offset += 7

        except Exception as e:
            error_text = self.small_font.render(f"Node Status Error: {str(e)[:30]}", True, self.color_dict['red'])
            self.status_surface.blit(error_text, (10, y_offset))
            y_offset += line_height

        # Vehicle status
        try:
            ET = self.temp_db.status_dict.get('ET', [])
            ED = self.temp_db.status_dict.get('ED', [])
            v_coords = self.temp_db.status_dict.get('v_coord', [])
            TW = self.temp_db.status_dict.get('TW', [])
            NT = self.temp_db.status_dict.get('NT', [])
            ND = self.temp_db.status_dict.get('ND', [])

            vehicles_title = self.medium_font.render("Vehicle Status:", True, self.color_dict['dark_green'])
            self.status_surface.blit(vehicles_title, (10, y_offset))
            y_offset += line_height

            # 显示卡车状态
            truck_status = {0: "In-transit", 1: "Waiting", 2: "Serving", 3: "Idle"}
            for k in range(self.temp_db.num_trucks):
                if k >= len(ET):
                    continue
                    
                # Vehicle header
                vehicle_text = self.small_font.render(f"Truck {k}:", True, self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red'])
                self.status_surface.blit(vehicle_text, (20, y_offset))
                y_offset += 18

                # Position
                pos_text = self.small_font.render(f"  Position: Node {v_coords[k] if k < len(v_coords) else 'N/A'}",
                                                True, self.color_dict['black'])
                self.status_surface.blit(pos_text, (20, y_offset))
                y_offset += 16

                # Truck status
                truck_text = self.small_font.render(
                    f"  Status: {truck_status.get(ET[k], 'Unknown')} -> Node {NT[k] if k < len(NT) else 'N/A'}",
                    True, self.color_dict['black'])
                self.status_surface.blit(truck_text, (20, y_offset))
                y_offset += 16

                # Load
                if k < len(TW):
                    load_text = self.small_font.render(f"  Load: {TW[k]:.2f}", True, self.color_dict['black'])
                    self.status_surface.blit(load_text, (20, y_offset))
                y_offset += 20

            # 显示无人机状态  
            drone_status = {0: "In-transit", 1: "Waiting", 2: "Serving", 3: "On-truck"}
            for k in range(self.temp_db.num_drones):
                if k >= len(ED):
                    continue
                    
                # Vehicle header
                vehicle_text = self.small_font.render(f"Drone {k}:", True, self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue'])
                self.status_surface.blit(vehicle_text, (20, y_offset))
                y_offset += 18

                # Position
                vehicle_idx = self.temp_db.num_trucks + k
                pos_text = self.small_font.render(f"  Position: Node {v_coords[vehicle_idx] if vehicle_idx < len(v_coords) else 'N/A'}",
                                                True, self.color_dict['black'])
                self.status_surface.blit(pos_text, (20, y_offset))
                y_offset += 16

                # Drone status
                drone_text = self.small_font.render(
                    f"  Status: {drone_status.get(ED[k], 'Unknown')} -> Node {ND[k] if k < len(ND) else 'N/A'}",
                    True, self.color_dict['black'])
                self.status_surface.blit(drone_text, (20, y_offset))
                y_offset += 16

                # Drone weight (if available)
                if 'DW' in self.temp_db.status_dict and k < len(self.temp_db.status_dict['DW']):
                    weight_text = self.small_font.render(f"  Weight: {self.temp_db.status_dict['DW'][k]:.2f}", True, self.color_dict['black'])
                    self.status_surface.blit(weight_text, (20, y_offset))
                y_offset += 20

        except Exception as e:
            error_text = self.small_font.render(f"Vehicle Status Error: {str(e)[:30]}", True, self.color_dict['red'])
            self.status_surface.blit(error_text, (10, y_offset))
            y_offset += line_height

        # Last actions
        if last_actions:
            actions_title = self.medium_font.render("Last Actions:", True, self.color_dict['dark_red'])
            self.status_surface.blit(actions_title, (10, y_offset))
            y_offset += line_height

            for i, action in enumerate(last_actions):
                action_str = str(action)[:50] + "..." if len(str(action)) > 50 else str(action)
                action_text = self.small_font.render(f"  Agent {i}: {action_str}", True, self.color_dict['black'])
                self.status_surface.blit(action_text, (20, y_offset))
                y_offset += 16

        # Last rewards
        if last_rewards:
            y_offset += 5
            rewards_title = self.medium_font.render("Last Rewards:", True, self.color_dict['dark_red'])
            self.status_surface.blit(rewards_title, (10, y_offset))
            y_offset += line_height

            for i, reward in enumerate(last_rewards):
                reward_text = self.small_font.render(f"  Agent {i}: {reward:.3f}", True, self.color_dict['black'])
                self.status_surface.blit(reward_text, (20, y_offset))
                y_offset += 16

        # Control instructions
        y_offset += 10
        control_title = self.medium_font.render("Controls:", True, self.color_dict['purple'])
        self.status_surface.blit(control_title, (10, y_offset))
        y_offset += line_height

        instructions = [
            "SPACE: Pause/Resume",
            "ENTER: Next Step (when paused)",
            "ESC: Quit",
            "Click: Node details"
        ]

        for instruction in instructions:
            instr_text = self.small_font.render(instruction, True, self.color_dict['black'])
            self.status_surface.blit(instr_text, (20, y_offset))
            y_offset += 16

    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"🎮 {'Paused' if self.paused else 'Resumed'} simulation")
                elif event.key == pygame.K_RETURN and self.paused:
                    return 'step'
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse clicks for node details
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.handle_mouse_click(mouse_x, mouse_y)
        return None

    def handle_mouse_click(self, mouse_x, mouse_y):
        """处理鼠标点击，显示节点详细信息"""
        try:
            coords = self.temp_db.get_val('n_coord')
            demands = self.temp_db.get_val('n_items')
            deadlines = self.temp_db.get_val('deadline')
            delta = self.temp_db.get_val('delta')

            for i, (x, y) in enumerate(coords):
                # 修正坐标转换，与draw_nodes保持一致
                screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)

                # Check if click is within node circle
                # 如果使用了居中布局，需要调整鼠标坐标
                if hasattr(self, 'content_start_x'):
                    adjusted_mouse_x = mouse_x - self.content_start_x
                    adjusted_mouse_y = mouse_y - self.content_start_y
                else:
                    adjusted_mouse_x = mouse_x - self.grid_padding
                    adjusted_mouse_y = mouse_y - self.grid_padding
                
                distance = np.sqrt(
                    (adjusted_mouse_x - screen_x) ** 2 + (adjusted_mouse_y - screen_y) ** 2)
                if distance <= self.marker_size + 5:
                    node_type = "Depot" if i in self.temp_db.depot_indices else "Customer"
                    status = "Visited" if delta[i] == 0 else "Unvisited"
                    print(f"\n🖱️  Clicked Node {i} ({node_type}):")
                    print(f"   Position: ({x}, {y})")
                    print(f"   Demand: {demands[i]:.2f}")
                    print(f"   Deadline: {deadlines[i]:.2f}")
                    print(f"   Status: {status}")
                    break
        except Exception as e:
            print(f"❌ Error handling mouse click: {e}")

    def visualize_step(self, episode, step, slow_down_pls=False, last_actions=None, last_rewards=None):
        """可视化当前步骤 - 增强路径追踪"""
        if not self.enabled:
            return
        # 检查是否开始新的episode
        if episode != self.current_episode:
            print(f"🔄 Starting new episode {episode}, clearing previous paths")
            
            # 清空当前episode的路径（每个episode重新开始）
            self.current_episode = episode
            for i in range(len(self.truck_paths)):
                self.truck_paths[i].clear()
                self.drone_paths[i].clear()
        
        self.current_step = step
        
        # # 记录当前位置到路径中
        # self._record_vehicle_positions()

        # 🔧 关键修复：在step=0时记录初始位置，其他时候在动作执行后记录
        if step == 0:
            # Episode开始时记录所有车辆的初始位置
            self._record_vehicle_positions()
            print(f"📍 Recorded initial positions for episode {episode}")
        else:
            # 步骤执行后记录新位置
            self._record_vehicle_positions()
        
        # 设置总训练步数用于显示
        self.total_training_steps = getattr(self, 'total_training_steps', 0) + 1

        # Create window if not exists
        if self.no_open_window:
            self.create_window()

        # Handle events
        event_result = self.handle_events()

        # Wait if paused
        while self.paused:
            event_result = self.handle_events()
            if event_result == 'step':
                break
            time.sleep(0.1)

        # Reset and draw
        self.reset_surfaces()
        self.draw_paths()  
        self.draw_nodes()
        self.draw_vehicles()
        self.draw_status_info(episode, step, last_actions, last_rewards)

        # 将所有图层按正确顺序叠加到主屏幕上
        # 使用content_start位置确保内容居中显示
        grid_x = self.content_start_x
        grid_y = self.content_start_y
        
        # 底层：网格和坐标轴
        self.screen.blit(self.grid_info_surface, (grid_x, grid_y))
        # 中层：节点和车辆
        self.screen.blit(self.grid_surface, (grid_x, grid_y))
        # 顶层：路径轨迹
        self.screen.blit(self.travel_surface, (grid_x, grid_y))
        
        # 下方：信息面板
        info_y = grid_y + self.grid_surface_dim[1] + self.axis_size
        self.screen.blit(self.info_surface, (grid_x, info_y))
        
        # 右侧：状态面板
        status_x = grid_x + self.grid_surface_dim[0] + self.axis_size + 15
        self.screen.blit(self.status_surface, (status_x, grid_y))

        # 更新屏幕显示
        pygame.display.flip()


        # # 强制等待用户输入 - 这是您要求的关键功能
        # if step == 0 or slow_down_pls:
        #     print(f"\n🎯 Episode {episode}, Step {step} - Visualization Updated")
        #     print("   Press ENTER to continue to next step...")
        #     input()  # 等待用户按Enter键
        # else:
        #     # 正常步骤也可以设置短暂延迟
        #     time.sleep(0.3)

    def convert_to_img_array(self):
        """重置可视化器 - 每个episode结束时调用"""
        print(f"🔄 Episode {self.current_episode} finished, clearing paths for next episode")
        
        # 清空当前episode的所有路径轨迹
        for i in range(len(self.truck_paths)):
            self.truck_paths[i].clear()
            self.drone_paths[i].clear()

        """将当前screen转换为numpy数组，供神经网络使用"""
        if not self.no_open_window:
            self.reset_surfaces()
        print("✅ Visualizer reset completed")

        # 获取screen的像素数据
        w, h = self.screen.get_size()
        raw = pygame.surfarray.array3d(self.screen)
        # pygame使用(width, height, 3)格式，转换为标准的(height, width, 3)
        return raw.swapaxes(0, 1)

    def reset(self):
        """重置可视化器"""
        if not self.enabled:
            return
        if not self.no_open_window:
            self.reset_surfaces()
        print("🔄 Visualizer reset")

    def close(self):
        """关闭可视化器"""
        if not self.enabled:
            return
        if not self.no_open_window:
            pygame.quit()
            self.no_open_window = True
            print("🚪 Visualizer closed")