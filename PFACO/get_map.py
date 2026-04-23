#!/usr/bin/env python
# mxm map, square matrix shape.
# S indicates the starting point, just one starting point is allowed.
# F indicates the final point, just one final point is allowed.
# E indicates if a point in the map is empty.
# O indicates if a point in the map is occupied.

import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation
from function import  list_to_tuple
import random

import matplotlib as mpl
# mpl.use('TKAgg')
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
from collections import deque

# import xlwt
# import xlrd

def find_enclosed_areas(grid):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    enclosed_areas = []
    
    def bfs(x, y):
        queue = deque([(x, y)])
        area = []
        while queue:
            cx, cy = queue.popleft()
            area.append((cx, cy))
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 1:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        return area
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                visited[i][j] = True
                enclosed_area = bfs(i, j)
                enclosed_areas.append(enclosed_area)
    return enclosed_areas

class Map:
    ''' Class used for handling the information provided by the input map '''
    # 每一个node有3个属性：1.位置（x，y）；2.edges; 3.spec={S,F,E,O};
                                   # edges: 1.可以取的点；2.phermone；3.probability
    class Nodes:
        ''' Class for representing the nodes used by the ACO algorithm '''
        def __init__(self, row, col, in_map, spec, start_node, terminal_node):
            self.node_pos = (row, col)
            self.spec = spec
            self.start = start_node             #  start
            self.terminal = terminal_node       # terminal
            self.d_ST = np.sqrt((self.start[0]-self.terminal[0])**2 + (self.start[1]-self.terminal[1])**2)
            self.edges, self.available = self.compute_edges(in_map)


        def compute_edges(self, map_arr):   # initial phermone and edges' length
            ''' class that returns the edges connected to each node '''
            imax = map_arr.shape[0]  # map_arr的行数
            jmax = map_arr.shape[1]  # map_arr的列数
            edges = []
            available = []
            if map_arr[self.node_pos[0]][self.node_pos[1]] == 1:        # node i : x = self.node_pos[0] ; y = self.node_pos[1]
                for dj in [-1, 0, 1]:           
                    for di in [-1, 0, 1]:      
                        newi = self.node_pos[0] + di        # newi = node j[0]
                        newj = self.node_pos[1] + dj        # newj = node j[1]
                        if (dj == 0 and di == 0):
                            continue
                        if (newj >= 0 and newj < jmax and newi >= 0 and newi < imax):
                            if map_arr[newi][newj] == 1:  # 无障碍物 且 一步可达
                                dis = np.sqrt((self.node_pos[0] - newi)**2 + (self.node_pos[1] - newj)**2)        
                                available.append((newi, newj))
                                
                                # 3.3 Adaptive phromone concentration setting
                                # 根据function(6)初始化信息素浓度分布  
                                d_Si = np.sqrt((self.start[0] - self.node_pos[0])**2 + (self.start[1] - self.node_pos[1])**2)
                                d_iT = np.sqrt((self.terminal[0] - self.node_pos[0])**2 + (self.terminal[1] - self.node_pos[1])**2)

                                d_Sj = np.sqrt((self.start[0] - newi)**2 + (self.start[1] - newj)**2)
                                d_jT = np.sqrt((self.terminal[0] - newi)**2 + (self.terminal[1] - newj)**2)
                                
                                if d_iT > d_jT:
                                    a = 2
                                else:
                                    a = 1

                                pheromone = a * (self.d_ST/(d_Si + d_iT) + self.d_ST/(d_Sj + d_jT)) * 1    # tau_0 = 1
                                # pheromone = a * ( self.d_ST/(d_Sj + d_jT) ) * 1    # tau_0 = 1
                                edges.append({'NextNode': (newi,newj), 'Pheromone': pheromone, 'eta': 1/(dis), 'Probability': 0.0})
            return edges, available



    def __init__(self, map_name):   
        self.in_map = self._read_map(map_name)  # in_map类型为str
        self.occupancy_map = self._map_2_occupancy_map()  # 输入in_map 将地图转化成int matrix
        self.available_nodes = self.add_available_nodes()  # 地图上所有空的点

        self.flag_enclosed = False
        # 查找被障碍物隔开的密闭空间
        enclosed_areas_full = find_enclosed_areas(self.occupancy_map.tolist())
        if len(enclosed_areas_full) > 1:  # 至少有两个子集=存在被障碍物割断的两个空间 
            self.flag_enclosed = True   # enclosed_areas is a list, every element means an arear of enclose_areas
            enclosed_areas = [sub_area for sub_area in enclosed_areas_full if len(sub_area) >= 2]
        # for idx, area in enumerate(enclosed_areas):
        #     print(f"Enclosed Area {idx + 1}: {area}")
        if self.flag_enclosed:      # 存在因障碍物而产生的密闭空间
            # 随机选择一个enclosed_area, 然后在这个范围内随机产生起点和终点（保证了起点和终点之间的联通性）
            sub_area_index = 0 #random.randint(0, (len(enclosed_areas)-1))
            self.available_nodes = enclosed_areas[sub_area_index]
            self.initial_node =  (9,5)#random.choice(enclosed_areas[sub_area_index])
            self.available_nodes.remove(self.initial_node)
            self.final_node = (2,9) #random.choice(enclosed_areas[sub_area_index])
        else:
            self.initial_node = (9,5) #random.choice(self.available_nodes)
            self.available_nodes.remove(self.initial_node)
            self.final_node = (2,9) #random.choice(self.available_nodes)

        self.barrier = self.add_obs_nodes()  # 添加障碍物
        self.nodes_array = self._create_nodes(self.initial_node, self.final_node)
        self.height = self.occupancy_map.shape[0]
        self.width = self.occupancy_map.shape[1]

    def add_available_nodes(self):
        available_nodes = []
        nodes = []
        points = np.where(self.in_map == '1')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            available_nodes.append(nodes)
            nodes = []
        return list_to_tuple(available_nodes)

    def _create_nodes(self, start_node, terminal_node):
        ''' Create nodes out of the initial map '''
        return [[self.Nodes(i, j, self.occupancy_map, self.in_map[i][j], start_node, terminal_node) 
                 for j in range(self.in_map.shape[1])] for i in range(self.in_map.shape[0])]
    # 读取map文件
    def _read_map(self, map_name):
        ''' Reads data from an input map txt file'''
        in_map = np.loadtxt( map_name, dtype=str )
        return in_map

    def add_obs_nodes(self):
        obs_nodes = []
        nodes = []
        points = np.where(self.in_map == '0')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            obs_nodes.append(nodes)
            nodes = []
        return list_to_tuple(obs_nodes)

    # str地图转化为int matrice
    def _map_2_occupancy_map(self):
        ''' Takes the matrix and converts it into a float array '''
        map_arr = np.copy(self.in_map)  # 复制一个in_map,命名为 map_arr
        map_arr[map_arr == 'O'] = 0  # 障碍
        map_arr[map_arr == 'E'] = 1  # 空的
        map_arr[map_arr == 'S'] = 1  # 开始
        map_arr[map_arr == 'F'] = 1  # 目的地
        return map_arr.astype(int)  # astype 函数用于array中数值类型转换

    #绘图
    def represent_map(self):
        ''' Represents the map '''
        # Map representation

        plt.plot(self.initial_node[1],self.initial_node[0], 'bo', markersize=10)
        #red 实心圆；markersize设置标记大小；marker设置标记形状
        plt.plot(self.final_node[1],self.final_node[0], 'r*', markersize=10)
        plt.imshow(self.occupancy_map, cmap='gray', interpolation = 'nearest')
        plt.grid(ls="--")
        plt.show()
        # plt.close()

    def plot_dis_iter(self, distance):
        fig = plt.figure(2)
        # plt.figure语()---在plt中绘制一张图片
        plt.title("Distance iteration graph")  # 距离迭代图
        plt.plot(range(1, len(distance) + 1), distance)
        plt.xlabel("Number of iterations")  # 迭代次数
        plt.ylabel("Distance value")  # 距离值
        plt.show()

    # 画出路径
    def represent_path(self, path): # 前景
        ''' Represents the path in the map '''
        # def update_points(num):
        #     '''更新数据点'''
        #     point_ani.set_data(x[num], y[num])
        #     text_pt.set_position((x[num], y[num]))
        #     text_pt.set_text("x=%.1f, y=%.1f" % (x[num], y[num]))
        #     return point_ani, text_pt,
        # # 存数据
        shape = list(copy.deepcopy(self.occupancy_map.shape))
        width = shape[0]
        height = shape[1]
        x = []
        y = []
        for p in path:
            x.append(p[1])  # x y反了，为什么
            y.append(p[0])
        # shape = list(copy.deepcopy(self.in_map.shape))
        # fig = plt.figure(tight_layout=True)

        # 画背景图
        plt.figure(figsize=(width,height))
        plt.plot(x, y)
        plt.grid(ls="--")
        # 画起始点
        plt.plot(self.initial_node[1],self.initial_node[0], 'bo', markersize=10)
        #red 实心圆；markersize设置标记大小；marker设置标记形状
        plt.plot(self.final_node[1],self.final_node[0], 'r*', markersize=10)
        plt.imshow(self.occupancy_map, cmap='Greys', interpolation = 'nearest')
        # plt.savefig("../Pure_aco/pureaco.png")
        plt.savefig("aco_path_eta.pdf")
        # plt.show()
        # plt.close()

        # 开始制作动画
        # fig = plt.figure()
        # point_ani, = plt.plot(x[0], y[0], "go")
        # plt.imshow(self.occupancy_map, cmap='gray', interpolation='nearest')
        # plt.grid(ls="--")
        # text_pt = plt.text(4, 0.8, '', fontsize=12, color='green')
        #
        # ani = animation.FuncAnimation(fig, update_points, np.arange(0, (len(x))), interval=1000, blit=True)
        # # ani.save('test.gif', writer='imagemagick', fps=1)
        # plt.show()
        # plt.close()


    def plot_iteration_len(self, iteration_len,iterations):
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        plt.rcParams['figure.figsize']=(5,4)

        plt.xlim(0,iterations)
        plt.ylim((min(iteration_len)-3), (max(iteration_len)+3))

        x_data = list(range((iterations)))

        y_data = iteration_len

        #输入折线图数据
        # plt.plot(x_data,y2_data,label='Path vs. AS on training set',linewidth=2,color='g')
        # plt.plot(x_data,y_data,label='Path vs. AS on test set',linewidth=2,color='b')

        plt.plot(x_data,y_data,label='expertACO',linewidth=2,color='green')
        # plt.plot(x_data,y_data1,label='ACO-8-8',linewidth=2,color='darkorange')

        # plt.plot(x_data,y_data3,label='Elite AS',linewidth=2,color='b')
        # plt.plot(x_data,y_data2,label='MMAS',linewidth=2,color='gold')
        # plt.plot(x_data,y_data5,label='PPACO',linewidth=2,color='darkcyan')
        # plt.plot(x_data,y_data4,label='LN-ACO',linewidth=2,color='r')
        # plt.plot(x_data,y_data6,label='LN-PPACO',linewidth=2,color='MediumVioletRed')

        # plt.plot(x_data,y2_data,label='Success ratio on training set',linewidth=2,color='g')
        # plt.plot(x_data,y_data,label='Success ratio on test set',linewidth=2,color='b')
        #横坐标为物品编号，纵坐标为库存量，线的名称为库存量，粗细为1，颜色为青色，标记为“o”所代表的图形（会在后面详细介绍），颜色为蓝色，大小为5


        font = {#'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        plt.xlabel("Number of iteration",font)
        #横坐标为物品编号

        plt.ylabel('The locally optimal path length',font)

        plt.rc('font')#,family='Times New Roman')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #纵坐标为各类指标
        # plt.title("商品详细信息")
        #折线图的名称

        x_major_locator=MultipleLocator(5)#以每15显示
        y_major_locator=MultipleLocator(2)#以每3显示
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)


        #图例说明
        plt.legend(prop = {'size':15},loc='upper right')#upper right lower
        plt.tight_layout()
        # #显示网格
        # plt.grid()
        plt.savefig("iterations.pdf")
        #显示图像
        # plt.show()





