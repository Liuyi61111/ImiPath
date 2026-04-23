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
mpl.use('Agg')
import matplotlib.pyplot as plt

class Map:
    ''' Class used for handling the information provided by the input map '''
    # 每一个node有3个属性：1.位置（x，y）；2.edges; 3.spec={S,F,E,O};
                                   # edges: 1.可以取的点；2.phermone；3.probability
    class Nodes:
        ''' Class for representing the nodes used by the ACO algorithm '''
        def __init__(self, row, col, in_map, spec):
            self.node_pos = (row, col)
            self.edges, self.avaliable = self.compute_edges(in_map)
            self.spec = spec

        def compute_edges(self, map_arr):
            ''' class that returns the edges connected to each node '''
            imax = map_arr.shape[0]  # map_arr的行数
            jmax = map_arr.shape[1]  # map_arr的列数
            # pheromone = []
            edges = []
            available = []
            if map_arr[self.node_pos[0]][self.node_pos[1]] == 1:
                for dj in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        newi = self.node_pos[0] + di
                        newj = self.node_pos[1] + dj
                        if (dj == 0 and di == 0):
                            continue
                        if (newj >= 0 and newj < jmax and newi >= 0 and newi < imax):
                            if map_arr[newi][newj] == 1:  # 无障碍物 且 一步可达
                                dis = np.sqrt((self.node_pos[0]-newi)**2+(self.node_pos[1]-newj)**2)
                                available.append((newi, newj))
                                edges.append({'NextNode': (newi,newj), 'Pheromone': 1.0, 'eta': 1/(dis), 'Probability': 0.0})
            return edges,available


    def __init__(self, map_name):
        self.in_map = self._read_map(map_name)  # in_map类型为str
        self.occupancy_map = self._map_2_occupancy_map()  # 输入in_map 将地图转化成int matrix
        self.avaliable_nodes = self.add_avaliable_nodes()  # 地图上所有空的点

        self.initial_node = tuple(random.sample(self.avaliable_nodes, 1)[0])
        # self.initial_node = (0,2)
        self.avaliable_nodes.remove(self.initial_node)
        self.final_node =  tuple(random.sample(self.avaliable_nodes, 1)[0])
        # self.final_node = (2,3)
        # self.barrier = self.add_obs_nodes()  # 添加障碍物
        self.nodes_array = self._create_nodes()
        self.height = self.occupancy_map.shape[0]
        self.width = self.occupancy_map.shape[1]

        # # A*用的
        # self.x_range = self.height +2 # 长 由于还要加边框
        # self.y_range = self.width +2 # 高
        # self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),(1, 0), (1, -1), (0, -1), (-1, -1)]
        # self.obs = self.obs_map()  # 转换后的地图数据


    def add_avaliable_nodes(self):
        avaliable_nodes = []
        nodes = []
        points = np.where(self.in_map == 'E')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            avaliable_nodes.append(nodes)
            nodes = []
        return list_to_tuple(avaliable_nodes)

    def _create_nodes(self):
        ''' Create nodes out of the initial map '''
        return [[self.Nodes(i, j, self.occupancy_map, self.in_map[i][j]) for j in
                 range(self.in_map.shape[1])] for i in range(self.in_map.shape[0])]
    # 读取map文件
    def _read_map(self, map_name):
        ''' Reads data from an input map txt file'''
        in_map = np.loadtxt('/root/PV_ACO_TIME_STEP/maps/' + map_name, dtype=str )
        return in_map

    def add_initial_node(self):
        initial_node = []
        nodes = []
        points = np.where(self.in_map == 'S')
        # A = len(points[1])
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            initial_node.append(nodes)
            nodes = []
        return initial_node

    def add_final_node(self):
        final_node = []
        nodes = []
        points = np.where(self.in_map == 'F')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            final_node.append(nodes)
            nodes = []
        return final_node

    def add_obs_nodes(self):
        obs_nodes = []
        nodes = []
        points = np.where(self.in_map == 'O')
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
        return map_arr.astype(np.int)  # astype 函数用于array中数值类型转换

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
        plt.savefig("Pure_aco/pureaco.png") 
        plt.show()
        plt.close()

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


