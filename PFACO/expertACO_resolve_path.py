#!/usr/bin/env python
import numpy as np

from get_map import Map
from board_game import Board, Game
import time
from expertACO import AntColony

def _len(node_list):
    l = len(node_list)
    sumlen = 0
    for i in range(l - 1):
        sumlen += np.sqrt(
            ((node_list[i + 1][0] - node_list[i][0]) ** 2) + (node_list[i + 1][1] - node_list[i][1]) ** 2)
    return sumlen

# p: is the pheromone's evaporation rate, [0-1]
# Q: is the pheromone adding constant.

if __name__ == '__main__':
    iterations = 10
    aco_params = {"rho": 0.2, "beta": 1, "Q": 2, "ant_no": 10}
    T_0 = 1             # the inital turn factor constant


    # map_path = ('mapFig17.txt')  # 地图文件
    # map_path = ('mapFig21.txt')  # 地图文件
    # map_path = ('mapFig25.txt')  # 地图文件
    # map_path = ('8.txt')  # 地图文件
    map_path = ('/mnt/p-aco_map10_20240527/MAP10/AlternatingGaps1_3.txt')  # 地图文件
    # start_node = (0,0)
    # terminal_node = (19,19)
    map = Map(map_path)
    board = Board(map)

    ACO_player = AntColony(board, aco_params['ant_no'], iterations, aco_params['rho'], aco_params['Q'])

# 开始计时
    t0 = time.time()
    # path, visited_area, iteration_len = ACO_player.calculate()
    path  = ACO_player.calculate()
    # 记时
    t1 = time.time()
    time_loss = t1-t0
    print('time loss:{}s'.format(time_loss))
# 生成路径
    print(path)
    print(_len(path))
# 画图
    # board.map.represent_path(path)
    # # board.map.plot_dis_iter(visited_area)
    # board.map.plot_iteration_len(iteration_len, iterations)

























