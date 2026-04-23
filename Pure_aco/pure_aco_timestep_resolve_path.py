#!/usr/bin/env python

from get_map_expert import Map
from board_game import Board, Game
import time
from pure_ant_colony import AntColony


# p: is the pheromone's evaporation rate, [0-1]
# Q: is the pheromone adding constant.

if __name__ == '__main__':
    iterations = 10
    Q = 1.0
    display = 1
    aco_params = {"rho": 0.3, "beta": 1, "Q": 1, "ant_no": 10} 

    return_pher = 0 # 是否显示pher分布
    # map_path = ('test_with_MCTS.txt')  # 地图文件
    # map_path = ('6X6.txt')  # 地图文件
    # map_path = ('6X6O.txt')  # 地图文件
    map_path = ('10X10.txt')  # 地图文件
    # map_path = ('16X16.txt')  # 地图文件
    # map_path = ('32X32.txt')  # 地图文件
    # map_path = ('map2.txt')  # 地图文件
    # map_path = ('small8X8.txt')  # 地图文件

    map = Map(map_path)

    board = Board(map)
    # game = Game(board)

    ACO_player = AntColony(board, aco_params['ant_no'], iterations, aco_params['rho'], aco_params['Q'])

# 开始计时
    t0 = time.time()
    path= ACO_player.calculate()
# 生成路径
    print(path)
    print(len(path))

# 画图
    # board.map.represent_path(path)

# 记时
    t1 = time.time()
    time_loss = t1-t0
    print('time loss:{}s'.format(time_loss))























