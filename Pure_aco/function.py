#from get_map import Map
# from ant_colony import MCTS
from plot_picture import plot_picture
import copy

# p: is the pheromone's evaporation rate, [0-1]
# Q: is the pheromone adding constant.
import numpy as np
def tuple_to_list(route):
    list_route =[]
    for i in route:
        a = list(i)
        list_route.append(a)
    return list_route

def list_to_tuple(route):
    tuple_route =[]
    for i in route:
        a = tuple(i)
        tuple_route.append(a)
    return tuple_route

def empty_node(board):
    avaliables = []
    move = board.actual_node  # 当前所在的位置
    map_empty = list_to_tuple(board.map_avaliable)  # 地图中可行的点
    # player_empty = tuple_to_list(board.map.nodes_array[int(move[0])][int(move[1])].edges) # 当前位置本来可以走的点
    player_empty = board.map.nodes_array[int(move[0])][int(move[1])].avaliable # 当前位置本来可以走的点
    for i in player_empty:
        if i in map_empty:
            avaliables.append(i)  # 如果当前位置周围还有没有走过的点，那么继续往下rollout
    return avaliables      # 如果avaliables为空，证明卡死；如果不为空，证明还可以继续往下走

