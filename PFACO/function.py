#from get_map import Map
# from ant_colony import MCTS
from plot_picture import plot_picture
import copy

# p: is the pheromone's evaporation rate, [0-1]
# Q: is the pheromone adding constant.
import numpy as np
# def tuple_to_list(route):
#     list_route =[]
#     for i in route:
#         a = list(i)
#         list_route.append(a)
#     return list_route
#
# def list_to_tuple(route):
#     tuple_route =[]
#     for i in route:
#         a = tuple(i)
#         tuple_route.append(a)
#     return tuple_route
#
# def empty_node(board):
#     availables = []
#     move = board.actual_node  # 当前所在的位置
#     map_empty = list_to_tuple(board.map_available)  # 地图中可行的点
#     # player_empty = tuple_to_list(board.map.nodes_array[int(move[0])][int(move[1])].edges) # 当前位置本来可以走的点
#     player_empty = board.map.nodes_array[int(move[0])][int(move[1])].available # 当前位置本来可以走的点
#     for i in player_empty:
#         if i in map_empty:
#             availables.append(i)  # 如果当前位置周围还有没有走过的点，那么继续往下rollout
#     return availables      # 如果availables为空，证明卡死；如果不为空，证明还可以继续往下走

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
    availables = []
    move = board.actual_node  # 当前所在的位置
    map_empty = list_to_tuple(board.map_availables)  # 地图中可行的点
    # player_empty = tuple_to_list(board.map.nodes_array[int(move[0])][int(move[1])].edges) # 当前位置本来可以走的点
    player_empty = board.map.nodes_array[int(move[0])][int(move[1])].available # 当前位置本来可以走的点
    for i in player_empty:
        if i in map_empty:
            availables.append(i)  # 如果当前位置周围还有没有走过的点，那么继续往下rollout
    return availables      # 如果availables为空，证明卡死；如果不为空，证明还可以继续往下走


# ant在搜索过程中是否产生了loop，基于每个结点只访问一次的前提
def available_nodes(ant,board):
    availables = []     # 是否有未访问过的节点
    visited = ant.visited_nodes  # ant已经访问过的点
    player_empty =  board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])].available # 当前位置本来可以走的点
    for i in player_empty:
        if i not in visited:
            availables.append(i)  # 如果当前位置周围还有没有走过的点，那么继续往下rollout
    ant.availables = availables   # 更新ant的可行空间
    if availables == []:
        return True   # 存在loop
    else:
        return False     # 如果availables为空，证明卡死；如果不为空，证明还可以继续往下走


def set_location_to_move(list, board):   # 坐(h,w)标  转  位置
    m = []
    for i in list:
        if len(i) != 2:
            return -1
        h = i[0]
        w = i[1]
        move = h * board.width + w
        if move not in range(board.width * board.height):
            return -1
        m.append(move)
    return m

def set_move_to_location(list, board):   #  转 (h,w)
    m=[]
    for i in list:
        h = i // board.width
        w = i % board.width
        m.append((h,w))
    return m

def location_to_move(location,board):   # 坐标 转 位置
    if len(location) != 2:
        return -1
    h = location[0]
    w = location[1]
    move = h * board.map.width + w
    if move not in range(board.map.width * board.map.height):
        return -1
    return move

def move_to_location( move,board):   # 位置 转 坐标
    """
    3*3 board's moves like:
    6 7 8     0 1 2
    3 4 5     3 4 5
    0 1 2     6 7 8
    and move 5's location is (1,2)
    """
    h = move // board.map.width
    w = move % board.map.width
    return [h, w]
