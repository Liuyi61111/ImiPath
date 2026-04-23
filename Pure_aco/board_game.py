#!/usr/bin/env python

import numpy as np
from operator import itemgetter
import copy

from function import tuple_to_list,list_to_tuple,empty_node

class Board(object):
    ''' Class used for handling the behaviour of the whole ant colony '''
    def __init__(self, in_map):
        self.map = in_map
        self.map_avaliable = self.map.avaliable_nodes
        self.start_pos = self.map.initial_node
        self.actual_node = self.map.initial_node
        self.final_node = self.map.final_node
        self.visited_nodes = []
        
        self.visited_nodes.append(self.actual_node)
        self.availables = self.map.nodes_array[int(self.actual_node[0])][int(self.actual_node[1])].avaliable
        self.final_node_reached = False
        self.loop_end = False

    def is_final_node_reached(self):  # play1 和 play2 进行交换；删除占据的位置；记录最后一步停到了哪里
        if self.actual_node == self.final_node:
            self.final_node_reached = True
        else:
            self.final_node_reached = False

    def game_end(self):  # 检查游戏是否结束了
        """Check whether the game is ended or not"""
        if self.final_node_reached or self.loop_end:
            return True
        else:
            return False

    def move_player(self, node_visit):
        ''' Moves ant to the selected node '''
        self.actual_node = node_visit
        self.visited_nodes.append(node_visit)  # 记录走过的路径
        self.map_avaliable.remove(node_visit)  # 当前棋盘中空的位置集
    

    def current_state(self):   # 返回当前玩家视角中的棋盘状况
        """return the board state from the perspective of the current player.
        state shape: 4*width*height"""
        square_state = np.zeros((4, self.width, self.height))  # 三维的 width X height的特征图；{（4，6，6）}
        # 第一维存储curr_player下棋信息，落子的地方置为1； 第二维存储oppo_player下子信息，落子位置为1；第三维存储最后一步的位置信息；
        if self.visited_nodes:  # 如果states不为空，执行下面的；为空，跳过
            square_state[0][self.start_pos[0], self.start_pos[1]] = 1.0     # star
            square_state[1][self.final_node[0], self.final_node[1]] = 1.0       # end
            # square_state[2][self.actual_node[0], self.actual_node[1]] = 1.0     # 上一次位移的position
            for i in self.visited_nodes:   # 之前走的步数所有步
                square_state[3][i[0], i[1]] = 1

    def do_move(self, node_visit):
        ''' Moves ant to the selected node '''
        if node_visit == self.final_node:       # 已经到达终点，更新信息
            self.actual_node = node_visit
            self.visited_nodes.append(node_visit)
            self.map_avaliable.remove(node_visit)
            self.final_node_reached = True
        else:
            # 1020 avaliable已经写出，这里是否会重复选择？
            if node_visit not in self.visited_nodes:  # 正常判断流程，防止每个node访问两次
                self.actual_node = node_visit
                self.visited_nodes.append(node_visit)  # 记录走过的路径
                self.map_avaliable.remove(node_visit)  # 当前棋盘中空的位置集
                self.availables = empty_node(self)
                # 更新avaliable nodes，在选择了这个node之后；如果avaliables为空，表示执行这个动作之后，陷入了死角且 没有找到终点呐
                if self.availables == []:
                    self.loop_end = True

class Game(object):   # 打印玩家每下一步棋子后棋局的装填； 两个玩家玩游戏；自己玩游戏
    """game server"""
    def __init__(self, board, **kwargs):
        self.board = board

    def start_self_play(self, aco_player, return_pher):
        """ start a self-play path planning task using a ACO player and store the self-play data: (state, mcts_probs, z) for training """
        while True:     # 每次迭代，找到一个确定的动作
            move = aco_player.get_action(self.board, return_pher)     # 一个确定要执行的动作； 所有可行nodes的概率，形状是1*（w*h）
            # if self.board.loop_end == False:
            # store the data
            # states.append(self.board.current_state())       # 从当前玩家的角度返回棋盘状态 state shape: 4*width*height
            # aco_pheromone.append(move_phers)
            self.board.do_move(move)        # 在执行确定的动作之后，判断是陷入死角还是到达终点；更新全局地图中的avaliable nodes信息
            # 整理数据，判断输赢，决定是否还要 go on simulation
            end = self.board.game_end()
            if end:
                if self.board.final_node_reached:       # 到达终点结束
                    print("catch end node")
                elif self.board.loop_end:
                    print("stop loop")
                return self.board.visited_nodes

# 前后不衔接 可行位置不对应，建立模拟和实施之间的联系



















