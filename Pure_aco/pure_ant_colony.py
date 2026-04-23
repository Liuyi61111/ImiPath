#!/usr/bin/env python
import numpy as np
import time
from operator import itemgetter
from function import empty_node,set_location_to_move, set_move_to_location, available_nodes,set_location_to_move,move_to_location,location_to_move


def softmax(x):
    if len(x) == 0:
        raise ValueError("Input array is empty")
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class AntColony(object):
    ''' Class used for handling the behaviour of the whole ant colony '''
    class Ant(object):
        ''' Class used for handling
            the ant's
            individual behaviour '''
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node= start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []   # 禁忌表？ 是否要进一步判断一个node是否被访问过？
            self.final_node_reached = False
            self.remember_visited_node(start_node_pos)

        def move_ant(self, node_to_visit):
            ''' Moves ant to the selected node '''
            self.actual_node = node_to_visit
            self.remember_visited_node(node_to_visit)
            if self.actual_node == self.final_node: 
                self.final_node_reached = True

        def remember_visited_node(self, node_pos):
            ''' Appends the visited node to
                the list of visited nodes '''
            self.visited_nodes.append(node_pos)

        def get_visited_nodes(self):
            ''' Returns the list of visited nodes '''
            return self.visited_nodes

    #     def is_final_node_reached(self):
    #         ''' Checks if the ant has reached the final destination '''
    #         if self.actual_node == self.final_node : # 只考虑没找到终点的情况，TODO: dead end情况？
    #             self.final_node_reached = True
    # # FIXME:if self.actual_node == self.final_node or self.availables == [] : # TODO:当前节点的相邻节点如果是空，即陷入陷阱，结束任务
    #             self.final_node_reached = True

        def enable_start_new_path(self):
            ''' Enables a new path search setting the final_node_reached variable to false '''
            self.final_node_reached = False

        def setup_ant(self):
            ''' Clears the list of visited nodes it stores the first one and selects the first one as initial'''
            self.visited_nodes[1:] =[]
            self.actual_node= self.start_pos


    def __init__(self, board, no_ants, iterations, evaporation_factor, pheromone_adding_constant):
        self.board = board
        # self.map = board.map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor    # 挥发因子
        self.pheromone_adding_constant = pheromone_adding_constant
        self.paths = []                     # 存储已经找到的路径
        self.loop_paths = []
        self.ants = []    # 初始化各个蚂蚁的start和end
        self.best_result = []               # 存储最优路径
        self.loop_result = []
        self.visited = []

    def create_ants(self):
        ''' Creates a list containin the total number of ants specified in the initial node '''
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.board.actual_node, self.board.final_node))
        return ants
    
    def select_next_node(self, actual_node):
        ''' Randomly selects the next node to visit '''
        # Compute the total sumatory of the pheromone of each edge
        total_sum = 0.0
        for edge in actual_node.edges:
            total_sum += (edge['Pheromone']**0.1)*(edge['eta']**3)
        
        # Calculate probability of each edge
        prob = 0
        edges_list = []
        p = []
        for edge in actual_node.edges:
            prob = (edge['Pheromone']**0.1)*(edge['eta']**3) / total_sum   # 这里没有η信息，η信息都一样
            edge['Probability'] = prob
            edges_list.append(edge)
            p.append(prob)
# 0630是否过于均匀了        
        probs = softmax(p)   
        # Clear probability values 初始化信息素表
        # for edge in actual_node.edges:  
        #     edge['Probability'] = 0.0
        return np.random.choice(edges_list, 1, p = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))[0]['NextNode']
    
    
    def pheromone_update(self):  # 信息素的更新分两类: 正反馈和负反馈
        ''' Updates the pheromone level of the each of the trails and sorts the paths by lenght '''
        #先挥发， 再涂抹
        for list in self.board.map.nodes_array:
            for nodes in list:
                for e in nodes.edges:
                    e['Pheromone'] = (1.0 - self.evaporation_factor) * e['Pheromone']
        self.sort_paths()
        # self.sort_loop_paths()
        for i, path in enumerate(self.paths):
            # path = set_move_to_location(path, self.board)
            for j, element in enumerate(path):
                # e = move_to_location(element,self.board)
                for edge in self.board.map.nodes_array[element[0]][element[1]].edges:
                    if (j+1) < len(path):
                        if edge['NextNode'] == path[j+1]:    # 如果这条路径经过了(x_j,y_j)->(x_j+1,y_j+1)
                            edge['Pheromone'] += self.pheromone_adding_constant / self._len(path)

    # def pheromone_update(self):
    #     ''' Updates the pheromone level of the each of the trails and sorts the paths by lenght '''
    #     # Sort the list according to the size of the lists
    #     self.sort_paths()
    #     for i, path in enumerate(self.paths):
    #         for j, element in enumerate(path):
    #             for edge in self.board.map.nodes_array[element[0]][element[1]].edges:  # 结点element的available nodes
    #                 if (j+1) < len(path):
    #                     if edge['NextNode'] == path[j+1]:    # 如果这条路径经过了(x_j,y_j)->(x_j+1,y_j+1)
    #                         # edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone'] + \
    #                         #                       self.pheromone_adding_constant /float(len(path))
    #                         edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone'] + \
    #                                             self.pheromone_adding_constant / self._len(path)
    #                         # if edge['Pheromone'] > 1:
    #                         #     edge['Pheromone'] = 1
    #                     else:
    #                         edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone']
    #                         # if edge['Pheromone'] > 1:
    #                         #     edge['Pheromone'] = 1

    def empty_paths(self):
        ''' Empty the list of paths '''
        self.paths[:]=[]
        
    def empty_loop_paths(self):
        ''' Empty the list of paths '''
        self.loop_paths[:]=[]

    def sort_paths(self):
        ''' Sorts the paths '''
        self.paths.sort(key=self._len)

    def add_to_path_results(self, in_path):
        ''' Appends the path to
            the results path list'''
        self.paths.append(in_path)

    def get_coincidence_indices(self, lst, element):
        ''' Gets the indices of the coincidences
            of elements in the path '''
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)

    def delete_loops(self, in_path):
        ''' Checks if there is a loop in the
            resulting path and deletes it '''
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            # reverse the list to delete elements from back to front of the list
            coincidences.reverse()
            for i, coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []

        return res_path
    
    def initial_ant(self):
        self.paths = []                     # 存储已经找到的路径
        self.ants = []    # 初始化各个蚂蚁的start和end
        self.best_result = []
        self.loop_result =[]

    def calculate(self):
        ''' Carries out the process to
            get the best path '''
        # Repeat the cicle for the specified no of times
        self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
        for i in range(self.iterations):
            for ant in self.ants:
                ant.setup_ant()
                while not ant.final_node_reached:
                    # Randomly selection of the node to visit；不选择重复节点
                    node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]))

                    # Move ant to the next node randomly selected
                    ant.move_ant(node_to_visit)

                    # Check if solution has been reached
                    # ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；

                # Add the resulting path to the path list
                # self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))

                self.visited.append(ant.visited_nodes)
                # Enable the ant for a new search
                ant.enable_start_new_path()

            # Update the global pheromone level
            self.pheromone_update()
            self.best_result = self.paths[0]
            # Empty the list of paths
            self.empty_paths()
            # print( 'Iteration: ',i, ' lenght of the path: ', len(self.best_result))
        return self.best_result         #, self.visited

            
    def select_next_node_pureaco_collect(self, board, actual_node, state, ant):
        ''' Randomly selects the next node to visit '''
        ant.loop = available_nodes(ant, board ) 
        # Compute the total sumatory of the pheromone of each edge
        if ant.loop == False:  # 存在可选位移
            total_sum = 0.0
            for edge in actual_node.edges:
                total_sum += edge['Pheromone']*(edge['eta']**3)
            
            # Calculate probability of each edge
            prob = 0
            edges_list = []
            p = []
            for edge in actual_node.edges:
                prob = edge['Pheromone']*(edge['eta']**3) / total_sum   # 这里没有η信息，η信息都一样
                edge['Probability'] = prob
                edges_list.append(edge)
                p.append(prob)
            
            probs = softmax(p)
            # Clear probability values 初始化信息素表

            for edge in actual_node.edges:  
                # i = edge
                edge['Probability'] = 0.0
            return np.random.choice(edges_list, 1, p)[0]['NextNode'] # np.random.choice(edges_list, 1, p = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))[0]['NextNode']
        else:
            return False

    def calculate_collect_data_pure_aco(self):
        # Repeat the cicle for the specified no of times
        self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
        for i in range(self.iterations):
            for ant in self.ants:
                ant.setup_ant()
                while not ant.final_node_reached:
                    # Randomly selection of the node to visit；不选择重复节点
                    node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]))

                    # Move ant to the next node randomly selected
                    ant.move_ant(node_to_visit)

                    # Check if solution has been reached
                    # ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；

                # Add the resulting path to the path list
                # self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
               
                # Enable the ant for a new search
                ant.enable_start_new_path()

            # Update the global pheromone level ,所有的ant都搜索过一遍后
            self.pheromone_update()
            # Update the global pheromone level ,所有的ant都搜索过一遍后
            if self.paths:
                self.best_result = self.paths[0]
                self.empty_paths()
                self.empty_loop_paths()
            else:
                self.loop_result = self.loop_paths[0]
                self.empty_paths()
                self.empty_loop_paths()

        states, p,  = [], []
        probs = np.zeros(self.board.width * self.board.height)

        if self.best_result:
            # self.best_result = set_move_to_location(self.paths[0],self.board)   # 以best_path 和paths列表这些信息整理数据
            win_z = np.zeros(len(self.best_result))
            # win_z[:] = sigmoid(a)
            win_z[:] = 1

            for j in range(len(self.best_result)-1):  # 逐步分析每一步
                probs = np.zeros(self.board.width * self.board.height)
                node = self.best_result[j]     # 第j步的结点
                states.append(self.current_state(self.best_result, node, j))   # 整理state
                actions, pheromone = self.collect_p(node, j)
                moves = set_location_to_move(actions, self.board)
                # p.append(pheromone)  # 根据访问次数给出p
                probs[list(moves)] = pheromone
                p.append(probs) 
            print("catch end node")
            return zip(states, p, win_z), win_z[-1]

        else:
            print('no ant find no available action in this iteration!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # self.best_result = set_move_to_location(self.loop_paths[-1], self.board)
            win_z = np.zeros(len(self.loop_result ))
            win_z[:] = -1

            for j in range(len(self.best_result)-1):  # 逐步分析每一步
                probs = np.zeros(self.board.width * self.board.height)
                node = self.best_result[j]     # 第j步的结点
                states.append(self.current_state(self.best_result, node, j))   # 整理state
                actions, pheromone = self.collect_p(node, j)
                moves = set_location_to_move(actions, self.board)
                # p.append(pheromone)  # 根据访问次数给出p
                probs[list(moves)] = pheromone  #probs要关注一个方向
                p.append(probs) 
            return zip(states, p, win_z), win_z[-1]


    def _len(self, node_list):
        l = len(node_list)
        sumlen = 0
        for i in range(l - 1):
            sumlen += np.sqrt(
                ((node_list[i + 1][0] - node_list[i][0]) ** 2) + (node_list[i + 1][1] - node_list[i][1]) ** 2)
        return sumlen
    
    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree."""
        self.board.actual_node = last_move       # 更新board的ant1玩家信息里的开始结点和可行结点
        self.board.availables = empty_node(self.board)        # 当前nodes，除去访问过一次的nodes，剩下的还没访问过的nodes有哪些



    def current_state( self, path, act,n ):   # 返回当前玩家视角中的棋盘状况
        """return the board state from the perspective of the current player.
        state shape: 4*width*height"""
        square_state = np.zeros((4, self.board.width, self.board.height))  # 三维的 width X height的特征图；{（4，6，6）}
        # 第一维存储curr_player下棋信息，落子的地方置为1； 第二维存储oppo_player下子信息，落子位置为1；第三维存储最后一步的位置信息；
         # 如果states不为空，执行下面的；为空，跳过
        square_state[0][self.board.start_pos[0], self.board.start_pos[1]] = 1.0     # star
        square_state[1][self.board.final_node[0], self.board.final_node[1]] = 1.0       # end
        
        for i in path[0:n+1:]:   # 之前走的步数所有步
            square_state[2][i[0], i[1]] = 1

        for i in self.board.map.barrier:
            square_state[3][i[0],i[1]] = 1
        return square_state 


    def collect_p(self, action, n):
        a = []  # 需要被统计的位移
        # visit_n = [0]*len(a)
        available_actions = self.board.map.nodes_array[int(action[0])][int(action[1])].available #本可以走的所以位置
        availabl_actions_pheromone = self.board.map.nodes_array[int(action[0])][int(action[1])].edges
    
        w = self.best_result[0:n:]
        for i in available_actions:
            if i not in self.best_result[0:n:]:
                a.append(i)    # a 实际可以选择的节点
        a1 = set_location_to_move(a, self.board) #位置
        # 信息素
        visit_n = []

        for j in a:
            for k in availabl_actions_pheromone:
                if k['NextNode'] == j:
                    m = k['Pheromone']
                    visit_n.append(m)
        # act_probs = softmax(np.array(tuple(visit_n)) + 1e-10)
        act_probs = []
        # x = np.array(visit_n)       # 0630拉大数据之间的差距
        # act_probs = x/x.sum()
        act_probs = softmax(np.array(visit_n) + 1e-10)
        return tuple(a), act_probs