#!/usr/bin/env python
import numpy as np
import time
from operator import itemgetter
from function import empty_node, set_location_to_move, set_move_to_location, available_nodes,set_location_to_move,move_to_location,location_to_move
import random
from collections import deque
from scipy.spatial import distance
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

def count_turns(path):
    if len(path) < 3:
        return 0  # 如果路径点少于3个，无法形成拐角，返回0
    turns = 0
    for i in range(1, len(path) - 1):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        x3, y3 = path[i+1]
        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if cross_product != 0:
            turns += 1
    return turns

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def generate_uniform_random():
    # 生成 [0, 1) 之间的随机数
    rand_num = random.random()
    return rand_num

class AntColony(object):
    ''' Class used for handling the behaviour of the whole ant colony '''
    class Ant(object):
        ''' Class used for handling
            the ant's
            individual behaviour '''
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node = start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []  # 禁忌表？ 是否要进一步判断一个node是否被访问过？
            self.visited_nodes.append(self.start_pos)
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
    #         if self.actual_node == self.final_node: # 只考虑没找到终点的情况，TODO: dead end情况？
    #             self.final_node_reached = True
    # # FIXME:if self.actual_node == self.final_node or self.availables == [] : # TODO:当前节点的相邻节点如果是空，即陷入陷阱，结束任务
    #             # self.final_node_reached = True

        def enable_start_new_path(self):
            ''' Enables a new path search setting the final_node_reached variable to false '''
            self.final_node_reached = False

        def setup_ant(self):
            ''' Clears the list of visited nodes it stores the first one and selects the first one as initial'''
            self.visited_nodes[1:] =[]
            self.actual_node = self.start_pos


    def __init__(self, board, number_ants, iterations, evaporation_factor, pheromone_adding_constant):
        self.board = board
        # self.map = board.map
        self.no_ants = number_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor    # 挥发因子
        self.pheromone_adding_constant = pheromone_adding_constant
        self.paths = []                     # 存储已经找到的路径
        self.loop_paths = []
        self.ants = []    # 初始化各个蚂蚁的start和end
        self.best_result = []               # 存储最优路径
        self.loop_result = []
        self.best_distance = []
        self.min_len = 10000                # 用来记录最优解
        self.visited=[]
        self.iteration_len = []
        

    def create_ants(self):
        ''' Creates a list containin the total number of ants specified in the initial node '''
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.board.actual_node, self.board.final_node))
        return ants
    
    def select_next_node(self, actual_node, visited_nodes):
        ''' Randomly selects the next node to visit '''
        # Compute the total sumatory of the pheromone of each edge
        T = 1 
        total_sum = 0.0
        for edge in actual_node.edges:
            x_j = edge['NextNode'] 
            # judge turning 
            if len(self.board.visited_nodes) > 2:  
                node_last = self.board.visited_nodes[-2]    # node_(i-1)
                if (node_last[0] - actual_node[0]) ==  ( actual_node[0] - x_j[0]) and  (node_last[1] - actual_node[1]) ==  ( actual_node[1] - x_j[1]):
                    T = 2
                else:
                    T = 1  
            total_sum += (edge['Pheromone']**1) * ((edge['eta'])**3)

        # Calculate probability of each edge
        prob = 0
        edges_list = []
        p = []
        for edge in actual_node.edges:
            x_j = edge['NextNode'] 
            # judge turning 
            if len(self.board.visited_nodes) > 2:  
                node_last = self.board.visited_nodes[-2]    # node_(i-1)
                if (node_last[0] - actual_node[0]) ==  ( actual_node[0] - x_j[0]) and  (node_last[1] - actual_node[1]) ==  ( actual_node[1] - x_j[1]):
                    T = 2
                else:
                    T = 1 
            # print('total_sum :%d'%total_sum )
            prob = (edge['Pheromone']**1) * ((edge['eta'])**3) * T / (total_sum)   # 这里没有η信息，η信息都一样
            edge['Probability'] = prob
            edges_list.append(edge)
            p.append(prob)
        # action = np.random.choice(edges_list, 1, p)[0]['NextNode'] 

        q = generate_uniform_random()
        if q < 0.4:     # function 11
            max_index = np.argmax(p)
            action = edges_list[max_index]['NextNode'] 
            # # TODO：判断是否在陷入了循环
            # if len(visited_nodes) > 5:
            #     while visited_nodes[-2]== visited_nodes[-4]== action and visited_nodes[-1]== visited_nodes[-3]:
            #         p[max_index] = 0
            #         max_index = np.argmax(p)
            #         action = edges_list[max_index]['NextNode']
        else:
            action = np.random.choice(edges_list, 1, p)[0]['NextNode'] 
            # # TODO：判断是否在陷入了循环
            # if len(visited_nodes) > 5:
            #     while visited_nodes[-2]== visited_nodes[-4]== action and visited_nodes[-1]== visited_nodes[-3]:
            #         action_index = next((i for i, item in enumerate (edges_list) if item['NextNode'] == action),None) 
            #         p[action_index] = 0
            #         action = np.random.choice(edges_list, 1, p)[0]['NextNode'] 

        return action                                                                                           # np.random.choice(edges_list, 1, p)[0]['NextNode']  

        # return np.random.choice(edges_list, 1, p)[0]['NextNode']
    
    
    def pheromone_update(self, current_itertaion, iteration_best, global_best_result):  # 信息素的更新分两类: 正反馈和负反馈
        ''' Updates the pheromone level of the each of the trails and sorts the paths by lenght '''
        #先挥发， 再涂抹
        for nodelist in self.board.map.nodes_array:
            for nodes in nodelist:
                for e in nodes.edges:
                    e['Pheromone'] = (1.0 - self.evaporation_factor) * e['Pheromone']

        # self.sort_paths() 
        num = int((1/2) * len(self.paths))
        for i, path in enumerate(self.paths):
            for j, element in enumerate(path):      # j是编号，element是节点
                for edge in self.board.map.nodes_array[element[0]][element[1]].edges:
                    # 
                    if (j+2) < len(path):   # 距离终点至少还有三个节点时候
                        # 90度对角产生拐角
                        if count_turns(path[j:j+3]) != 0 and \
                            distance.euclidean(path[j], path[j+1])==distance.euclidean(path[j+1], path[j+2])==1 and \
                            np.sqrt((path[j][0]-path[j+2][0])**2 + (path[j][1]-path[j+2][1])**2)< \
                                ( np.sqrt((path[j][0]-path[j+1][0])**2 + (path[j][1]-path[j+1][1])**2) + np.sqrt((path[j+1][0] - path[j+2][0])**2 + (path[j+1][1] - path[j+2][1])**2)): # 存在直行转折，切换到45度路线
                            del path[j+1] # 只删除中间节点
                        
                        # 45度对角产生拐角
                        if count_turns(path[j:j+3]) != 0 and \
                            distance.euclidean(path[j], path[j+1])== distance.euclidean(path[j+1], path[j+2]) == np.sqrt(2) and\
                                self.board.map.nodes_array[path[j+1][0]][path[j][1]] == 1\
                                and np.sqrt((path[j][0]-path[j+2][0])**2 + (path[j][1]-path[j+2][1])**2)< \
                                ( np.sqrt((path[j][0]-path[j+1][0])**2 + (path[j][1]-path[j+1][1])**2) + np.sqrt((path[j+1][0] - path[j+2][0])**2 + (path[j+1][1] - path[j+2][1])**2)): # 存在直行转折，切换到45度路线
                            path[j+1]= (path[j+1][0],path[j][1]) # 替换中间节点为新的4号节点                    

                    elif (j+2) == len(path):  # 距离终点只有三个节点时候
                        if count_turns(path[j:-1]) != 0 and \
                            distance.euclidean(path[j], path[j+1])==1 and np.sqrt((path[j][0]-path[j+2][0])**2 + (path[j][1]-path[j+2][1])**2)< \
                                ( np.sqrt((path[j][0]-path[j+1][0])**2 + (path[j][1]-path[j+1][1])**2) + np.sqrt((path[j+1][0] - path[j+2][0])**2 + (path[j+1][1] - path[j+2][1])**2)): # 存在直行转折，切换到45度路线
                            del path[j+1] # 只删除中间节点
                        
                        # 写对角产生拐角
                        if count_turns(path[j:-1]) != 0 and \
                            distance.euclidean(path[j], path[j+1])== np.sqrt(2) and\
                                self.board.map.nodes_array[path[j+1][0]][path[j][1]] == 1\
                                and np.sqrt((path[j][0]-path[j+2][0])**2 + (path[j][1]-path[j+2][1])**2)< \
                                ( np.sqrt((path[j][0]-path[j+1][0])**2 + (path[j][1]-path[j+1][1])**2) + np.sqrt((path[j+1][0] - path[j+2][0])**2 + (path[j+1][1] - path[j+2][1])**2)): # 存在直行转折，切换到45度路线
                            path[j+1]= (path[j+1][0],path[j][1]) # 替换中间节点为新的4号节点 

                        # distance.euclidean(path[j], path[j+1])
                    elif (j+1) < len(path):         # 距离终点至少有一个节点时（包括情况距离终点有1-2个节点）
                        turning_count = count_turns(path)                        
                        if edge['NextNode'] == path[j+1]:    # 如果这条路径经过了(x_j,y_j)->(x_j+1,y_j+1)
                            edge['Pheromone'] += self.pheromone_adding_constant* global_best_result/ (self._len(path) + turning_count)      # lilly 0416
        
        elite_list = []

        while len(elite_list)<50:
            index = int(len(elite_list)/2)
            if (index+1) > len(iteration_best):
                a = iteration_best[0]
            else:
                a = iteration_best[index]
            for e in range(2):
                elite_list.append(a)
        
        # print(current_itertaion)
        # for p in iteration_best:
        #     print(self._len(p))

        for i, path in enumerate(elite_list):
            for j, element in enumerate(path):
                for edge in self.board.map.nodes_array[element[0]][element[1]].edges:
                    if (j+1) < len(path):
                        turning_count = count_turns(path)  
                        if edge['NextNode'] == path[j+1]:    # 如果这条路径经过了(x_j,y_j)->(x_j+1,y_j+1)
                            edge['Pheromone'] += self.pheromone_adding_constant* global_best_result / (self._len(path) + turning_count)        

    
    def empty_paths(self):
        ''' Empty the list of paths '''
        self.paths = []
        
    def empty_loop_paths(self):
        ''' Empty the list of paths '''
        self.loop_paths = []

    def sort_paths(self):
        ''' Sorts the paths '''
        self.paths.sort(key=self._len)
        l = self._len(self.paths[0])
        if l <= self.min_len:
            self.best_result = self.paths[0]
            self.min_len = l   # local best
        
        

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
        self.loop_result = []
        self.best_distance = []

    def calculate(self):
        ''' Carries out the process to
            get the best path '''
        # Repeat the cicle for the specified no of times
        self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
        iteration_best = []  #长度为10；  0.05*self.no_ants ;    5是number of ants 的 0.05   # global variable
        global_best_result = self.board.map.height * self.board.map.width
        
        for i in range(self.iterations):
            for ant in self.ants:
                while True:
                    ant.setup_ant()
                    t0 = time.time()                
                    while not ant.final_node_reached:# TODO：要判断一下找不到的情况
                        # Randomly selection of the node to visit；
                        node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]), ant.visited_nodes)
                        # Move ant to the next node randomly selected
                        ant.move_ant(node_to_visit)
                        # Check if solution has been reached
                        # ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；
                        t1 = time.time()
                        # print("time is %f"%(t1-t0))
                        if (t1-t0) > 30:
                            break
                    if not ant.final_node_reached:
                        print("ant did not reach final node, restarting search")
                        continue
                    else:
                        # Add the resulting path to the path list
                        self.visited.append(ant.visited_nodes)
                        self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                        # Enable the ant for a new search
                        ant.enable_start_new_path()
                        break


            self.sort_paths()       # 整理self.paths, based on length
            if self.min_len <= global_best_result:
                global_best_result = self.min_len
                iteration_best.sort(key=self._len, reverse=True)
                if len(iteration_best) > 25: 
                    if self._len(iteration_best[0])>self._len(self.best_result) :
                        iteration_best[0] = self.best_result
                else:                   
                    iteration_best.append(self.best_result)              
            iteration_best.sort(key=self._len, reverse=True)                       
            self.pheromone_update(i, iteration_best, global_best_result)

            self.iteration_len.append(self._len(self.best_result))
            self.best_result = self.paths[0]
            # Empty the list of paths
            self.empty_paths()
            # print( 'Iteration: ',i, ' lenght of the path: ', self._len(self.best_result))
        return self.best_result #, self.iteration_len #, self.visited , self.iteration_len


    # def calculate_collect_data_pure_aco(self):
    #     # Repeat the cicle for the specified no of times
    #     self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
    #     iteration_best = []  #长度为10；  0.05*self.no_ants ;    5是number of ants 的 0.05   # global variable
    #     global_best_result = self.board.map.height * self.board.map.width
       
    #     for i in range(self.iterations):
    #         for ant in self.ants:
    #             ant.setup_ant()
    #             while not ant.final_node_reached:
    #                 # Randomly selection of the node to visit；不选择重复节点
    #                 node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]))
    #                 # Move ant to the next node randomly selected
    #                 ant.move_ant(node_to_visit)
    #                 # Check if solution has been reached
    #                 ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；
    #             # Add the resulting path to the path list
    #             self.visited.append(ant.visited_nodes)
    #             self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
               
    #             # Enable the ant for a new search
    #             ant.enable_start_new_path()
            
    #         self.sort_paths()       # 整理self.paths, based on length
    #         if self.min_len <= global_best_result:
    #             global_best_result = self.min_len
    #             iteration_best.sort(key=self._len, reverse=True)
    #             if len(iteration_best) > 25: 
    #                 if self._len(iteration_best[0])>self._len(self.best_result) :
    #                     iteration_best[0] = self.best_result
    #             else:                   
    #                 iteration_best.append(self.best_result)              
    #         iteration_best.sort(key=self._len, reverse=True)                       
    #         self.pheromone_update(i, iteration_best, global_best_result)

    #         # Update the global pheromone level ,所有的ant都搜索过一遍后
    #         if self.paths:
    #             self.iteration_len.append(self._len(self.best_result))
    #             self.best_result = self.paths[0]
    #             self.empty_paths()
    #             self.empty_loop_paths()
    #         else:
    #             self.loop_result = self.loop_paths[0]
    #             self.empty_paths()
    #             self.empty_loop_paths()

    #     states, p,  = [], []
    #     probs = np.zeros(self.board.width * self.board.height)

    #     if self.best_result:
    #         # self.best_result = set_move_to_location(self.paths[0],self.board)   # 以best_path 和paths列表这些信息整理数据
    #         win_z = np.zeros(len(self.best_result))
    #         # win_z[:] = sigmoid(a)
    #         win_z[:] = 1

    #         for j in range(len(self.best_result)-1):  # 逐步分析每一步
    #             probs = np.zeros(self.board.width * self.board.height)
    #             node = self.best_result[j]     # 第j步的结点
    #             states.append(self.current_state(self.best_result, node, j))   # 整理state
    #             actions, pheromone = self.collect_p(node, j)
    #             moves = set_location_to_move(actions, self.board)
    #             # p.append(pheromone)  # 根据访问次数给出p
    #             probs[list(moves)] = pheromone
    #             p.append(probs) 
    #         print("catch end node")
    #         return zip(states, p, win_z), win_z[-1]

    #     else:
    #         print('no ant find no available action in this iteration!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #         win_z = np.zeros(len(self.loop_result ))
    #         win_z[:] = -1
    #         for j in range(len(self.best_result)-1):  # 逐步分析每一步
    #             probs = np.zeros(self.board.width * self.board.height)
    #             node = self.best_result[j]     # 第j步的结点
    #             states.append(self.current_state(self.best_result, node, j))   # 整理state
    #             actions, pheromone = self.collect_p(node, j)
    #             moves = set_location_to_move(actions, self.board)
    #             # p.append(pheromone)  # 根据访问次数给出p
    #             probs[list(moves)] = pheromone  #probs要关注一个方向
    #             p.append(probs) 
    #         return zip(states, p, win_z), win_z[-1]

# inference process
    # def calculate(self):             
    #     ''' Carries out the process to
    #         get the best path '''
    #     # Repeat the cicle for the specified no of times
    #     self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
    #     iteration_best = []  #长度为10；  0.05*self.no_ants ;    5是number of ants 的 0.05   # global variable
    #     global_best_result = self.board.map.height * self.board.map.width
        
    #     for i in range(self.iterations):
    #         for ant in self.ants:
    #             ant.setup_ant()
    #             while not ant.final_node_reached:
    #                 # Randomly selection of the node to visit；不选择重复节点
    #                 node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]))
    #                 # Move ant to the next node randomly selected
    #                 ant.move_ant(node_to_visit)
    #                 # Check if solution has been reached
    #                 ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；
    #             # Add the resulting path to the path list
    #             self.visited.append(ant.visited_nodes)
    #             self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))               
    #             # Enable the ant for a new search
    #             ant.enable_start_new_path()          
    #         self.sort_paths()       # 整理self.paths, based on length
    #         if self.min_len <= global_best_result:
    #             global_best_result = self.min_len
    #             iteration_best.sort(key=self._len, reverse=True)
    #             if len(iteration_best) > 25: 
    #                 if self._len(iteration_best[0])>self._len(self.best_result) :
    #                     iteration_best[0] = self.best_result
    #             else:                   
    #                 iteration_best.append(self.best_result)
    #         iteration_best.sort(key=self._len, reverse=True)       
    #         self.pheromone_update(i, iteration_best, global_best_result)
    #         self.iteration_len.append(self._len(self.best_result))
    #         self.best_result = self.paths[0]
    #         # Empty the list of paths
    #         self.empty_paths()
    #         # print( 'Iteration: ',i, ' lenght of the path: ', self._len(self.best_result))
    #     return self.best_result, self.visited, self.iteration_len


# multiprocess-fail
    # def calculate(self):
    #     ''' Carries out the process to get the best path '''
    #     # Repeat the cicle for the specified no of times
    #     self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
    #     with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
    #         for i in range(self.iterations):
    #             futures = [executor.submit(self.run_ant, ant, i) for ant in self.ants]
    #             for future in concurrent.futures.as_completed(futures):
    #                 pass  # 等待所有蚂蚁完成当前迭代的运行
    #             # Update the global pheromone level
    #             local_best_len = self.pheromone_update(i)
    #             self.iteration_len.append(local_best_len)
    #             self.best_result = self.paths[0]
    #             # Empty the list of paths
    #             self.empty_paths()
    #             print('Iteration: ', i, ' lenght of the path: ', self._len(self.best_result))
    #     return self.best_result, self.visited 
    # def run_ant(self, ant, iteration):
    #     ant.setup_ant()
    #     while not ant.final_node_reached:
    #         # Randomly selection of the node to visit；不选择重复节点
    #         node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]), iteration 
    #         # Move ant to the next node randomly selected
    #         ant.move_ant(node_to_visit
    #         # Check if solution has been reached
    #         ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；
    #     self.visited.append(ant.visited_nodes)
    #     # Add the resulting path to the path list
    #     self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
    #     # Enable the ant for a new search
    #     ant.enable_start_new_path()


    

    def calculate_collect_data_pure_aco(self):
        # Repeat the cicle for the specified no of times
        self.ants = self.create_ants()      # 初始化各个蚂蚁的start和end
        iteration_best = []  #长度为10；  0.05*self.no_ants ;    5是number of ants 的 0.05   # global variable
        global_best_result = 1000

        for i in range(self.iterations):
            for ant in self.ants:
                # ant.setup_ant()
                # while not ant.final_node_reached:
                #     # Randomly selection of the node to visit；不选择重复节点
                #     node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]), ant.visited_nodes)
                #     # Move ant to the next node randomly selected
                #     ant.move_ant(node_to_visit)
                #     # Check if solution has been reached
                #     # ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；
                # # Add the resulting path to the path list
                # self.visited.append(ant.visited_nodes)
                # self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
               
                # # Enable the ant for a new search
                # ant.enable_start_new_path()
                while True:
                    ant.setup_ant()
                    t0 = time.time()                
                    while not ant.final_node_reached:# TODO：要判断一下找不到的情况
                        # Randomly selection of the node to visit；
                        node_to_visit = self.select_next_node((self.board.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]), ant.visited_nodes)
                        # Move ant to the next node randomly selected
                        ant.move_ant(node_to_visit)
                        # Check if solution has been reached
                        # ant.is_final_node_reached()   # TODO:两种情况：1 找到终点；2 dead end；
                        t1 = time.time()
                        # print("time is %f"%(t1-t0))
                        if (t1-t0) > 30:
                            break
                    if not ant.final_node_reached:
                        print("ant did not reach final node, restarting search")
                        continue
                    else:
                        # Add the resulting path to the path list
                        self.visited.append(ant.visited_nodes)
                        self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                        # Enable the ant for a new search
                        ant.enable_start_new_path()
                        break



            self.sort_paths()       # 整理self.paths, based on length
            if self.min_len <= global_best_result:
                global_best_result = self.min_len
                iteration_best.sort(key=self._len, reverse=True)
                if len(iteration_best) > 25: 
                    if self._len(iteration_best[0])>self._len(self.best_result) :
                        iteration_best[0] = self.best_result
                else:                   
                    iteration_best.append(self.best_result)              
            iteration_best.sort(key=self._len, reverse=True)                       
            self.pheromone_update(i, iteration_best, global_best_result)
            
            self.iteration_len.append(self._len(self.best_result))
            self.best_result = self.paths[0]
            # Empty the list of paths
            self.empty_paths()
            # # Update the global pheromone level ,所有的ant都搜索过一遍后
            # if self.paths:
            #     self.iteration_len.append(self._len(self.best_result))
            #     self.best_result = self.paths[0]
            #     self.empty_paths()
            #     self.empty_loop_paths()
            # else:
            #     self.iteration_len.append(self._len(self.best_result))
            #     self.loop_result = self.loop_paths[0]
            #     self.empty_paths()
            #     self.empty_loop_paths()

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



    def current_state( self, path, act, n ):   # 返回当前玩家视角中的棋盘状况
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

