#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
import torch.optim as optim
import torch
from multiprocessing import Process, Lock, Manager

import time

from pv_ant_select import PV_AntColony     
from policy_value_net_pytorch import PolicyValueNet

# from Pure_aco.pure_ant_colony import AntColony 
# from AS_aco.AS import AntColony as AntColony2


from PFACO.expertACO import AntColony

from get_map import Map
from board_game import Board, Game
from policy_value_net_pytorch import PolicyValueNet 
import os
import copy
from elo_rating_system import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.init()


class TrainPipeline():
    def __init__(self, init_model = None):
        path_MAP10 = ('./MAP10')
        pathDir = os.listdir(path_MAP10)
        self.map_path = random.choice(pathDir)
        # self.map_path = (sample_map)
        # self.map_path = ('%d.txt'%(random.choice([1,2,3,4,5,6])))
        self.map = Map(self.map_path)
        self.board = Board(self.map)

        self.pure_number_ants = 100
        self.pure_iterations = 200 

        self.number_ants = 1
        self.iterations = 1   
        self.evaporation_factor = 0.2  
        self.pheromone_adding_constant = 2   
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0         
        self.n_game = 20 
        
        self.buffer_size = 20000
        self.batch_size = 512 

        self.data_buffer = deque(maxlen=self.buffer_size)      

        self.play_batch_size = 1  
        self.epochs = 5  

        self.check_freq = 50 
        self.best_win_ratio = 0.0 
        self.episode_len = 0
        self.kl_targ = 0.02

        if init_model:
            self.policy_value_net = PolicyValueNet(self.map.width,
                                                   self.map.height,
                                                   model_file = init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.map.width,
                                                   self.map.height)
            
        self.ACO_player = PV_AntColony(self.board, self.number_ants, self.iterations, self.evaporation_factor, self.pheromone_adding_constant, self.policy_value_net.policy_value_fn)
        self.pure_aco =  AntColony(self.board, self.pure_number_ants, self.pure_iterations, self.evaporation_factor, self.pheromone_adding_constant) 
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping play_data: [(state, prob, winner_z), ..., ...]"""
        extend_data = []
        for state, porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_prob = np.rot90(np.flipud(porb.reshape(self.map.height, self.map.width)), i)
                extend_data.append((equi_state, np.flipud(equi_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_prob = np.fliplr(equi_prob)
                extend_data.append((equi_state, np.flipud(equi_prob).flatten(), winner))
        return extend_data


    def collect_selfplay_data_multi(self, data_buffer):
        data_bottle0= []
        data_bottle1 = []
        play_data0 = []

        for j in range(self.play_batch_size):
            path_MP20 = ('./MAP10')
            pathDir = os.listdir(path_MP20)
            self.map_path = random.choice(pathDir)
            self.map = Map(self.map_path)
            # # self.map_path = ('%d.txt'%(random.choice([1,2,3,4,5,6])))
            # self.map = Map(self.map_path)
            # self.board = Board(self.map)

            # self.map_path = ('%d.txt'%(random.choice([1,2,3,4,5,6])))
            new_map = Map(self.map_path)
            m = copy.deepcopy(new_map)

            self.pure_aco.board = Board(m)
            # play_data, winner = self.ACO_player.calculate_collect_data()
            play_data, winner = self.pure_aco.calculate_collect_data_pure_aco() 
            play_data = list(play_data)[:]        
            self.episode_len = len(play_data)     

            if winner < 0:  
                data_bottle0.append(play_data)
            else:
                data_bottle1.append(play_data)

        lenght = min(len(data_bottle1), len(data_bottle0) ) 
        if len(data_bottle0) != lenght: 
            for p in range(lenght):  
                tem = data_bottle0.pop()
                play_data0.append(tem)  
        else: 
            for k in range(len(data_bottle0)):  
                tem = data_bottle0.pop()
                play_data0.append(tem) 

        for l in play_data0:
            t = self.get_equi_data(l) 
            data_buffer.extend(t)


        if len(data_bottle1) != 0:
            for l in data_bottle1:     
                play_data1 = self.get_equi_data(l)  
                data_buffer.extend(play_data1)


    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size) 
        state_batch = [data[0] for data in mini_batch]
        probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs): 
            loss, entropy = self.policy_value_net.train_step(state_batch, probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)      
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))  
            if kl > self.kl_targ * 4: 
                break

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5

        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.6f},"
               "lr_multiplier:{:.6f},"
               "loss:{},"
               "entropy:{},"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy))

        return kl, self.lr_multiplier, loss, entropy

    def policy_evaluate_multi(self, win_cnt, lock):  
        winner = -3
        for i in range(self.n_game):
            print("第%d次对局" % (i + 1))
            path_MAP10 = ('./MAP10')
            pathDir = os.listdir(path_MAP10)
            self.map_path = random.choice(pathDir)
            print(self.map_path)
            new_map = Map(self.map_path)
            print(new_map.initial_node)
            print(new_map.final_node)

            m = copy.deepcopy(new_map) 
            board = Board(m)

            pure_aco = AntColony(board, self.pure_number_ants, self.pure_iterations, self.evaporation_factor,self.pheromone_adding_constant)  # 没有神经网络
            # print("pure aco start." )            
            route_play1 = pure_aco.calculate() 
            # print("pure aco end." )

            policy_value_net = PolicyValueNet(self.map.width, self.map.height, model_file='best_policy.model')
            current_paco_player = PV_AntColony(board, 1, 1, self.evaporation_factor, self.pheromone_adding_constant, policy_value_net.policy_value_fn)
            route_play2 = current_paco_player.calculate() 


            len1 = self._len(route_play1)
            len2 = self._len(route_play2)

            if route_play2[-1] != board.final_node and route_play1[-1] != board.final_node: 
                winner = 0 
            elif route_play2[-1] != board.final_node and route_play1[-1] == board.final_node:
                winner = 0 
            elif route_play1[-1] != board.final_node and route_play2[-1] == board.final_node:
                winner = 1 
            elif route_play1[-1] == board.final_node and route_play2[-1] == board.final_node and len1 > len2:
                winner = 1
            elif route_play1[-1] == board.final_node and route_play2[-1] == board.final_node and len1 < len2:
                winner = 2 
            elif route_play1[-1] == board.final_node and route_play2[-1] == board.final_node and len1 == len2:
                winner = 1 

            win_cnt[winner] += 1

            lock.acquire()

            print(route_play1, self._len(route_play1))
            # print(route_play2, self._len(route_play2))
            lock.release()

    
    def elo(self, win_cnt):
        i = Implementation()
        i.addPlayer("pure_aco-player")
        i.addPlayer("self-player")  # ,rating=None
        rating1,rating2 = i.recordMatch("pure_aco-player", "self-player", draw=win_cnt[0],win=win_cnt[1], lose=win_cnt[2])
        print(rating1, rating2)

        self.save_int2txt(rating2, 'ELO/rating.txt')
        win_ratio = 1.0 * (win_cnt[1] ) / (sum(win_cnt)) 
        catch_the_end_ratio = 1.0 * (win_cnt[1] + win_cnt[2]) / (sum(win_cnt)) 
        self.save_int2txt(win_ratio, 'ELO/win_ratio.txt')  
        self.save_int2txt(catch_the_end_ratio, 'ELO/catch_the_end_ratio.txt')
        self.save_int2txt(win_cnt, 'ELO/win_lose_draw.txt') 
        
        print("num_playouts:{},deadend:{}, win&tie: {}, lose: {}".format(
            sum(win_cnt),
            win_cnt[0], win_cnt[1], win_cnt[2]))

        print("win_ratio:{}".format(win_ratio))
        print("catch_the_end_ratio:{}".format(catch_the_end_ratio))
        self.save_int2txt(win_ratio, 'logs/win_ratio.txt')
        self.save_int2txt(catch_the_end_ratio, 'logs/catch_the_end_ratio.txt')
        # return rating
        return catch_the_end_ratio, win_ratio


    def _len(self, node_list):  
        l = len(node_list)
        sumlen = 0
        for i in range(l - 1):
            sumlen += np.sqrt(
                ((node_list[i + 1][0] - node_list[i][0]) ** 2) + (node_list[i + 1][1] - node_list[i][1]) ** 2)
        return sumlen

    def save_int2txt(self,param,path):
        with open(path,'a') as fin:     
            fin.write(str(param))
            fin.write("\n")

    def run(self):
        """run the training pipeline"""
        self.policy_value_net.save_model('./initial.model')    
        self.policy_value_net.save_model('./best_policy.model')
        print(" test code.")

        # evaluate the algorithm
        manager = Manager()
        win_cnt = manager.list([0,0,0])
        # win_cnt = defaultdict(int) 
        p_obj = []
        lock = Lock()

        for j in range(5):        
            p = Process(target = self.policy_evaluate_multi, args = (win_cnt, lock))
            p.start()
            p_obj.append(p)
        for obj in p_obj:
            obj.join()

        catch_the_end_ratio, self.best_win_ratio = self.elo(win_cnt)

        try:
            i = 0
            while True:
                i += 1
                p1_obj = []
                queue_list = Manager().list()
                for k in range(8):        
                    p1 = Process(target = self.collect_selfplay_data_multi, args = (queue_list,))
                    p1.start()
                    p1_obj.append(p1)
                        
                for obj in p1_obj:
                    obj.join()

                a = deque(queue_list)
                print("lenght of queue_list:{}".format(len(a)) )
                            
                self.data_buffer.extend(a)  

                if len(self.data_buffer) < self.buffer_size:
                    print("data not enough!")                

                print("batch i:{}, episode_len:{}".format( i+1, self.episode_len))  

                if len(self.data_buffer) > self.batch_size:
                    for k in range(8):
                        kl, lr_multiplier, loss, entropy= self.policy_update()
                        self.save_int2txt(loss, 'logs/loss.txt')
                        self.save_int2txt(kl, 'logs/kl.txt')
                        self.save_int2txt(entropy, 'logs/entropy.txt')
                        self.save_int2txt(lr_multiplier, 'logs/lr.txt')
                        self.policy_value_net.save_model('models/'+str(i)+'.model')

                if (i+1) % self.check_freq == 0:  
                    print("current self-play batch: {}".format(i+1))
                    
                    manager = Manager()
                    win_cnt = manager.list([0,0,0])
                    p_obj = []
                    lock = Lock()
                    for b in range(5):        
                        p = Process(target = self.policy_evaluate_multi, args = (win_cnt,lock))
                        p.start()
                        p_obj.append(p)
                    for obj in p_obj:
                        obj.join()
                    catch_the_end_ratio, win_ratio  = self.elo(win_cnt)
                    self.policy_value_net.save_model('./current_policy.model') 
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model('./best_policy.model')  
                        
                        if (self.best_win_ratio == 0.7 and (i+1) < 5000): 
                            self.pure_number_ants += 25
                            self.pure_iterations += 50                         
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            print('\n\rquit')

def clear_folder(path):
    """
    check whether the input path is a folder, if the folder doesn't exist, create it.
    otherwise clear all the files in the folder recursively
    """
    if os.path.isdir(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                clear_folder(c_path)
            else:
                os.remove(c_path)
    else:
        os.mkdir(path)

if __name__ == '__main__':
    setup_seed(7)
    torch.multiprocessing.set_start_method('spawn',force=True)
    training_pipeline = TrainPipeline( )
    clear_folder(r'models/')       
    clear_folder(r'logs/')
    clear_folder(r'play_data/')
    clear_folder(r'ELO/')
    training_pipeline.run()
