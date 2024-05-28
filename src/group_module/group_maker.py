import random
import pickle
import os
import numpy as np
import copy
np.random.seed(2024)
def banlance(D_distribution, clients_data_distribution):
    D = copy.deepcopy(D_distribution)
    for i in range(len(D)):
        D[i] += clients_data_distribution[i]
    sum_sample = sum(D)
    mathcal_Q = 0
    for i in range(len(D)):
        mathcal_Q += (D[i] / sum_sample - 1 / abs(len(D))) ** 2
    return mathcal_Q

class Group_Maker():
    def __init__(self, clients_data_distribution, clients_representer, options):
        self.clients_data_distribution = clients_data_distribution
        self.clients_representer = clients_representer
        self.options = options
        if self.options['method_division'] == 1:
            self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groupfile', 'dirichlet')
        else:
            self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groupfile', 'pathology')
        os.makedirs(self.script_dir, exist_ok=True)
        if self.options['is_real_class'] == True:
            self.suffix = 'dn_{}_noc_{}_dir_{}_'.format(
                                            self.options['dataset_name'],
                                            self.options['num_of_clients'],
                                            self.options['dirichlet'],
                                            )
        else:
            self.suffix = 'dn_{}_noc_{}_dir_{}_{}'.format(
                                            self.options['dataset_name'],
                                            self.options['num_of_clients'],
                                            self.options['dirichlet'],
                                            "estimate"
                                            )
        self.load_or_generate_group()

    def get_group(self, is_real_class):
        if is_real_class:
            a = []
            G = {}
            k = 0
            clients_index = [_ for _ in range(self.options['num_of_clients'])]
            while len(clients_index) != 0:
                group = []
                D_distribution = [0 for _ in range(len(self.clients_data_distribution[0]))]
                g1 = random.choice(clients_index)
                for i in range(len(D_distribution)):
                  D_distribution[i] += self.clients_data_distribution[g1][i]
                group.append(g1)
                clients_index.remove(g1)
                # while (len(group) < (self.options['num_of_clients'] / 10)) and (sum(D_distribution) < 800):
                while (len(group) < (self.options['num_of_clients'] / 10)):
                    more_banlance_client = clients_index[0]
                    for j in clients_index:
                        # 将j和group组合, 然后判断最小
                        # If group + 第一个客户端的距离 比 group + 第二个客户端的距离 更平衡，那么就选择第一个客户端;
                        if banlance(D_distribution, self.clients_data_distribution[j]) \
                                < banlance(D_distribution, self.clients_data_distribution[more_banlance_client]):
                            more_banlance_client = j
                    group.append(more_banlance_client)
                    for i in range(len(D_distribution)):
                        D_distribution[i] += self.clients_data_distribution[more_banlance_client][i]
                    clients_index.remove(more_banlance_client)
                a.append(D_distribution)
                G[k] = group
                k += 1
            return G
        else:
            a = []
            G = {}
            k = 0
            clients_index = [_ for _ in range(self.options['num_of_clients'])]
            while len(clients_index) != 0:
                group = []
                D_distribution = [0 for _ in range(len(self.clients_representer[0]))]
                g1 = random.choice(clients_index)
                D_distribution += self.clients_representer[g1]
                group.append(g1)
                clients_index.remove(g1)
                # while (len(group) < (self.options['num_of_clients'] / 10)) and (sum(D_distribution) < 800):
                while (len(group) < (self.options['num_of_clients'] / 10)):
                    more_banlance_client = clients_index[0]
                    for j in clients_index:
                        # 将j和group组合, 然后判断最小
                        # If group + 第一个客户端的距离 比 group + 第二个客户端的距离 更平衡，那么就选择第一个客户端;
                        if banlance(D_distribution + self.clients_representer[j]) \
                                > banlance(D_distribution + self.clients_representer[more_banlance_client]):
                            more_banlance_client = j
                    group.append(more_banlance_client)
                    D_distribution += self.clients_representer[more_banlance_client]
                    clients_index.remove(more_banlance_client)
                a.append(D_distribution)
                G[k] = group
                print(len(group))
                print(D_distribution)
                k += 1
            return G

    def load_or_generate_group(self, filename='group.pkl'):
        filename = self.suffix + filename
        print("self.script_dir", self.script_dir)
        try:
            # 尝试加载已保存的文件
            self.load_group(os.path.join(self.script_dir, filename))
            print("Group loaded from file.")
        except FileNotFoundError:
            # 如果文件不存在，生成新的group并保存
            self.G = self.get_group(self.options['is_real_class'])
            self.save_group(os.path.join(self.script_dir, filename))
            print("New group generated and saved to file.")


    def save_group(self, filename='group.pkl'):
 
        with open(os.path.join(self.script_dir, filename), 'wb') as file:
            pickle.dump(self.G, file)

    def load_group(self, filename='group.pkl'):

        with open(os.path.join(self.script_dir, filename), 'rb') as file:
            self.G = pickle.load(file)