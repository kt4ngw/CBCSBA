from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.group_module.client_evaluator import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src.fed_client.fedgroup_client import FedGroupClient
import random
from src.group_module.group_maker import Group_Maker
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import numpy as np
import copy
from tqdm import tqdm
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
class Proposed(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'])
        super(Proposed, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model, self.optimizer,)
        self.options['model_size'] = model.get_model_size()
        group_method = {}
        clients_representer = None
        self.client_evaluator = None
        self.clients_data_distribution = self.combine_clients_data_distribution(self.clients)
        print(self.clients_data_distribution)
        # clients_representer = self.get_client_representer(self.clients)
        self.group = Group_Maker(self.clients_data_distribution, clients_representer, self.options)
        self.group_num = len(self.group.G)

    def combine_clients_data_distribution(self, clients):
        clients_data_distribution = []
        for client in clients:
            clients_data_distribution.append(client.class_distribution)

        return clients_data_distribution

    def train(self):

        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))

        # Fetch latest flat model parameter
        # self.latest_global_model = self.get_model_parameters()
        for round_i in tqdm(range(self.num_round)):
            # print("{}, {}".format(round_i, self.latest_global_model))
            # Test latest model on train data
            # self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_testdata(round_i)
            selected_clients = self.select_group(self.group.G, round_i)
            # Choose K clients prop to data size
            # selected_clients = self.select_group(self.group.G, round_i)
            # print(selected_clients)
            D = self.get_each_class_vloume(selected_clients)
            # Solve minimization locally
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)

            # 分配带宽 ！！！
            bandwidth_allocation_result = self.bandwidth_allocation.proposed_bandwidth_allocation(selected_clients, round_i)
            energy_cost = self.cost.get_energy_sum(selected_clients, bandwidth_allocation_result, round_i)
            latency_cost = self.cost.get_latency_sum(selected_clients, bandwidth_allocation_result, round_i)
            local_latency = latency_cost[1]
            upload_latency = latency_cost[2]            
            self.cost.accumulated_latency += latency_cost[0]
            self.cost.accumulated_energy += energy_cost[0]
            local_energy = energy_cost[1]
            upload_energy = energy_cost[2]  
            
            self.metrics.update_cost(round_i, local_latency, upload_latency, self.cost.accumulated_latency, \
                                    local_energy, upload_energy, self.cost.accumulated_energy)                 
            self.metrics.update_class_vloume_round(D)

            print("latency ",  self.cost.accumulated_latency)
            print("energy",  self.cost.accumulated_energy)


            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            
            #self.optimizer.adjust_learning_rate(round_i)
            self.optimizer.soft_decay_learning_rate()
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        self.metrics.write()

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = FedGroupClient(self.options, i, self.model, self.optimizer, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.clients_system_attr, self.max_iterations)
            all_client.append(local_client)
        return all_client


    def get_client_representer(self, clients):
        # clients_representer = []

        # 1. 模型预训练
        self.client_evaluator = ClientEvaluator(self.dataset.testData, self.dataset.testLabel, self.model,
                                                 extract=["confidence"], epochs=10, variance_explained=1.0)

        # 2. 测试数据 得到置信度 概率分布
        evalutions = self.client_evaluator.evaluate(clients)
        is_need_evalutions = evalutions['confidence']
        # 3. 乘以本地样本数量
        # 4. 得到本地的标签
        clients_own_label = self.get_clients_own_labels(clients, is_need_evalutions)

        return clients_own_label

    def get_clients_own_labels(self, clients, is_need_evalutions):
        clients_own_label = []
        for i, client in enumerate(clients, start=0):
            client_label_num = len(client.local_dataset) * is_need_evalutions[i]
            clients_own_label.append(client_label_num)
        return clients_own_label

    # old
    # def select_group(self, G, round_i):
    #     # 计算每个组 所消耗的能耗和时延
    #     weight_cost = self.the_lower_cost_of_each_round(self.group.G, self.clients_system_attr, round_i)
    #     print(weight_cost)
    #     random_ = np.random.uniform(0, 1)
    #     if random_ > 0.9:
    #         min_index = weight_cost.index(min(weight_cost))
    #     else:
    #         min_index= np.random.choice(len(self.group.G), 1, replace=False,)[0] 
    #     print(min_index)
    #     # index = np.random.choice(len(self.group.G), 1, replace=False,)

    #     select_clients = []
    #     for i in G[min_index]:
    #         select_clients.append(self.clients[i])
    #     return select_clients
    def select_group(self, G, round_i):
        # 计算每个组 所消耗的能耗和时延
        weight_cost = self.the_lower_cost_of_each_round(self.group.G, self.clients_system_attr, round_i)
        # print(weight_cost)
        sorted_index_weight_cost = sorted(range(len(weight_cost)), key=lambda i: weight_cost[i])
        index = []
        random_ = np.random.uniform(0, 1)
        g_num = int(self.options['c_fraction'] * self.options['num_of_clients'] / 20)
        if random_ > 0.9:
            min_index = sorted_index_weight_cost[0: g_num]
        else:
            min_index = np.random.choice(len(self.group.G), g_num, replace=False,)
        index.extend(min_index)
       # print(index)
        # index = np.random.choice(len(self.group.G), 1, replace=False,)
        select_clients = []
        for g in index:
            for i in G[g]:
                select_clients.append(self.clients[i])
        return select_clients   

    def the_lower_cost_of_each_round(self, G, ClientAttr, round_i):
        
        weight_cost = [0 for i in range(self.group_num)] 
        G_data_vloume = [0 for i in range(self.group_num)] 
        for g in G:
            energy = 0
            latency = 0
            data_vloume = 0
            for client in G[g]:
                energy += self.clients[client].getLocalEngery(round_i) # 计算每个客户端的能耗 和 时延 
                latency += self.clients[client].getLocalDelay(round_i)
                data_vloume += len(self.clients[client].local_dataset)
            weight_c = (self.options['weight'] * (latency) + ((1 - self.options['weight']) * energy))
            weight_cost[g] = weight_c
        return weight_cost  

    def select_clients(self,):
        
        select_clients = []
        group = []
        clients_index = [_ for _ in range(self.options['num_of_clients'])]
        D_distribution = [0 for _ in range(10)]
        g1 = random.choice(clients_index)
        group.append(g1)
        clients_index.remove(g1)
        for i in range(10):
            D_distribution[i] += self.clients[g1].class_distribution[i]
        while len(group) < self.options['num_of_clients'] / 10:

            more_banlance_client = clients_index[0]
            for j in clients_index:
                if banlance(D_distribution, self.clients_data_distribution[j]) \
                                < banlance(D_distribution, self.clients_data_distribution[more_banlance_client]):
                    more_banlance_client = j
            group.append(more_banlance_client)
            for i in range(len(D_distribution)):
                D_distribution[i] += self.clients_data_distribution[more_banlance_client][i]
            clients_index.remove(more_banlance_client)
        for i in group:
            select_clients.append(self.clients[i])
        return select_clients
 

