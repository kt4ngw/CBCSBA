from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm

from torch.optim import SGD, Adam
import copy
class FedAvgTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(FedAvgTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model, self.optimizer,)
    
    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))
        #print("第一轮的模型", self.latest_global_model['fc2.bias'])
        for round_i in range(self.num_round):
            # self.w_last_global = copy.deepcopy(self.latest_global_model)
            # print("{}, {}".format(round_i, self.latest_global_model))
            # Test latest model on train data
            # self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_testdata(round_i)
            # Choose K clients prop to data size
            selected_clients = self.select_clients()
            # print(selected_clients)
            D = self.get_each_class_vloume(selected_clients)
            # Solve minimization locally
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)
            
            bandwidth_allocation_result = self.bandwidth_allocation.equal_allocation(selected_clients)

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
            print("energy",  self.cost.accumulated_energy )
            # comp cost
            # self.comptuing_delay_energy(selected_clients)
            # self.metrics.update_cost(round_i, self.cost.delay_Sum, self.cost.energy_Sum)
            # Track communication cost
            #self.metrics.extend_communication_stats(round_i, stats)
            # Update latest model
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            # - - - - - - - - - - #
            # 获取梯度
            # grad_last_local_last_round = {}
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         grad_last_local_last_round[name] = param.grad.clone().detach()
            # print("grad", grad_last_local_last_round)
            # - - - - - - - - - - #
            self.optimizer.soft_decay_learning_rate()
            #self.optimizer.soft_decay_learning_rate()
            #self.optimizer.inverse_prop_decay_learning_rate(round_i)
        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        self.metrics.write()

    def select_clients(self):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients

