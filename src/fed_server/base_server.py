import numpy as np
import torch
import time
from src.fed_client.base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
from src.utils.metrics import Metrics
from src.cost import Cost, ClientAttr
criterion = F.cross_entropy
from src.allocation_alo.bandwidth_allocation import Bandwidth_Allocation
from src.utils.torch_utils import *
class BaseFederated(object):

    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power, model=None, optimizer=None, name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer
        self.clients_system_attr = ClientAttr(cpu_frequency, B, transmit_power)
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.num_round = options['round_num']
        self.per_round_c_fraction = options['c_fraction']
        self.max_iterations = self.get_max_iterations()
        self.clients = self.setup_clients(self.dataset, self.clients_label)
        self.clients_num = len(self.clients)
        self.name = '_'.join([name, f'wn{int(self.per_round_c_fraction * self.clients_num)}',
                              f'tn{len(self.clients)}'])
        self.latest_global_model = copy.deepcopy(self.get_model_parameters())
        # self.options['model_size'] = self.model.get_model_size()
        self.clients_own_datavolume = [len(client.local_dataset) for client in self.clients]  
        self.metrics = Metrics(options, self.clients, self.name)
        self.label_composition_truth = self.get_clients_label_composition_truth(self.clients, self.dataset, self.clients_label)
        self.bandwidth_allocation = Bandwidth_Allocation(options, B)
        self.cost = Cost()

        #print("self.max_iterations", self.max_iterations)

    @staticmethod
    def move_model_to_gpu(model, options):
        if options['gpu'] >= 0:
            device = options['gpu']
            torch.cuda.set_device(device)
            # torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def get_max_iterations(self):
        max_iterations = 0
        for client in self.clients_label:
            if len(client) // self.options['batch_size'] > max_iterations:
                max_iterations = len(client) // self.options['batch_size']
        return max_iterations
   
    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def get_model_gradients(self):
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        return gradients

    def set_model_gradients(self, gradient_dict):
        for name, param in self.model.named_parameters():
            if name in gradient_dict:
                param.grad = gradient_dict[name]
                
    def train(self):
        """The whole training procedure
        No returns. All results all be saved.
        """
        raise NotImplementedError

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = Client(self.options, i, self.model, self.optimizer, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.clients_system_attr, self.max_iterations)
            all_client.append(local_client)

        return all_client

    def local_train(self, round_i, select_clients, ):
        local_model_paras_set = []
        stats = []
        for i, client in enumerate(select_clients, start=1):
            client.set_model_parameters(self.latest_global_model)

            local_model_paras, stat = client.local_train()
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, client.id, i, int(self.per_round_c_fraction * self.clients_num),
                       stat['loss'], stat['acc']*100, stat['time'], ))
        return local_model_paras_set, stats


    def aggregate_parameters(self, local_model_paras_set):

        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = 0
        for var in averaged_paras:
            averaged_paras[var] = 0
        for num_sample, local_model_paras in local_model_paras_set:
            for var in averaged_paras:
                averaged_paras[var] += num_sample * local_model_paras[var]
            train_data_num += num_sample
        for var in averaged_paras:
            averaged_paras[var] = averaged_paras[var] / train_data_num
        return averaged_paras

    def test_latest_model_on_testdata(self, round_i):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.global_test(use_test_data=True)
        end_time = time.time()

        if True:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_test_data['acc'],
                   stats_from_test_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_test_stats(round_i, stats_from_test_data)

    def global_test(self, use_test_data=True):
        assert self.latest_global_model is not None
        self.set_model_parameters(self.latest_global_model)
        testData = self.dataset.testData
        testLabel = self.dataset.testLabel
        testDataLoader = DataLoader(TensorDataset(torch.tensor(testData), torch.tensor(testLabel)), batch_size=100, shuffle=False)
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)
        
        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total,}
        # a = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
        return stats


    def get_clients_label_composition_truth(self, clients, dataset, clients_label):
        class_num = 10
        if self.options['dataset_name'] == "emnist":
            class_num = 47
        if self.options['dataset_name'] == "cifar100":
            class_num = 20
        clients_own_label_turth = []
        train_label = dataset.trainLabel
        for i, client in enumerate(clients, start=0):
            result = np.zeros(class_num)
            for j in range(len(train_label[clients_label[i]])):
                result[train_label[clients_label[i]][j]] += 1
            clients_own_label_turth.append(result)
        return clients_own_label_turth

    def get_each_class_vloume(self, selected_clients):
        class_num = 10
        if self.options['dataset_name'] == "emnist":
            class_num = 47
        if self.options['dataset_name'] == "cifar100":
            class_num = 20
        D = [0 for _ in range(class_num)]
        for i in selected_clients:
            for j in range(class_num):
                D[j] += i.class_distribution[j]
        return D

    def getEngery(self, select_clients):
        for client in select_clients:
            self.cost.energy_Sum += client.getSumEngery()
        return self.cost.energy_Sum

    def getDelay(self, select_clients):
        maxD1 = 0
        maxD2 = 0
        for client in select_clients :
            if client.getLocalDelay() > maxD1:
                maxD1 = client.getLocalDelay()
            if client.getUploadDelay() > maxD2:
                maxD2 = client.getUploadDelay()
        self.cost.delay_Sum += (maxD1 + maxD2)
        return self.cost.delay_Sum