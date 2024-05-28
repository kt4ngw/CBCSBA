from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
from src.cost import ClientAttr, Cost
import math

from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to


criterion = F.cross_entropy
mse_loss = nn.MSELoss()
from src.utils.torch_utils import *
class Client():
    def __init__(self, options, id, model, optimizer, local_dataset, system_attr, max_iterations):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        self.optimizer = optimizer
        self.gpu = options['gpu']
        self.attr_dict = system_attr.get_client_attr(self.id)
        self.class_distribution, self.class_is_own = self.get_class()
        self.iterations = self.get_iterations()
        #print(self.iterations)
        self.max_iterations = max_iterations
    
    def get_iterations(self, ):
        iterations = math.ceil(len(self.local_dataset) / self.options['batch_size'])
        return iterations

    def get_class(self):
        class_num = 10
        if self.options['dataset_name'] == "emnist":
            class_num = 47
        class_distribution = [0 for _ in range(class_num)] # 多少个类别
        class_own = [0 for _ in range(class_num)]
        for i in range(len(self.local_dataset)):
            site = self.local_dataset[i][1]
            class_distribution[site] += 1
        for i in range(len(class_distribution)):
            if class_distribution[i] > 0:
                class_own[i] = 1
        return class_distribution, class_own

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

        
    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_parameters(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)
    
    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_gradients(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:            
            x = self.flatten_data(x)
            if self.gpu >= 0:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    # def get_model_gradients(self):
    #     gradients = {}
    #     for name, param in self.model.named_parameters():
    #         if param.grad is not None:
    #             gradients[name] = param.grad.clone().detach()
    #     return gradients

    # def set_model_gradients(self, gradient_dict):
    #     for name, param in self.model.named_parameters():
    #         if name in gradient_dict:
    #             param.grad = gradient_dict[name]
    



    def local_train(self, ):
        begin_time = time.time()
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats

    def local_update(self, local_dataset, options, ):
        # batch_size=options['batch_size']
        if options['batch_size'] == -1:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            if len(local_dataset) < options['batch_size']:
                localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
            else:
                sampler = RandomSampler(local_dataset, replacement=False, num_samples=1 * options['batch_size'])
                localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)
                # localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
           # local_model_paras = self.get_model_parameters()   
        print(self.get_flat_model_params())
        local_model_paras = copy.deepcopy(self.get_model_parameters())
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict
    

    def getLocalEngery(self, round_i):
        if len(self.local_dataset) < self.options['batch_size']:
            dataset_len = len(self.local_dataset)
        else:
            dataset_len = self.options['batch_size']
        localEngery = (10 ** -26) * (self.attr_dict['cpu_frequency'][round_i] * 10 ** 9) ** 2 * self.options['C'] * dataset_len * self.options['local_epoch']
        return localEngery

    def getUploadEngery(self, round_i, bandwidth):
        uploadEngery = self.attr_dict['transmit_power'] * self.getUploadDelay(round_i, bandwidth)
        return uploadEngery

    def getLocalDelay(self, round_i):
        if len(self.local_dataset) < self.options['batch_size']:
            dataset_len = len(self.local_dataset)
        else:
            dataset_len = self.options['batch_size']
        localDelay = (self.options['C'] * dataset_len * self.options['local_epoch']) / (self.attr_dict['cpu_frequency'][round_i] * 10 ** 9)
        return localDelay

    def getUploadDelay(self, round_i, bandwidth):
        R_K =  bandwidth * 1000000 * np.log2(1 + self.attr_dict['transmit_power'] * 8) # 1M bit / s / self.B
        uploadDelay = self.options['model_size'] / (R_K / 8 / 1024 / 1024) # 100KB 0.1M  # 1S
        return uploadDelay

    # def getSumEngery(self, round_i):
    #     return self.getUploadEngery(round_i) + self.getLocalEngery(round_i)

    # def getSumDelay(self, round_i):
    #     return self.getUploadDelay(round_i) + self.getLocalDelay(round_i)

    def get_loss(self, local_dataset, options):
        if options['batch_size'] == 100:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        train_loss = train_acc = train_total = 0
        with torch.no_grad():
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size    
        return train_loss / train_total   

    def get_pre_loss(self,):
        pre_loss = self.get_loss(self.local_dataset, self.options)
        return pre_loss
