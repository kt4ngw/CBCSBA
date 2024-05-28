from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
criterion = F.cross_entropy
mse_loss = nn.MSELoss()
from .base_client import Client
import os

class FedGroupClient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset, system_attr, max_iterations ):

        super(FedGroupClient, self).__init__(options, id, model, optimizer, local_dataset, system_attr, max_iterations )

    # def local_train(self, ):
    #     begin_time = time.time()
    #     local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
    #     end_time = time.time()
    #     stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
    #     stats.update(dict)
    #     return (len(self.local_dataset), local_model_paras), stats


    # def local_update(self, local_dataset, options, ):
    #     localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
    #     self.model.train()
    #     train_loss = train_acc = train_total = 0
    #     for epoch in range(options['local_epoch']):
    #         train_loss = train_acc = train_total = 0
    #         for X, y in localTrainDataLoader:
    #             if self.gpu >= 0:
    #                 X, y = X.cuda(), y.cuda()
    #             pred = self.model(X)
    #             loss = criterion(pred, y)
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #             _, predicted = torch.max(pred, 1)
    #             correct = predicted.eq(y).sum().item()
    #             target_size = y.size(0)
    #             train_loss += loss.item() * y.size(0)
    #             train_acc += correct
    #             train_total += target_size
    #         local_model_paras = self.get_model_parameters()

    #     return_dict = {"id": self.id,
    #                    "loss": train_loss / train_total,
    #                    "acc": train_acc / train_total}
    #     return local_model_paras, return_dict

    # def set_model_parameters(self, model_parameters_dict):
    #     state_dict = self.model.state_dict()
    #     for key, value in state_dict.items():
    #         state_dict[key] = model_parameters_dict[key]
    #     self.model.load_state_dict(state_dict)



    def pre_train(self, model, local_dataset, epochs):
        save_path = f'pre_train/model{self.id}.pt'
        # 检查是否已经存在预训练的模型，如果存在则直接加载
        if os.path.exists(save_path):
            print(f"Pre-trained model found at {save_path}. Loading...")
            local_model_paras = torch.load(save_path)
            model.load_state_dict(local_model_paras)
            return local_model_paras
        else:
            print(f"No pre-trained model found. Training...")        
        
        pre_dataloader = DataLoader(local_dataset, batch_size=self.options['batch_size'], shuffle=True)
        model.train()
        for epoch in range(epochs):
            train_loss = train_acc = train_total = 0
            for X, y in pre_dataloader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_model_paras = copy.deepcopy(self.model.state_dict())
        save_path = f'pre_train/model{self.id}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建父目录
        torch.save(local_model_paras, save_path)  # 保存模型参数
        return local_model_paras
