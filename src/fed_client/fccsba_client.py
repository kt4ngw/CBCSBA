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

class FccsbaClient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset, system_attr, max_iterations ):

        super(FccsbaClient, self).__init__(options, id, model, optimizer, local_dataset, system_attr, max_iterations )
        self.rho = None   
        self.beta = None
        self.gama = None



    def update_rho_beta(self, local_model_paras, loss_last_global, w_last_global, grad_last_global, loss_now_local):
        grad_local = self.get_model_gradients()
        tmp_norm = 0.0
        for w, w_t in zip(local_model_paras.values(), w_last_global.values()):
            # print("w", w.dtype)
            # print("w_{t}, w", w_t.dtype)
            w = torch.tensor(w, dtype=torch.float32)
            w_t = torch.tensor(w_t, dtype=torch.float32)
            tmp_norm += torch.norm(w - w_t, p=2).item()
            # tmp_norm += (w - w_t).norm(2)
        # print(tmp_norm)
        # Compute rho
        if tmp_norm > 1e-10:
            loss_last_global = torch.tensor(loss_last_global, dtype=torch.float32)
            loss_now_local = torch.tensor(loss_now_local, dtype=torch.float32)
            self.rho = torch.norm(loss_last_global - loss_now_local, p=2) / tmp_norm
        else:
            self.rho = 0
        # compute beta
        # c = self.grad_last_global - grad_local\
        c = 0.0
        for w, w_t in zip(grad_last_global.values(), grad_local.values()):
            # print("w", w.dtype)
            # print("w_{t}, w", w_t.dtype)
            # w = torch.tensor(w, dtype=torch.float32)
            # w_t = torch.tensor(w_t, dtype=torch.float32)
            c += torch.norm(w - w_t, p=2).item()       
        if tmp_norm > 1e-10:
            self.beta = c /  tmp_norm
        else:
            self.beta = 0
        
        
    def local_train(self,):
        begin_time = time.time()
        loss_last_global = self.get_loss(self.local_dataset, self.options)
        w_last_global = copy.deepcopy(self.get_model_parameters()) 
        grad_last_global = copy.deepcopy(self.get_model_gradients())
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, )

        loss_now_local = dict['loss']

        self.update_rho_beta(local_model_paras, loss_last_global, w_last_global, grad_last_global, loss_now_local)
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats, (self.rho, self.beta) , (len(self.local_dataset), loss_now_local)

    def local_update(self, local_dataset, options, ):
        if options['batch_size'] == 100:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
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
        local_model_paras = copy.deepcopy(self.get_model_parameters())
        # print("更新后", local_model_paras['fc2.bias'])
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict
