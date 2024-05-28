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


class FedProx_CLient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset, system_attr,  ):

        super(FedProx_CLient, self).__init__(options, id, model, optimizer, local_dataset, system_attr, )

        self.global_model_parameters = None
        self.mu = 0.01

    def local_train(self, ):
        begin_time = time.time()
        self.global_model_parameters = copy.deepcopy(self.get_model_parameters()) 
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats


    def local_update(self, local_dataset, options, ):
        if options['batch_size'] == 100:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.get_model_parameters().values(), self.global_model_parameters.values()):
                    proximal_term += torch.norm(w - w_t, p=2).item()
                loss = criterion(pred, y) + (self.mu / 2) * proximal_term
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
            local_model_paras = copy.deepcopy(self.get_model_parameters())

        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict
