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


class Scaffold_CLient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset, ):

        super(Scaffold_CLient, self).__init__(options, id, model, optimizer, local_dataset, )

        self.global_model_parameters = None
        self.c_local = None

    def local_train(self, c_global):
        begin_time = time.time()
        local_model_paras, dict, delta = self.local_update(self.local_dataset, self.options, c_global, )
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats, delta

    def local_update(self, local_dataset, options, c_global, ):
        if options['batch_size'] == 100:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        if self.c_local == None:
            self.c_local = list([param.detach().clone() for param in c_global])
        old_model_parameters = [param.detach().clone() for param in self.model.parameters()]
        c_diff = [-c_l + c_g for c_l, c_g in zip(self.c_local, c_global)]
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                # client_drift
                for para, c_d in zip(self.model.parameters(), c_diff):
                    para.grad += c_d.data
                self.optimizer.step()
                self.optimizer.zero_grad()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
            local_model_paras = self.get_model_parameters()
        # need to computing {c_plus, y_delta, c_delta}
        y_local_delta = []
        c_local_plus = []
        c_local_delta = []
        # computing {y_local_delta}
        for param_l, param_g in zip(self.model.parameters(), old_model_parameters):
            y_local_delta.append(param_l - param_g)
        # print("y_locala_delta", y_local_delta)
        coeff =  1.0 / ((options['local_epoch'] * options['lr']))

        # computing {c_plus}
        for c_l, c_g, diff in zip(self.c_local, c_global, y_local_delta):
            c_local_plus.append(c_l - c_g + coeff * (-diff))

        # compute {c_delta}
        for c_p, c_l in zip(c_local_plus, self.c_local):
            c_local_delta.append(c_p - c_l)

        # update {c_i}
        self.c_local = list([param.detach().clone() for param in c_local_plus])

        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict, (y_local_delta, c_local_delta)

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)
