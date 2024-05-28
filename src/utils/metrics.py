import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image
import random
import torch



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

class Metrics(object):
    def __init__(self, options, clients, name=''):
        self.options = options

        num_rounds = options['round_num'] + 1
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        # global_test_data
        self.loss_on_g_test_data = [0] * num_rounds
        self.acc_on_g_test_data = [0] * num_rounds 
        # local and upload e and d
        self.local_latency = [0] * num_rounds 
        self.local_energy = [0] * num_rounds 

        self.upload_latency = [0] * num_rounds 
        self.upload_energy = [0] * num_rounds 
        self.D = []

        # cost time and delay
        self.accumulation_delay = [0] * num_rounds
        self.accumulation_energy = [0] * num_rounds 

        if self.options['method_division'] == 1:
            self.result_path = mkdir(os.path.join('./result/dirichlet', str(self.options['dataset_name']).lower() + str(self.options['dirichlet'])))
        elif self.options['method_division'] == 2:
            self.result_path = mkdir(os.path.join('./result/pathology', str(self.options['dataset_name']).lower()))
        if self.options['batch_size'] == 0:
            suffix = '{}_sd{}_lr{}_ne{}'.format(name,
                                            options['seed'],
                                            options['lr'],
                                            options['round_num'],
                                               )
        else:
            suffix = '{}_sd{}_lr{}_ne{}_bs{}_weight{}'.format(name,
                                            options['seed'],
                                            options['lr'],
                                            options['round_num'],
                                            options['batch_size'],
                                            options['weight'],
                                                )
        # self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algorithm'],
        #                                      options['model_name'], suffix)
        self.exp_name = '{}_{}_{}_{}'.format(options['server'],
                                             options['model_name'], options['opti'], suffix)

        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        test_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(test_event_folder)

    def update_communication_stats(self, round_i, stats):
        id, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']
        self.bytes_written[id][round_i] += bytes_w
        self.client_computations[id][round_i] += comp
        self.bytes_read[id][round_i] += bytes_r

    def extend_communication_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_communication_stats(round_i, stats)

    def update_test_stats(self, round_i, eval_stats):
        self.loss_on_g_test_data[round_i] = eval_stats['loss']
        self.acc_on_g_test_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def update_cost(self, round_i, local_latency, upload_latency, accumulated_latency,\
                                    local_energy, upload_energy, accumulated_energy):
        self.accumulation_delay[round_i] = accumulated_latency
        self.local_latency[round_i] = local_latency
        self.upload_latency[round_i] = upload_latency
        self.accumulation_energy[round_i] = accumulated_energy
        self.local_energy[round_i] = local_energy
        self.upload_energy[round_i] = upload_energy

    def update_class_vloume_round(self, D):
        self.D.append(D)

    def write(self):
        metrics = dict()
        metrics['dataset'] = self.options['dataset_name']
        metrics['loss_on_g_test_data'] = self.loss_on_g_test_data
        metrics['acc_on_g_test_data'] = self.acc_on_g_test_data
        
        metrics['accumulation_delay'] = self.accumulation_delay
        metrics['accumulation_energy'] = self.accumulation_energy
        metrics['local_delay'] = self.local_latency 
        metrics['upload_delay'] = self.upload_latency
        metrics['local_energy'] = self.local_energy
        metrics['upload_energy'] = self.upload_energy
        metrics['class_vloume'] = self.D
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf, indent=8)