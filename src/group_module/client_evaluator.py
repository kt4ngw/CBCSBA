import copy
import pickle
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict
from src.utils.utils import save_pickle
from src.group_module import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# class ClientEvaluation:
#
#     def __init__(self, representers, extracted):
#         self.representers = representers
#         self.extracted = extracted
#
#     @property
#     def representers(self):
#         return self.representers
#     @property
#     def extracted(self):
#         return self.extracted



class ClientEvaluator():
    can_extract = ["confidence", "classifierLast", "classifierLast2", "classifierAll"] # 暂时用第一个！
    def __init__(self, test_data, test_label, model, extract: List[str], epochs: int, variance_explained: float):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data = test_data
        self.test_label = test_label
        self.model = model
        self.extract = extract  # ["confidence", "classifierLast", "classifierLast2", "classifierAll"]
        self.epochs = epochs
        self.test_dataloader = DataLoader(TensorDataset(torch.tensor(self.test_data),
                                                torch.tensor(test_label)), num_workers=0, batch_size=1)
        self.variance_explained = variance_explained
        self.init_global_model = copy.deepcopy(self.model.state_dict())

    def evaluate(self, clients,):

        if os.path.exists(os.path.join(self.script_dir, "representers.pkl")):
            with open(os.path.join(self.script_dir, "representers.pkl"), "rb") as file:
                representers = pickle.load(file)
            return representers
        evaluations = {}
        representers = {e: list() for e in self.extract} # {'confidence': [], 'classifierLast': [], 'classifierLast2': [], 'classifierAll': []}
        # 在这里 执行一个预训练...
        for client in clients:
            # print("得到模型参数")
            client.set_model_parameters(self.init_global_model)
            local_model_paras = client.pre_train(self.model, client.local_dataset, self.epochs, )

            for to_extract in self.extract:
                client_representer = self.get_representer(client, to_extract, self.model)
            representers[to_extract].append(client_representer) #

        # for to_extract in self.extract:
        #     reduced = self.reduce_representers(representers[to_extract], to_extract)
        #     evaluations[to_extract] = ClientEvaluation(reduced, to_extract)
        # Save representers
        with open(os.path.join(self.script_dir, "representers.pkl"), "wb") as file:
            pickle.dump(representers, file)
        return representers # 多个评估结果

    def reduce_representers(self, representers, to_extract):

        return representers


    def get_representer(self, client, to_extract: str, model):
        if to_extract == "confidence":
            return self.get_confidence_prediction(client, model)

    def get_confidence_prediction(self, client, model):
        """
        Evaluates the client model (after the pre-training phase) to extract confidence vectors
        :param client: the client from which to extract the confidence vector
        :return: the confidence vector for the client
        """
        model.eval()
        n_classes = len(set(self.test_label))
        conf_vector = np.zeros(n_classes)
        testDataLoader = DataLoader(TensorDataset(torch.tensor(self.test_data), torch.tensor(self.test_label)), batch_size=1,
                                    shuffle=False)
        with torch.no_grad():
            for exemplar, target in testDataLoader:
                if client.gpu:
                    exemplar, target = exemplar.cuda(), target .cuda()
                logits = model(exemplar)[0].detach().cpu().numpy()
                logits = softmax(logits)
                target = target.detach().cpu().numpy()
                conf_vector[target] += logits[target]
        conf_vector = conf_vector / np.sum(conf_vector)
        return conf_vector