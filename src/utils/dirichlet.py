import numpy as np


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    np.random.seed(2024)
    print(train_labels)
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    # 首先，为每个客户端分配一个样本
    for i in range(n_clients):
        random_class = np.random.choice(n_classes)
        random_sample = np.random.choice(class_idcs[random_class])
        client_idcs[i].append(random_sample)
    # 分配余下的样本给每个客户端
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # print(k_idcs)

        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            # print(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int)))
            # print(A

            client_idcs[i] += idcs.tolist()

    client_idcs = [np.array(idcs) for idcs in client_idcs]
    # 假设 client_idcs 是一个包含多个子列表的列表
    result = []
    for i, idcs in enumerate(client_idcs):
        result.append(idcs.tolist())
    return client_idcs, result