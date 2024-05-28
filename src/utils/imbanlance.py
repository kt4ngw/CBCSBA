# import numpy as np
#
# def split_imbalance(train_labels, number_of_clients):
#     each_client_data_number = int(len(train_labels) / number_of_clients)
#     np.random.seed(2024)
#     n_classes = train_labels.max() + 1
#     index_of_diff_class = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
#     print(index_of_diff_class)
#     index_own_by_clients = [[] for _ in range(number_of_clients)]
#     for i in range(number_of_clients):
#         assign_which_class = int(i / (number_of_clients / n_classes))
#         assign_which_class_area = int(i % (number_of_clients / n_classes))
#         index_own_by_clients[i].append(index_of_diff_class[assign_which_class] \
#                                        [assign_which_class_area * each_client_data_number: \
#                                         (assign_which_class_area + 1) * each_client_data_number])
#     index_own_by_clients = [np.array(idcs[0]) for idcs in index_own_by_clients]
#     return index_own_by_clients
import numpy as np

def split_imbalance(train_labels, number_of_clients):
    each_client_data_number = int(len(train_labels) / number_of_clients)
    np.random.seed(2025)
    n_classes = train_labels.max() + 1
    index_of_diff_class = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    print(index_of_diff_class)
    index_own_by_clients = [[] for _ in range(number_of_clients)]
    for i in range(number_of_clients):
        assign_which_class = int(i / (number_of_clients / n_classes))
        # assign_which_class_area = int(i % (number_of_clients / n_classes))
        random_choose_index = np.random.choice(index_of_diff_class[assign_which_class][:], size=each_client_data_number, replace=False)
        index_own_by_clients[i] += random_choose_index.tolist()
    index_own_by_clients = [np.array(idcs) for idcs in index_own_by_clients]
    print(index_own_by_clients)
    return index_own_by_clients




def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split  #
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst

def choose_two_digit(split_data_lst, uesr):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        if uesr == 199 or uesr == 198:
            lst = np.random.choice(available_digit, 2, replace=True).tolist()
        else:
            lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst

np.random.seed(2025)
def split_imbalance_two(train_labels, number_of_clients):
    np.random.seed(2025)
    index_of_diff_class = [np.argwhere(train_labels == y).flatten() for y in range(10)]
    split_mnist_traindata = []
    index_own_by_clients = [[] for _ in range(number_of_clients)]
    for digit in index_of_diff_class:
        split_mnist_traindata.append(data_split(digit, 40))  # 第一类 分成20份

    for user in range(number_of_clients):
        print(user, np.array([len(v) for v in split_mnist_traindata]))
        for d in choose_two_digit(split_mnist_traindata, user):
            l = len(split_mnist_traindata[d][-1])
            index_own_by_clients[user] += split_mnist_traindata[d].pop().tolist()
    index_own_by_clients = [np.array(idcs) for idcs in index_own_by_clients]
    return index_own_by_clients