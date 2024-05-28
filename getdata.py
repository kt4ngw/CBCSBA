import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import platform
from torchvision import transforms
from PIL import Image
class GetDataSet():
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName

        self.trainData = None
        self.trainLabel = None
        self.trainDataSize = None

        self.testData = None
        self.testLabel = None
        self.testDataSize = None

        if self.dataSetName == 'MNIST' or self.dataSetName == 'mnist':
            self.mnistDataDistribution()
            print("mnist!!")
        elif self.dataSetName == 'EMNIST' or self.dataSetName == 'emnist':
            self.emnistDataDistribution()
            print("Emnist!!")
        elif self.dataSetName == 'CIFAR10' or self.dataSetName == 'cifar10':
            self.cifar10DataDistribution()
            print("cifar10!!")
        elif self.dataSetName == 'FASHIONMNIST' or self.dataSetName == 'fashionmnist':
            self.fashionmnistDataDistribution()
            print("fashion!!")

    # def mnistDataDistribution(self, isIID):
    #
    #     trainingData = datasets.CIFAR10(
    #         root="data",
    #         train=True,
    #         download=True,
    #         transform=ToTensor(),
    #     )
    #     trainData = []
    #     trainLabel = []
    #     for X, y in trainingData:
    #         trainData.append(X.tolist())
    #         trainLabel.append(y)
    #     self.trainDataSize = len(trainData)
    #     # ----------------------------------------------------------- #
    #     testingData = datasets.CIFAR10(
    #         root="data",
    #         train=False,
    #         download=True,
    #         transform=ToTensor(),
    #     )
    #     testData = []
    #     testLabel = []
    #     for X, y in testingData:
    #         testData.append(X.tolist())
    #         testLabel.append(y)
    #     self.testDataSize = len(testData)
    #     self.testData = torch.tensor(testData)
    #     self.testLabel = torch.tensor(testLabel)
    #     # ----------------------------------------------------------- #
    #
    #     if isIID == True:
    #         self.trainData = torch.tensor(trainData)
    #         self.trainLabel = torch.tensor(trainLabel)
    #         print(1)
    #
    #     else:
    #         trainDataT = np.array(trainData, dtype='float32')
    #         trainLabelT = np.array(trainLabel, dtype='int64')
    #         self.trainData = trainDataT
    #         self.trainLabel = trainLabelT
    #     print(self.trainData.shape)

    def emnistDataDistribution(self, ):
        data_dir = r'./data/EMNIST'
        train_images_path = os.path.join(data_dir, 'emnist-balanced-train.csv')
        test_images_path = os.path.join(data_dir, 'emnist-balanced-test.csv.gz')
        import pandas as pd
        import numpy as np

        # 读取 CSV 文件
        train_images = pd.read_csv(train_images_path)
        test_images = pd.read_csv(train_images_path)
        # 提取标签
        train_labels = train_images.iloc[:, 0].values
        test_labels = test_images.iloc[:, 0].values
        # 提取图像数据并转换为 numpy 数组
        train_images = train_images.iloc[:, 1:].values
        train_images = train_images.astype(np.float32)  # 将图像数据转换为 float32 类型
        train_images = np.reshape(train_images, (-1, 1, 28, 28))  # 将图像数据重新整形为 28x28 的数组

        test_images = test_images.iloc[:, 1:].values
        test_images = test_images.astype(np.float32)  # 将图像数据转换为 float32 类型
        test_images = np.reshape(test_images , (-1,  1,  28, 28))  # 将图像数据重新整形为 28x28 的数组
        # 打印标签和图像的形状
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        self.trainData = train_images
        self.trainLabel = train_labels
        self.testData = test_images
        self.testLabel = test_labels
        print(self.trainData.shape)
        print(self.trainLabel.shape)

        balance_testData = []
        balance_testlabel = []
        class_index = [np.argwhere(self.testLabel == y).flatten() for y in range(self.testLabel.max() + 1)]
        min_number = min([len(class_) for class_ in class_index])
        for number in range(self.testLabel.max() + 1):
            balance_testData.append(self.testData[class_index[number][:min_number]])
            balance_testlabel += [number] * min_number
        print(min_number)
        self.testData = np.concatenate(balance_testData, axis=0)
        self.testLabel = np.array(balance_testlabel)
        self.testLabel = torch.tensor(self.testLabel).to(torch.int64)
    def mnistDataDistribution(self, ):

        data_dir = r'./data/MNIST/raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        self.trainData = train_images
        self.trainLabel = np.argmax(train_labels == 1, axis = 1)
        self.testData = test_images
        self.testLabel = np.argmax(test_labels == 1, axis = 1)
        print(self.trainData.shape)
        balance_testData = []
        balance_testlabel = []
        class_index = [np.argwhere(self.testLabel == y).flatten() for y in range(self.testLabel.max() + 1)]
        min_number = min([len(class_) for class_ in class_index])
        for number in range(self.testLabel.max() + 1):
            balance_testData.append(self.testData[class_index[number][:min_number]])
            balance_testlabel += [number] * min_number 

        self.testData = np.concatenate(balance_testData, axis=0)
        self.testLabel = np.array(balance_testlabel)
        self.testLabel = torch.tensor(self.testLabel).to(torch.int64)



    def fashionmnistDataDistribution(self, ):
        print("执行了吗？")
        data_dir = r'./data/FashionMNIST/raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        self.trainData = train_images
        self.trainLabel = np.argmax(train_labels == 1, axis = 1)
        self.testData = test_images
        self.testLabel = np.argmax(test_labels == 1, axis = 1)
        print(self.trainData.shape)
        print(self.trainLabel.shape)


    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')

        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_labels(self, filename):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return self.dense_to_one_hot(labels)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def cifar10DataDistribution(self):
        cifar10_dir = 'data/cifar-10-batches-py'
        print(self.trainLabel)
        self.trainData, self.trainLabel, self.testData, self.testLabel = self.load_CIFAR10(cifar10_dir)

    def load_CIFAR10(self, ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del xs, ys
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

        X_train = np.multiply(Xtr, 1.0 / 255.0)
        X_test = np.multiply(Xte, 1.0 / 255.0)
        # Resize images to 224x224

        # X_train = Xtr
        # X_test = Xte
        # X_train = torch.Tensor(Xtr).permute(0, 1, 2, 3) / 255.0
        # X_test = torch.Tensor(Xte).permute(0, 1, 2, 3) / 255.0
        return X_train, Ytr, X_test, Yte


    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 1, 2, 3, ).astype("float32")

            Y = np.array(Y).astype("int64")
            return X, Y

    def load_pickle(self, f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return pickle.load(f)
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))


# g = GetDataSet("EMNIST")
# print(g.trainData)
# print(g.trainLabel)
