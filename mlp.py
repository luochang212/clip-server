import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    @staticmethod
    def use_svg_display():
        """Use the svg format to display a plot in Jupyter.

        Defined in :numref:`sec_calculus`"""
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
    
    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.

        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
        axes.set_xscale(xscale), axes.set_yscale(yscale)
        axes.set_xlim(xlim),     axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class SimpleMLP:

    def __init__(self,
                 input_channel,
                 output_channel,
                 hidden_layers=[128, 32],
                 dropout_probs=[0.2, 0],
                 batch_size=256,
                 num_epochs=20,
                 ylim=[0.0, 1.0],
                 test_size=0.3,
                 random_state=37):

        assert hidden_layers is None or (len(hidden_layers) == len(dropout_probs))

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_layers = hidden_layers
        self.dropout_probs = dropout_probs
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.ylim = ylim
        self.test_size = test_size
        self.random_state = random_state
        self.embd_col_name = None
        self.label_col_name = None
        self.net = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'device: {self.device}')

    def split_dateset(self, df, test_size, random_state):
        if self.embd_col_name is None or self.label_col_name is None:
            raise Exception('`embd_col_name` or `label_col_name` is None.')

        X, y = df[self.embd_col_name].tolist(), df[self.label_col_name].tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_iter = DataLoader(train_dataset,
                                self.batch_size,
                                shuffle=True)
        test_iter = DataLoader(test_dataset,
                               self.batch_size,
                               shuffle=True)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        return X_tensor, train_iter, test_iter

    def init_model(self):
        layers = []
        input_dim = self.input_channel

        # 添加隐藏层
        for i, hidden_dim in enumerate(self.hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            # 若 dropout 概率大于 0，添加 dropout 层
            if self.dropout_probs[i] > 0:
                layers.append(nn.Dropout(self.dropout_probs[i]))

            input_dim = hidden_dim

        # 添加输出层
        layers.append(nn.Linear(input_dim, self.output_channel))

        net = nn.Sequential(*layers)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        net.apply(init_weights)

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())

        net = net.to(self.device)

        return net, loss, optimizer

    @staticmethod
    def accuracy(y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(dim=1)
        cmp = y_hat.type(y.dtype) == y

        return float(cmp.type(y.dtype).sum())

    def evaluate_accuracy(self, data_iter):
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()
        metric = Accumulator(2)  # 正确预测数、预测总数

        with torch.no_grad():
            for X, y in data_iter:
                metric.add(self.accuracy(self.net(X), y), y.numel())
        return metric[0] / metric[1]

    def train_epoch(self, train_iter, loss, updater):

        if isinstance(self.net, torch.nn.Module):
            self.net.train()

        metric = Accumulator(3)
        for X, y in train_iter:
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.net(X)
            l = loss(y_hat, y)

            updater.zero_grad()
            l.mean().backward()
            updater.step()

            metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=self.ylim,
                            legend=['train loss', 'train acc', 'test acc'])

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))

        train_loss, train_acc = train_metrics

        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }

    def predict(self, X):
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()

        pred_list = []
        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i+self.batch_size].to(self.device)
            with torch.no_grad():
                output = self.net(X_batch).argmax(dim=1)
            pred_list += output.tolist()

        return pred_list

    def main(self,
             df,
             embd_col_name,
             label_col_name):
        self.embd_col_name = embd_col_name
        self.label_col_name = label_col_name

        net, loss, optimizer = self.init_model()
        self.net = net

        X_tensor, train_iter, test_iter = self.split_dateset(df,
                                                             test_size=self.test_size,
                                                             random_state=self.random_state)
        metrics = self.train(train_iter=train_iter,
                             test_iter=test_iter,
                             loss=loss,
                             num_epochs=self.num_epochs,
                             updater=optimizer)

        return self.predict(X_tensor), metrics

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)


class MLP(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 hidden_layers=[128, 32],
                 dropout_probs=[0.2, 0]):
        super().__init__()

        # 定义网络结构
        layers = []
        input_dim = input_channel
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout_probs[i] > 0:
                layers.append(nn.Dropout(dropout_probs[i]))

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_channel))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrainMLP(SimpleMLP):
    def __init__(self, net):
        super().__init__(input_channel=None,
                         output_channel=None,
                         hidden_layers=None,
                         dropout_probs=None)
        self.net = net

    def init_model(self):

        # 初始化网络权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        self.net.apply(init_weights)

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())

        net = self.net.to(self.device)

        return net, loss, optimizer
