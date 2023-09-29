import mindspore
from mindspore import Tensor
from mindspore import nn, ops
import math


class Bottleneck(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, pad_mode='pad', padding=1, has_bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.8)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), 1)
        return out


# single layer
class SingleLayer(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, pad_mode='pad', padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.8)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv1(self.relu(x))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), 1)
        return out


# transition layer
class Transition(nn.Cell):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, has_bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.8)  # 0.2->
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.avg_pool2d(out, 2, 2, pad_mode='same')  # ceil_mode=True
        return out


class DenseNet(nn.Cell):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, pad_mode='pad', padding=3, stride=2, has_bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.pool(out)  # , ceil_mode=True
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out


if __name__ == '__main__':
    import numpy as np

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    # 创建测试环境
    params = {
        'densenet': {
            'growthRate': 24,
            'reduction': 0.5,
            'bottleneck': True,
            'use_dropout': True
        },
        'encoder':{
            'input_channel': 1,
            'out_channel': 684
        }
    }
    densenet = DenseNet(params)
    x = Tensor(np.ones((8, 1, 128, 823)).astype(np.float32))
    y = densenet(x)
