from mindspore import nn, ops
from mindspore import Tensor
import mindspore
import numpy as np


class ChannelAtt(nn.Cell):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc = nn.SequentialCell(
                nn.Dense(channel, channel//reduction),
                nn.ReLU(),
                nn.Dense(channel//reduction, channel),
                nn.Sigmoid())

    def construct(self, x):
        b, c, h, w = x.shape
        # y = self.avg_pool(x).view(b, c)
        # y = Tensor(np.random.randn(8, 512, 1, 1).astype(np.float32)).view(b, c)
        # y = nn.AvgPool2d((h, w))(x).view(b, c)
        y = ops.mean(x, axis=(2, 3), keep_dims=True).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CountingDecoder(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.trans_layer = nn.SequentialCell(
            nn.Conv2d(self.in_channel, 512, kernel_size=kernel_size, pad_mode='pad', padding=kernel_size//2, has_bias=False),
            nn.BatchNorm2d(512))
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.SequentialCell(
            nn.Conv2d(512, self.out_channel, kernel_size=1, has_bias=False),
            nn.Sigmoid())

    def construct(self, x, mask):
        b, c, h, w = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)
        if mask is not None:
            x = x * mask
        x = x.view(b, self.out_channel, -1)
        x1 = ops.reduce_sum(x, axis=-1)
        return x1, x.view(b, self.out_channel, h, w)


if __name__ == '__main__':
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    import numpy as np
    cnn_features = Tensor(np.ones((8, 684, 8, 52)).astype(np.float32))
    counting_mask = Tensor(np.ones((8, 1, 8, 52)).astype(np.float32))
    counting_decoder = CountingDecoder(684, 111, 3)
    a = counting_decoder(cnn_features, counting_mask)