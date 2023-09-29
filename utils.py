import mindspore
from mindspore import nn, ops
from mindspore import Tensor
import numpy as np
import math
import yaml
import os
from difflib import SequenceMatcher

class CustomFlatten(nn.Cell):
    def __init__(self, axis):
        super(CustomFlatten, self).__init__()
        self.axis = axis

    def construct(self, x):
        shape = mindspore.ops.shape(x)
        shape_before = shape[:self.axis]
        shape_after = shape[self.axis+1:]
        new_shape = shape_before + (-1,)
        return mindspore.ops.reshape(x, new_shape)


def gen_counting_label(labels, channel, tag):
    b, t = labels.shape
    counting_labels = ops.zeros((b, channel), mindspore.float32)
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    return counting_labels


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('尝试UTF-8编码....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    if not params['experiment']:
        print('实验名不能为空!')
        exit(-1)
    if not params['train_image_path']:
        print('训练图片路径不能为空！')
        exit(-1)
    if not params['train_label_path']:
        print('训练label路径不能为空！')
        exit(-1)
    if not params['word_path']:
        print('word dict路径不能为空！')
        exit(-1)
    if 'train_parts' not in params:
        params['train_parts'] = 1
    if 'valid_parts' not in params:
        params['valid_parts'] = 1
    if 'valid_start' not in params:
        params['valid_start'] = 0
    if 'word_conv_kernel' not in params['attention']:
        params['attention']['word_conv_kernel'] = 1
    return params


def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_checkpoint(model, optimizer, word_score, ExpRate_score, epoch, optimizer_save=False, path='checkpoints', multi_gpu=False, local_rank=0):
    filename = f'{os.path.join(path, model.name)}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'
    if optimizer_save:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.state_dict()
        }
    mindspore.save_checkpoint(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path):
    state = mindspore.load(path)
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')
    model.load_state_dict(state['model'])


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        # _, word_pred = word_probs.max(2)
        word_pred = word_probs.argmax(2)
    # word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
    #             len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
    #                for s1, s2, s3 in zip(word_label.to_tensor().detach().asnumpy(),
    #                                      word_pred.to_tensor().detach().asnumpy(),
    #                                      mask.cpu().detach().asnumpy())]
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
                len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(Tensor.from_numpy(word_label).detach().asnumpy(),
                                         Tensor.from_numpy(word_pred).detach().asnumpy(),
                                         mask.cpu().detach().asnumpy())]
    
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate


if __name__ == '__main__':
    pass