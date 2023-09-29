from densenet import DenseNet
from encoder import CountingDecoder
from decoder import AttDecoder
from utils import gen_counting_label
import mindspore
from mindspore import Tensor
import numpy as np
from mindspore import nn, ops


class CAN(nn.Cell):
	def __init__(self, params=None):
		super(CAN, self).__init__()
		self.params = params
		self.use_label_mask = params['use_label_mask']
		self.encoder = DenseNet(params=self.params)
		self.in_channel = params['counting_decoder']['in_channel']
		self.out_channel = params['counting_decoder']['out_channel']
		self.counting_decoder1 = CountingDecoder(self.in_channel, self.out_channel, 3)
		self.counting_decoder2 = CountingDecoder(self.in_channel, self.out_channel, 5)
		self.decoder = AttDecoder(params=self.params)
		self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
		self.counting_loss = nn.SmoothL1Loss(reduction='mean')

		"""经过cnn后 长宽与原始尺寸比缩小的比例"""
		self.ratio = params['densenet']['ratio']

	def construct(self, images, images_mask, labels, labels_mask, is_train=True):
		cnn_features = self.encoder(images)

		counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
		counting_labels = gen_counting_label(labels, self.out_channel, True)
		print('\t can step 1')
		counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
		print('\t can step 2')
		counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
		print('\t can step 3')
		counting_preds = (counting_preds1 + counting_preds2) / 2
		counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) + self.counting_loss(counting_preds, counting_labels)
		print('\t can step 4')
		word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=is_train)
		print('\t can step 5')
		word_loss = self.cross(word_probs.view(-1, word_probs.shape[-1]), labels.view(-1))
		print('\t can step 6')
		word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (
			labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss
		return word_average_loss + counting_loss, word_probs, counting_preds, word_average_loss, counting_loss


if __name__ == '__main__':
	mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
	from dataset import get_crohme_dataset
	from utils import load_config

	config_file = 'config.yaml'
	params = load_config(config_file)
	can = CAN(params)
	images, images_mask, labels, labels_mask = next(iter(get_crohme_dataset(params)[0]))
	for i in range(1):
		loss, probs, counting_preds, word_loss, counting_loss = can(images.astype(np.float32), images_mask.astype(np.int32), labels.astype(np.int32), labels_mask.astype(np.int32), is_train=True)
		print(loss)