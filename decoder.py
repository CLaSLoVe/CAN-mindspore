import mindspore
from mindspore import nn, ops
from mindspore import Tensor
import numpy as np
import math
from attention import Attention
from utils import CustomFlatten


class PositionEmbeddingSine(nn.Cell):
	"""
	This is a more standard version of the position embedding, very similar to the one
	used by the Attention is all you need paper, generalized to work on images.
	"""

	def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
		super().__init__()
		self.num_pos_feats = num_pos_feats
		self.temperature = temperature
		self.normalize = normalize
		if scale is not None and normalize is False:
			raise ValueError("normalize should be True if scale is passed")
		if scale is None:
			scale = 2 * math.pi
		self.scale = scale
		self.flatten_op = CustomFlatten(axis=3)

	def construct(self, x, mask):
		y_embed = mask.cumsum(1, dtype=mindspore.float32)
		x_embed = mask.cumsum(2, dtype=mindspore.float32)
		if self.normalize:
			eps = 1e-6
			y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
			x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

		dim_t = mindspore.numpy.arange(self.num_pos_feats, dtype=mindspore.float32)
		dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

		pos_x = x_embed[:, :, :, None] / dim_t
		pos_y = y_embed[:, :, :, None] / dim_t
		pos_x = self.flatten_op(ops.stack((ops.sin(pos_x[:, :, :, 0::2]), ops.cos(pos_x[:, :, :, 1::2])), axis=4))
		pos_y = self.flatten_op(ops.stack((ops.sin(pos_y[:, :, :, 0::2]), ops.cos(pos_y[:, :, :, 1::2])), axis=4))
		pos = ops.transpose(ops.concat((pos_y, pos_x), axis=3), (0, 3, 1, 2))
		return pos


class AttDecoder(nn.Cell):
	def __init__(self, params):
		super(AttDecoder, self).__init__()
		self.params = params
		self.input_size = params['decoder']['input_size']
		self.hidden_size = params['decoder']['hidden_size']
		self.out_channel = params['encoder']['out_channel']
		self.attention_dim = params['attention']['attention_dim']
		self.dropout_prob = params['dropout']
		self.word_num = params['word_num']
		self.counting_num = params['counting_decoder']['out_channel']

		"""经过cnn后 长宽与原始尺寸比缩小的比例"""
		self.ratio = params['densenet']['ratio']

		# init hidden state
		self.init_weight = nn.Dense(self.out_channel, self.hidden_size)
		# word embedding
		self.embedding = nn.Embedding(self.word_num, self.input_size)
		# word gru
		self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
		# attention
		self.word_attention = Attention(params)
		self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim, kernel_size=params['attention']['word_conv_kernel'], pad_mode='pad', padding=params['attention']['word_conv_kernel'] // 2)

		self.word_state_weight = nn.Dense(self.hidden_size, self.hidden_size)
		self.word_embedding_weight = nn.Dense(self.input_size, self.hidden_size)
		self.word_context_weight = nn.Dense(self.out_channel, self.hidden_size)
		self.counting_context_weight = nn.Dense(self.counting_num, self.hidden_size)
		self.word_convert = nn.Dense(self.hidden_size, self.word_num)

		if params['dropout']:
			self.dropout = nn.Dropout(params['dropout_ratio'])

	def construct(self, cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=True):
		batch_size, num_steps = labels.shape
		height, width = cnn_features.shape[2:]
		word_probs = ops.zeros((batch_size, num_steps, self.word_num), mindspore.float32)
		images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

		word_alpha_sum = ops.zeros((batch_size, 1, height, width), mindspore.float32)
		word_alphas = ops.zeros((batch_size, num_steps, height, width), mindspore.float32)
		hidden = self.init_hidden(cnn_features, images_mask)
		counting_context_weighted = self.counting_context_weight(counting_preds)

		cnn_features_trans = self.encoder_feature_conv(cnn_features)
		position_embedding = PositionEmbeddingSine(256, normalize=True)
		pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])
		cnn_features_trans = cnn_features_trans + pos

		if is_train:
			for i in range(num_steps):
				word_embedding = self.embedding(labels[:, i - 1]) if i else self.embedding(
					ops.ones(batch_size, mindspore.int64))
				hidden = self.word_input_gru(word_embedding, hidden)
				word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden, word_alpha_sum, images_mask)

				current_state = self.word_state_weight(hidden)
				word_weighted_embedding = self.word_embedding_weight(word_embedding)
				word_context_weighted = self.word_context_weight(word_context_vec)

				if self.params['dropout']:
					word_out_state = self.dropout(
						current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
				else:
					word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

				word_prob = self.word_convert(word_out_state)
				word_probs[:, i] = word_prob
				word_alphas[:, i] = word_alpha
		else:
			word_embedding = self.embedding(ops.ones(batch_size, mindspore.int64))
			for i in range(num_steps):
				hidden = self.word_input_gru(word_embedding, hidden)
				word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden, word_alpha_sum, images_mask)

				current_state = self.word_state_weight(hidden)
				word_weighted_embedding = self.word_embedding_weight(word_embedding)
				word_context_weighted = self.word_context_weight(word_context_vec)

				if self.params['dropout']:
					word_out_state = self.dropout(
						current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
				else:
					word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

				word_prob = self.word_convert(word_out_state)
				# _, word = word_prob.max(axis=1, return_indices=True)
				word = word_prob.argmax(1)
				word_embedding = self.embedding(word)
				word_probs[:, i] = word_prob
				word_alphas[:, i] = word_alpha
		return word_probs, word_alphas

	def init_hidden(self, features, feature_mask):
		A = ops.reduce_sum(ops.reduce_sum((features * feature_mask).astype(mindspore.float32), axis=-1), axis=-1)
		B = ops.reduce_sum(ops.reduce_sum(feature_mask.astype(mindspore.float32), axis=-1), axis=-1)
		average = A.astype(mindspore.float32) / B.astype(mindspore.float32)
		average = self.init_weight(average)
		return ops.tanh(average)


if __name__ == '__main__':
	mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

	# Define the parameters
	params = {
		'decoder': {'input_size': 256, 'hidden_size': 256},
		'encoder': {'input_channel': 1, 'out_channel': 684},
		'attention': {'attention_dim': 512, 'word_conv_kernel': 1},
		'word_num': 111,
		'counting_decoder': {'out_channel': 111},
		'densenet': {'ratio': 16},
		'dropout_ratio': 0.5,
		'dropout': True,
		'optimizer': "Adadelta",
		'lr': 0.01,
		'weight_decay': 0.000001
	}

	# Create dummy input tensors
	cnn_features = Tensor(np.random.randn(8, 684, 8, 52).astype(np.float32))
	labels = Tensor(np.random.randint(low=0, high=10, size=(8, 50), dtype=np.int32))
	counting_preds = Tensor(np.random.random((8, 111)), dtype=mindspore.float32)
	images_mask = Tensor(np.ones((8, 1, 128, 823), dtype=np.float32))
	labels_mask = Tensor(np.ones((8, 50), dtype=np.float32))

	# Create an instance of the AttDecoder model
	model = AttDecoder(params)

	# Test forward pass
	word_probs, word_alphas = model(cnn_features, labels, counting_preds, images_mask, labels_mask)
	#
	# # Print the output shapes
	# print("Word Probs shape:", word_probs.shape)
	# print("Word Alphas shape:", word_alphas.shape)
