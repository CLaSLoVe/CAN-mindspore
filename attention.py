from mindspore import nn, ops
from mindspore import Tensor


class Attention(nn.Cell):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']
        self.hidden_weight = nn.Dense(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, pad_mode='pad', padding=5, has_bias=False)
        self.attention_weight = nn.Dense(512, self.attention_dim, has_bias=False)
        self.alpha_convert = nn.Dense(self.attention_dim, 1)

    def construct(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(ops.transpose(alpha_sum_trans, (0,2,3,1)))
        alpha_score = ops.tanh(query[:, None, None, :] + coverage_alpha + ops.transpose(cnn_features_trans, (0,2,3,1)))
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = ops.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_sum = alpha[:,None,:,:] + alpha_sum
        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)
        return context_vector, alpha, alpha_sum
