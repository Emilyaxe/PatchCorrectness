import torch.nn as nn
import torch.nn.functional as F
from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from ConvolutionForward import ConvolutionLayer
from Multihead_Combination import MultiHeadedCombination
from TreeConv import TreeConv
from gcnn import GCNN
from LayerNorm import LayerNorm
import torch
class sAttention(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.lineara = nn.Linear(hidden, hidden)
        self.linearb = nn.Linear(hidden, hidden)
        self.linearc = nn.Linear(hidden, 1)


    def forward(self, a, b):
        att = self.linearc(self.lineara(a).unsqueeze(dim=0) + self.linearb(b))
        att = F.softmax(att.squeeze(dim=-1), dim=-1)
        #print(att.size(), b.size())
        return torch.sum(att.unsqueeze(dim=-1) * b, dim=0)#torch.matmul(att, b)
