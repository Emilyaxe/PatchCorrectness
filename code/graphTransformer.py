import torch.nn as nn

from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from ConvolutionForward import ConvolutionLayer
from TreeConv import TreeConv
from gcnn2 import GCNNSplit
from relAttention import relAttention
class graphTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention1 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention2 = relAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.conv_forward = ConvolutionLayer(dmodel=hidden, layernum=hidden)
        self.Tconv_forward = GCNNSplit(dmodel=hidden)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer5 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, left, leftmask, edgeem):
        x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        #x = self.sublayer5(x, lambda _x: self.combination.forward(_x, _x, charem))
        x = self.sublayer3(x, lambda _x: self.attention2.forward(_x, left, left, mask=leftmask, relem=edgeem))
        #x = self.sublayer2(x, lambda _x: self.attention2(_x, left, left, mask=nlmask, relem=edgeem))
        x = self.sublayer4(x, self.feed_forward)
        return self.dropout(x)
