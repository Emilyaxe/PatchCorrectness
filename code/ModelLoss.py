import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from rightnTransfomer import rightTransformerBlock
from Embedding import Embedding
from Multihead_Attention import MultiHeadedAttention
from postionEmbedding import PositionalEmbedding
import numpy as np
class TreeAttEncoder(nn.Module):
    def __init__(self, args):
        super(TreeAttEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])
        self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])


    def forward(self, input_code, input_codechar, inputAd):
        codemask = torch.gt(input_code, 0)
        charEm = self.char_embedding(input_codechar)
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        #print(charEm.shape)
        x = self.token_embedding(input_code.long())
        for trans in self.transformerBlocksTree:
            x = trans.forward(x, codemask, charEm, inputAd, True)
        for trans in self.transformerBlocks:
            x = trans.forward(x, codemask, charEm)
        return x
class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(6)])
        self.token_embedding = Embedding(args.Nl_Vocsize, self.embedding_size)
        '''self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])'''


    def forward(self, input_nl, input_nlchar):
        nlmask = torch.gt(input_nl, 0)
        charEm = self.char_embedding(input_nlchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_nl.long())
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, charEm)
        return x
class CodeEncoder(nn.Module):
    def __init__(self, args):
        super(CodeEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        '''self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])'''


    def forward(self, input_nl, input_nlchar, lineEm):
        nlmask = torch.gt(input_nl, 0)
        charEm = self.char_embedding(input_nlchar.long())
        #print(input_nlchar.size())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_nl.long(), True, lineEm)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, charEm)
        return x
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.word_len = args.WoLen
        self.nl_len = args.NlLen
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.resLen = args.Code_Vocsize
        self.transformerBlocks = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(6)])
        self.encoder = NlEncoder(args)
        self.finalLinear = nn.Linear(self.embedding_size, 1024)
        self.resLinear = nn.Linear(1024, self.resLen)
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.CrossEntry = nn.CrossEntropyLoss()
        '''self.adm = np.zeros([args.CodeLen, args.CodeLen])
        self.adh = np.zeros([args.CodeLen, args.CodeLen])
        for i in range(args.CodeLen - 1):
            self.adm[i, i + 1] = 1
            self.adh[i + 1, i] = 1
        self.adm = torch.from_numpy(self.adm).float()
        self.adh = torch.from_numpy(self.adh).float()'''
        self.zeros = torch.zeros(args.CodeLen, self.resLen)
        #self.register_buffer('zeros', zeros)
    def getSquareLoss(self, a, b, mask):
        tmpLoss = a.unsqueeze(1) - b.unsqueeze(2)
        tmpLoss = torch.sum(tmpLoss * tmpLoss, dim=-1)
        tmpLoss = tmpLoss.masked_fill(mask==0, 1e9)
        tmpLoss = torch.min(tmpLoss, dim=-1)[0]
        tmpLoss = torch.sum(tmpLoss, dim=-1)
        return tmpLoss
        
    def forward(self, input_nl, input_nlchar, input_code, input_codechar, inputres, antimask, adm, adh):
        codemask = antimask.unsqueeze(0).repeat(input_code.size(0), 1, 1).unsqueeze(1)
        resmask = torch.gt(inputres, 0)
        nlmask = torch.gt(input_nl, 0)
        charEm = self.encoder.char_embedding(input_codechar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        encodes = self.encoder(input_nl, input_nlchar)
        x = self.token_embedding(input_code.long())
        for trans in self.transformerBlocks:
            x = trans.forward(x, codemask, encodes, nlmask, charEm)
        x = self.finalLinear(x)
        x = self.resLinear(x)
        res = torch.softmax(x, dim=-1)
        x = x.permute(0, 2, 1)
        loss = self.CrossEntry(x, inputres)
        loss = loss.masked_fill(resmask == 0, 0.0)
        totalloss = torch.mean(loss)
        #bleuloss
        zeros = self.zeros.repeat(input_code.size(0), 1, 1).cuda()
        #print(self.zeros.size(), inputres.size())
        onehot = zeros.scatter_(2, inputres.unsqueeze(-1).long(), 1)
        #print(resmask.unsqueeze(1).size(), resmask.unsqueeze(2).size())
        lossmask = torch.mul(resmask.unsqueeze(1), resmask.unsqueeze(2))
        #print(lossmask.size(), self.adm.size(), self.adh.size())
        adm = adm.unsqueeze(0).repeat(input_code.size(0), 1, 1).cuda() * lossmask.float()
        adh = adh.unsqueeze(0).repeat(input_code.size(0), 1, 1).cuda() * lossmask.float()
        res1 = torch.matmul(adm, res)
        onehot1 = torch.matmul(adm, onehot)
        #res2 = torch.matmul(self.adm, res1)
        #onehot2 = torch.matmul(self.adm, onehot1)
        #res3 = torch.matmul(self.adm, res2)
        #onehot3 = torch.matmul(self.adm, onehot2)
        #1-gram
        #loss1 = self.getSquareLoss(res, onehot, lossmask)
        #2-gram
        lossmask = torch.matmul(torch.matmul(adm, lossmask.float()), adh)
        loss2 = self.getSquareLoss(res + res1, onehot + onehot1, lossmask).sum()
        #3-gram
        #lossmask = torch.matmul(torch.matmul(self.adm, lossmask.float()), self.adh)
        #loss3 = self.getSquareLoss(res + res1 + res2, onehot1 + onehot2 + onehot3, lossmask)
        #4-gram
        #lossmask = torch.matmul(torch.matmul(self.adm, lossmask.float()), self.adh)
        #loss4 = self.getSquareLoss(res + res1 + res2 + res3, onehot + onehot1 + onehot2 + onehot3, lossmask)
        loss = loss2#loss1 + loss2 + loss3 + loss4
        return loss, res#totalloss, res
class LineEncoder(nn.Module):
    def __init__(self, args):
        super(LineEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = CodeEncoder(args)
        self.conv1_1 = nn.Conv2d(self.embedding_size, self.embedding_size, (3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(self.embedding_size, self.embedding_size, (3, 3), padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2_1 = nn.Conv2d(self.embedding_size, self.embedding_size, (3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(self.embedding_size, self.embedding_size, (3, 3), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.linear = nn.Linear(5 * 5 * self.embedding_size, self.embedding_size)
        self.lineEmbedding = PositionalEmbedding(self.embedding_size)
        self.activate = nn.ReLU()

    def forward(self, inputCode, inputCodeChar):
        lineEm = self.lineEmbedding(inputCode)
        lineEms = torch.split(lineEm, 1, dim=1)
        codelines = torch.split(inputCode, 1, dim=1)
        codelineschar = torch.split(inputCodeChar, 1, dim=1)
        codelinesEncoding = []
        codelinesmask = []
        linemask = torch.gt(torch.sum(inputCode, dim=-1), 0)
        for i in range(len(codelines)):
            codelinesEncoding.append(self.codeEncoder(codelines[i].squeeze(1), codelineschar[i].squeeze(1), lineEms[i].squeeze(1)))
        codeImage = torch.stack(codelinesEncoding, 1)
        codeImage = self.activate(self.conv1_1(codeImage.permute(0, 3, 1, 2)))
        codeImage = self.activate(self.conv1_2(codeImage))
        codeImage = self.maxpool1(codeImage)
        codeImage = self.activate(self.conv2_1(codeImage))
        codeImage = self.activate(self.conv2_2(codeImage))
        codeImage = self.maxpool2(codeImage)
        finalOutput = self.linear(codeImage.permute(0, 2, 3, 1).flatten(start_dim=1))
        return finalOutput

class JointEmbber(nn.Module):
    def __init__(self, args):
        super(JointEmbber, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = LineEncoder(args)
        self.margin = args.margin
        self.nlEncoder = NlEncoder(args)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.poolConvcode = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolnl = nn.MaxPool1d(args.NlLen)
        self.maxPoolcode = nn.MaxPool1d(args.CodeLen)
    def scoring(self, qt_repr, cand_repr):
        sim = F.cosine_similarity(qt_repr, cand_repr)
        return sim
    def nlencoding(self, inputnl, inputnlchar):
        nl = self.nlEncoder(inputnl, inputnlchar)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        return nl
    def codeencoding(self, inputcode, inputcodechar):
        code = self.codeEncoder(inputcode, inputcodechar)
        return code
    def forward(self, inputnl, inputnlchar, inputcode, inputcodechar, inputcodeneg, inputcodenegchar):
        #print(inputcode.size(), inputcodechar.size())
        nl = self.nlencoding(inputnl, inputnlchar)
        code = self.codeencoding(inputcode, inputcodechar)
        codeneg = self.codeencoding(inputcodeneg, inputcodenegchar)
        good_score = self.scoring(nl, code)
        bad_score = self.scoring(nl, codeneg)
        loss = (self.margin - good_score + bad_score).clamp(min=1e-6).mean()
        return loss, good_score, bad_score










