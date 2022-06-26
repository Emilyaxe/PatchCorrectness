import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from Searchnode import Node
from scipy import sparse
import math
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
class Graph:
    def __init__(self):
        self.row = []
        self.col = []
        self.val = []
        self.edge = {}
        self.rowNum = 0
        self.colNum = 0
    def addEdge(self, r, c, v):
        if (r, c) in self.edge:
            #print(r, c)
            return
        self.edge[(r, c)] = len(self.row)
        self.row.append(r)
        self.col.append(c)
        self.val.append(v)
        self.edge[(c, r)] = len(self.row)
        self.row.append(c)
        self.col.append(r)
        self.val.append(v)
    def editVal(self, r, c, v):
        self.val[self.edge[(r, c)]] = v
    def updateval(self, index, v):
        self.val[index] = v
    def normlize(self):
        r = {}
        c = {}
        for i  in range(len(self.row)):
            if self.row[i] not in r:
                r[self.row[i]] = 0
            r[self.row[i]] += 1
            if self.col[i] not in c:
                c[self.col[i]] = 0
            c[self.col[i]] += 1
        for i in range(len(self.row)):
            self.val[i] = 1 / math.sqrt(r[self.row[i]]) * 1 / math.sqrt(c[self.col[i]])
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", path = '', proj="Math", testid=0, lst=[]):
        if dataName == 'train':
            self.train_path = path
        elif dataName == 'test':
            self.test_path = path
        self.val_path = "validateSet_200.pkl"  # "validD.txt"
        print(path)
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        if os.path.exists('nl_voc.pkl'):
            self.Load_Voc()
        else:
            self.init_dic()
        self.Codes = []
        self.ids = []
        self.Nls = []
        if dataName == "train":
            if os.path.exists("%sdata.pkl"%self.dataName):
                self.data = pickle.load(open("%sdata.pkl"%self.dataName, "rb"))
                self.order = pickle.load(open('%sorder.pkl'%self.dataName, 'rb'))
            else:
                self.data = self.preProcessData(open(self.train_path, "rb"))
        elif dataName == 'test':
            if os.path.exists("testdata.pkl"):
                self.data = pickle.load(open("testdata.pkl", "rb"))
                self.order = pickle.load(open('%sorder.pkl'%self.dataName, 'rb'))
            else:
                self.data = self.preProcessData(open(self.test_path, "rb"))
        elif dataName == 'val':
            if os.path.exists("valdata.pkl"):
                self.data = pickle.load(open("valdata.pkl", "rb"))
                self.order = pickle.load(open('%sorder.pkl'%self.dataName, 'rb'))
            else:
                self.data = self.preProcessData(open(self.val_path, "rb"))
            #self.data = pickle.load(open("testdata.pkl", "rb"))

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
    def init_dic(self):
        print("initVoc")
        f = open(self.train_path, 'rb')
        data = pickle.load(f)
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            Codes.append(data[x]['tree'].split())
        Codes.append(['fixtest', 'bugtest', 'special'])
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff = 0)
        self.Nl_Voc = code_voc.word2id
        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def tokenize_for_bleu_eval(self, code):
        code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
        #code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens
    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans
    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index(x) < self.Code_Len and x != "(" and x != ")":
                ans.append(len(self.Nl_Voc) + codetoken.index(x))
            else:
                if x in self.Nl_Voc:
                    ans.append(self.Nl_Voc[x])
                else:
                    ans.append(1)
        for x in ans:
            if x >= len(self.Nl_Voc) + self.Code_Len:
                print(codetoken, nltoken)
                exit(0)
        return ans
    def preProcessData(self, dataFile):
        liness = pickle.load(dataFile)#dataFile.readlines()
        inputNodes = []
        inputAd = []
        inputRes = []     # label
        inputPos = []      # posibility
        inputNlChar = []
        order = []
        inputPatch = []
        inputAdTreetoTest = []
        inputAdTesttoTest = []
        maxl = []
        for x in tqdm(liness):
            #if x in ['Lang58b_Patch26', 'Lang55b_Patch25', 'Chart5b_Patch7', 'Math93b_Patch207', 'Math50b_Patch46', 'Lang44b_Patch21']:
            #W    continue
            #if x in ['Chart5b_Patch7'] and self.dataName == 'test':
            #    continue
            order.append(x)
            x = liness[x]
            xs = x
            inputpos = x['prob']
            if x['label'] not in ['0', '1']:
                print(x['label'])
                assert(0)
            inputRes.append(int(x['label']))
            tree = x['tree']
            inputpos = self.pad_seq(inputpos, self.Nl_Len)
            nl = tree.split()
            node = Node(nl[0], 0)
            currnode = node
            idx = 1
            nltmp = [nl[0]]
            nodes = [node]
            for j, x in enumerate(nl[1:]):
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(x.father.id)
                        nladdata.append(1)
                    for s in x.father.child:
                        if x.id < self.Nl_Len and s.id < self.Nl_Len:
                            nladrow.append(x.id)
                            nladcol.append(s.id)
                            nladdata.append(1)
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)
            nl = nltmp
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)    
            inputnls = self.pad_seq(self.Get_Em(nl, self.Nl_Voc), self.Nl_Len)
            nlad = sparse.coo_matrix((nladdata, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            inputNodes.append(inputnls)
            inputPos.append(inputpos)
            inputAd.append(nlad)
            inputtest = ['special']
            grapht = Graph()
            grapht2t = Graph()
            #print(len(xs['cover']))
            for x in xs['fcover']:
                test = xs['fcover'][x]
                inputtest.append('fixtest')
                #inputtest.append('bugtest')
                #grapht2t.addEdge(len(inputtest) -2, len(inputtest) - 1, 2)
                #grapht2t.addEdge(len(inputtest) -1, len(inputtest) - 2, 3)
                #print(test)
                if 'fixed' in test:
                    for y in test['fixed']:
                        grapht.addEdge(len(inputtest) - 1, y, 1)
                        #grapht.addEdge(y, self.Nl_Len + len(inputtest) - 1, 2)
                if 'buggy' in test:
                    for y in test['buggy']:
                        if (len(inputtest) - 1, y) in grapht.edge:
                            grapht.editVal(len(inputtest) - 1, y, 2)
                        else: 
                            grapht.addEdge(len(inputtest) - 1, y, 3)
                        #grapht.addEdge(y, self.Nl_Len + len(inputtest) - 1, 4)
            num = 0
            for x in xs['pcover']:
                test = xs['pcover'][x]
                inputtest.append('bugtest')
                #inputtest.append('bugtest')
                #grapht2t.addEdge(len(inputtest) -2, len(inputtest) - 1, 2)
                #grapht2t.addEdge(len(inputtest) -1, len(inputtest) - 2, 3)
                #print(test)
                if 'fixed' in test:
                    for y in test['fixed']:
                        grapht.addEdge(len(inputtest) - 1, y, 1)
                        #grapht.addEdge(y, self.Nl_Len + len(inputtest) - 1, 2)
                if 'buggy' in test:
                    for y in test['buggy']:
                        if (len(inputtest) - 1, y) in grapht.edge:
                            grapht.editVal(len(inputtest) - 1, y, 2)
                        else: 
                            grapht.addEdge(len(inputtest) - 1, y, 3)
                        #grapht.addEdge(y, self.Nl_Len + len(inputtest) - 1, 4)
                if 'fixed' in test and 'buggy' in test:
                    lst1 = set(test['buggy'])
                    lst2 = set(test['fixed'])
                    if lst1 == lst2:
                        num += 1
            maxl.append(len(inputtest))
            inputPatch.append(self.pad_seq(self.Get_Em(inputtest, self.Nl_Voc), 500))
            inputAdTreetoTest.append(sparse.coo_matrix((grapht.val, (grapht.row, grapht.col)), shape= (500, self.Nl_Len)))
            inputAdTesttoTest.append(num)

        batchs = [inputNodes, inputPos, inputAd, inputNlChar, inputRes, inputPatch, inputAdTreetoTest, inputAdTesttoTest]
        self.data = batchs
        self.order = order
        print(np.max(maxl))
        open("%sdata.pkl"%self.dataName, "wb").write(pickle.dumps(batchs, protocol=4))
        open('%sorder.pkl'%self.dataName, 'wb').write(pickle.dumps(order, protocol=4))
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range(len(self.data)):
                if i == 2 or i == 6:
                    #torch.FloatTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])).float()
                    #torch.FloatTensor(self.data[i][offset].data)
                    #torch.FloatTensor(self.data[i][offset].data)
                    #ans.append(self.data[i][offset])
                    #ans.append(torch.sparse.FloatTensor(torch.LongTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])), torch.FloatTensor(self.data[i][offset].data).float(), torch.Size([self.Nl_Len,self.Nl_Len])))
                    #open('tmp.pkl', 'wb').write(pickle.dumps(self.data[i][offset]))
                    #assert(0)
                    ans.append(self.data[i][offset].toarray().astype(np.int64))
                    #print(self.data[i][offset].toarray()[0, 2545])
                    #assert(0)
                else:
                    ans.append(np.array(self.data[i][offset]).astype(np.int64))
        else:
            for i in range(len(self.data)):
                if i == 4:
                    continue
                ans.append(np.array(self.data[i][offset]))
            negoffset = random.randint(0, len(self.data[0]) - 1)
            while negoffset == offset:
                negoffset = random.randint(0, len(self.data[0]) - 1)
            if self.dataName == "train":
                ans.append(np.array(self.data[2][negoffset]))
                ans.append(np.array(self.data[3][negoffset]))
        return ans
    def __len__(self):
        return len(self.data[0])
    def Get_Train(self, batch_size, Maskid = -1, shuffles=True):
        data = self.data
        loaddata = data
        batch_nums = int(len(data[0]) / batch_size)
        if True:
            if shuffles:
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                shuffle = np.arange(len(loaddata[0]))
            if batch_nums * batch_size == len(data[0]):
                batch_nums -= 1
            for i in range(batch_nums + 1):
                #print(shuffle[i])
                if shuffle[i] != Maskid and not shuffles:
                    continue
                if shuffles and shuffle[i] == Maskid:
                    continue 
                ans = []
                bias4_0 = 0
                bias4_1 = 0
                bias5_0 = 0
                bias5_1 = 0
                bias6_0 = 0
                bias6_1 = 0
                for j in range(len(data)):
                    if j != 3:
                        tans = []
                        for k in shuffle[batch_size * i: batch_size * (i + 1)]:
                            tans.append(data[j][k])
                        ans.append(torch.from_numpy(np.concatenate(tans, axis=0)))
                        #tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        #ans.append(torch.from_numpy(np.array(tmpd)))
                    elif j == 3:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            if idx >= len(data[0]):
                                continue
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([data[j][shuffle[idx]].row[p] + bias4_0, data[j][shuffle[idx]].col[p] + bias4_1])
                                v.append(data[j][shuffle[idx]].val[p])
                            bias4_0 += data[j][shuffle[idx]].rowNum
                            bias4_1 += data[j][shuffle[idx]].colNum
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([bias4_0, bias4_1])))
                    elif j == 5:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            if idx >= len(data[0]):
                                continue
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([data[j][shuffle[idx]].row[p] + bias5_0, data[j][shuffle[idx]].col[p] + bias5_1])
                                #print(idx, p)
                                v.append(data[j][shuffle[idx]].val[p])
                            bias5_0 += data[j][shuffle[idx]].rowNum
                            bias5_1 += data[j][shuffle[idx]].colNum
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([bias5_0, bias5_1])))
                    elif j == 8:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            if idx >= len(data[0]):
                                continue
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([data[j][shuffle[idx]].row[p] + bias6_0, data[j][shuffle[idx]].col[p] + bias6_1])
                                v.append(data[j][shuffle[idx]].val[p])
                            bias6_0 += data[j][shuffle[idx]].rowNum
                            bias6_1 += data[j][shuffle[idx]].colNum
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([bias6_0, bias6_1])))
                yield ans
            '''
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size: ]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans'''
            
class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1
#from run.py import args
if __name__ == '__main__':
    SumDataset('train', args)
