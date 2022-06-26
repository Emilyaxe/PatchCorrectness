import torch
import random

from yaml import nodes
from Dataset import SumDataset
import numpy as np
import pickle
from tqdm import tqdm
from Searchnode import Node
from scipy import sparse
import time
import math
import statistics
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':500,
    'CodeLen':3200,
    'SentenceLen':10,
    'batch_size':120,
    'embedding_size':256,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'seed':19970316
})
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

def mainProcess():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    patchdir = 'crosspatch3'
    testnum = 100
    train_pkldir = './result/%s_pkldir/trainSet_%s.pkl'%(patchdir,str(testnum))
    T1 = time.time()
    buildModifyGraph(open(train_pkldir, "rb"))
    T2 = time.time()
    print('Modify:%fms' % ((T2 - T1)*1000/941.0))

    T1 = time.time()
    buildCoverageGraph(open(train_pkldir, "rb"))
    T2 = time.time()
    print('Coverage:%fms' % ((T2 - T1)*1000/941.0))

def pad_seq(self, seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [self.PAD_token] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq

def buildCoverageGraph(dataFile):
    liness = pickle.load(dataFile)#dataFile.readlines()
    nodelist = []
    edgelist = []
    for x in tqdm(liness):
        x = liness[x]
        xs = x
        grapht = Graph()
        inputtest = ['special']

        edgeset = set()
        nodeset = set()
        for x in xs['fcover']:
            test = xs['fcover'][x]
            inputtest.append('fixtest')

            if 'fixed' in test:
                for y in test['fixed']:
                    grapht.addEdge(len(inputtest) - 1, y, 1)
                    # nodeset.add(len(inputtest) - 1)
                    # nodeset.add(y)
                    # edgeset.add(addEdge(len(inputtest)-1, y))


            if 'buggy' in test:
                for y in test['buggy']:
                    if (len(inputtest) - 1, y) in grapht.edge:
                        grapht.editVal(len(inputtest) - 1, y, 2)
                    else: 
                        grapht.addEdge(len(inputtest) - 1, y, 3)

                        # nodeset.add(len(inputtest) - 1)
                        # nodeset.add(y)
                        # edgeset.add(addEdge(len(inputtest)-1, y))

        for x in xs['pcover']:
            test = xs['pcover'][x]
            inputtest.append('bugtest')
            if 'fixed' in test:
                for y in test['fixed']:
                    grapht.addEdge(len(inputtest) - 1, y, 1)

                    # nodeset.add(len(inputtest) - 1)
                    # nodeset.add(y)
                    # edgeset.add(addEdge(len(inputtest)-1, y))
            if 'buggy' in test:
                for y in test['buggy']:
                    if (len(inputtest) - 1, y) in grapht.edge:
                        grapht.editVal(len(inputtest) - 1, y, 2)
                    else: 
                        grapht.addEdge(len(inputtest) - 1, y, 3)
                        # nodeset.add(len(inputtest) - 1)
                        # nodeset.add(y)
                        # edgeset.add(addEdge(len(inputtest)-1, y))
        testad = sparse.coo_matrix((grapht.val, (grapht.row, grapht.col)), shape= (500, args.NlLen))
    #     nodelist.append(len(nodeset))
    #     edgelist.append(len(edgeset))
    # print('Coverage: Node Size: %f, Edge Size: %f'%(statistics.mean(nodelist), statistics.mean(edgelist)))


def addEdge(a,b):
    tmplist = []
    tmplist.append(a)
    tmplist.append(b)
    tmplist.sort()
    #print(tmplist[0]+tmplist[1])
    return str(tmplist[0])+ '-' + str(tmplist[1])


def buildModifyGraph(dataFile):
    liness = pickle.load(dataFile)#dataFile.readlines()
    print(len(liness))

    order = []
    nodelist = []
    edgelist = []
    for x in tqdm(liness):
        order.append(x)
        x = liness[x]

        tree = x['tree']
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
        edgeset = set()
        nodeset = set()
        for x in nodes:
            if x.father:
                if x.id < args.NlLen and x.father.id < args.NlLen:
                    nladrow.append(x.id)
                    nladcol.append(x.father.id)
                    nladdata.append(1)


                    # nodeset.add(x.id)
                    # nodeset.add(x.father.id )
                    # pair = addEdge(x.id, x.father.id)
                    # edgeset.add(pair)
                for s in x.father.child:
                    if x.id < args.NlLen and s.id < args.NlLen:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)

                        # nodeset.add(x.id)
                        # nodeset.add(s.id)
                        # pair = addEdge(x.id, s.id)
                        # edgeset.add(pair)
            for s in x.child:
                if x.id < args.NlLen and s.id < args.NlLen:
                    nladrow.append(x.id)
                    nladcol.append(s.id)
                    nladdata.append(1)


                    # nodeset.add(x.id)
                    # nodeset.add(s.id)
                    # pair = addEdge(x.id, s.id)
                    # edgeset.add(pair)
        nl = nltmp
        nlad = sparse.coo_matrix((nladdata, (nladrow, nladcol)), shape=(args.NlLen, args.NlLen))
    #     nodelist.append(len(nodeset))
    #     edgelist.append(len(edgeset))
    # print('Modify: Node Size: %f, Edge Size: %f'%(statistics.mean(nodelist), statistics.mean(edgelist)))
    
    





if __name__ == '__main__':
    mainProcess()
