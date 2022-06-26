import torch
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
#from annoy import AnnoyIndex
from nltk import word_tokenize
import pickle
from ScheduledOptim import ScheduledOptim
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import random
from memory_profiler import profile
from copy import deepcopy
#import wandb
#wandb.init(project="codesum")
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
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ["CUDA_VISIBLE_DEVICES"]="4, 5, 6, 0"
def save_model(model, dirs = "checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))
def evalscore(trans, ground):
    score = 0
    recall = 0
    for i in range(len(trans)):
        ori = ground[i][0]
        pre = 0
        lll = []
        for key in trans[i]:
            if key not in  ["Unknown", "unknown"]:
                lll.append(key)
        trans[i] = lll
        for t in range(len(trans[i])):
            word = trans[i][t]
            if word in ori:
                pre += 1
        score += float(pre) / max(1, len(trans[i]))
        pre = 0 
        for t in range(len(ori)):
            word = ori[t]
            if word in trans[i]:
                pre += 1
        recall += float(pre) / len(ori) 
    score /= len(trans)
    recall /= len(trans)
    return score, recall
use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def getAntiMask(size):
  ans = np.zeros([size, size])
  for i in range(size):
    for j in range(0, i + 1):
      ans[i, j] = 1.0
  return ans
def getAdMask(size):
  ans = np.zeros([size, size])
  for i in range(size - 1):
      ans[i, i + 1] = 1.0
  return ans
def getAhMask(size):
  ans = np.zeros([size, size])
  for i in range(size - 1):
      ans[i + 1, i] = 1.0
  return ans
def evalacc(model, devloader, test_set):
    model = model.eval()
    scores = []
    preds = []
    masks = []
    realpreds = []
    ress = []
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        with torch.no_grad():
            loss, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7])
            pred = pre[:,1]
            preds.append(pred)
            ress.append(devBatch[4])
            realpred = torch.argmax(pre, dim=-1)
            negativemask = torch.eq(devBatch[4], 0)
                        
            masks.append(negativemask)
            realpreds.append(realpred)
            realpred = torch.cat(realpreds, dim=0)
            pred = torch.cat(preds, dim=0)
            negativemask = torch.cat(masks, dim=0)
            resp = torch.cat(ress, dim=0)
            print(realpred.size(), len(test_set))
            for i in range(1, 2, 1):
                predtrue = pred.masked_fill(resp == 0, 1)
                i = 0.1 * i
                marco = torch.min(predtrue)#.min()
                idxs = torch.argmin(predtrue, dim=-1).item()
                idxsort = torch.argsort(predtrue, dim=-1)
                for x in idxsort[:10]:
                    print(x, test_set.order[x.item()], predtrue[x])
                            #print('less score is %d' % )
                print(torch.eq(realpred, resp).sum().float() / len(test_set))
                            #print(realpred, realpred.size(), pred)
                print(torch.eq(realpred, resp).masked_fill(negativemask==1, 0).sum().float() / (len(test_set) - negativemask.sum().float()))
                negativepred = torch.ge(pred, marco)
                pred1 = pred.masked_fill(negativepred == 1, 1)
                pred1 = pred1.masked_fill(negativepred == 0, 0)
                res = torch.eq(pred1, resp) * negativemask
                            #if i == 0.5:
                scores.append((res.sum().float() / negativemask.sum().float()).item())
                print(i, 'recall is %f'%(res.sum().float() / negativemask.sum().float()).item())
                print((len(test_set) - negativemask.sum().float()))
                print(i, 'precision is %f'%(torch.eq(pred1, resp).masked_fill(negativemask==1, 0).sum().float() / (len(test_set) - negativemask.sum().float()).item()))
    return scores[0]
@profile(precision=4, stream=open('memory_profiler.log', 'w+'))
def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_set = SumDataset(args, "train")
    test_set = SumDataset(args, 'test')
    val_set = SumDataset(args, 'val')

    args.Vocsize = len(train_set.Char_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
    devloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False, num_workers=1)
    valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False, num_workers=1)
    model = NlEncoder(args)
    #torch.cuda.set_device(4)

    if use_cuda:
        print('using GPU')
        model = model.cuda()
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1, 2, 3])
    #load_model(model)
    #wandb.watch(model)
    #model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1,  2, 3, 4, 5, 6, 7])
    #nlem = pickle.load(open("embedding.pkl", "rb"))
    #codeem = pickle.load(open("code_embedding.pkl", "rb"))
    #model.encoder.token_embedding.token.em.weight.data.copy_(gVar(nlem))
    #model.token_embedding.token.em.weight.data.copy_(gVar(codeem))
    #model.token_embedding.token.em.weight.data.copy_(gVar(codeem))
    maxl = 1e9
    #optimizer = ScheduledOptim(adamod.AdaMod(model.parameters(), lr=1e-3, beta3=0.999), args.embedding_size, 4000)
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=5e-5), args.embedding_size, 4000)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    bestModel = None        
    minloss = 1e9
    devBatch = None
    tacc = -1
    for x in train_set.Get_Train(1, id, False):
        devBatch = x
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):
            if j % 3000 == 0:
                #print(len(dev_set))
                #devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=len(dev_set),
                #                              shuffle=False, drop_last=True, num_workers=1)
                

                acc = evalacc(model, devloader, test_set)#scores[0]
                acc1 = evalacc(model, valloader, val_set)
                print('current accuracy is %f max acc is (%f, %f) test acc is %f'%(acc, maxAcc, tacc, acc1))
                #assert(0)
                if maxAcc < acc:
                    maxAcc = acc
                    tacc = acc1
                    print("find better acc " + str(maxAcc))
                    save_model(model)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], dBatch[7])
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            j += 1
import sys
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    train()
    '''res = {}
    for i in range(100):
        a = train(i)
        res[i] = a 
        open('res.pkl', 'wb').write(pickle.dumps(res))'''
    #test()
    #trainsearch()
    #combinetrain()



