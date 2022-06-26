from __future__ import division
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
import time

#import wandb
#wandb.init(project="codesum")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':500,
    'CodeLen':3200,
    'SentenceLen':10,
    'batch_size':60,
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
    #print(negativemask.sum())
    # max_f1 = 0
    # threds = 0
    # for i in range(1, 10, 1):
    #     marco = 0.1 * i 
    #     premask = torch.ge(pred, marco)
    #     predtrue = pred.masked_fill(premask == 1, 1)
    #     predtrue = predtrue.masked_fill(premask==0, 0)
    #     print('marco is %f'%marco)
    #     F1, _ = getRecall(predtrue, negativemask, resp, test_set)
    #     if max_f1 < F1:
    #         max_f1 = F1
    #         threds = marco
    predtrue = pred.masked_fill(resp == 0, 1)
    # corindex = (predtrue < 1).nonzero()
    # median_marco = predtrue.gather(0, corindex.squeeze(-1)).median()


    idxsort = torch.argsort(predtrue, dim=-1)

    # for x in idxsort[:10]:
    #     print(x, test_set.order[x.item()], predtrue[x])

    marco = torch.min(predtrue)
    print('marco is %f'%marco.item())
    negativepred = torch.ge(pred, marco)
    pred1 = pred.masked_fill(negativepred == 1, 1)
    pred1 = pred1.masked_fill(negativepred == 0, 0)
    F1, recall = getRecall(pred1, negativemask, resp, test_set)

    return F1, marco.item(), recall


         

def getRecall(realpred, negativemask, resp, test_set):
    TP = torch.eq(realpred, resp).masked_fill(negativemask==0, 0).sum().item()
    TN = torch.eq(realpred, resp).masked_fill(negativemask==1, 0).sum().item()
    FP = (len(test_set) - negativemask.sum() - TN).item()
    FN = (negativemask.sum() - TP).item()

    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))

    if (TP + FP) == 0:
        precision = 10
    else:
        precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)

    if precision + recall == 0:
        F1 = 10
    else:
        F1 = 2 * precision * recall / (precision +  recall)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    return F1, recall
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
    patchdir = 'crossbug'
    testnum = 100
    train_pkldir = './result/%s_pkldir/trainSet_%s.pkl'%(patchdir,str(testnum))
    test_pkldir = './result/%s_pkldir/testSet_%s.pkl'%(patchdir,str(testnum))
    train_set = SumDataset(args, "train", path=train_pkldir)
    test_set = SumDataset(args, 'test', path=test_pkldir)
    #val_set = SumDataset(args, 'val')

    args.Vocsize = len(train_set.Char_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
    devloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False, num_workers=1)
    # valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size,
    #                                           shuffle=False, drop_last=False, num_workers=1)
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
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=1e-4), args.embedding_size, 4000)
    maxAcc = 0
    betterepoch = 0
    bestthreds = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    bestModel = None        
    minloss = 1e9
    devBatch = None
    tacc = -1
    train_start = time.time()
    for x in train_set.Get_Train(1, id, False):
        devBatch = x
    for epoch in range(81):
        j = 0
        for dBatch in tqdm(data_loader):
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], dBatch[7])
            loss = torch.mean(loss)

            if j % 3000 == 0:
                f1, threds, recall = evalacc(model, devloader, test_set)#scores[0]
                #f1, threds, recall = evalacc(model, devloader, test_set)#scores[0]
                #acc1 = evalacc(model, valloader, val_set)
                print('current recall, threds and epoch (%f, %f, %f)  '%(recall, threds, epoch))
                #assert(0)
                if epoch%10 == 0:
                    with open("finalresult/dynamic_pre100_%s_%s_4.txt"%(patchdir,str(testnum)), mode='a+', encoding='utf-8') as file_obj:
                        file_obj.write(str(f1) +  ',' + str(threds) + ',' + str(recall)+ ',' + str(epoch)  + '\n')
                # if maxAcc < f1 and f1 < 1:
                #     maxAcc = f1
                #     #tacc = acc1
                #     bestthreds= threds
                #     betterepoch = epoch
                #     #print("find better acc %s and epoch %s " + str(maxAcc) + ' ' + str(epoch))
                #     # with open("finalresult/%s_%s_2.txt"%(patchdir,str(testnum)), mode='a+', encoding='utf-8') as file_obj:
                #     #     file_obj.write(str(maxAcc) + ',' + str(epoch) + '\n')
                    save_model(model)
            test_start = time.time()
            f1, threds, recall = evalacc(model, devloader, test_set)#scores[0]
            test_end = time.time()
            print('test_time (%.2f)m' % (test_end - test_start))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            j += 1
    train_end = time.time()
    print('train_time (%.2f)m' % (train_end - train_start))
    
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



