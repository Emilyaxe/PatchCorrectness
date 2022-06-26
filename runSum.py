import torch
from torch import optim
from DatasetSum import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':40,
    'CodeLen':300,
    'batch_size':64,
    'embedding_size':256,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100
})
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def save_model(model):
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint")
    torch.save(model.state_dict(), 'checkpoint/best_model.ckpt')


def load_model(model):
    assert os.path.exists('checkpoint/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load('checkpoint/best_model.ckpt'))
use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    #print(type(tensor))
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
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
def train():
    train_set = SumDataset(args, "train")
    print(len(train_set.data[0]))
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    dev_set = SumDataset(args, "val")
    print(len(dev_set.data[3]))
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
    model = Decoder(args)
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    maxAcc = 0
    antimask = gVar(getAntiMask(args.NlLen))
    for epoch in range(100000):
        model = model.train()
        for j, dBatch in tqdm(enumerate(data_loader)):
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], antimask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 300 == 0:
                devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
                model = model.eval()
                accs = []
                for devBatch in tqdm(devloader):
                    for i in range(len(devBatch)):
                        devBatch[i] = gVar(devBatch[i])
                    with torch.no_grad():
                        _, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], antimask)
                        pred = pre.argmax(dim=-1)
                        resmask = torch.gt(devBatch[5], 0)
                        acc = (torch.eq(pred, devBatch[5]) * resmask).float().mean(dim=-1)
                        resTruelen = torch.sum(resmask, dim=-1).float()
                        acc = acc * args.NlLen / resTruelen
                        accs.append(acc.mean().item())
                acc = np.mean(accs)
                if maxAcc < acc:
                    maxAcc = acc
                    print("find better acc " + str(maxAcc))
                    save_model(model)
                model = model.train()

if __name__ == "__main__":
     train()




