import subprocess
from tqdm import tqdm
import time
import os
import pickle
import sys
card = [0, 1, 4, 5, 6, 7]
singlenum = 6
#lst = list(range(11))
'''for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        cardn =int(j / singlenum)
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python3 run.py " + str(lst[totalnum * i + j]) + " " + pr, shell=True)#subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card) + " python3 testDefect4j.py " + lst[12 * i + j], stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, shell=True)#subprocess.run(["python3", "run.py"])
                #p = subprocess.Popen("python3 run.py", stdout=subprocess.PIPE, shell=True)#subprocess.run(["python3", "run.py"])
        jobs.append(p)
        time.sleep(10)
                #jobs.append(1)
    for p in jobs:
        p.wait()'''
jobs = []
cardjob = {}
cardd = {}
for x in card:
    cardd[x] = 0
for i in range(0, 40):
    acard = -1
    while acard == -1:
        for x in cardd:
            cardd[x] = 0
        for x in jobs:
            if x.poll() is None:
                cardd[cardjob[x]] += 1
        for x in cardd:
            if cardd[x] < singlenum:
                acard = x
                break
        time.sleep(1)
    p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(acard) + " python3 run.py " + str(i) + " " + "train", shell=True)
    cardjob[p] = acard
    jobs.append(p)
    time.sleep(10)
