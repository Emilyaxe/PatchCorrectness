import json
import random
lst = json.loads(open('../Dcorrect', 'r').read())
lst1 = json.loads(open('../Ddifferent', 'r').read())
lst2 = json.loads(open('../Doverfitting', 'r').read())
lst3 = json.loads(open('../Doverfitting2', 'r').read())
dic = {}
dic2 = {}
correctpatch = []
for x in lst:
    correctpatch.append((x, 'Dcorrect'))
    tmp = x.split('-')
    dic.setdefault(tmp[1]+tmp[2], []).append((x, 'Dcorrect'))
for x in lst1:
    correctpatch.append((x, 'Ddifferent'))
    tmp = x.split('-')
    dic.setdefault(tmp[1]+tmp[2], []).append((x, 'Ddifferent'))
overfitpatch = []
for x in lst2:
    overfitpatch.append((x, 'Doverfitting'))
    tmp = x.split('-')
    dic2.setdefault(tmp[1]+tmp[2], []).append((x, 'Doverfitting'))
for x in lst3:
    overfitpatch.append((x, 'Doverfitting2'))
    tmp = x.split('-')
    dic2.setdefault(tmp[1]+tmp[2], []).append((x, 'Doverfitting2'))
foldnum = len(correctpatch) // 10
visit = {}
correctset = {}
sortkey = sorted(list(dic.keys()), key=lambda x:len(dic[x]), reverse = True)
for x in sortkey:
    assert(len(dic[x]) < 10)
    visitm = {}
    for res in dic[x]:
        for i in range(10):
            if i in visitm:
                continue
            if i not in correctset or len(correctset[i]) < foldnum:
                correctset.setdefault(i, []).append(res)
                visit[res[0]] = 1
                visitm[i] = 1
                break
for x in correctpatch:
    if x[0] not in visit:
        correctset.setdefault(10, []).append(x)
pluspatch = {}
visit2 = {}
foldnum = len(overfitpatch) // 10

for i in range(10):
    plus = []
    while len(plus) < foldnum:
        idx = random.randint(0, len(overfitpatch) - 1)
        if idx not in visit2:
            plus.append(overfitpatch[idx])
            visit2[idx] = 1
    pluspatch[i] = plus
for idx in range(len(overfitpatch)):
    if idx not in visit2:
        pluspatch.setdefault(10, []).append(overfitpatch[idx])
final = {}
for i in correctset:
    final.setdefault(i, {})['correct'] = correctset[i]
    final.setdefault(i, {})['plus'] = pluspatch[i]
import pickle
open('datasetsplit.pkl', 'wb').write(pickle.dumps(final))

