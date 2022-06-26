import pickle
import sys
pr = sys.argv[1]
def splitCamel(token):
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
p = pickle.load(open(pr + 'res.pkl', 'rb'))
s = 0
b = {}
for x in p:
    if len(p[x]) != 0:
        s += 1
        for v in p[x]:
            if v not in b:
                b[v] = 0
            b[v] += 1
print(s)
print(b)