import pickle
import os
import sys
import numpy as np
versionNum = {'Lang': 65, 'Time': 27, 'Chart': 26, 'Math': 106, 'Closure': 133, 'Mockito': 38,
                   'CommonsCli': 24, 'CommonsCodec': 22, 'CommonsCsv': 12, 'CommonsJXPath': 14,
                   'JacksonCore': 13, 'JacksonDatabind': 39, 'JacksonXml': 5, 'Jsoup': 63}
proj = sys.argv[1]
t = {}
for i in range(0, 40):
    if not os.path.exists(proj + 'res%d.pkl'%i):
        continue
    p = pickle.load(open(proj + 'res%d.pkl'%i, 'rb'))
    for x in p:
        print(x)
        ps = np.array(p[x][0])
        t[x] = ps[:,1]
open(proj + 'res.pkl', 'wb').write(pickle.dumps(t))
print(len(t))
