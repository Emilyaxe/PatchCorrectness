# coding=utf-8
import json
import os
import pickle

import javalang

from Searchnode import Node

lst = ['trainSet', 'testSet']
line_node_name = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement',
                  'ContinueStatement_ter', 'LocalVariableDeclaration', 'condition', 'control', 'BreakStatement',
                  'ContinueStatement', 'ReturnStatement', "parameters", 'StatementExpression', 'return_type']


# os.environ["CUDA_VISIBLE_DEVICES"]="1, 4"
def get_local_var(node):
    varnames = []
    if node.name == 'VariableDeclarator':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    if node.name == 'FormalParameter':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    if node.name == 'InferredFormalParameter':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    for x in node.child:
        varnames.extend(get_local_var(x))
    return varnames


n = 0


def setid(root):
    global n
    root.id = n
    n += 1
    for x in root.child:
        setid(x)


def getNodeById(root, line):
    if root.position:
        if root.position == line and root.name != 'IfStatement' and root.name != 'ForStatement':
            return root
    for x in root.child:
        t = getNodeById(x, line)
        if t:
            return t
    return None


def getNodeById2(root, line):
    if root.position2:
        if root.position2 == line and root.name != 'IfStatement' and root.name != 'ForStatement':
            return root
    for x in root.child:
        t = getNodeById2(x, line)
        if t:
            return t
    return None


def solveLongTree(root, subroot, maxsize):
    global n
    m = 'None'
    troot = 'None'
    for x in root.child:
        if x.name == 'name':
            m = x.child[0].name
    if len(root.getTreestr().strip().split()) >= maxsize:
        tmp = subroot
        if len(tmp.getTreestr().split()) >= maxsize:
            assert (0)
        lasttmp = None
        while True:
            if len(tmp.getTreestr().split()) >= maxsize:
                break
            lasttmp = tmp
            tmp = tmp.father
        index = tmp.child.index(lasttmp)
        ansroot = Node(tmp.name, 0)
        ansroot.child.append(lasttmp)
        ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
        while True:
            b = True
            afternode = tmp.child.index(ansroot.child[-1]) + 1
            if afternode < len(tmp.child) and ansroot.num + tmp.child[afternode].getNum() < maxsize:
                b = False
                ansroot.child.append(tmp.child[afternode])
                ansroot.num += tmp.child[afternode].getNum()
            prenode = tmp.child.index(ansroot.child[0]) - 1
            if prenode >= 0 and ansroot.num + tmp.child[prenode].getNum() < maxsize:
                b = False
                ansroot.child = [tmp.child[prenode]] + ansroot.child
                ansroot.num += tmp.child[prenode].getNum()
            if b:
                break
        troot = ansroot
    else:
        troot = root
    n = 0
    setid(troot)
    varnames = get_local_var(troot)
    fnum = -1
    vnum = -1
    vardic = {}
    vardic[m] = 'meth0'
    typedic = {}
    for x in varnames:
        if x[1].name == 'VariableDeclarator':
            vnum += 1
            vardic[x[0]] = 'loc' + str(vnum)
            t = -1
            for s in x[1].father.father.child:
                # print(s.name)
                if s.name == 'type':
                    t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert (t != -1)
            typedic[x[0]] = t
        else:
            fnum += 1
            vardic[x[0]] = 'par' + str(fnum)
            t = -1
            for s in x[1].child:
                if s.name == 'type':
                    t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert (t != -1)
            typedic[x[0]] = t
    return troot, vardic, typedic


def addter(root):
    if len(root.child) == 0:
        root.name += "_ter"
    for x in root.child:
        addter(x)
    return


def setProb(r, p):
    r.possibility = p  # max(min(np.random.normal(0.8, 0.1, 10)[0], 1), 0)
    for x in r.child:
        setProb(x, p)


def getSubroot(treeroot):
    global line_node_name
    currnode = treeroot
    lnode = None
    mnode = None
    while currnode:
        if currnode.name in line_node_name:
            lnode = currnode
            break
        currnode = currnode.father
    currnode = treeroot
    while currnode:
        if currnode.name == 'MethodDeclaration' or currnode.name == 'ConstructorDeclaration':
            mnode = currnode
            break
        currnode = currnode.father
    return lnode, mnode


def getLineNode(root, block, add=True):
    ans = []
    block = block + root.name
    # print(root.name, 'lll')
    for x in root.child:
        if x.name in linenode:
            if 'info' in x.getTreestr() or 'assert' in x.getTreestr() or 'logger' in x.getTreestr() or 'LOGGER' in x.getTreestr() or 'system.out' in x.getTreestr().lower():
                continue
            x.block = block
            ans.append(x)
        else:
            # print(x.name)
            s = ""
            if not add:
                s = block
                # tmp = getLineNode(x, block)
            else:
                s = block + root.name
            # print(block + root.name + "--------")
            tmp = getLineNode(x, block)
            '''if x.name == 'then_statement' and tmp == []:
              print(tmp)
              print(x.father.printTree(x.father))
              assert(0)'''
            ans.extend(tmp)
    return ans


def getroottree(tokens, isex=False):
    if isinstance(tokens[0], tuple):
        root = Node(tokens[0][0], 0)
    else:
        root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def getroottreewithLine(tokens, isex=False):
    if isinstance(tokens[0], tuple):
        root = Node(tokens[0][0], 0)
    else:
        lst = tokens[0].split('🚀')
        root = Node(lst[0], 0)
        root.position = eval(lst[1])
        root.position2 = eval(lst[2])
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                lst = x.split('🚀')
                nnode = Node(lst[0], idx)
                nnode.position = eval(lst[1])
                nnode.position2 = eval(lst[2])
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def ismatch(root, subroot):
    index = 0
    # assert(len(subroot.child) <= len(root.child))
    # print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while index < len(root.child) and root.child[index].name != x.name:
            index += 1
        if index == len(root.child):
            return False
        if not ismatch(root.child[index], x):
            return False
        index += 1
    return True


def findSubtree(root, subroot):
    if root.name == subroot.name:
        if ismatch(root, subroot):
            return root
    for x in root.child:
        tmp = findSubtree(x, subroot)
        if tmp:
            return tmp
    return None


def generateAST(tree):
    sub = []
    if not tree:
        return ['None', '^']
    if isinstance(tree, str):
        tmpStr = tree
        tmpStr = tmpStr.replace(" ", "").replace(":", "")
        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
            tmpStr = "<string>"
        if len(tmpStr) == 0:
            tmpStr = "<empty>"
        if tmpStr[-1] == "^":
            tmpStr += "<>"
        sub.append(tmpStr)
        sub.append("^")
        return sub
    elif isinstance(tree, list):
        if len(tree) == 0:
            sub.append("empty")
            sub.append("^")
        else:
            for ch in tree:
                subtree = generateAST(ch)
                sub.extend(subtree)
        return sub
    position = None
    if hasattr(tree, 'position'):
        # assert(0)
        position = tree.position
    curr = type(tree).__name__
    # print(curr)
    if True:
        if False:
            assert (0)  # sub.append((str(getLiteral(tree.children)))
        else:
            sub.append((curr, position))
            try:
                for x in tree.attrs:
                    if x == "documentation":
                        continue
                    if not getattr(tree, x):
                        continue
                    '''if x == 'prefix_operators':
                        node = getattr(tree, x)
                        print(type(node))
                        print(len(node))
                        print(node[0])
                        assert(0)
                    if type(getattr(tree, x)).__name__ not in nodes:
                        print(type(getattr(tree, x)).__name__)
                        continue'''
                    sub.append(x)
                    node = getattr(tree, x)
                    if isinstance(node, list):
                        if len(node) == 0:
                            sub.append("empty")
                            sub.append("^")
                        else:
                            for ch in node:
                                subtree = generateAST(ch)
                                sub.extend(subtree)
                    elif isinstance(node, javalang.tree.Node):
                        subtree = generateAST(node)
                        sub.extend(subtree)
                    elif not node:
                        continue
                    elif isinstance(node, str):
                        tmpStr = node
                        tmpStr = tmpStr.replace(" ", "").replace(":", "")
                        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
                            tmpStr = "<string>"
                        if len(tmpStr) == 0:
                            tmpStr = "<empty>"
                        if tmpStr[-1] == "^":
                            tmpStr += "<>"
                        sub.append(tmpStr)
                        sub.append("^")
                    elif isinstance(node, set):
                        for ch in node:
                            subtree = generateAST(ch)
                            sub.extend(subtree)
                    elif isinstance(node, bool):
                        sub.append(str(node))
                        sub.append("^")
                    else:
                        print(type(node))
                        assert (0)
                    sub.append("^")
            except AttributeError:
                assert (0)
                pass
        sub.append('^')
        return sub
    else:
        print(curr)
    return sub


import traceback
from tqdm import tqdm

errors = {}
fnames = {}
datas = []
dnodes = []
anodes = []


def setProb2(root, v):
    root.possibility = v
    for x in root.child:
        setProb2(x, v)


def setProb(root):
    if root.name == 'Delete_S':
        setProb2(root, 1)
    elif root.name == 'Add_S':
        setProb2(root, 2)
    elif root.name == 'Modify':
        setProb2(root[0], 1)
        setProb2(root[1], 2)
    else:
        root.possibility = 3
        for x in root.child:
            setProb(x)


def changetree(root1, root2):
    if root1.getTreestr() == root2.getTreestr():
        root1.position2 = root2.position
        return None, None
    if root1.name != root2.name:
        # dnodes.append(root1)
        # anodes.append(root2)
        return root1, root2
        print('delete1 ' + root1.getTreestr())
        print('add1 ' + root2.getTreestr())
    elif len(root1.child) == len(root2.child):
        for i in range(len(root1.child)):
            if root1.child[i].getTreestr() != root2.child[i].getTreestr():
                a, b = changetree(root1.child[i], root2.child[i])
                if a is not None:
                    tmpNode = Node('Modify_S', 0)
                    tmpNode.child = [a, b]
                    root1.child[i] = tmpNode
            else:
                root1.child[i].position2 = root2.child[i].position
    else:
        index1 = 0
        index2 = 0
        child = []
        while index1 < len(root1.child) and index2 < len(root2.child):
            if root1.child[index1].getTreestr() == root2.child[index2].getTreestr():
                root1.child[index1].position2 = root2.child[index2].position
                child.append(root1.child[index1])
                root1.child[index1].expanded = True
                idx = -1
                for j in range(index1 - 1, -1, -1):
                    if root1.child[j].expanded:
                        idx = j
                        break
                    root1.child[j].expanded = True
                if idx + 1 != index1:
                    node = Node('Delete_S', 1)
                    node.child = root1.child[idx + 1: index1]
                    child.append(node)
                index1 += 1
                index2 += 1
            else:
                idx2 = -1
                for j in range(index2, len(root2.child)):
                    if root1.child[index1].getTreestr() == root2.child[j].getTreestr():
                        root1.child[index1].position2 = root2.child[j].position
                        idx2 = j
                        break
                if idx2 != -1:
                    # for k in range(index2, idx2):
                    idx = -1
                    for j in range(index1 - 1, -1, -1):
                        if root1.child[j].expanded:
                            idx = j
                            break
                        root1.child[j].expanded = True
                    # assert(idx != -1)
                    if idx + 1 != index1:
                        node = Node('Delete_S', 1)
                        node.child = root1.child[idx + 1: index1]
                        child.append(node)
                    if index2 != idx2:
                        node = Node('Add_S', 1)
                        node.child = root2.child[index2:idx2]
                        child.append(node)
                    child.append(root1.child[index1])
                    index1 += 1
                    index2 = idx2 + 1
                    continue
                else:
                    # print('test', root1.child[index1].getTreestr())
                    index1 += 1

        if index1 < len(root1.child):
            node = Node('Delete_S', 1)
            node.child = root1.child[index1:]
            child.append(node)
            idx = len(root2.child)
            for j in range(len(root2.child) - 1, -1, -1):
                idx = j
                if root2.child[j].expanded:
                    break
                root2.child[j].expanded = True
            if idx < len(root2.child) - 1:
                node = Node('Add_S', 1)
                node.child = root2.child[idx + 1: len(root2.child)]
            child.append(node)

        if index2 < len(root2.child):
            node = Node('Add_S', 1)
            # node.child = root1.child[index2:]  # root2.child[index2:]
            node.child = root2.child[index2:]
            child.append(node)
            idx = len(root1.child)
            for j in range(len(root1.child) - 1, -1, -1):
                idx = j
                if root1.child[j].expanded:
                    break
                root1.child[j].expanded = True
            if idx < len(root1.child) - 1:
                node = Node('Delete_S', 1)
                node.child = root1.child[idx + 1:len(root1.child)]
            child.append(node)
        root1.child = child
    return None, None


def turnposition(root):
    root.position2 = root.position
    root.position = None
    for x in root.child:
        turnposition(x)


def getModify(root):
    if root.name == 'Modify_S':
        return root
    if root.name == 'Delete_S':
        return root
    if root.name == 'Add_S':
        return root
    for x in root.child:
        tmp = getModify(x)
        if tmp is not None:
            return tmp
    return None


def collectLine(root):
    ans = []
    if root.position is not None:
        ans.append(root.position)
    for x in root.child:
        ans.extend(collectLine(x))
    return ans


def collectLine2(root):
    ans = []
    if root.position2 is not None:
        ans.append(root.position2)
    for x in root.child:
        ans.extend(collectLine(x))
    return ans


def most_change(plinecover):
    pcover_score = {}
    for key in plinecover:
        # print(key)
        if 'buggy' in plinecover[key] and 'fixed' in plinecover[key]:
            if max(len(set(plinecover[key]['buggy'])), len(set(plinecover[key]['fixed']))) == 0:
                score = 0.0
            else:
                score = 1.0 - len(set(plinecover[key]['buggy']) & set(plinecover[key]['fixed'])) / max(
                    len(set(plinecover[key]['buggy'])),
                    len(set(plinecover[key]['fixed'])))
        else:
            score = 1.0
        pcover_score[key] = score  # score 为修改前后发生变化的大小，若修改前后发生变化越大，分数值越大
    pcover_score = sorted(pcover_score.items(), key=lambda d: d[1], reverse=True)
    return pcover_score


def process_long_list(longlist):
    # if len(longlist) < 500:
    #     return longlist
    filter = {}
    result = []
    for line in longlist:
        # lineid = int(line)
        if line in filter:
            filter[line] = filter[line] + 1
        else:
            filter[line] = 1
        if filter[line] <= 10:
            result.append(line)
    return filter


max_test = 0
min_test = 10000
max_list = 0
num_count = {}
potential_long = ['Closure_32.src.patch', 'Closure_110.src.patch', 'Math_55.src.patch', 'Lang_5.src.patch',
                  'Math_29.src.patch']
test_num = []
patchdir = 'crosspatch3'
num = 50
for x in lst:
    for data_line in tqdm(open('./result/%s/%s' % (patchdir, x), 'r').read().split('\n')):
        if data_line == "":
            continue
        data = json.loads(data_line)
        if not os.path.exists('./result/%s_pkldir' % (patchdir)):
            os.mkdir('./result/%s_pkldir' % (patchdir))
        wf = open('./result/%s_pkldir/%s_%s.pkl' % (patchdir, x, str(num)), 'wb')
        newdata = {}
        # infodata = {}
        for datas in tqdm(data):
            # if datas['patchName'] != 'Math93b_Patch207':
            #     continue
            codelines = datas['combinedMethod'].splitlines()
            # print(datas['combinedMethod'])
            oldcode = []
            addcode = []
            deletelines = {}
            addlines = {}
            normallines = {}
            for idx, code in enumerate(codelines):
                if len(code) == 0:
                    addcode.append('')
                    oldcode.append('')
                    normallines[idx] = len(oldcode)
                elif code[0] == '+':
                    addcode.append(code[1:])
                    addlines[idx] = len(addcode)
                # elif '-    }'in code and idx == len(codelines) - 1:
                #    addcode.append(code[1:])

                elif code[0] == '-':
                    oldcode.append(code[1:])
                    deletelines[idx] = len(oldcode)
                else:
                    oldcode.append(code)
                    addcode.append(code)
                    normallines[idx] = len(oldcode)
            # print("\n".join(oldcode), addcode)
            code = '\n'.join(oldcode)
            newcode = '\n'.join(addcode)
            try:
                tokens = javalang.tokenizer.tokenize(code)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
                result = generateAST(tree)
                root = getroottree(result)
                print('---------------\n')
                tokens = javalang.tokenizer.tokenize(newcode)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
                newroot = getroottree(generateAST(tree))
                turnposition(newroot)
                # print(root.printTreeWithLine(root))#print(code, newcode)
                changetree(root, newroot)
                # print('test1', root.printTreeWithLine(root))
                root = getroottreewithLine(root.printTreeWithLine(root).split())
                subroot = getModify(root)
                if 'dummyMethod' in root.getTreestr():
                    subroot = root
                if subroot is None:
                    print(root.printTree(root))
                    assert (0)
                # print(root.printTreeWithLine(root))

                root, vardic, _ = solveLongTree(root, subroot, 1000)
                setProb(root)
                # print(root.name)
                # print(root.printTree(root))
                # assert(0)
                ##trace info
                # filepre = '%s/%s/' % (x[:-1], patchid)
                # if not os.path.exists(filepre):
                #     continue
                alineb = collectLine(root)
                alinef = collectLine2(root)
                # lst = os.listdir(filepre)
                # tcover = {}

                pcover = {}
                fcover = {}
                plinecover = {}
                if 'failingTests' in datas:
                    failingTests = datas['failingTests']
                else:
                    failingTests = []
                    fcover['1'] = {}
                    pcover['1'] = {}
                    fcover['1']['fixed'] = []
                    fcover['1']['buggy'] = []
                    pcover['1']['fixed'] = []
                    pcover['1']['buggy'] = []
                    continue

                if 'buggyTraceInfo' in datas:
                    buggyTraceInfo = datas['buggyTraceInfo']
                else:
                    buggyTraceInfo = []

                if 'fixedTraceInfo' in datas:
                    fixedTraceInfo = datas['fixedTraceInfo']
                else:
                    fixedTraceInfo = []

                for key in buggyTraceInfo:
                    # cover = {}
                    tmp = []
                    commonline = []
                    # line_cover = process_long_list(buggyTraceInfo[key])
                    line_cover = buggyTraceInfo[key]
                    for line in line_cover:
                        # lst = line.split('#')
                        lineid = int(line.split('#')[1])
                        if lineid in normallines:
                            commonline.append(lineid)
                            lineid = normallines[lineid]
                            if lineid not in alineb:
                                continue
                            node = getNodeById(root, lineid)
                        elif lineid in addlines:
                            assert (0)
                        elif lineid in deletelines:
                            lineid = deletelines[lineid]
                            if lineid not in alineb:
                                continue
                            node = getNodeById(root, lineid)
                        # print(lineid)
                        linenode, _ = getSubroot(node)
                        if linenode is None:
                            continue
                        # print(line, codelines[int(lst[1])])
                        # print(root.printTreeWithLine(root))
                        tmp.append(linenode.id)
                        # tmp.append(str(linenode.id) + "#" + str(line_cover[line]))

                    if max_list < len(tmp):
                        max_list = len(tmp)

                    if key in failingTests:
                        if key in fcover:
                            fcover[key]['buggy'] = tmp
                        else:
                            fcover[key] = {}
                            fcover[key]['buggy'] = tmp
                    else:
                        if key in pcover:
                            pcover[key]['buggy'] = tmp
                            plinecover[key]['buggy'] = commonline
                        else:
                            pcover[key] = {}
                            plinecover[key] = {}
                            pcover[key]['buggy'] = tmp
                            plinecover[key]['buggy'] = commonline

                for key in fixedTraceInfo:
                    commonline = []
                    tmp = []
                    # line_cover = process_long_list(fixedTraceInfo[key])
                    line_cover = fixedTraceInfo[key]
                    for line in line_cover:
                        # lst = line.split('#')
                        lineid = int(line.split('#')[1])
                        if lineid in normallines:
                            commonline.append(lineid)
                            lineid = normallines[lineid]
                            if lineid not in alineb:
                                continue
                            node = getNodeById(root, lineid)
                        elif lineid in addlines:
                            lineid = addlines[lineid]
                            if lineid not in alinef:
                                continue
                            node = getNodeById2(root, lineid)
                        elif lineid in deletelines:
                            assert (0)
                        # node = getNodeById2(root, lineid)
                        linenode, _ = getSubroot(node)
                        if linenode is None:
                            continue
                        tmp.append(linenode.id)
                        # tmp.append(str(linenode.id) + "#" + str(line_cover[line]))
                    # cover['fixed'] = tmp

                    if max_list < len(tmp):
                        max_list = len(tmp)
                    if key in failingTests:
                        if key in fcover:
                            fcover[key]['fixed'] = tmp
                        else:
                            fcover[key] = {}
                            fcover[key]['fixed'] = tmp
                    else:
                        if key in pcover:
                            pcover[key]['fixed'] = tmp
                            plinecover[key]['fixed'] = commonline
                        else:
                            pcover[key] = {}
                            plinecover[key] = {}
                            pcover[key]['fixed'] = tmp
                            plinecover[key]['fixed'] = commonline

                for key in fcover:
                    if 'fixed' not in fcover[key]:
                        fcover[key]['fixed'] = []
                    if 'buggy' not in fcover[key]:
                        fcover[key]['buggy'] = []
                for key in pcover:
                    if 'fixed' not in pcover[key]:
                        pcover[key]['fixed'] = []
                    if 'buggy' not in pcover[key]:
                        pcover[key]['buggy'] = []

                pcover_score = most_change(plinecover)
                pcover_limit = {}
                # num = 500 - len(failingTests)
                # num = 400
                if len(pcover_score) <= num:
                    pcover_limit = pcover
                else:
                    i = 1
                    for key in pcover_score:
                        if i > num:
                            break
                        pcover_limit[key[0]] = pcover[key[0]]
                        # pcover_limit[key] = pcover[key]
                        i = i + 1
                if max_test < len(pcover_limit) + len(fcover):
                    max_test = len(pcover_limit) + len(fcover)

                newdata[datas['patchName']] = (
                    {'tree': root.printTreeWithVar(root, vardic), 'label': datas['label'],
                     'prob': root.getTreeProb(root),
                     'pcover': pcover_limit, 'fcover': fcover})
                # if len(pcover) + len(fcover) > max_test:
                #     max_test = len(pcover) + len(fcover)
                # if len(pcover) + len(fcover) < min_test:
                #     min_test = len(pcover) + len(fcover)
                # n = 0
                # setid(root)
                print('PatchName %s, treewithid %s' % (datas['patchName'], root.printTree(root)))
                print('PatchName %s, pcover %s, fcover %s' % (datas['patchName'], len(pcover), len(fcover)))
                tmplist = []

            except:
                # print(datas['patchName'])
                # print(datas['combinedMethod'])
                traceback.print_exc()
                if 'Closure-92' in datas['patchName'] or 'Closure-93' in datas['patchName']:
                    continue
                if datas['patchName'] == 'Closure_65.src.patch':
                    continue
                if datas['patchName'] in potential_long:
                    continue
                print(datas['combinedMethod'])
                print(datas['patchName'])
                errors.setdefault(x, []).append(datas['patchName'])
                assert (0)
                pass
                errors.setdefault(x, []).append(patchid)
    num_count[x] = len(newdata)
    print('%s  Size %s : ' % (x, len(newdata)))
    wf.write(pickle.dumps(newdata, protocol=1))
print(errors)
print(fnames)
print(num_count)
print('num: %f'%num)
print(potential_long)
print(max_test)

# with open('test_num.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(test_num)

# print('max_list %s: ' % max_list)
# print('max_text %s: ' % max_test)
# print('min_test %s:' % min_test)

