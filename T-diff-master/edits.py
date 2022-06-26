import unittest
from util.tree import *
from treediff import *
import pickle
data = pickle.load(open('../TrainSet4kui.pkl', 'rb'))
def getroottree(tokens, isex=False):
    root = TreeNode(tokens[0])
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = TreeNode(x)
            currnode.add_child(nnode)
            nnode.set_father(currnode)
            currnode = nnode
        else:
            currnode = currnode.father()
    return root
for x in data:
    oroot = getroottree(data[x]['oldtree'].split())
    nroot = getroottree(data[x]['newtree'].split())
    tree1 = Tree(oroot)
    tree1.build_caches()
    tree2 = Tree(nroot)
    tree2.build_caches()
    distance, mapping = computeDiff(tree1, tree2) 
    print produceHumanFriendlyMapping(mapping, tree1, tree2)
    print(distance)
    print(mapping)