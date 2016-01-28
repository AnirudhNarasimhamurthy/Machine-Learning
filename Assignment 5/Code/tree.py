from collections import deque

class node(object):
    totalnodes=0
    def __init__(self,entries,en,rem,templ ):
        node.totalnodes+=1 #total nodes in the tree
        self.index= node.totalnodes #index for each node in BFS fashion
        self.feat = -1  #feature selected for the decision node
        self.remfeaturelist = rem #remaining features left out that should summated to find information gain.
        self.subtrees = [] #children for each label node can be added here
        self.entries = entries #all rows in dataset whose label match this decision node selection.
        self.entropy= en # Entropy for present node
        self.label=templ #if the node is label and not a feature decision.
        
    @staticmethod
    def increment2nodes():
        node.totalnodes+=2
        
    def printpreorder(self):
        leng=len(self.subtrees)
        if self.label=="":
            print("Node",str(self.index),",Feature=",str(self.feat))
        else:
            print("Node",str(self.index),"Label=",self.label)
        if leng!=0:
            for i in range(leng):
                if i==0:
                    print("Node",str(self.index),",child[+]=>")
                else:
                    print("Node",str(self.index),"child[-]=>")
                self.subtrees[i].printpreorder()
        else:
            print("Node",str(self.index),",Leaf - No children")
            
"""
root = node(None)
root.feat=1
(root.subtrees).append(node(None))
root.subtrees[0].feat=2
(root.subtrees).append(node(None))
root.subtrees[1].feat=3
root.subtrees[0].subtrees.append(node(None))
root.subtrees[0].subtrees[0].feat=4

queue = deque([root])
while len(queue)!=0 :
    for i in range(len(queue)):
        print(queue[i].feat)
    delnode=queue.popleft()
    if len(delnode.subtrees)!=0:
        #add all nodes
        for i in range(len(delnode.subtrees)):
            queue.append(delnode.subtrees[i])    
    print("dfs=",delnode.feat)

"""