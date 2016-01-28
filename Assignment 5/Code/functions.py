import re
import math
from tree import *

# Functions start here
vectors=[]
labels=[]
secondvectors=[]

def majorityerror(pos,neg):
    if pos==0 and neg==0:
        return 0
    total=pos+neg
    return 1-max((float)(pos/total),(float)(neg/total))
    
def entropy(pos,neg):
    if pos==0 or neg==0:
        return 0
    total=pos+neg
    x=pos/float(total)
    y=neg/float(total)
    return -1 *(x * math.log(x,2))  -1* (y *math.log(y,2)) 
def infogain(countplus,countminus,numfeatureplusplus,numfeatureminusplus,originalentropy):
    counttotal=countplus+countminus
    entropyplus=entropy(numfeatureplusplus,countplus-numfeatureplusplus)
    entropyminus=entropy(numfeatureminusplus,countminus-numfeatureminusplus)
    expectedentropy=(countplus/counttotal)*entropyplus
    expectedentropy+=(countminus/counttotal)*entropyminus
    return originalentropy-expectedentropy,entropyplus,entropyminus
    
def infogainmaj(countplus,countminus,numfeatureplusplus,numfeatureminusplus,originalentropy):
    counttotal=countplus+countminus
    entropyplus=majorityerror(numfeatureplusplus,countplus-numfeatureplusplus)
    entropyminus=majorityerror(numfeatureminusplus,countminus-numfeatureminusplus)
    expectedentropy=(countplus/counttotal)*entropyplus
    expectedentropy+=(countminus/counttotal)*entropyminus
    return originalentropy-expectedentropy,entropyplus,entropyminus


def printintlist(list_of_ints):
    return "List="+(", ".join(str(x) for x in list_of_ints))
    
def countlabelplus(labelsindices):
    global labels
    numplus=0
    total=len(labelsindices)
    for label in labelsindices:
        if labels[label]==1:
            numplus+=1
    return numplus,total-numplus

def countlabelplusandfeatureplus(featurenum,labelsindices):
    global labels
    global vectors
    numplus=0
    numfeatureminusplus=0
    numfeatureplusplus=0
    total=len(labelsindices)
    featureplus=[]
    featureminus=[]
    for label in labelsindices:
        if vectors[label].get(featurenum)==1:
            numplus+=1
            if labels[label]==1:
                numfeatureplusplus+=1
            featureplus.append(label)
        else:
            if labels[label]==1:
                numfeatureminusplus+=1
            featureminus.append(label)
    return numplus,total-numplus,numfeatureplusplus,numfeatureminusplus,featureplus,featureminus

def parseuntillabel(root,featlist):
    if root.label!="":
        # print("label=",root.label)
        if str(root.label)=="+":
            return 1
        elif str(root.label)=="-":
            return 0
    if featlist.get(root.feat)==1:
        # print("tree0",root.subtrees[0].feat)
        return parseuntillabel(root.subtrees[0],featlist)
    else:
        # print("tree1",root.subtrees[1].feat)
        return parseuntillabel(root.subtrees[1],featlist)

## Functions end here
    
