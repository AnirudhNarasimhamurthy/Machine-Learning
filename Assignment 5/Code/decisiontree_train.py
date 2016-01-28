import re
import math
import random
import sys
from functions import *
from functions_svm import *

maxwindex=0

u=1
if len(sys.argv)>1:
     maxlevel=float(sys.argv[1])
else:
    C=1 #hyper parameter
    p0=1
    maxlevel=4

print 'Creating 100 decision trees of depth' ,maxlevel, 'and predicting mean and variance.....'
print'========================================='

#r=p0
k=10

with open("badges-train-features.txt", "r") as ins:
    array = []
    for line in ins:
        dummy=(line.rstrip('\n')).split(" ")
        labels.append(int(dummy[0]))
        vector={}
        for word in dummy[1:]:
            indexvalue=word.split(":")
            if word=="":
                break
            if maxwindex < int(indexvalue[0]):
                maxwindex=int(indexvalue[0])
            if int(indexvalue[1])!=0:
                vector[int(indexvalue[0])]=int(indexvalue[1])
        vectors.append(vector)
totalvectors=len(vectors)

#TEST DATA 
testlabels=[]
testvectors=[]
with open("badges-test-features.txt", "r") as ins:
    array = []
    for line in ins:
        dummy=(line.rstrip('\n')).split(" ")
        testlabels.append(int(dummy[0]))
        testvector={}
        for word in dummy[1:]:
            indexvalue=word.split(":")
            if word=="":
                break
            if int(indexvalue[1])!=0:
                testvector[int(indexvalue[0])]=int(indexvalue[1])
        testvectors.append(testvector)
totaltestvectors=len(testvectors)




accuracylist=[]

secondvectors= [{} for _ in range(totalvectors)] #this is final generated vector out of the present ith tree

for treecounter in range(100):
   
    #Calculate entropy and information gain
    subindices=random.sample(range(totalvectors), int(totalvectors/2)) #list(range(totalvectors))
    allindices=list(range(totalvectors))
    countplus,countminus=countlabelplus(subindices)
    # print("first counts",countplus,countminus)
    firstentropy=entropy(countplus,countminus)
    # print("firstentropy=",str(firstentropy))
    
    #reinit total nodes in tree
    node.totalnodes=0
    
    root = node(subindices,firstentropy,allindices,"")
    queue = deque([root])
    
    #BFS starts
    #maxlevel=8
    ITERATION=1
    nodesinmaxlevel=math.pow(2,maxlevel)
    nodesinmaxminus2level=math.pow(2,maxlevel-2)
    while len(queue)!=0 :
        # print("\nqueuelen=",len(queue)," ITERATION=",ITERATION)
        ITERATION+=1
        if ITERATION>nodesinmaxlevel:
            break
        delnode=queue.popleft()
        # print("List=",(", ".join(str(x) for x in delnode.remfeaturelist)))
        # print("Index=",delnode.index)
        if delnode.label!="":    #if label is not empty
            #print("Label=",delnode.label)
            xdummy=1
            #node.increment2nodes()
        else:
            #select feature to use
            maxiGain=0
            maxifeature=0
            maxifeatureplus=[]
            maxifeatureminus=[]
            maxicountplus=0
            maxicountminus=0
            maxinumfeatureplusplus=0
            maxinumfeatureminusplus=0
            entropyminus=0
            entropyplus=0
            for featurenum in delnode.remfeaturelist:
                countplus,countminus,numfeatureplusplus,numfeatureminusplus,entriesplus,entriesminus=countlabelplusandfeatureplus(featurenum,delnode.entries)
                iGain,tempentropyplus,tempentropyminus=infogain(countplus,countminus,numfeatureplusplus,numfeatureminusplus,delnode.entropy)
                # print("counts:",countplus,",",countminus,",",numfeatureplusplus,",",numfeatureminusplus)
                # print("Igain F",featurenum,"=",str(iGain))
                if maxiGain <= iGain:
                    maxiGain=iGain
                    maxifeature=featurenum
                    maxifeatureplus=entriesplus
                    maxifeatureminus=entriesminus
                    entropyminus=tempentropyminus
                    entropyplus=tempentropyplus
                    maxinumfeatureplusplus=numfeatureplusplus
                    maxinumfeatureminusplus=numfeatureminusplus
                    maxicountplus=countplus
                    maxicountminus=countminus
            if len(delnode.remfeaturelist)==0: #if there are no features left out for summation
                maxiGain=delnode.entropy
            # print("delnode.entropy =",str(delnode.entropy))
            # print("maxiGain =",str(maxiGain))
            # print("maxifeature =",str(maxifeature))
            # print("entropyminus =",str(entropyminus))
            # print("entropyplus =",str(entropyplus))
            # print("len(maxifeatureplus) =",str(len(maxifeatureplus)))
            # print("len(maxifeatureminus) =",str(len(maxifeatureminus)))
            # print("maxinumfeatureplusplus =",str(maxinumfeatureplusplus))
            # print("maxinumfeatureminusplus =",str(maxinumfeatureminusplus))
            # print("maxicountplus =",str(maxicountplus))
            # print("maxicountminus =",str(maxicountminus))
            if maxiGain==0:
                if len(maxifeatureplus)>0 and len(maxifeatureminus)==0:
                    delnode.label="+"
                if len(maxifeatureminus)>0 and len(maxifeatureplus)==0:
                    delnode.label="-"
                # print("Gain=0,CalculatedLabel=",delnode.label)
            else:
                #setting present node values for later use
                delnode.feat=maxifeature
                newremlist=list(delnode.remfeaturelist)
                if len(delnode.remfeaturelist)!=0: #if there are features left out for summation
                    newremlist.remove(delnode.feat)
                labelplus=""
                labelminus=""
                if entropyplus==0:
                    labelplus="+"
                elif ITERATION>nodesinmaxminus2level: #For maxlevel-2, calculate max of plus or minus labels and assign to leaf
                    if maxinumfeatureplusplus > maxicountplus-maxinumfeatureplusplus:
                        labelplus="+"
                    else:
                        labelplus="-"
                if entropyminus==0:
                    labelminus="-"
                elif ITERATION>nodesinmaxminus2level: #For maxlevel-2, calculate max of plus or minus labels and assign to leaf
                    if maxinumfeatureminusplus > maxicountminus-maxinumfeatureminusplus:
                        labelminus="+"
                    else:
                        labelminus="-"
                # print("labelplus=",labelplus)
                # print("labelminus=",labelminus)
                (delnode.subtrees).append(node(maxifeatureplus,entropyplus,newremlist,labelplus))
                (delnode.subtrees).append(node(maxifeatureminus,entropyminus,newremlist,labelminus))
                # print("Gain!=0,CalculatedLabel=",delnode.label)
                
                #add all nodes for BFS
                if len(delnode.subtrees)!=0:
                    for i in range(len(delnode.subtrees)):
                        queue.append(delnode.subtrees[i])    
    
    #validate test data
    correct=0
    for examplei in range(totaltestvectors):
        decidedlabel=parseuntillabel(root,testvectors[examplei])
        if decidedlabel==testlabels[examplei]:
            correct+=1
    accuracylist.append(100*correct/float(totaltestvectors))
    # print("Accuracy on test data: ",100*(correct)/totalvectors,"\\\\")

    #For each example in the data set, the value of the i th feature will be the prediction of the i th decision tree.
    for examplei in range(0,totalvectors):
        decidedlabel=parseuntillabel(root,vectors[examplei])
        #final_labels.append(decidedlabel)
        if decidedlabel==1:
            secondvectors[examplei][treecounter]=decidedlabel
        else:
            secondvectors[examplei][treecounter]=0
    # root.printpreorder()
    # print("Total number of nodes in tree",treecounter,":",root.totalnodes)
    # break
    
print 'Accuracy list is :', accuracylist
mmean=mymean(accuracylist)
print("Mean of the accuracy over 100 decision trees is:",mmean)
print("Variance of the accuracy over 100 decision trees is:",vari(accuracylist,mmean))

# Writing the feature vector to a file 

if maxlevel==4:
	file_name='badges-ensemble-features-train_4.txt'
elif maxlevel==8:
	file_name='badges-ensemble-features-train_8.txt'	
elif maxlevel==20:
	file_name='badges-ensemble-features-train_20.txt'	
		
f = open(file_name, 'w' )

for i in range(0, len(secondvectors)):
	
		final_label=labels[i]
		if final_label==1:
			final_label='+1'
		f.write(str(final_label)+' ')
		for j in range(0, 100):
			val2=str(secondvectors[i][j])
			if (j!=99):
				f.write(str(j+1)+':'+ val2+' ')
			else:
				f.write(str(j+1)+':'+ val2)
		f.write('\n')	

f.close()
print'========================================='
print 'Process completed !!!'
vectors=secondvectors
