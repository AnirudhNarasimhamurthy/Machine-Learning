import math
import util

# Decision tree data structure

class Tree:
	def __init__(self, name,parent=None):
		self.parent = parent
		self.children = []
		self.label = None
		self.entropyf1=0
		self.entropyf2=0
		self.name=''
		self.information_gain=0
		self.majority_errorf1=0
		self.majority_errorf2=0
		self.splitFeatureValue = None
		self.splitFeature = None

# Storing the nodes of the tree in queue for processing 
root=Tree('root')
queue=util.Queue()
queue.push(root)
while not queue.isEmpty():
	node=queue.pop()


f1=Tree('feature1')
f2=Tree('feature2')
f3=Tree('feature3')
f4=Tree('feature4')

# Taking the count of vowels
def vowels_count(string):
    count=0
    for c in string:
        if c in "aeiou":
           count = count+1
    return count	

#Global variables declaration

label=[]
firstnames,secondnames=[],[]
#secondnames=[]
feature1,feature2,feature3,feature4=[],[],[],[]
feature_function=[]
spacefree_name=[]
features_entropy=[]
pplus_count=0 
pplus1_count=0
pminus_count=0 
pminus1_count=0
pminus0_count=0
pplus0_count=0
pplus11_count,pplus10_count,pminus11_count,pminus10_count=0,0,0,0
total_count=0
featurex1_count=0
featurex0_count=0
entropy_featurex1=0
entropy_featurex0=0
featurex21_count=0
featurex20_count=0
#cumulative_entropiesfx=0
entropy=0
pplus=0.0
pminus=0.0
major_error=0

#Reading the input file and converting it to a list of names 
with open('badges-train.txt', 'r') as inputFile:
	data = inputFile.readlines()
	print 'Number of rows in the file',len(data)

#Obtaining the first,second and third names	
for names in data:
	
	
	spacefree_name.append(names.strip())
	label.append(names.split(' ')[0])
	
	firstnames=names.split(' ')[1]
	secondnames=names.split(' ')[2]
	
	# Feature function 1: If the last letter in the name is between a-m, then 0 else 1	
	
	ascii_lastletter=ord(names[-2:-1])
	#print 'Last character is :', names[-1]
	if(ascii_lastletter < 110):
		feature1.append(0)
	else:
		feature1.append(1)	
	
	# Feature function 2: If the first letter of the second name is between a-m ,then 1 else 0
	
	ascii_value=ord(secondnames[:1])
	if (ascii_value < 110):
		feature2.append(1)
	else:
		feature2.append(0)	
	
	# Feature function 3: If the count of vowels in the name is odd,then 1 else 0
		
	vowel_count=vowels_count(names)	
	if(vowel_count%2==1):
		feature3.append(1)
	else:
		feature3.append(0)	
	
	# Feature function 4: If the string length is even,then 1 else 0
		
	str_len=len(spacefree_name)
	if(str_len%2==0):
		feature4.append(1)
	else:
		feature4.append(0)		
		
		
for i in range(0,len(data)):
	#print 'Name is :', data[i], 'Label is', label[i],'length is:', len(spacefree_name[i]), 'Feature list vector is :', feature1[i],feature2[i],feature3[i],feature4[i]
	x=1

#Count of positive and negative examples in whole dataset
def proportion_positivenegative():

	global label
	global pplus_count,pminus_count
	global total_count
	for i in label:
		if  i=='+':
			pplus_count=pplus_count+1
		elif i=='-': 
			pminus_count=pminus_count+1
	total_count=len(label)
	pplus=pplus_count/float(total_count)
	pminus=pminus_count/float(total_count)
	
	return pplus,pminus

# Entropy for the whole dataset
def datasetentropy(pplus,pminus):

	global entropy
	if pplus==0 or pminus==0:
		return 0
	entropy= -1 * (pplus * math.log(pplus,2)  + pminus * math.log(pminus,2)) 
	#print 'Entropy of dataset is :', entropy
	return entropy


#Majority error for the whole dataset	
def majorityerror_dataset(pplus,pminus):

	global major_error
	
	major_error=1-max(pplus,pminus)
	print 'Majority error of dataset is:', major_error
	return major_error

# Information gain for individual feature functions

plus,minus=proportion_positivenegative()
global_entropy=datasetentropy(plus,minus)
print 'Entropy of the dataset is :',global_entropy


def information_gain_root(feature):

	global featurex1_count
	global featurex0_count
	global pplus1_count
	global pminus1_count
	global pplus0_count
	global pminus0_count
	#global cumulative_entropiesfx
	
	for i in range(0,len(feature)):
		if feature[i]==1:
			featurex1_count=featurex1_count+1
			if label[i]=='+':		
				pplus1_count=pplus1_count+1
			elif label[i]=='-':
				pminus1_count=pminus1_count+1
		elif feature[i]==0:
			featurex0_count=featurex0_count+1
			if label[i]=='+':		
				pplus0_count=pplus0_count+1
			elif label[i]=='-':
				pminus0_count=pminus0_count+1


	#print 'pplus1 count is :', pplus1_count , 'pminus1_count is :', pminus1_count
	#print 'pplus0 count is :', pplus0_count , 'pminus0_count is :', pminus0_count

	pplus1=pplus1_count/float(featurex1_count)
	pminus1=pminus1_count/float(featurex1_count)
	pplus0=pplus0_count/float(featurex0_count)
	pminus0=pminus0_count/float(featurex0_count)
	
	#print 'pplus1',pplus1,'pplus0',pplus0,'pminus1',pminus1,'pminus0',pminus0
	#print 'Featurex1 count is :', featurex1_count, 'pplus1 is:',pplus1,'pminus1 is ',pminus1
	#print 'Featurex0 count is :', featurex0_count, 'pplus0 is:',pplus0,'pminus1 is ',pminus0

	entropy_featurex1=datasetentropy(pplus1,pminus1)
	entropy_featurex0=datasetentropy(pplus0,pminus0)
	#entropy_featurex1= -1 * (pplus1 * math.log(pplus1,2) + pminus1 * math.log(pminus1,2))
	#entropy_featurex0= -1 * (pplus0 * math.log(pplus0,2) + pminus0 * math.log(pminus0,2))
	#print 'Entropy featurex with zero is :', entropy_featurex0
	#print 'Entropy featurex with one is :', entropy_featurex1							

	#f1_len=len(feature1)
	#print 'f1 length is :',f1_len
	cumulative_entropiesfx=(featurex1_count/float(200)) * entropy_featurex1 + (featurex0_count/float(200)) * entropy_featurex0				
				
	featurex_ig= global_entropy-cumulative_entropiesfx 
	#print 'entropy total is :', entropy,'cumulative:',cumulative_entropiesfx
	#print 'feature x information gain is :',featurex_ig			


	featurex0_count=0
	featurex1_count=0
	pminus0_count=0
	pminus1_count=0
	pplus1_count=0
	pplus0_count=0
	#cumulative_entropiesfx=0
	
	return featurex_ig,entropy_featurex1,entropy_featurex0	

f1.information_gain,f1.entropyf1,f1.entropyf0=information_gain_root(feature1)	
f2.information_gain,f2.entropyf1,f2.entropyf0=information_gain_root(feature2)	
f3.information_gain,f3.entropyf1,f3.entropyf0=information_gain_root(feature3)	
f4.information_gain,f4.entropyf1,f3.entropyf0=information_gain_root(feature4)	

#features_entropy=[f1.entropy,f2.entropy,f3.entropy,f4.entropy]	
	

print 'Information gain of feature 1 using entropy is:',f1.information_gain
print 'Information gain of feature 2 using entropy is:',f2.information_gain
print 'Information gain of feature 3 using entropyis:',f3.information_gain
print 'Information gain of feature 4 using entropy is:',f4.information_gain
#print 'Feature 1 value 1 entropy is :',f1.entropyf1


root=Tree('')	
root.information_gain=max(f1.information_gain,f2.information_gain,f3.information_gain,f4.information_gain)
root.name='Abc'
print 'Root element selected via entropy is:', root.information_gain	
			

def information_gain_branches(root_f,feature):

	global f1entropy
	featurex0_count,featurex1_count=0,0
	featurex00_count,featurex01_count=0,0
	featurex1_ig,featurex0_ig=0,0
	pplus11_count,pplus10_count,pplus01_count,pplus00_count=0,0,0,0
	pminus11_count,pminus10_count,pminus01_count,pminus00_count=0,0,0,0
	root0_count,root1_count=0,0
	
	for i in range(1, len(root_f)):
		if root_f[i]==1:
			root1_count=root1_count+1
			if feature[i]==1:
				featurex1_count=featurex1_count+1
				if label[i]=='+':		
			 		pplus11_count=pplus11_count+1
				elif label[i]=='-':
					pminus11_count=pminus11_count+1
			elif feature[i]==0:
				featurex0_count=featurex0_count+1
				if label[i]=='+':	
			 		pplus10_count=pplus10_count+1
				elif label[i]=='-':
					pminus10_count=pminus10_count+1	
					
		elif root_f[i]==0:
			root0_count=root0_count+1
			if feature[i]==1:
				featurex01_count=featurex01_count+1
				if label[i]=='+':
					pplus01_count=pplus01_count+1
				elif label[i]=='-':
					pminus01_count=pminus10_count+1

			elif feature[i]==0:
				featurex00_count=featurex00_count+1
				if label[i]=='+':
					pplus00_count=pplus00_count+1
				if label[i]=='-':
			 		pminus00_count=pminus00_count+1
			
					
	pplus11=pplus11_count/float(featurex1_count)
	pminus11=pminus11_count/float(featurex1_count)
	pplus10=pplus10_count/float(featurex0_count)
	pminus10=pminus10_count/float(featurex0_count)				
	
	#print 'pplus11',pplus11,'pplus10',pplus10,'pminus11',pminus11,'pminus10',pminus10
	
	entropy_featurexr1=datasetentropy(pplus11,pminus11)
	entropy_featurexr0=datasetentropy(pplus10,pminus10)
	
	"""print 'entropy_featurexr1',entropy_featurexr1
	print 'entropy_featurexr1',entropy_featurexr0"""
	
	#entropy_featurexr1= -1 * (pplus11 * math.log(pplus11,2) + pminus11 * math.log(pminus11,2))
	#entropy_featurexr0= -1 * (pplus10 * math.log(pplus10,2) + pminus10 * math.log(pminus10,2))
	
	cumulative_entropiesf1x=(featurex1_count/float(root1_count)) * entropy_featurexr1 + (featurex0_count/float(root0_count)) * entropy_featurexr0				
	#ig,entropy=information_gain_root(root)
	 
	"""print 'featurex1 count is :', featurex1_count
	print 'featurex0 count is :', featurex0_count
	print 'root0 count is :', root0_count
	print 'root1 count is :', root1_count"""
	 
	"""print 'cumulative_entropiesf1x',cumulative_entropiesf1x"""
	root_entropy=f1.entropyf1 
	featurex1_ig= root_entropy-cumulative_entropiesf1x 
	
	pplus01=pplus01_count/float(featurex01_count)
	pminus01=pminus01_count/float(featurex01_count)
	pplus00=pplus00_count/float(featurex00_count)
	pminus00=pminus00_count/float(featurex00_count)
	
	entropy_featurexr01=datasetentropy(pplus01,pminus01)
	entropy_featurexr00=datasetentropy(pplus00,pminus00)
	
	cumulative_entropiesf0x=(featurex01_count/float(root0_count)) * entropy_featurexr01 + (featurex00_count/float(root0_count)) * entropy_featurexr00				
	featurex1_ig= f1.entropyf2-cumulative_entropiesf0x 
	
	
	#print 'entropy total is :', features_entropy[0],'cumulative:',cumulative_entropiesf1x
	#print 'feature x information gain is :',featurex1_ig			
						
						
	return featurex1_ig,featurex0_ig
	

secondlevel_ig31,secondlevel_ig30=	information_gain_branches(feature1,feature3)	
secondlevel_ig41,secondlevel_ig40=information_gain_branches(feature1,feature4)	
secondlevel_ig21,secondlevel_ig20=information_gain_branches(feature1,feature2)

"""print 'Information gain with respect to feature 1=1 for feature 2 using entropy  is:',secondlevel_ig21	
print 'Information gain with respect to feature1=1 for feature 3 using entropy is:',secondlevel_ig31
print 'Information gain with respect to feature 1=1 for feature 4 using entropy  is:',secondlevel_ig41

print 'Information gain with respect to feature 1=0 for feature 2 using entropy is:',secondlevel_ig20	
print 'Information gain with respect to feature1=0 for feature 3 using entropy is:',secondlevel_ig30
print 'Information gain with respect to feature 1=0 for feature 4 using entropy  is:',secondlevel_ig40"""




# Calling the majority error function on the dataset to obtain the majority error
dataset_me=majorityerror_dataset(plus,minus)



# Calculating information gain for the individual features using majority error measure
def majorityerror_features(feature):
	
	global featurex1_count
	global featurex0_count
	global pplus1_count
	global pminus1_count
	global pplus0_count
	global pminus0_count
	
	for i in range(0,len(feature)):
		if feature[i]==1:
			featurex1_count=featurex1_count+1
			if label[i]=='+':		
				pplus1_count=pplus1_count+1
			elif label[i]=='-':
				pminus1_count=pminus1_count+1
		elif feature[i]==0:
			featurex0_count=featurex0_count+1
			if label[i]=='+':		
				pplus0_count=pplus0_count+1
			elif label[i]=='-':
				pminus0_count=pminus0_count+1
				
	pplus1=pplus1_count/float(featurex1_count)
	pminus1=pminus1_count/float(featurex1_count)
	pplus0=pplus0_count/float(featurex0_count)
	pminus0=pminus0_count/float(featurex0_count)
	
	majority_errorfx1=1-max(pplus1,pminus1)
	majority_errorfx0=1-max(pplus0,pminus0)
	
	cumulative_errorfx=(featurex1_count/float(200)) * majority_errorfx1 + (featurex0_count/float(200)) * majority_errorfx0				
				
	featurex_ig= major_error-cumulative_errorfx
	#print 'majority error total is :', major_error,'cumulative:',cumulative_errorfx
	#print 'feature x information gain is :',featurex_ig			

	
	featurex0_count=0
	featurex1_count=0
	pminus0_count=0
	pminus1_count=0
	pplus1_count=0
	pplus0_count=0
	
	return featurex_ig,majority_errorfx1,majority_errorfx0	

f1.information_gain,f1.majority_errorf1,f1.majority_errorf2=majorityerror_features(feature1)
f2.information_gain,f2.majority_errorf1,f2.majority_errorf2=majorityerror_features(feature2)
f3.information_gain,f3.majority_errorf1,f3.majority_errorf2=majorityerror_features(feature3)	
f4.information_gain,f4.majority_errorf1,f4.majority_errorf2=majorityerror_features(feature4)

print 'information gain using majority error on feature 1',f1.information_gain	
print 'information gain using majority error on feature 2',f2.information_gain
print 'information gain using majority error on feature 3',f3.information_gain	
print 'information gain using majority error on feature 4',f4.information_gain	

root_node=max(f1.information_gain,f2.information_gain,f3.information_gain,f4.information_gain)

print' Root node selected via majority error is :', root_node


#Calculating information gain for a feature with respect to another feature using majority error measure 

def majorityerror_branches_features(feature, feature2):
	
	global featurex1_count
	global featurex0_count
	global featurex21_count
	global featurex20_count
	featurex00_count=0
	featurex01_count=0
	global pplus11_count
	global pminus11_count
	global pplus10_count
	global pminus10_count
	
	for i in range(0,len(feature)):
		if feature[i]==1:
			featurex1_count=featurex1_count+1
			if feature2[i]==1:
				featurex21_count=featurex21_count+1
				if label[i]=='+':		
					pplus11_count=pplus11_count+1
				elif label[i]=='-':
					pminus11_count=pminus11_count+1
			elif feature2[i]==0:
				featurex20_count=featurex20_count+1
				if label[i]=='+':		
					pplus10_count=pplus10_count+1
				elif label[i]=='-':
					pminus10_count=pminus10_count+1				
					
		elif feature[i]==0:
			featurex0_count=featurex0_count+1
			if feature2[i]==1:
				featurex01_count=featurex01_count+1
				if label[i]=='+':		
					pplus01_count=pplus11_count+1
				elif label[i]=='-':
					pminus01_count=pminus11_count+1
			elif feature2[i]==0:
				featurex00_count=featurex00_count+1
				if label[i]=='+':		
					pplus00_count=pplus10_count+1
				elif label[i]=='-':
					pminus00_count=pminus10_count+1				
					
				
	pplus11=pplus11_count/float(featurex21_count)
	pminus11=pminus11_count/float(featurex21_count)
	
	majority_errorfx11=1-max(pplus11,pminus11)
	
	pplus10=pplus10_count/float(featurex20_count)
	pminus10=pminus10_count/float(featurex20_count)
	
	
	majority_errorfx10=1-max(pplus10,pminus10)
	
	cumulative_errorfx=(featurex21_count/featurex1_count) * majority_errorfx11 + (featurex20_count/featurex1_count) * majority_errorfx10				
	featurex_ig= f1.majority_errorf1-cumulative_errorfx
	
				
	pplus01=pplus01_count/float(featurex01_count)
	pminus01=pminus01_count/float(featurex01_count)
	
	majority_errorfx01=1-max(pplus01,pminus01)
	
	pplus00=pplus00_count/float(featurex00_count)
	pminus00=pminus00_count/float(featurex00_count)
	
	majority_errorfx00=1-max(pplus00,pminus00)
	
	cumulative_errorfx00=(featurex01_count/featurex0_count) * majority_errorfx01 + (featurex00_count/featurex0_count) * majority_errorfx00				
	featurex0_ig=f1.majority_errorf2-cumulative_errorfx  
	
	
	
	featurex0_count=0
	featurex1_count=0
	pminus0_count=0
	pminus1_count=0
	pplus1_count=0
	pplus0_count=0
	
	return featurex_ig,featurex0_ig	

ig_merrorf21,ig_merrorf20=majorityerror_branches_features(feature1,feature2)
ig_merrorf31,ig_merrorf30=majorityerror_branches_features(feature1,feature3)
ig_merrorf41,ig_merrorf40=majorityerror_branches_features(feature1,feature4)	


print 'information gain using majority error for feature2 with respect to feature 1=1',ig_merrorf21	
print 'information gain using majority error on feature3 with respect to feature1=1',ig_merrorf31	
print 'information gain using majority error on feature4 with respect to feature1=1',ig_merrorf41	

x=max(ig_merrorf21	,ig_merrorf31	,ig_merrorf41)	
print 'feature 1= 1 branch is :', x

print 'information gain using majority error for feature2 with respect to feature 1=0',ig_merrorf20	
print 'information gain using majority error on feature3 with respect to feature1=0',ig_merrorf30	
print 'information gain using majority error on feature4 with respect to feature1=0',ig_merrorf40

y=max(ig_merrorf20	,ig_merrorf30	,ig_merrorf40)	
print 'feature 1= 0 branch is :', y

					
