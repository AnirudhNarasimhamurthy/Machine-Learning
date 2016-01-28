from time import clock
from numpy import dot
import random
from random import shuffle
import sys

expected=[]
expected_test=[]
semi_cleaned=[]
fully_cleaned=[]
semi_cleaned_test=[]
fully_cleaned_test=[]
x=[]
x_test=[]
w=[]
w_mp=[]
value=[]
z=[]
temp=[]
training_data=[]
start_time=clock()
finalaccuracy_list=[]
heldout_set=[]
overall_accuracy=[]
feature_vector_size=100

mistake_count_train=0
mistake_count_test=0
mistake_marginp_train=0
mistake_marginp_test=0




def start_and_end(epoch,i):

	start_index=0
	end_index=0
	
	if epoch==0:
		start_index=0
	elif epoch==1:
		start_index=i
	elif epoch==2:
		start_index=i*2
	elif epoch==3:
		start_index=i*3
	elif epoch==4:
		start_index=i*4
	elif epoch==5:
		start_index=i*5
	elif epoch==6:
		start_index=i*6
	elif epoch==7:
		start_index=i*7
	elif epoch==8:
		start_index=i*8
	elif epoch==9:
		start_index=i*9
	
	end_index=start_index+i-1	
		 
	return start_index,end_index
	
if len(sys.argv)>1:
    dt_depth=float(sys.argv[1])
else:
    dt_depth=4
    

if dt_depth==4:
	train_file= 'badges-ensemble-features-train_4.txt'
	test_file=  'badges-ensemble-features-test_4.txt' 

elif dt_depth==8:	

	train_file= 'badges-ensemble-features-train_8.txt'
	test_file=  'badges-ensemble-features-test_8.txt' 

elif dt_depth==20:	

	train_file= 'badges-ensemble-features-train_20.txt'
	test_file=  'badges-ensemble-features-test_20.txt' 	
	

#Reading the training file and converting it to a list of names 
with open(train_file, 'r') as inputFile:
	data = inputFile.readlines()
	training_size=len(data)
	#print 'Number of rows in the file',len(data)

#Reading the test file and converting it to a list of names 
with open(test_file, 'r') as inputFile:
	data2 = inputFile.readlines()
	test_size=len(data2)
	#print 'Number of rows in the file',len(data2)


#Extracting the expected labels for all the training examples and storing it in list	
for values in data:
	expected.append(values[:2])	
expected=map(int,expected)


#Extracting the expected labels for all the test examples and storing it in list	
for values in data2:
	expected_test.append(values[:2])	
expected_test=map(int,expected_test)

#print 'Expected training labels is :', expected
#print 'Expected test labels is :', expected_test


#Data cleaning for training examples


max_train=0
for values in data:
	values=values[3:]
	#values=values.replace(' ','')
	values=values.replace('\n','')
	values=values.split(' ')
	finalv={}
	for vect in values:
		vect=vect.split(':')
		# Converting it into a integer vector 
		vect=map(float,vect) 
		finalv[vect[0]]=vect[1]  
		#print 'Vector is:', vect	
		#semi_cleaned.append(vect)
	semi_cleaned.append(finalv)

# Looping through dictionary items and creating the feature vector

for i in range(0, len(semi_cleaned)):
	x.append(semi_cleaned[i].values())	

#print 'X is :', x[8]

#Data cleaning for test examples

semi_cleaned=[]
for values in data2:
	values=values[3:]
	values=values.replace('\n','')
	values=values.split(' ')
	finalv={}
	for vect in values:
		vect=vect.split(':')
		# Converting it into a integer vector 
		vect=map(float,vect) 
		finalv[vect[0]]=vect[1]  
		#print 'Vector is:', vect	
	semi_cleaned.append(finalv)

# Looping through dictionary items and creating the feature vector


for i in range(0, len(semi_cleaned)):
	x_test.append(semi_cleaned[i].values())	

#print 'X is :', x_test[36]

print 'Starting the cross validation experiments for 10 folds using 20 combinations of rho_0 and C......'
 
			
accuracy=0
T=10
mistake_count_train=0
t=0
k_fold=10


for i in range(0,training_size):
	training_data+=[(expected[i], x[i])]

rho=[0.001,0.01,0.1,1]
clist=[0.1,1,10,100,1000]

for u in range(0, len(rho)):
	for v in range(0,len(clist)):
		rho_0=rho[u]
		C=clist[v]

		for kfolds in range(0, k_fold):
	
			heldoutset_size=training_size / k_fold 
			start,end=start_and_end(kfolds,heldoutset_size)
			
			
			# Randomly generating the weight vector

			for i in range(0,feature_vector_size):
					w.append(0)	
	
			for epoch in range(0, T):
			
				#if epoch !=0:
				shuffle(training_data)
	
				for i in range(0,training_size):

					if i in range(start,end+1):
						continue
				
					else:
						yi=training_data[i][0]
						xi=training_data[i][1]
						result= yi * ( dot(w,xi) )
						if (result <= 1):
							denom_temp= rho_0 * t / float(C)
							denom=1+denom_temp
							r=rho_0 / float(denom)
							gradient= w - C* dot(yi,xi)
							w= w- r * gradient
						
						else:
							denom_temp= rho_0 * t / float(C)
							denom=1+denom_temp
							r=rho_0 / float(denom)
							gradient= w 
							w=w-r*gradient	
					
						t=t+1
	
				#print 'Heldout set is :', heldout_set
	
				#print 'Len of heldout set is :', len(heldout_set)
				#print 'Helodut set of a particular index is :', heldout_set[2]
		
			for i in range(start,end+1):
				result=  dot(w,x[i])
				if (result <= 0):
					label=-1
				else:
					label=+1
				if label !=expected[i]:
					mistake_count_train=mistake_count_train+1

			#print 'Mistake count is :', mistake_count_train
			a=heldoutset_size-mistake_count_train
			b=heldoutset_size
			#print 'a is :', a, 'b is :', b
			accuracy=a/float(b) *100
			#print 'accuracy is :', accuracy
			finalaccuracy_list.append(accuracy)
			accuracy=0
			heldout_set=[]
			w=[]
			mistake_count_train=0

	

		#print ' The weight vector after training is :', w
		#print 'Accuracy list for individual run is :', finalaccuracy_list

		avg_accuracy=0	
		for i in range(0, len(finalaccuracy_list)):
			avg_accuracy+=(finalaccuracy_list[i])

		avg_accuracy=avg_accuracy /float(k_fold)
		overall_accuracy.append(avg_accuracy)

		finalaccuracy_list=[]
		avg_accuracy=0			

print 'Overall Average accuracy for different combinations of hyper parameters are :'
print '===================================='
for i in range(len(rho)):
	for j in range(len(clist)):
		print 'rho_0=', rho[i], 'C=', clist[j], 'Overall Accuracy :',overall_accuracy[i+j]
		

print '==========================='
print ' Experiments completed !!! '	
		
		
 
