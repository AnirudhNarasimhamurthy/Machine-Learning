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
feature_vector_size=100

mistake_count_train=0
mistake_count_test=0
mistake_marginp_train=0
mistake_marginp_test=0

if len(sys.argv)>1:
    dt_depth=float(sys.argv[1])
    rho_0=float(sys.argv[2])
    C=float(sys.argv[3])
else:
    dt_depth=4
    rho_0=0.001
    C=100
    

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
		#print 'Values is :', values
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

# Randomly generating the weight vector

for i in range(0,feature_vector_size):
	w.append(0)
	
#print 'Weight vector is :', w	

#rho_0=0.001
#C=100
T=30
t=0
for i in range(0,training_size):

	training_data+=[(expected[i], x[i])]


for epoch in range(0, T):

	if epoch != 0:
		shuffle(training_data)
	
	for i in range(0,training_size):
		yi=training_data[i][0]
		xi=training_data[i][1]
		result= yi * ( dot(w,xi) )
		if (result <= 1):
			#print 'Before updating weight vector was :', w
			denom_temp= rho_0 * t / float(C)
			denom=1+denom_temp
			r=rho_0 / float(denom)
			gradient= w - C* dot(yi,xi)
			w= w - r * gradient
		
		else:
			denom_temp= rho_0 * t / float(C)
			denom=1+denom_temp
			r=rho_0 / float(denom)
			gradient= w 
			w= w - r*gradient	
			
		t=t+1


#print ' The weight vector after training is :', w

# Accuracy on training data 


for i in range(0,training_size):

	result= dot(w,x[i]) 
	if (result <= 0):
		label=-1
	else:
		label=+1
	if label !=expected[i]:
		mistake_count_train=mistake_count_train+1

#print 'Training size is :', training_size
a=training_size - mistake_count_train
b=float(training_size)
#print 'a is :', a, 'b is :', b
accuracy=a/b * 100


print 'Final Accuracy on training data is :',a, '/', b, '=', accuracy	


# Accuracy on test data 
for i in range(0,test_size):
	
	result= dot(w,x_test[i]) 
	if (result <= 0):
		label=-1
	else:
		label=+1
	if label !=expected_test[i]:
		mistake_count_test=mistake_count_test+1

#print 'Test size is :', test_size
a=test_size - mistake_count_test
b=float(test_size)
#print 'a is :', a, 'b is :', b
accuracy=a/b * 100

print 'Final Accuracy on test data is :',a, '/', b, '=', accuracy		

