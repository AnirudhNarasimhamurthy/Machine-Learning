from time import clock
from numpy import dot
import random
from random import shuffle

expected=[]
expected_test=[]
semi_cleaned=[]
fully_cleaned=[]
semi_cleaned_test=[]
fully_cleaned_test=[]
training_data=[]
x=[]
x_test=[]
w=[]
w_mp=[]
value=[]
z=[]
mistake_counter=[]
mp_mistake_counter=[]

start_time=clock()


mistake_count_train=0
mistake_count_test=0
mistake_marginp_train=0
mistake_marginp_test=0
total_mistakes=0
mp_total_mistakes=0


#Reading the training file and converting it to a list of names 
with open('a1a.train', 'r') as inputFile:
	data = inputFile.readlines()
	training_size=len(data)
	print 'Number of training examples',len(data)

#Reading the test file and converting it to a list of names 
with open('a1a.test', 'r') as inputFile:
	data2 = inputFile.readlines()
	test_size=len(data2)
	print 'Number of test examples',len(data2)


#Extracting the expected labels for all the training examples and storing it in list	
for values in data:
	expected.append(values[:2])	
expected=map(float,expected)


#Extracting the expected labels for all the test examples and storing it in list	
for values in data2:
	expected_test.append(values[:2])	
expected_test=map(float,expected_test)



#Data cleaning for training examples


max_train=0
for values in data:
	values=values[3:]
	values=values.replace(':1 ',',')[:-2]
	temp=values[-3:]
	if(temp > max_train):
		max_train=temp
	semi_cleaned.append(values)

#print' Required vector is :', cleaned	
#print 'Max value or feature vector size for training is :', max_train

features_train=int(max_train)

for vect in semi_cleaned:
	vect= vect.split(',')
	""" Converting it into a integer vector """
	vect=map(float,vect)   
	#print 'Vector is:', vect	
	fully_cleaned.append(vect)

#Obtaining the feature vector or x	

for j in range(0, training_size):
	y=fully_cleaned[j]
	#print 'y is :', y
	for i in range(0,features_train):
		z.append(0)
	
	for i in range(0,features_train):
		if(i in y):
			#print' Setting the index value as 1 for z[',i,']'
			z[i-1]=1
	x.append(z)
	z=[]
	
#print 'Feature vector is :',x
	

#Data cleaning for test examples
max=0

for values in data2:
	values=values[3:]
	values=values.replace(':1 ',',')[:-2]
	temp=values[-3:]
	#print 'Temp is :', temp
	if(temp > max):
		max=temp
	#print 'Values is :', values
	semi_cleaned_test.append(values)

#print' Required vector is :', cleaned	
#print 'Max value or feature vector size is :', max

features_test=int(max)

for vect in semi_cleaned_test:
	vect= vect.split(',')
	""" Converting it into a integer vector """
	vect=map(float,vect)   
	#print 'Vector is:', vect	
	fully_cleaned_test.append(vect)

#Obtaining the feature vector or x	

for j in range(0, test_size):
	y=fully_cleaned_test[j]
	#print 'y is :', y
	for i in range(0,features_test):
		z.append(0)
	
	for i in range(0,features_test):
		if(i in y):
			#print' Setting the index value as 1 for z[',i,']'
			z[i-1]=1
	x_test.append(z)
	z=[]



#Margin Perceptron

#Training examples

# Randomly generating the weight vector

for i in range(0,features_train):
	w_mp.append(random.uniform(-1,1))

# Generating the bias

b=random.uniform(-1,1)

# Hyper parameters --- Please make changes here to the hyper parameters and run the code

#mu= random.uniform(0,5)
mu=4
r=1
epoch=3
shuff=random.randint(0,1)

print 'Aggressive Margin Perceptron'
print '--------------------------------------------'
print 'Bias:', b

print  '\nHyper parameters'
print '----------------'
print 'r=',r ,'mu=',mu,'epoch=',epoch



# With Shuffle

for i in range(0,training_size):

	training_data+=[(expected[i], x[i])]


for k in range(0, epoch):


	#shuffle(training_data)

	
	for i in range(0,training_size):
	#print 'Expected label here is:', expected[i]
		y=training_data[i][0]
		xdash=training_data[i][1]
		#print 'Training data is :', x
		#print ' Expected label is:', y

		result= y * ( dot(w_mp,xdash) +b )
		eta= (mu - result ) / float( dot(xdash,xdash) +1)
		#print 'Eta is :', eta
		if (result <= mu):
		
			w_mp= w_mp+ (eta* dot(y,xdash))
			b= b + eta*y
			mistake_marginp_train=mistake_marginp_train+1
		
	mp_mistake_counter.append(mistake_marginp_train)
	mistake_marginp_train=0	


print 'No of updates in each epoch', mp_mistake_counter

for i in mp_mistake_counter:
		mp_total_mistakes+=i	
		

print '\nTraining Set'
print '------------------------------------'

print 'Total number of examples on training set is :',training_size*epoch	
print 'The number of mistakes/updates made on training data is :', mp_total_mistakes

# Applying the updated weight vector on the training examples to obtain the accuracy

mistake_marginp_train=0
for i in range(0,training_size):
	#print 'Expected label here is:', expected[i]
	f=expected[i]
	g=dot(w_mp,x[i]) +b
	result=  dot(w_mp,x[i]) +b 
	if (result <= 0):
		label=-1
	else:
		label=+1
	if label !=expected[i]:
		mistake_marginp_train=mistake_marginp_train+1

print '\nApplying the updated weight vector on the training examples to obtain the accuracy'
print '------------------------------------------------------------------------------------'
print 'Total number of examples on training set is :',training_size	
print 'The number of mistakes made  using the updated weight vector is :', mistake_marginp_train
accuracy= (training_size-mistake_marginp_train) / float(training_size)
print 'Accuracy on the training set is :', accuracy * 100 ,'%'
print '************************************************************************************'


# Applying the updated weight vector on Test Examples to obtain the accuracy

""" The training sample had a feature vector size of 119 and hence the weight vector was also created with the same dimesnion
Since the test examples have a feature vector size of 123, I am randomly adding 4 weights at the end to the weight cvector
obtained after training."""

w_mp=map(str,w_mp)

for i in range(1,5):
	w_mp.append(random.uniform(-1,1))

w_mp=map(float,w_mp)
#print 'Weight vector is :', w	


for i in range(0,test_size):

	result=  dot(w_mp,x_test[i]) +b 
	#print 'Dot product Result + Bias is:', dot(w,x[i]) +b
	#print ' Expected label is :',expected[i]
	#print 'Result is :', result
	if (result <= 0):
		label=-1
	else:
		label=+1
	if(label !=expected_test[i]):
		mistake_marginp_test=mistake_marginp_test+1

print '\nApplying the updated weight vector on Test Examples to obtain the accuracy'
print '-----------------------------------------------------------------------------'
print 'Total number of examples on test set is :',test_size	
print 'The number of mistakes made using the updated weight vector is :',mistake_marginp_test
accuracy= (test_size-mistake_marginp_test) / float(test_size)
print 'Accuracy on the test set is :', accuracy * 100, '%'
print '******************************************************************************'

running_time=clock() - start_time
print 'Time taken in seconds is :',running_time 	


