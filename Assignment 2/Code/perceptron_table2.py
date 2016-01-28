from random import choice
from numpy import array,dot,random


training_data = [
    (array([1,0,0,0]), 0),
    (array([1,1,0,0]), 0),
    (array([1,0,1,1]), 1),
    (array([0,1,0,0]), 0),
    (array([0,1,1,0]), 0),
    (array([1,1,1,0]), 0),
    (array([0,1,1,1]), 1),
]


test_data = [
    (array([1,0,0,0]), 0),
    (array([1,1,0,0]), 0),
    (array([1,0,1,1]), 1),
    (array([0,1,0,0]), 0),
    (array([0,1,1,0]), 0),
    (array([1,1,1,0]), 0),
    (array([0,1,1,1]), 1),
]

mistake_count_train=0
mistake_count_test=0
mistake_counter_train=[]
mistake_counter_test=[]
weight_vector=[]
feature_vector=4
no_of_mistakes_test=0
no_of_mistakes_train=0
w=[]
w_final=[]
w1,w2,w3,w4=[],[],[],[]
final_weightvect=[]
w1_sum,w2_sum,w3_sum,w4_sum=0,0,0,0

#print 'length of training data is :', len(training_data)

for i in range(0,feature_vector):
	w.append(random.uniform(-1,1))
print 'The weight vector generated for the given training data is :',w	

b=random.uniform(0.1,0.9)

print 'Bias is :', b
r=0.1  # Never less than 5 on training
#r=0.01  # Never less than 5
#r=1
for i in range(0,len(training_data)):
	x=training_data[i][0]
	expected=training_data[i][1]
	result=expected * ( dot(w,x) +b )
	if(result <= 0):
		mistake_count_train=mistake_count_train+1
		w= w+ r*expected*x
		b=b+r*expected
		print 'Incorrect prediction for :',x	
mistake_counter_train.append(mistake_count_train)
print 'The number of mistakes made on training set is:', mistake_count_train
print 'The updated bias is :', b
print 'The updated Weight vector at the end of one pass is :',w		
mistake_count_train=0

"""for i in range(0,len(test_data)):
	x=test_data[i][0]
	expected=test_data[i][1]
	#print ' x is :', x , 'expected is :', expected
	result= dot(w,x) +b 
	if( result <=0):
		label=0
	else:
		label=1	
	#print 'Result is:', result
	if(label!= expected):
		mistake_count_test=mistake_count_test+1
		#print ' Expected :', expected , 'Actual label :', label , 'Vector is:', x
	
print 'The number of mistakes made in test is:', mistake_count_test

accuracy=(len(test_data) - mistake_count_test) / float(len(test_data))	
print 'Accuracy on test set is :',accuracy"""




"""

For r=0.1 and 10 iterations 

 Average mistakes on training set is : 5.4
 Average mistakes on test set is : 4.3
The accuracy is : 38.5714285714
 Final weight vector is : [-0.0019387079744494495, -0.12013367037422258, 0.22756412904146273, 0.1639919477493966]


For r=0.5 

 Average mistakes on training set is : 5.4
 Average mistakes on test set is : 4.5
The accuracy is : 35.7142857143
 Final weight vector is : [0.0009203869657466113, 0.31687533004417706, 0.24390747932494997, 0.22356195129796808]
Anirudhs-MacBook-Pro:adult Anirudh$ 


For r=0.01

 Average mistakes on training set is : 5.5
 Average mistakes on test set is : 4.2
The accuracy is : 40.0
 Final weight vector is : [0.043048516897662002, 0.013157444610471924, 0.12737532126680484, -0.013139466982234671]

For r=1
 Average mistakes on training set is : 5.2
 Average mistakes on test set is : 4.6
The accuracy is : 34.2857142857
 Final weight vector is : [0.44134759533180679, 0.3349220084464119, 0.40020181179004111, 0.35148857665293182]

"""

			
		