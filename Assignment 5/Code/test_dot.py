from numpy import dot
import numpy as np
#import statistics 
w=[1,1,1,1,2,1,1]
xi=[1,2,3,4,5,6,7]
yi=[-1,1,1,1,-1,1,1]
r=1
temp1= dot(yi,xi)
temp2=r-temp1
w= w-temp2 
temp3=dot(yi[1],xi)
temp4=w - temp3

print 'Temp1 is:', temp1
print 'Temp2 is :', temp2
print 'Temp3 is :', temp3
print 'Temp4 is :', temp4
print 'W is :', w

abc=[1,2,3,5,6,7]
result=np.var(abc)

print 'Result is :', result

def grades_variance(my_list, average):
    variance = 0
    for i in my_list:
        variance += (average - my_list[i]) ** 2
    return variance / len(my_list)