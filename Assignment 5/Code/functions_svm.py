def dotproduct(s,x):
    product={}
    for key in x:
        product[key]=s*x[key]
    return product

def crossproduct(x,y):
    prod=0
    for key in y:
        if key in x:
            prod+=x[key]*y[key]
    return prod
    
def addvectors(x,y):
    sum={}
    for key in x:
        if key in y:
            sum[key]=x[key]+y[key]
        else:
            sum[key]=x[key]
    for key in y:
        if key not in x:
            sum[key]=y[key]
    return sum

def subvectors(x,y):
    sum={}
    for key in x:
        if key in y:
            sum[key]=x[key]-y[key]
        else:
            sum[key]=x[key]
    for key in y:
        if key not in x:
            sum[key]=-y[key]
    return sum
    
def sgn(x):
    print("sign(",x,")")
    return 1 if (x>=0) else -1

def mymean(listofnumbers):
    sum=0
    for x in listofnumbers:
        sum+=x
    return sum/float(len(listofnumbers))
    
def vari(listofnumbers,mymean):
    sum=0
    for x in listofnumbers:
        sum+=(x-mymean)**2
    return sum/float(len(listofnumbers))
    
# print(vectors[0])
# print(vectors[1])
# print(dotproduct(3,vectors[1]))
# print(sgn(3),sgn(-3))
# print("cross:",crossproduct(vectors[0],vectors[1]))
# print("add:",addvectors(vectors[0],vectors[1]))
