from numba import jit,vectorize,cuda,prange
import numpy as np
from timeit import default_timer as timer
q = 0
d = np.array([None]*1)






@jit(["int32(int32,int32,int32,int32)"])
def dev(a,b,c,f):
    print(d.shape)
    q = 0
    for i in range(a):
        for p in range (b):
            for j in range(c):
                for k in range(f):
                    o  = i*p*j
                    q = q + o
##        d[0] = q
        return q




##@vectorize(["int32(int32,int32,int32,int32)"])
##def mul(a,b,c,f):
##    print(d.shape)
##    q = 0
##    for i in range(a):
##        for p in range (b):
##            for j in range(c):
##                for k in range(f):
##                    o  = i*p*j
##                    q = q + o
##   
##    return  q
##





a = 100
b = 100
c = 100
f = 100
start = timer()
p = dev(a,b,c,f)
print("Bhai atlo time laagyo jit maa",timer() -  start)
##e = mul(a,b,c,f)
##start = timer()
##print("Bhai atlo time laagyo  vectorize maa",timer() -  start)
