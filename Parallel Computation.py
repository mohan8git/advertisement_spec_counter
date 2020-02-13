#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import face_recognition
import numba
from numba import jit,prange,cuda
from timeit  import default_timer as timer
temp1 = []
nv = []
heha = timer()
@jit
##def getting_centre_coordinates(face_locations):
##    global temp1
##    
##    for i in range(len(face_locations)):
##            temp1 = list(temp1)
##            t1,t2,t3,t4=face_locations[i]
##            temp=centre(t1,t2,t3,t4)
##            temp1.append(temp) 
##            temp1= np.array(temp1)
##    return temp1
        

@jit(nopython = True, target = "cuda")
def FindPoint(left, top,right, bottom, cx, cy) :
    if (cx > left and cx < right and 
        cy > top and cy < bottom) :
        return True
    else : 
        return False

@jit(nopython = True, target = "cuda")
def  centre(top,right,bottom,left):
    y = bottom + int((top-bottom)*0.5)
    x = left + int((right - left)*0.5)
    return  [x,y]

@jit(nopython = True, target = "cuda")
def get_total_viewers(outputs,p):
    global nv
    for i in range(len(outputs)):
        for j in range(len(p)):
            a,b,c,d,e= outputs[i]
            flag=FindPoint(a,b,c,d,p[j][0],p[j][1])
                        #print(flag)
            if flag == False:
                if e not in nv:
                    nv.append(e)
            else:
                print("mubaarakho tumhaareko baccha hua")
    return nv


# In[3]:


face_locations = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
start = timer()
face_locations = np.array(face_locations)
# print("Convert karvaama {} time laagyo".format(timer()-start))

start = timer()  
for i in range(len(face_locations)):
    temp1 = list(temp1)
    t1,t2,t3,t4=face_locations[i]
    temp=centre(t1,t2,t3,t4)
    temp1.append(temp)
    temp1= np.array(temp1)
print("Centres find karwaana atlo time laagyo, GPU wagar")
    # print("Centre Co-ordinates find karvaa maa {} time laagyo".format(timer()-start))


# In[4]:


outputs = [[10,15,25,30,1],[52,69,63,14,2],[96,58,47,41,3],[104,250,630,140,4]]
outputs = np.array(outputs)
start = timer()
get_total_viewers(outputs,p)
stop = timer()
print("Bas laa? atlo j time laagyo?",stop-start)
nv
print(timer() - heha)

