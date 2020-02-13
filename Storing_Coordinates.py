


import numpy as np



def make_matrix(t):
    p = np.array([0,0,0,t,0,0,0,0]).reshape(2,4)
    return p
o = make_matrix(2)
a = [[[1,2,3,4,1],[5,6,7,8,2]],[[9,10,11,12,2],
    [13,14,15,16,3]],[[17,18,19,20,3],[21,22,23,24,1],
    [25,26,27,28,2]],[[29,30,31,32,5],[33,34,35,36,4]],
    [[37,38,39,40,3]],[[41,42,43,44,7],[45,46,47,48,8]],
    [[49,50,51,52,6],[53,54,55,56,12]],[[57,58,59,60,18],
    [61,62,63,64,14]]]
a =  np.array(a)
q = [[1,2],[2,3],[3,1,2],[5,4],[3],[7,8],[6,12],[18,14]]
l = []





existing_list = []
container = []
# for frame_coordinates in a:
def storing(frame_coordinates):
    global existing_list
    global container
    for single_coordinates in frame_coordinates:
        temp = single_coordinates[-1]
        if temp not in existing_list:
            existing_list.append(temp)
            new = make_matrix(temp)
            new = np.vstack((new,single_coordinates[0:4]))
            container.append(new)
            
            return container
        else:
            ind = existing_list.index(temp)
            container[ind] = np.vstack((container[ind],single_coordinates[0:4]))
            
            return container
            
for k in a:
    storing(k)







# In[ ]:




