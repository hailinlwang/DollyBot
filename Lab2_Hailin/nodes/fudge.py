import numpy as np
# 40.5,25
# 42,26
data = np.load('shortest_path copy.npy')
print("data.shape")
print(data)
print(data.shape)
print(type(data))

data = np.append(data,np.array([40.5,25.0,3.2583721e+00]).reshape((3,1)),axis=1)
data = np.append(data,np.array([42.0,26.0,5.2345321e+00]).reshape((3,1)),axis=1)
print(data)
print(data.shape)
print(type(data))

np.save('shortest_path_final_modified.npy', data)
