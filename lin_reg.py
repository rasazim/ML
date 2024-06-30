import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
plt.rcParams['font.weight']= 'bold'
plt.style.use('seaborn-whitegrid')

train_size = 97

file = "Data Sets/linear_dataset.csv"

data = pd.read_csv(file)
data_arr=np.array(data)
x = np.append(data_arr[0:train_size,0:1],np.ones(shape=(train_size,1)),axis=1)
x=x.astype("float64")
y = data_arr[0:train_size,1:2]
y=y.astype("float64")

n=2
m=train_size


T = np.row_stack(np.array([0]*n))

grate=0.01
def j(T,x,y,m):
    return np.transpose((np.row_stack(np.matmul(x,T))-y)/m)

for _ in range(9000):
    J=j(T,x,y,m)
    T=T-grate*np.transpose(np.matmul(J,x))
print(T)
T = np.linalg.pinv(x) @ y
#T = np.linalg.inv(np.transpose(x)@x)@np.transpose(x)@y
print(T)










