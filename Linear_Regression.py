
# coding: utf-8

# In[ ]:


import pandas as pd
import os
#print(os.listdir('../Machine_Learning/data_set/'))
names = ['preg', 'plas', 'Bpres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data=pd.read_csv('data_set/diabetes.csv',names=names)
#print(data.describe())
print(data.shape)
x=data[['preg']].values
y=data[['plas']].values
print(x)


# In[ ]:



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
#print(x_test,111111111)
print(y_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
print(predictions)
print(metrics.mean_squared_error(y_test,predictions))

