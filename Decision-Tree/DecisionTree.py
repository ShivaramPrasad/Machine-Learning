
# coding: utf-8

# In[43]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree 


# In[37]:


#get Data..
iris = load_iris()

#to make simple i am going to del this particular data and take that data after to test my model..
test_index = [0,50,100]


# In[56]:


#training data...
train_data = np.delete(iris.data, test_index , axis = 0)
train_target = np.delete(iris.target, test_index)


# In[59]:


#testing data..
test_data = iris.data[test_index]
test_target = iris.target[test_index]


# In[57]:


#DecisionTree Classifier..
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


# In[64]:


#testing the fit model..
print("Predicted Output->F(X):{0}, Target Output->Y:{1}".format(clf.predict(test_data), test_target))


# In[66]:


#Calculate the accuracy..
from sklearn.metrics import accuracy_score
pred = clf.predict(test_data)
print(accuracy_score(test_target, pred))

