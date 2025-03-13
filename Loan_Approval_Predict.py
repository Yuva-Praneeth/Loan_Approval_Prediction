#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[28]:


dataset=pd.read_csv(r" Dataset path ")


# In[29]:


dataset.head()


# In[30]:


dataset.shape


# In[31]:


dataset.isna().sum()


# In[32]:


dataset = dataset.dropna()
dataset.shape


# In[33]:


dataset.reset_index(inplace=True)


# In[34]:


dataset


# In[35]:


dataset['Dependents'].unique()


# In[36]:


dataset['Dependents'].value_counts()


# In[37]:


dataset['Dependents']= dataset['Dependents'].replace(to_replace='3+',value=4)


# In[38]:


dataset['Dependents'].value_counts()


# In[39]:


sns.countplot(x='Education',hue ='Loan_Status',data =dataset)


# In[40]:


sns.histplot(data=dataset , x='LoanAmount',kde=True)
plt.show()


# In[41]:


dataset.replace({'Married':{'Yes':1,'No':0},'Gender':{'Male':1,'Female':0},'Education':{'Graduate':1,'Not Graduate':0},'Self_Employed':{'Yes':1,'No':0},'Property_Area':{'Rural':0,'Urban':1,'Semiurban':2},} , inplace=True)


# In[42]:


dataset


# In[43]:


dataset['Dependents'] = dataset['Dependents'].astype('int')


# In[44]:


X=dataset.iloc[: ,2:-1].values
X


# In[45]:


y=dataset.iloc[:,-1].values


# In[46]:


y


# In[47]:


X


# In[48]:


correlation =dataset.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(correlation,cmap='coolwarm',annot=True,square=True)
plt.show()


# In[49]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size =0.25,random_state=42)


# In[50]:


X_train.shape


# In[51]:


X_test.shape


# In[52]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[53]:


k_class =KNeighborsClassifier(n_neighbors=3)
k_class.fit(X_train,y_train)


# In[54]:


k_y_pred =k_class.predict(X_test)
sns.heatmap(confusion_matrix(y_test,k_y_pred),annot=True, cmap="Blues")


# In[74]:


k_a=accuracy_score(y_test,k_y_pred)
print("Accuracy:",k_a)

k_precision = metrics.precision_score(y_test, k_y_pred, pos_label='Y') 
print("Precision score:",k_precision)
k_recall = metrics.recall_score(y_test, k_y_pred, pos_label='Y') 
print("Recall score:",k_recall)
k_F1=metrics.f1_score(y_test,k_y_pred, pos_label='Y')
print("F1 score:",k_F1)


# In[75]:


from sklearn.svm import SVC
s_class=SVC(kernel='rbf',random_state=42)
s_class.fit(X_train,y_train)


# In[76]:


s_y_pred =s_class.predict(X_test)


# In[79]:


sns.heatmap(confusion_matrix(y_test,s_y_pred),annot=True,cmap = "Greens_r")
s_a=accuracy_score(y_test,k_y_pred)
print("accuracy",s_a)

s_precision = metrics.precision_score(y_test, s_y_pred, pos_label='Y') 
print("Precision score:",s_precision)
s_recall = metrics.recall_score(y_test, s_y_pred, pos_label='Y') 
print("Recall score:",s_recall)
s_F1=metrics.f1_score(y_test,s_y_pred, pos_label='Y')
print("F1 score:",s_F1)


# In[80]:


from sklearn.ensemble import RandomForestClassifier
classfier=RandomForestClassifier(n_estimators =30, criterion ='entropy')
classfier.fit(X_train,y_train)


# In[81]:


y_pred =classfier.predict(X_test)


# In[82]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)


# In[83]:


R_a=accuracy_score(y_test,y_pred)
print(R_a)


r_precision = metrics.precision_score(y_test, y_pred, pos_label='Y')
print("Precision score:",r_precision)
r_recall = metrics.recall_score(y_test, y_pred, pos_label='Y') 
print("Recall score:",r_recall)
r_F1=metrics.f1_score(y_test,y_pred, pos_label='Y')
print("F1 score:",r_F1)


# In[84]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state=43)
classifier.fit(X_train, y_train)


# In[85]:


d_y_pred = classifier.predict(X_test)


# In[86]:


sns.heatmap(confusion_matrix(y_test,d_y_pred),annot=True,cmap = "Blues")


# In[88]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, d_y_pred) 
print(cm)
d_a = metrics.accuracy_score(y_test, d_y_pred) 
print("Accuracy score:",d_a)
d_precision = metrics.precision_score(y_test, d_y_pred, pos_label='Y') 
print("Precision score:",d_precision)
d_recall = metrics.recall_score(y_test, d_y_pred, pos_label='Y') 
print("Recall score:",d_recall)
d_F1=metrics.f1_score(y_test,d_y_pred, pos_label='Y')
print("F1 score:",d_F1)


# In[89]:



accuracy_scores = [k_a,s_a,R_a,d_a]
algorithm_names = ['KNN','SVM', 'Random Forest', 'Desicion Tree'  ]


plt.barh(algorithm_names, accuracy_scores)
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
plt.title('Accuracy of Different Algorithms')
plt.show()


# In[90]:


plotdata = pd.DataFrame({

   
    "Accuracy score":[0.5833333333333334,0.5833333333333334,0.7833333333333333,0.6833333333333333],
    "Precision score":[0.6739130434782609, 0.6833333333333333, 0.78,0.7619047619047619],
    "Recall score":[0.7560975609756098,1.0,0.95121951219512190,0.7804878048780488],
    "F1 score":[0.7126436781609194,0.8118811881188119,0.8571428571428571,0.7710843373493976]},
  
     index=["KNN", "SVM", "Random Forest","Desicion Tree"])

plotdata.plot(kind="bar",figsize=(15, 8))

plt.title("Metrics Plot")

plt.xlabel("Algorithm Used")


# In[91]:


from sklearn import tree
plt.figure(figsize=(30,30))
tree.plot_tree(classifier,class_names=['0','1'], filled=True,rounded=True,fontsize=12)


# In[ ]:




