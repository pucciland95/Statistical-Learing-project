
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[13]:


#import of the training dataset
df = pd.read_csv("C:/Users/nicco/Desktop/Niccolo/Universita/Doppia laurea Cina/Esami/1_semestre/Statistical_Learning_and_Inference/Project/train.csv")


# In[14]:


df.head()


# In[15]:


from sklearn.preprocessing import MinMaxScaler
#Da implementare


# In[16]:


#Shuffle of the dataset
df = df.sample(frac=1)


# In[17]:


#Division of the Feature data and the target data
col = [x for x in df.columns if x!='categories' and x!='id'] 

X = df[col]
y = df['categories']


# In[18]:


#Normalization of the data
from sklearn import preprocessing 
X_clone = X.copy()
X_normalized = preprocessing.scale(X_clone)


# In[19]:


#Calculate the number of data for each classes
classSizes = (df.groupby('categories').size())
print(classSizes)


# In[21]:


#Splitting of the datset in x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=21, stratify=y)

x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(X_normalized, y, test_size = 0.1, random_state=21, stratify=y)

print("The lenght of the X_training dataset is: ", len(x_train))
print("The lenght of the X_test dataset is: ", len(x_test))
print("The lenght of the y_training dataset is: ", len(y_train))
print("The lenght of the y_training dataset is: ", len(y_test))


# # KNN

# In[ ]:


#Fitting of the model and prediction with KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

eval_error = []
training_error = []

training_error_normalized = []
eval_error_normalized = []

for x in range(1,20):
    KNN = KNeighborsClassifier(n_neighbors = x, n_jobs = -1)
    
    #Training for normalized data
    KNN_norm = KNN.fit(x_train_norm, y_train)
    KNN_training_predictions_norm = KNN_norm.predict(x_train_norm)
    KNN_evaluation_predictions_norm = KNN_norm.predict(x_test_norm)
    
    #Training for normal data
    KNN = KNN.fit(x_train, y_train)
    KNN_training_predictions = KNN.predict(x_train)
    KNN_evaluation_predictions = KNN.predict(x_test)
    
    error_df = list()
    error_df_normalized = list()
    
    error_df.append(pd.Series({'train' : sum(y_train == KNN_training_predictions)/len(y_train),
                                'eval' : sum(y_test == KNN_evaluation_predictions)/len(y_test)}))
    
    print("The actual number of neighbours considered are: ", x)
    print("This is the training accuracy: ", error_df[0][0] * 100)
    print("This is the evaluation accuracy: ", error_df[0][1] * 100)
    
    error_df_normalized.append(pd.Series({'train' : sum(y_train_norm == KNN_training_predictions_norm)/len(y_train_norm),
                                           'eval' : sum(y_test_norm == KNN_evaluation_predictions_norm)/len(y_test_norm)}))
    
    print("This is the training accuracy for the normalized data: ", error_df_normalized[0][0] * 100)
    print("This is the evaluation accuracy for the normalized data: ", error_df_normalized[0][1] * 100)
    print("\n")
    
    
    training_error.append(error_df[0][0])
    eval_error.append(error_df[0][1])
    
    training_error_normalized.append(error_df_normalized[0][0])
    eval_error_normalized.append(error_df_normalized[0][1])


# In[ ]:


#Misclassification of 
misclassified_elements_training = 1 - training_error
misclassified_elements_evaluation = 1 - eval_error

misclassified_elements_training_normalized = 1 - training_error_normalized
misclassified_elements_evaluation_normalized = 1 - eval_error_normalized



plt.plot(np.arange(19), misclassified_elements_training)
plt.plot(np.arange(19), misclassified_elements_evaluation)
plt.plot(np.arange(19), misclassified_elements_training_normalized)
plt.plot(np.arange(19), misclassified_elements_evaluation_normalized)
plt.show()


# In[ ]:


#Choosen KNN model
KNN_chosen = KNeighborsClassifier(n_neighbors = 8 )
KNN_chosen = KNN_chosen.fit(x_train, y_train)
predictions = KNN_chosen.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

