#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


# In[4]:


path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'


# In[6]:


df = pd.read_csv(path)
df.head()


# In[7]:


df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])


# In[8]:


df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


# In[12]:


df_sydney_processed.drop('Date',axis=1,inplace=True)


# In[13]:


df_sydney_processed = df_sydney_processed.astype(float)


# In[14]:


features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
features, Y


# In[15]:


# Q1) Use the train_test_split function to split the features and Y dataframes with a test_size of 0.2 and the random_state set to 10
# Use the train_test_split function to split the features and Y dataframes with a test_size of 0.2 and the random_state set to 10.
from sklearn.model_selection import train_test_split

# Assuming you have features (X) and the target variable (Y) as separate dataframes
# Specify test_size and random_state parameters
X_train, X_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=10)
Y_train


# In[16]:


# Q2) Create and train a Linear Regression model called LinearReg using the training data (x_train, y_train).
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
LinearReg = LinearRegression()

# Train the model using the training data
LinearReg.fit(X_train, Y_train)


# In[17]:


# Q3) Now use the predict method on the testing data (x_test) and save it to the array predictions
# Use the trained LinearReg model to make predictions on the testing data
predictions = LinearReg.predict(X_test)

# Now, the 'predictions' array contains the predicted values for the testing data.
predictions


# In[19]:


# Q4) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate Mean Absolute Error (MAE)
LinearRegression_MAE = mean_absolute_error(Y_test, predictions)

# Calculate Mean Squared Error (MSE)
LinearRegression_MSE = mean_squared_error(Y_test, predictions)

# Calculate R-squared (R2) score
LinearRegression_R2 = r2_score(Y_test, predictions)

# Print the results
print(f"LinearRegression MAE: {LinearRegression_MAE}")
print(f"LinearRegression MSE: {LinearRegression_MSE}")
print(f"LinearRegression R2: {LinearRegression_R2}")


# In[21]:


# Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate MAE, MSE, and R2
LinearRegression_MAE = mean_absolute_error(Y_test, predictions)
LinearRegression_MSE = mean_squared_error(Y_test, predictions)
LinearRegression_R2 = r2_score(Y_test, predictions)

# Create a DataFrame to display the metrics
report_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R2'],
    'Linear Regression': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
})

# Display the DataFrame
print(report_df)


# In[22]:


# Q6) Create and train a KNN model called KNN using the training data (x_train, y_train) with the n_neighbors parameter set to 4.
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN model with n_neighbors set to 4
KNN = KNeighborsClassifier(n_neighbors=4)

# Train the KNN model using the training data
KNN.fit(X_train, Y_train)

# The KNN model (KNN) is now trained and ready to make predictions.


# In[42]:


# Q7) Now use the predict method on the testing data (x_test) and save it to the array predictions.
# Use the trained KNN model to make predictions on the testing data
predictions = KNN.predict(X_test)
predictions

# Now, the 'predictions' array contains the predicted values for the testing data.


# In[24]:


# Q8) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function.
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

# Calculate Accuracy Score
KNN_Accuracy_Score = accuracy_score(Y_test, predictions)

# Calculate Jaccard Index (binary classification)
KNN_JaccardIndex = jaccard_score(Y_test, predictions)

# Calculate F1 Score (binary classification)
KNN_F1_Score = f1_score(Y_test, predictions)

# Print the results
print(f"KNN Accuracy Score: {KNN_Accuracy_Score}")
print(f"KNN Jaccard Index: {KNN_JaccardIndex}")
print(f"KNN F1 Score: {KNN_F1_Score}")


# In[26]:


# Q9) Create and train a Decision Tree model called Tree using the training data (x_train, y_train).
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree model
Tree = DecisionTreeClassifier()

# Train the Decision Tree model using the training data
Tree.fit(X_train, Y_train)


# In[43]:


# Q10) Now use the predict method on the testing data (x_test) and save it to the array predictions.
# Use the trained Decision Tree model to make predictions on the testing data
predictions = Tree.predict(X_test)
predictions


# In[29]:


# Q11) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function.
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

# Calculate Accuracy Score
Tree_Accuracy_Score = accuracy_score(Y_test, predictions)

# Calculate Jaccard Index (binary classification)
Tree_JaccardIndex = jaccard_score(Y_test, predictions)

# Calculate F1 Score (binary classification)
Tree_F1_Score = f1_score(Y_test, predictions)

# Print the results
print(f"Tree Accuracy Score: {Tree_Accuracy_Score}")
print(f"Tree Jaccard Index: {Tree_JaccardIndex}")
print(f"Tree F1 Score: {Tree_F1_Score}")


# In[44]:


# Q12) Use the train_test_split function to split the features and Y dataframes with a test_size of 0.2 and the random_state set to 1.

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=1)
Y_test


# In[31]:


# Q13) Create and train a LogisticRegression model called LR using the training data (x_train, y_train) with the solver parameter set to liblinear.
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model with the solver set to "liblinear"
LR = LogisticRegression(solver='liblinear')

# Train the Logistic Regression model using the training data
LR.fit(X_train, Y_train)

# The Logistic Regression model (LR) is now trained and ready to make predictions.


# In[45]:


# Q14) Now, use the `predict` and `predict_proba` methods on the testing data (`x_test`) and save it as 2 arrays `predictions` and `predict_proba`.
# Use the trained Logistic Regression model to make binary predictions on the testing data
predictions = LR.predict(X_test)

# Use the trained Logistic Regression model to get probability estimates on the testing data
predict_proba = LR.predict_proba(X_test)

predict_proba
# Now, 'predictions' contains the binary predictions (0 or 1),
# and 'predict_proba' contains the probability estimates for each class.


# In[37]:


# Q15) Using the predictions, predict_proba and the y_test dataframe calculate the value for each metric using the appropriate function.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss

# Calculate Accuracy Score
LR_Accuracy_Score = accuracy_score(Y_test, predictions)

# Calculate Jaccard Index (binary classification)
LR_JaccardIndex = jaccard_score(Y_test, predictions)

# Calculate F1 Score (binary classification)
LR_F1_Score = f1_score(Y_test, predictions)

# Calculate Log Loss
LR_Log_Loss = log_loss(Y_test, predict_proba)

# Print the results
print(f"LR Accuracy Score: {LR_Accuracy_Score}")
print(f"LR Jaccard Index: {LR_JaccardIndex}")
print(f"LR F1 Score: {LR_F1_Score}")
print(f"LR Log Loss: {LR_Log_Loss}")


# In[38]:


# Q16) Create and train a SVM model called SVM using the training data (x_train, y_train).
from sklearn.svm import SVC

# Create an SVM model
SVM = SVC()

# Train the SVM model using the training data
SVM.fit(X_train, Y_train)

# The SVM model (SVM) is now trained and ready to make predictions.


# In[46]:


# Q17) Now use the predict method on the testing data (x_test) and save it to the array predictions.
# Use the trained SVM model to make predictions on the testing data
predictions = SVM.predict(X_test)
predictions
# Now, the 'predictions' array contains the predicted values for the testing data.


# In[40]:


# Q18) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function.
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

# Calculate Accuracy Score
SVM_Accuracy_Score = accuracy_score(Y_test, predictions)

# Calculate Jaccard Index (binary classification)
SVM_JaccardIndex = jaccard_score(Y_test, predictions)

# Calculate F1 Score (binary classification)
SVM_F1_Score = f1_score(Y_test, predictions)

# Print the results
print(f"SVM Accuracy Score: {SVM_Accuracy_Score}")
print(f"SVM Jaccard Index: {SVM_JaccardIndex}")
print(f"SVM F1 Score: {SVM_F1_Score}")


# In[41]:


# Q19) Show the Accuracy,Jaccard Index,F1-Score and LogLoss in a tabular format using data frame for all of the above models.
import pandas as pd
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss

# Calculate metrics for each model
metrics = {
    'Model': ['Linear Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Logistic Regression', 'Support Vector Machine'],
    'Accuracy': [LinearRegression_MAE, KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [0, KNN_JaccardIndex, 0, LR_JaccardIndex, SVM_JaccardIndex],
    'F1 Score': [0, KNN_F1_Score, 0, LR_F1_Score, SVM_F1_Score],
    'Log Loss': [0, 0, 0, LR_Log_Loss, 0]
}

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame(metrics)

# Display the DataFrame
print(metrics_df)


# In[ ]:




