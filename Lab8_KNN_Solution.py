# ARTI308 - Machine Learning | Lab 8 K Nearest Neighbors
# Data file: KNN_Project_Data.csv

# K Nearest Neighbors Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get the Data

df = pd.read_csv('KNN_Project_Data.csv')

# Check the head of the dataframe.**

df.head()

# EDA

sns.pairplot(df, hue='TARGET CLASS')

# Standardize the Variables

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler() object called scaler.**

scaler = StandardScaler()

# Fit scaler to the features.**

scaler.fit(df.drop('TARGET CLASS', axis=1))

# Use the .transform() method to transform the features to a scaled version.**

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30, random_state=42)

# Using KNN

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN model instance with n_neighbors=1**

knn = KNeighborsClassifier(n_neighbors=1)

# Fit this KNN model to the training data.**

knn.fit(X_train, y_train)

# Predictions and Evaluations

# Use the predict method to predict values using your KNN model and X_test.**

pred = knn.predict(X_test)

# Create a confusion matrix and classification report.**

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

# Choosing a K Value

error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Now create the following plot using the information from your for loop.**

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Retrain with new K Value

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# End of notebook
