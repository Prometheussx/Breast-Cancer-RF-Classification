# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29, 2023
@author: erdem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Inclusion
data = pd.read_csv("Cancer_Data.csv")

# Remove unnecessary columns
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Separate malignant (M) and benign (B) data
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# Scatter Plot
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Malignant", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="Benign", alpha=0.3)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

# Update diagnosis values to 1 for Malignant and 0 for Benign
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# Get target (y) and feature (x_data) data
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("Decision Tree Score:", dt.score(x_test, y_test))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)
print("Random Forest Algorithm Result:", rf.score(x_test, y_test))

# Get predictions and true values
y_prediction = rf.predict(x_test)
y_true = y_test

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_prediction)

# Confusion Matrix Visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)

plt.xlabel("Predicted (Y_Predicted)")
plt.ylabel("Actual (Y_True)")
