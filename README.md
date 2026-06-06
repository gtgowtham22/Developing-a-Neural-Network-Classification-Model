# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
## STEP 1:
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

## STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

## STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

## STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

## STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

## STEP 6:
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.





## PROGRAM

### Name: SHAGILAN U

### Register Number: 212224040303

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from google.colab import drive
drive.mount('/content/drive')
# Load dataset
data = pd.read_csv('/content/drive/MyDrive/dq3/customers (1).csv')
data.head()
data.columns
# Drop ID column as it's not useful for classification
data = data.drop(columns=["ID"])
# Handle missing values
data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)
# Encode categorical variables
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])
# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3
# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#Initialize model
input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])

print("Name:GOWTHAM G T ")
print("Register No:212224110017")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)

with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

print("Name:GOWTHAM G T")
print("Register No:212224110017")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
```


### Dataset Information
<img width="1115" height="219" alt="image" src="https://github.com/user-attachments/assets/ff90ade1-0792-4c26-9ce5-9697d6d9bf65" />


### OUTPUT
<img width="516" height="383" alt="image" src="https://github.com/user-attachments/assets/e2b65cff-9beb-41be-8da1-6e5ab6fa341a" />


## Confusion Matrix

<img width="591" height="501" alt="image" src="https://github.com/user-attachments/assets/1f05821b-93ac-4416-abb7-8d98f4348560" />



## Classification Report
<img width="516" height="383" alt="image" src="https://github.com/user-attachments/assets/dfce6c41-4829-40fb-95a0-c7a35e35a135" />


### New Sample Data Prediction
<img width="324" height="80" alt="image" src="https://github.com/user-attachments/assets/27804678-5a49-4d0e-b207-d9718836ad9c" />


## RESULT
Neural network classification model for the given dataset is successfully developed.

