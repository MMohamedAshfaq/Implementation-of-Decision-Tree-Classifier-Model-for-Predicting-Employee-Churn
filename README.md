# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: FRANKLIN.F
RegisterNumber:  212224240041
*/
```
```py
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### DATA HEAD:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/bc753f9a-8a09-4815-89ae-79ec8e165e8a">

<br>
<br>
<br>
<br>
<br>
<br>

### DATASET INFO:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/f760819f-e78e-4a12-8b8c-32daeacb3410">

### NULL DATASET:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/67c8a973-c928-44e6-be58-ffd98b45057b">

### VALUES COUNT IN LEFT COLUMN:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/01f07495-4958-4186-bee8-c6632195895e">

### DATASET TRANSFORMED HEAD:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/1138eafe-ce6f-4fde-85e9-71daf0b64c00">

### X.HEAD:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/569580d7-a8b0-4d20-a637-73bf2f4bc9fe">

### ACCURACY:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/2dfb7e6d-2f2c-4b47-a09f-f766553e9dc5">

### DATA PREDICTION:
<img width="1115" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141175025/0e11798e-c2ab-44ee-875b-982d940a1851">

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
