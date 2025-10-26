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
Developed by: M.MOHAMED ASHFAQ
RegisterNumber:  212224240090
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
<img width="1604" height="296" alt="8ml1" src="https://github.com/user-attachments/assets/8ce692f3-5e97-490a-a6e8-248c2f89be7f" />

### DATASET INFO:
<img width="1000" height="422" alt="8ml2" src="https://github.com/user-attachments/assets/383b2db4-7c5c-4788-a0cd-3c424ef72f37" />


### NULL DATASET:
<img width="976" height="280" alt="8ml3" src="https://github.com/user-attachments/assets/0ea65049-e13f-43e7-8ba6-253410bb358f" />


### VALUES COUNT IN LEFT COLUMN:
<img width="1458" height="122" alt="8ml4" src="https://github.com/user-attachments/assets/44943bf8-d501-454b-8a7f-7dadb0058aad" />


### DATASET TRANSFORMED HEAD:
<img width="1604" height="252" alt="8ml5" src="https://github.com/user-attachments/assets/556dad8f-9ef0-4994-a62c-4cafd7ef4795" />


### X.HEAD:
<img width="1518" height="244" alt="8ml6" src="https://github.com/user-attachments/assets/0e99cf23-7329-4454-ba4e-d8b7dad9b316" />


### ACCURACY:
<img width="600" height="58" alt="8ml7" src="https://github.com/user-attachments/assets/44d1b8a6-046a-489d-bafa-661825128c9d" />


### DATA PREDICTION:
<img width="1622" height="128" alt="8ml8" src="https://github.com/user-attachments/assets/387317ed-8bda-4ddd-bc5a-ff76e31b1706" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
