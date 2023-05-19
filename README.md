# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
 
2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mohanapriya U
RegisterNumber: 212220040091
*/
```
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])

data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y = data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


## Output:
![decision tree classifier model](sam.png)

1.Data Head

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/651da90d-a013-42f3-ab0b-f013af5f4c80)


2.Data Info

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/1baf24e7-1de8-4151-a1ba-0ca080085e18)

3.Data isnull

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/39075883-c237-430e-bd28-4b432567341b)

4.Data Left

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/2ca2f264-1bcb-4f19-ac39-006395385f70)

5.X Head

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/e9dd78cd-1aa4-4b9e-a647-f52b43a32a6c)

6.Data fit

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/6522581c-91a4-49e7-9cb8-5ad8432b9dda)

7.Accuracy

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/42c8e938-ea38-4454-aba0-bbb8713c4c75)

8.Predicted Values

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133958624/9c1a01fb-6efd-4a8d-9ec4-2b8679608c85)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
