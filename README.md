# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data Loading and Exploration
3. Data Preparation
4. Train-Test Split
5. Text Vectorization
6. Model Training
7. Making Predictions
8. Evaluation
9.  End

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARSHANA M V
RegisterNumber:  212224240053
*/
```
```
import pandas as pd
data = pd.read_csv("spam.csv",encoding = 'Windows-1252') 
data.head()
data.isnull().sum()
x = data["v2"].values
y = data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:
<img width="1395" height="383" alt="image" src="https://github.com/user-attachments/assets/b4f6cf23-a18f-4336-ab10-a1bab5ff0112" />
<img width="451" height="239" alt="image" src="https://github.com/user-attachments/assets/c006af26-b684-4908-89bf-789ba9b9d0bb" />
<img width="980" height="113" alt="image" src="https://github.com/user-attachments/assets/65f581aa-f1a4-4173-974f-f89e9b63cd35" />
<img width="1614" height="142" alt="image" src="https://github.com/user-attachments/assets/e7bcef35-c93d-49cb-8245-35b7dd70395f" />





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
