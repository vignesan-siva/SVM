import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\classification\\Social_Network_Ads.csv")

x=data.iloc[:,[2,3]].values
y=data.iloc[:,[-1]].values

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#training the data of svm model

from sklearn.svm import SVC
classifier=SVC(C=10,kernel='rbf',gamma=0.2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

#find confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
acc=(sum(np.diag(cm))/len(y_test))
acc