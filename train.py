from cv2 import *
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score as acc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs
import joblib
data=pd.read_csv('mydata2.csv')
x=data.iloc[:,1:]
y=data.iloc[:,0]
#pca=PCA(0.97).fit(x)
#px=pca.transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=51)
rf=rfc(n_estimators=250,max_depth=None)
rf.fit(x,y)
ypred=rf.predict(xtest)
st=acc(ytest,ypred)*100
print('Accuracy : %.2f'%st)
rf.fit(x,y)
joblib.dump(rf,'mymodel5.pkl')
