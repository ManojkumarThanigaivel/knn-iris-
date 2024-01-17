import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
df=pd.read_csv(r"C:\Users\Manojkumar\Downloads\iris.csv",index_col=0)
df.replace({"setosa":0,"versicolor":1,"virginica":2},inplace=True)
s=StandardScaler()
s.fit(df.drop('Species',axis=1))
ss=s.transform(df.drop('Species',axis=1))
df_t=pd.DataFrame(ss,columns=df.columns[:-1])
a={}
for i in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_predict(knn,df_t,df['Species'],cv=10)
    a[i]=score.mean()
max_key = max(a, key=a.get)
print(max_key)
x_train,x_test,y_train,y_test=train_test_split(ss,df['Species'],test_size=0.30)
knn=KNeighborsClassifier(n_neighbors=max_key)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
