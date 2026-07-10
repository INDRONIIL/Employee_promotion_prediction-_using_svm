import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv(r'employee_promotion_prediction.csv')
print(df.head())

print(df.info())
print(df.shape)

df = df.drop(columns=['gender','education_level','marital_status','city_tier','department','employment_type'])
print(df.head())

X = df.drop('promoted',axis=1)
y = df['promoted']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf')
model.fit(X_train,X_test)

prediction = model.predict(X_test)

print("Predicted:",prediction)
print("Actual:",y_test)

print("Accuracy:",accuracy_score(y_test,prediction))
print("Confusion_matrix:",confusion_matrix(y_test,prediction))
print("Classification_report:",classification_report(y_test,prediction))

