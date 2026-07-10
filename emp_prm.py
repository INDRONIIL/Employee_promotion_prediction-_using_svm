import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

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
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Predicted:",prediction)
print("Actual:",y_test)

print("Accuracy:",accuracy_score(y_test,prediction))
print("Confusion_matrix:",confusion_matrix(y_test,prediction))
print("Classification_report:",classification_report(y_test,prediction))

df['promoted'].value_counts().plot(kind='bar')
plt.title("Promotion Distribution")
plt.xlabel("Promoted")
plt.ylabel("Count")
plt.show()

plt.scatter(df['age'],df['promoted'])
plt.title("Age vs Promotion")
plt.xlabel("Age")
plt.ylabel("Promoted")
plt.show()

plt.hist(df['salary'],bins=20)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Employees")
plt.show()

plt.scatter(df['salary'],df['promoted'])
plt.title("salary vs promotion")
plt.xlabel("Salary")
plt.ylabel("Promotion")
plt.show()

plt.scatter(df['performance_score'],df['promoted'])
plt.title("Performnce score vs promotion")
plt.xlabel("Performance score")
plt.ylabel("Promotion")
plt.show()

plt.scatter(df['years_in_current_role'],df['promoted'])
plt.title("years_in_current_role vs promotion")
plt.xlabel("years_in_current_role")
plt.ylabel("promotion")
plt.show()

df.groupby('promoted')['salary'].mean().plot(kind='bar')
plt.title("Average Salary by Promotion")
plt.ylabel("Average Salary")
plt.show()