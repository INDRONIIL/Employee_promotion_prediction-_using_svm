import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'employee_promotion_prediction.csv')
print(df.head())

print(df.info())
print(df.shape)

df = df.drop(columns=['gender','education_level','marital_status','city_tier','department','employment_type'])
print(df.head())

X = df.drop('promoted',axis=1)
y = df['promoted']