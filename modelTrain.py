import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import lightgbm as lgb
import joblib

df1 = pd.read_csv('data.csv')
X = df1.drop(columns=['Churn', 'location_code'], axis=1)
y = df1['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

clf = lgb.LGBMClassifier(random_state = 0, n_estimators=400, max_depth=8, verbose=-1)

clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

joblib.dump(clf, 'lgb.pkl')

print("Train score:", accuracy_score(y_train, train_pred), f1_score(y_train, train_pred), precision_score(y_train, train_pred), recall_score(y_train, train_pred))
print("Test score:", accuracy_score(y_test, test_pred), f1_score(y_test, test_pred), precision_score(y_test, test_pred), recall_score(y_test, test_pred))