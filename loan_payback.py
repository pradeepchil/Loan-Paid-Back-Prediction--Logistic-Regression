import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
pd.set_option('display.max_columns', None)

# print(train.head())
# print(test.head())

# print(f'Train Data:{train.isna().sum()}')
# print(f'Test Date :{test.isna().sum()}')
# print()
# print('Duplicates')
# print(f'Train Data:{train.duplicated().sum()}')
# print(f'Test Date :{test.duplicated().sum()}')

train_cat_columns = train.select_dtypes(include='object').columns

encoders = {}


for col in train_cat_columns:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))  # fit on both
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

# print(train.info())
# print(test.info())

# train_corr = train.corr()
# plt.figure(figsize=(16,8))
# sns.heatmap(train_corr, cmap='GnBu', annot=True, fmt='.2f')
# plt.title('Correlation Between The Features')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# print(train.columns)

red_cols = ['id', 'annual_income','loan_amount','gender', 'marital_status','education_level','loan_purpose']

train = train.drop(columns=red_cols,axis=1)
test = test.drop(columns=red_cols,axis=1)

# train_corr = train.corr()
# plt.figure(figsize=(16,8))
# sns.heatmap(train_corr, cmap='GnBu', annot=True, fmt='.2f')
# plt.title('Correlation Between The Features')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# print(train.columns)

X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
print(f'Accuracy Score :{accuracy*100:.2f}%')
print(f'F1 Score :{f1*100:.2f}%')
print(f'Precision Score :{precision*100:.2f}%')
print(f'Recall Score :{recall*100:.2f}%')

