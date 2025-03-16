import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# 붓꽃 데이터 csv 파일
file_path = "C:\\AI\\iris.csv"  
df = pd.read_csv(file_path)

# 붓꽃 종류 확인 (그래프)
sns.countplot(data = df, x="Name")
plt.xlabel("iris")
plt.ylabel("Count")
plt.title("label")
plt.show()

# 붓꽃 종류 확인 (숫자)
print(df['Name'].value_counts())

# 결측치 확인
print(df.isnull().sum())

# 결측치 제거
df = df.dropna() 
print(df.isnull().sum())

# 특성과 라벨 분리 
X = df.drop(columns=['Name'])
y = df['Name']

# 훈련/테스트 데이터 분할 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정규화 (SVM, LR, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 리스트 (DT, RF는 정규화 X / SVM, LR, KNN은 정규화 O)
models_no_scaling = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

models_with_scaling = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# DT, RF 학습 및 평가 
for name, model in models_no_scaling.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.4f}")
    # 혼동 행렬 출력
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {name}:")
    print(cm)

# SVM, LR, KNN 학습 및 평가 (정규화 O)
for name, model in models_with_scaling.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.4f}")
    # 혼동 행렬 출력
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {name}:")
    print(cm)