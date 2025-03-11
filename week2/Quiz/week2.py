import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 모바일 데이터 csv 파일
file_path = "C:\\AI\\mobile.csv"  
df = pd.read_csv(file_path)

# 데이터프레임 확인
print(df.head())
print(df.columns)

# 특성과 라벨 분리 
X = df.iloc[:, :-1]  # 특성 데이터 
y = df.iloc[:, -1]   # 클래스 라벨 

# 훈련/테스트 데이터 분할 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정규화가 필요한 모델을 위한 데이터 변환
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 리스트 (정규화 필요 없음: DT, RF / 정규화 필요: SVM, LR)
models_no_scaling = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

models_with_scaling = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

# 정규화가 필요 없는 모델 학습 및 평가
for name, model in models_no_scaling.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.4f}")

# 정규화가 필요한 모델 학습 및 평가
for name, model in models_with_scaling.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.4f}")