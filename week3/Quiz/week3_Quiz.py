import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# car csv 파일
file_path = "C:\\AI\\car_evaluation.csv"  
df = pd.read_csv(file_path)

# 문자형 데이터를 숫자로 변환 (엔코딩)
encoder = LabelEncoder()
df_encoded = df.apply(encoder.fit_transform)

# 입력(X)과 라벨(y) 분리
X = df_encoded.iloc[:, :-1]  # 마지막 열 제외
y = df_encoded.iloc[:, -1]   # 마지막 열 (라벨)

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