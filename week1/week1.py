import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
    
# 붓꽃 데이터 CSV 파일 읽기
file_path = "C:\\AI\\iris.csv"  # 본인이 iris.csv를 저장한 경로를 입력합니다.
df = pd.read_csv(file_path)

# 데이터프레임 확인
print(df.head())
print(df.columns)

# 특성과 라벨 분리 
X = df.iloc[:, :-1]  # 특성 데이터 
y = df.iloc[:, -1]   # 클래스 라벨 

# 훈련/테스트 데이터 분할 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화 (SVM과 로지스틱 회귀를 위해 필요)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 리스트
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

# 모델 학습 및 평가
for name, model in models.items():
    model.fit(X_train, y_train)  # 모델 학습
    y_pred = model.predict(X_test)  # 예측
    acc = accuracy_score(y_test, y_pred)  # 정확도 평가
    print(f"{name} 정확도: {acc:.4f}")