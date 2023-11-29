# Import thư viện
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu giả định
np.random.seed(42)
hours_studied = np.random.normal(5, 1.5, 100)
exam_result = (hours_studied * 1.5 + np.random.normal(0, 2, 100)) > 7

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(hours_studied.reshape(-1, 1), exam_result, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình hồi quy logistic
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Tính độ chính xác và ma trận nhầm lẫn

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Hiển thị kết quả

print("Độ chính xác:", accuracy)
print("Ma trận nhầm lẫn:")
print(conf_matrix)

# Hiển thị đường quyết định của mô hình trên biểu đồ

plt.scatter(X_test, y_test, color='black', marker='o', label='Thực tế')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Dự đoán')
plt.plot(sorted(X_test), model.predict_proba(sorted(X_test.reshape(-1, 1)))[:, 1], color='blue', linewidth=3, label='Đường quyết định')
plt.title('Hồi quy Logistic - Dự đoán đỗ kỳ thi')
plt.xlabel('Số giờ ôn tập')
plt.ylabel('Dự đoán (Đỗ: 1, Trượt: 0)')
plt.legend()
plt.show()
