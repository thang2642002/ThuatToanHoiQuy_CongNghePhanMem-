import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tạo dữ liệu mẫu
np.random.seed(0)
X = 5 * np.random.rand(100, 1)  # Diện tích của căn nhà
y = 3 * X + np.sin(X) + np.random.randn(100, 1)  # Giá nhà, với mối quan hệ phi tuyến tính và nhiễu ngẫu nhiên

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # Chuyển đổi thành mảng 2D

# Xây dựng mô hình SVR
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train_scaled)

# Dự đoán giá nhà trên tập kiểm thử
X_test_scaled = scaler_X.transform(X_test)
y_pred_scaled = svr_model.predict(X_test_scaled)

# Chuyển ngược lại để có thể so sánh với giá trị thực tế
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))  # Chuyển đổi thành mảng 2D


# Trực quan hóa kết quả
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.scatter(X_test, y_pred, color='blue', label='Dự đoán')
plt.xlabel('Diện tích căn nhà')
plt.ylabel('Giá nhà')
plt.legend()
plt.title('Support Vector Regression (SVR)')
plt.show()
