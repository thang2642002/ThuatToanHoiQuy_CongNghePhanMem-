# Import thư viện
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu giả định
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình hồi quy Ridge
alpha = 1.0  # Giá trị của siêu tham số alpha
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = ridge_model.predict(X_test)

# Tính Mean Squared Error trên tập kiểm tra
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Hiển thị đồ thị hồi quy Ridge
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Dự đoán (Ridge)')
plt.title('Hồi quy Ridge')
plt.xlabel('Biến độc lập')
plt.ylabel('Biến phụ thuộc')
plt.legend()
plt.show()
