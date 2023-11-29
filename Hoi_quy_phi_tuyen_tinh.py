# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Tạo dữ liệu mẫu
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Diện tích của căn nhà
y = 4 + 3 * X + np.random.randn(100, 1)  # Giá nhà, với mối quan hệ tuyến tính và nhiễu ngẫu nhiên

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán giá nhà trên tập kiểm thử
y_pred = model.predict(X_test)

# Trực quan hóa kết quả
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Dự đoán')
plt.xlabel('Diện tích căn nhà')
plt.ylabel('Giá nhà')
plt.legend()
plt.title('Hồi quy Phi tuyến tính đơn biến')
plt.show()
