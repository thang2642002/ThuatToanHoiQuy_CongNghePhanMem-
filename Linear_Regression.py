# Import thư viện
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu huấn luyện giả định
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Reshape X để có dạng ma trận 2D, vì scikit-learn mong đợi input có dạng (n_samples, n_features)
X = X.reshape(-1, 1)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình với dữ liệu
model.fit(X, y)

# Dự đoán giá trị mới
new_data = np.array([[6]])
prediction = model.predict(new_data)

# In trọng số (hệ số) và sai số
print("Hệ số (slope):", model.coef_)
print("Sai số (intercept):", model.intercept_)

# Hiển thị dữ liệu huấn luyện và đường hồi quy tuyến tính
plt.scatter(X, y, color='blue', label='Dữ liệu huấn luyện')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Đường hồi quy tuyến tính')
plt.scatter(new_data, prediction, color='green', marker='o', s=100, label='Dự đoán cho mới')

plt.title('Hồi quy Tuyến tính')
plt.xlabel('Biến độc lập')
plt.ylabel('Biến phụ thuộc')
plt.legend()
plt.show()
