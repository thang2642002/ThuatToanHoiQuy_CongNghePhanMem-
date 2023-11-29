# Import thư viện
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])
# Áp dụng đa thức bậc 2
poly_features = PolynomialFeatures(degree=2) 
X_poly = poly_features.fit_transform(X)

# Khởi tạo mô hình tuyến tính 
model = LinearRegression()

# Fit mô hình 
model.fit(X_poly, Y)

# Dự đoán
Y_pred = model.predict(X_poly)

# Hiển thị biểu đồ
plt.scatter(X, Y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, Y_pred, color='red', label='Dự đoán', linewidth=2)
plt.title('Hồi Quy Đa Thức: Chiều cao dựa trên độ tuổi') 
plt.xlabel('Độ tuổi')
plt.ylabel('Chiều cao') 
plt.legend()
plt.show()
