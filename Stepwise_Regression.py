# Import thư viện
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
Y = np.array([10, 20, 25, 30, 35])

# Khởi tạo mô hình tuyến tính 
model = LinearRegression()

# Áp dụng hồi quy từng bước
selector = RFE(model, n_features_to_select=1) 
selector = selector.fit(X.T, Y)

# Dự đoán
Y_pred = selector.predict(X.T)


# Hiển thị biểu đồ
plt.scatter(X[0], Y, color='blue', label='Dữ liệu thực tế') 
plt.plot(X[0], Y_pred, color='red', label='Dự đoán', linewidth=2) 
plt.title('Hồi Quy Từng Bước: Doanh số bán hàng') 
plt.xlabel('Quảng cáo')
plt.ylabel('Doanh số') 
plt.legend()
plt.show()
