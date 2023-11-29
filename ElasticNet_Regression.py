# Import thư viện
import numpy as np
from sklearn.linear_model import ElasticNet 
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Y = np.array([10, 20, 30])


# Khởi tạo mô hình ElasticNet với alpha=1 và l1_ratio=0.5
model = ElasticNet(alpha=1, l1_ratio=0.5)

# Fit mô hình 
model.fit(X, Y)

# Dự đoán
Y_pred = model.predict(X)


# Hiển thị kết quả print("Dự đoán:", Y_pred)

# Hiển thị biểu đồ
plt.scatter(X[:, 0], Y, color='blue', label='Dữ liệu thực tế') 
plt.plot(X[:, 0], Y_pred, color='red', label='Dự đoán', linewidth=2)
plt.title('Hồi Quy ElasticNet: Dự đoán giá nhà')
plt.xlabel('Biến độc lập 1') 
plt.ylabel('Biến phụ thuộc') 
plt.legend()
plt.show()

