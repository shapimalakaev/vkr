from python_files.data import data
data_ = data.copy()
from sklearn.model_selection import train_test_split 
y_mat = data_.pop('Соотношение матрица-наполнитель')
X_mat = data_
X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(X_mat, y_mat,
                                                                   test_size = 0.3,
                                                                   random_state = 7)  

# from python_files.matrix.mat_split_data import X_train_mat, X_test_mat, y_train_mat, y_test_mat