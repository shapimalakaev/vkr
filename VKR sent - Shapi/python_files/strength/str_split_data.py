from python_files.data import data
data_ = data.copy()
from sklearn.model_selection import train_test_split 
y_str = data_.pop('Прочность при растяжении, МПа')
X_str = data_
X_train_str, X_test_str, y_train_str, y_test_str = train_test_split(X_str, y_str,
                                                                   test_size = 0.3,
                                                                   random_state = 7)  

#from python_files.strength.str_split_data import X_train_str, X_test_str, y_train_str, y_test_str