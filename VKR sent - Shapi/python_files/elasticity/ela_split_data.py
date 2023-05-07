from python_files.data import data
data_ = data.copy()
from sklearn.model_selection import train_test_split 
y_ela = data_.pop('Модуль упругости при растяжении, ГПа')
X_ela = data_
X_train_ela, X_test_ela, y_train_ela, y_test_ela = train_test_split(X_ela, y_ela,
                                                                   test_size = 0.3,
                                                                   random_state = 7)  

# from python_files.elasticity.ela_split_data import X_train_ela, X_test_ela, y_train_ela, y_test_ela