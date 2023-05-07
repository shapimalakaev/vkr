from python_files.data import data
from python_files.elasticity.ela_col_list import ela_x_col_list, ela_col_list_norm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data_ = data.copy()

# Тип вычисления верхней и нижней границы выбросов
from enum import Enum
class OutlierBoundaryType(Enum):
    SIGMA = 1
    QUANTILE = 2
    IQR = 3

# Функция вычисления верхней и нижней границы выбросов
def get_outlier_boundaries(df, col, outlier_boundary_type: OutlierBoundaryType):
    if outlier_boundary_type == OutlierBoundaryType.SIGMA:
        K1 = 3
        lower_boundary = df[col].mean() - (K1 * df[col].std())
        upper_boundary = df[col].mean() + (K1 * df[col].std())

    elif outlier_boundary_type == OutlierBoundaryType.QUANTILE:
        lower_boundary = df[col].quantile(0.025)
        upper_boundary = df[col].quantile(0.975)

    elif outlier_boundary_type == OutlierBoundaryType.IQR:
        K2 = 1.5
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        lower_boundary = df[col].quantile(0.25) - (K2 * IQR)
        upper_boundary = df[col].quantile(0.75) + (K2 * IQR)

    else:
        raise NameError('Unknown Outlier Boundary Type')
        
    return lower_boundary, upper_boundary    

# Разделим исходный датасет на обучающую и тестовую выборки и будем работать с выбросами только на обучающей выборке
y = data_.pop('Модуль упругости при растяжении, ГПа')
X = data_

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 7)

# соберем всю обучающую выборку в новую переменную df_train 
df_train = X_train.copy() 
df_train['Модуль упругости при растяжении, ГПа'] = y_train.copy()

# сначала обработаем выбросы в признаке со скошенным исходным распределением ('Поверхностная плотность, г/м2')
# Вычисление по межквартильному интервалу верхней и нижней границы в признаке со скошенным исходным распределением  
lower_boundary_, upper_boundary_ = get_outlier_boundaries(df_train, 'Поверхностная плотность, г/м2', OutlierBoundaryType.IQR)

# Флаги для удаления выбросов в признаке со скошенным исходным распределением 
outliers_ = np.where(df_train['Поверхностная плотность, г/м2'] > upper_boundary_, True, 
                     np.where(df_train['Поверхностная плотность, г/м2'] < lower_boundary_, True, False))

# Удаление выбросов на основе флага в признаке со скошенным исходным распределением
df_train_trimmed = df_train.loc[~(outliers_), ]

# удалим выбросы в признаках с исходным распределением близким к нормальному  
for col in ela_col_list_norm:

    # Вычисление по правилу "3 сигм" верхней и нижней границы 
    lower_boundary, upper_boundary = get_outlier_boundaries(df_train_trimmed, col, OutlierBoundaryType.SIGMA)
    
    # Флаги для удаления выбросов 
    outliers = np.where(df_train_trimmed[col] > upper_boundary, True, 
                        np.where(df_train_trimmed[col] < lower_boundary, True, False))
    
    # Удаление выбросов на основе флага
    df_train_trimmed_3S = df_train_trimmed.loc[~(outliers), ]
    
# выделим целевой признак в отдельную переменную после обработки выбросов
y_train_trimmed_3S = df_train_trimmed_3S.pop('Модуль упругости при растяжении, ГПа')
X_train_trimmed_3S = df_train_trimmed_3S

# формируем DataFrame на основе массива
y_train_trimmed_3S = pd.DataFrame(y_train_trimmed_3S, columns=['Модуль упругости при растяжении, ГПа'])

# удалим выбросы в признаках с исходным распределением близким к нормальному  
for col in ela_col_list_norm:

    # Вычисление по квантилям (2.5% и 97.5%) верхней и нижней границы 
    lower_boundary, upper_boundary = get_outlier_boundaries(df_train_trimmed, col, OutlierBoundaryType.QUANTILE)
    
    # Флаги для удаления выбросов 
    outliers = np.where(df_train_trimmed[col] > upper_boundary, True, 
                        np.where(df_train_trimmed[col] < lower_boundary, True, False))
    
    # Удаление выбросов на основе флага
    df_train_trimmed_QT = df_train_trimmed.loc[~(outliers), ]
    
# выделим целевой признак в отдельную переменную после обработки выбросов
y_train_trimmed_QT = df_train_trimmed_QT.pop('Модуль упругости при растяжении, ГПа')
X_train_trimmed_QT = df_train_trimmed_QT

# формируем DataFrame на основе массива
y_train_trimmed_QT = pd.DataFrame(y_train_trimmed_QT, columns=['Модуль упругости при растяжении, ГПа'])

df_train_sub = df_train.copy()     
# Вычисление по межквартильному интервалу верхней и нижней границы в признаке со скошенным исходным распределением
lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub, 'Поверхностная плотность, г/м2', OutlierBoundaryType.IQR)
# Изменение данных
df_train_sub['Поверхностная плотность, г/м2'] = np.where(df_train_sub['Поверхностная плотность, г/м2'] > upper_boundary, upper_boundary, 
                np.where(df_train_sub['Поверхностная плотность, г/м2'] < lower_boundary, lower_boundary, df_train_sub['Поверхностная плотность, г/м2']))

# Замена выбросов граничными значениями в признаках с исходным распределением близким к нормальному  
for col in ela_col_list_norm:    
    # Вычисление верхней и нижней границы
    lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub, col, OutlierBoundaryType.SIGMA)
    # Изменение данных
    df_train_sub[col] = np.where(df_train_sub[col] > upper_boundary, upper_boundary, 
                        np.where(df_train_sub[col] < lower_boundary, lower_boundary, df_train_sub[col]))
# cохраним результат в новую переменную
df_train_sub_bound_3S = df_train_sub

# выделим целевой признак в отдельную переменную после обработки выбросов
y_train_sub_bound_3S = df_train_sub_bound_3S.pop('Модуль упругости при растяжении, ГПа')
X_train_sub_bound_3S = df_train_sub_bound_3S

# формируем DataFrame на основе массива
y_train_sub_bound_3S = pd.DataFrame(y_train_sub_bound_3S, columns=['Модуль упругости при растяжении, ГПа'])

# повторяем обработку выбросов для признака 'Поверхностная плотность, г/м2' сохранив результат в нлвую переменную 'df_train_sub_'
df_train_sub_ = df_train.copy()     
# Вычисление по межквартильному интервалу верхней и нижней границы в признаке со скошенным исходным распределением
lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub_, 'Поверхностная плотность, г/м2', OutlierBoundaryType.IQR)
# Изменение данных
df_train_sub_['Поверхностная плотность, г/м2'] = np.where(df_train_sub_['Поверхностная плотность, г/м2'] > upper_boundary, upper_boundary, 
            np.where(df_train_sub_['Поверхностная плотность, г/м2'] < lower_boundary, lower_boundary, df_train_sub_['Поверхностная плотность, г/м2']))

# Замена выбросов граничными значениями в признаках с исходным распределением близким к нормальному  
for col in ela_col_list_norm:    
    # Вычисление верхней и нижней границы
    lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub_, col, OutlierBoundaryType.QUANTILE)
    # Изменение данных
    df_train_sub_[col] = np.where(df_train_sub_[col] > upper_boundary, upper_boundary, 
                        np.where(df_train_sub_[col] < lower_boundary, lower_boundary, df_train_sub_[col]))
# cохраним результат в новую переменную
df_train_sub_bound_QT = df_train_sub_

# выделим целевой признак в отдельную переменную после обработки выбросов
y_train_sub_bound_QT = df_train_sub_bound_QT.pop('Модуль упругости при растяжении, ГПа')
X_train_sub_bound_QT = df_train_sub_bound_QT

# формируем DataFrame на основе массива
y_train_sub_bound_QT = pd.DataFrame(y_train_sub_bound_QT, columns=['Модуль упругости при растяжении, ГПа'])

# В признаке со скошенным исходным распределением заменим выбросы по моде
df_train_sub_med = df_train.copy()     
# Вычисление по межквартильному интервалу верхней и нижней границы в признаке со скошенным исходным распределением
lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub_med, 'Поверхностная плотность, г/м2', OutlierBoundaryType.IQR)
# Изменение данных
df_train_sub_med['Поверхностная плотность, г/м2'] = np.where(df_train_sub_med['Поверхностная плотность, г/м2'] > upper_boundary,
                                                             df_train_sub_med['Поверхностная плотность, г/м2'].mode(), 
        np.where(df_train_sub_med['Поверхностная плотность, г/м2'] < lower_boundary, df_train_sub_med['Поверхностная плотность, г/м2'].mode(),
                    df_train_sub_med['Поверхностная плотность, г/м2']))

# Замена выбросов граничными значениями в признаках с исходным распределением близким к нормальному  
for col in ela_col_list_norm:    
    # Вычисление верхней и нижней границы
    lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub_med, col, OutlierBoundaryType.SIGMA)
    # Изменение данных
    df_train_sub_med[col] = np.where(df_train_sub_med[col] > upper_boundary, df_train_sub_med[col].median(), 
                        np.where(df_train_sub_med[col] < lower_boundary, df_train_sub_med[col].median(), df_train_sub_med[col]))
# cохраним результат в новую переменную
df_train_sub_med_3S = df_train_sub_med

# выделим целевой признак в отдельную переменную после обработки выбросов
y_train_sub_med_3S = df_train_sub_med_3S.pop('Модуль упругости при растяжении, ГПа')
X_train_sub_med_3S = df_train_sub_med_3S

# формируем DataFrame на основе массива
y_train_sub_med_3S = pd.DataFrame(y_train_sub_med_3S, columns=['Модуль упругости при растяжении, ГПа'])

# В признаке со скошенным исходным распределением заменим выбросы по моде
df_train_sub_med_ = df_train.copy()     
# Вычисление по межквартильному интервалу верхней и нижней границы в признаке со скошенным исходным распределением
lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub_med_, 'Поверхностная плотность, г/м2', OutlierBoundaryType.IQR)
# Изменение данных
df_train_sub_med_['Поверхностная плотность, г/м2'] = np.where(df_train_sub_med_['Поверхностная плотность, г/м2'] > upper_boundary,
                                                             df_train_sub_med_['Поверхностная плотность, г/м2'].mode(), 
        np.where(df_train_sub_med_['Поверхностная плотность, г/м2'] < lower_boundary, df_train_sub_med_['Поверхностная плотность, г/м2'].mode(),
                    df_train_sub_med_['Поверхностная плотность, г/м2']))

# Замена выбросов граничными значениями в признаках с исходным распределением близким к нормальному  
for col in ela_col_list_norm:    
    # Вычисление верхней и нижней границы
    lower_boundary, upper_boundary = get_outlier_boundaries(df_train_sub_med_, col, OutlierBoundaryType.QUANTILE)
    # Изменение данных
    df_train_sub_med_[col] = np.where(df_train_sub_med_[col] > upper_boundary, df_train_sub_med_[col].median(), 
                        np.where(df_train_sub_med_[col] < lower_boundary, df_train_sub_med_[col].median(), df_train_sub_med_[col]))
# cохраним результат в новую переменную
df_train_sub_med_QT = df_train_sub_med_

# выделим целевой признак в отдельную переменную после обработки выбросов
y_train_sub_med_QT = df_train_sub_med_QT.pop('Модуль упругости при растяжении, ГПа')
X_train_sub_med_QT = df_train_sub_med_QT

# формируем DataFrame на основе массива
y_train_sub_med_QT = pd.DataFrame(y_train_sub_med_QT, columns=['Модуль упругости при растяжении, ГПа'])

# from python_files.elasticity.ela_outliers import X_train_trimmed_3S, y_train_trimmed_3S
# from python_files.elasticity.ela_outliers import X_train_trimmed_QT, y_train_trimmed_QT
# from python_files.elasticity.ela_outliers import X_train_sub_bound_3S, y_train_sub_bound_3S
# from python_files.elasticity.ela_outliers import X_train_sub_bound_QT, y_train_sub_bound_QT
# from python_files.elasticity.ela_outliers import X_train_sub_med_3S, y_train_sub_med_3S
# from python_files.elasticity.ela_outliers import import X_train_sub_med_QT, y_train_sub_med_QT