from python_files.data import data
mat_x_col_list = data.columns.drop(['Соотношение матрица-наполнитель'])
mat_col_list_norm = mat_x_col_list.drop(['Поверхностная плотность, г/м2', 'Угол нашивки, град'])
mat_col_list_std = mat_x_col_list.drop(mat_col_list_norm)

# from python_files.matrix.mat_col_list import mat_x_col_list, mat_col_list_norm, mat_col_list_std