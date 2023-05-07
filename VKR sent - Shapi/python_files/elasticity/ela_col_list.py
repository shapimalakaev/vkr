from python_files.data import data
ela_x_col_list = data.columns.drop(['Модуль упругости при растяжении, ГПа'])
ela_col_list_norm = ela_x_col_list.drop(['Поверхностная плотность, г/м2', 'Угол нашивки, град'])
ela_col_list_std = ela_x_col_list.drop(ela_col_list_norm)

# from python_files.elasticity.ela_col_list import ela_x_col_list, ela_col_list_norm, ela_col_list_std