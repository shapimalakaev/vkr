from python_files.data import data
str_x_col_list = data.columns.drop(['Прочность при растяжении, МПа'])
str_col_list_norm = str_x_col_list.drop(['Поверхностная плотность, г/м2', 'Угол нашивки, град'])
str_col_list_std = str_x_col_list.drop(str_col_list_norm)

# from python_files.strength.str_col_list import str_x_col_list, str_col_list_norm, str_col_list_std