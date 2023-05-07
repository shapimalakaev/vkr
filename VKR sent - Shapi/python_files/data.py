import pandas as pd
bp = pd.read_excel('X_bp.xlsx', index_col = 0)
nup = pd.read_excel('X_nup.xlsx', index_col = 0)
data = bp.join(nup, how = 'inner')

# from python_files.data import data