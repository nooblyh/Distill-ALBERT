import pandas as pd
fs = ['train', 'dev', 'test']
for i in fs:
    csv_table=pd.read_table(i+'.tsv',sep='\t')
    csv_table.to_csv(i+'.csv',index=False)
