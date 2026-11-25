from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from src.models import COLUMNS
root=Path(__file__).resolve().parent
with ZipFile(root/'data/source.zip') as z:
    name=next(n for n in z.namelist() if n.endswith('.xls'))
    df=pd.read_excel(BytesIO(z.read(name)),header=1,engine='xlrd')
assert df.shape==(30000,25)
assert df.columns[0]=='ID' and list(df.columns[1:24])==COLUMNS
assert set(df.iloc[:,-1])=={0,1}
df.columns=['ID']+COLUMNS+['default'];df.astype(int).to_csv(root/'data/credit.csv',index=False)
print(df.shape,df.default.value_counts().to_dict())
