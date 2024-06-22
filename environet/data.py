import numpy as np
import os
os.system("pip install -q pandas==1.3.5")
import pandas as pd

class cfg:
    dir_0 = "./meteo_1317.xlsx"
    dir_1 = "./poll_1013.xlsx"
    dir_2 = "./poll_1416.xlsx"
    station = "Στ. ΕΓΝΑΤΙΑΣ"

df0 = pd.read_excel(cfg.dir_0,sheet_name=None,skiprows=6,header=0)
df0 = df0["DATA"].iloc[1:].reset_index(drop=True)
df0.Date = pd.to_datetime(df0.Date)
df0 = df0.drop(columns=["Time"]).groupby(df0.Date).mean().reset_index(drop=True)
df0.set_axis(
              ["date","wsv","wdv","tout","rhout"],
              axis=1,
              inplace=True,
             )

df1 = pd.read_excel(cfg.dir_1,sheet_name=None,skiprows=0,header=0)
df1 = df1[cfg.station]
df1.drop(columns = df1.columns[[0,1,3,11]].values,inplace=True)
df1.set_axis(
              ["date","so2","pm10","pm25","co","no","no2","o3","temp","rh"],
              axis=1,
              inplace=True,
             )

df2 = pd.read_excel(cfg.dir_2,sheet_name=None,skiprows=0,header=0)
df2 = df2[cfg.station]
df2.drop(columns = df2.columns[[0,1,3,11]].values,inplace=True)
df2.set_axis(
              ["date","so2","pm10","pm25","co","no","no2","o3","temp","rh"],
              axis=1,
              inplace=True,
             )

df = pd.concat([
                df1[(df1.date>="2013-01-01")&(df1.date<="2013-12-31")],
                df2[(df2.date>="2014-01-01")&(df2.date<="2015-12-31")],
                ],
               axis=0,
              )

df = pd.merge(
               df,
               df0[(df0.date>="2013-01-01")&(df0.date<="2015-12-31")],
               how="outer",
               on="date",
              )

df.fillna(np.nan,inplace=True)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
for col in df.columns.values:
    if col != "date": df[col] = df[col].astype(np.float64)

df.to_csv("./data.csv",index=False)

hdf = pd.merge(
               df2[(df2.date>="2016-01-01")&(df2.date<="2016-12-31")],
               df0[(df0.date>="2016-01-01")&(df0.date<="2016-12-31")],
               how="outer",
               on="date",
              )

hdf.fillna(np.nan,inplace=True)
hdf.replace(r'^\s*$', np.nan, regex=True, inplace=True)
for col in df.columns.values:
    if col != "date": hdf[col] = hdf[col].astype(np.float64)

hdf.to_csv("./holdout.csv",index=False)