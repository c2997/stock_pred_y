import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import lightgbm as lgb
import warnings
from sklearn.preprocessing import KBinsDiscretizer
warnings.filterwarnings('ignore')
import jquantsapi
import shap

dir_path = 'stock_data'

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

filepath_stock_fin      = os.path.join(dir_path, 'stock_fin_load_20260102.csv.gz')
filepath_stock_price    = os.path.join(dir_path, 'stock_price_load_20260102.gz')
filepath_stock_list     = os.path.join(dir_path, 'stock_list_load_20260102.gz')
filepath_stock_indices  = os.path.join(dir_path, 'stock_indices_load_20260102.gz')

MAIL_ADDRESS = ""
PASSWORD     = ""
cli          = jquantsapi.Client(mail_address=MAIL_ADDRESS, password=PASSWORD)

HISTORICAL_DATA_YEARS = 10

start_dt          = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(HISTORICAL_DATA_YEARS*365))
end_dt            = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_dt_yyyymmdd = start_dt.strftime("%Y%m%d")
end_dt_yyyymmdd   = end_dt.strftime("%Y%m%d")

print( f'{start_dt_yyyymmdd}から{end_dt_yyyymmdd}までの記録を使用します。' )

if os.path.exists(filepath_stock_list):
    stock_list_load = pd.read_csv(filepath_stock_list, compression='gzip')
else:
    stock_list_load = cli.get_listed_info()
    stock_list_load.to_csv(filepath_stock_list, compression='gzip', index=False)

print(stock_list_load)

if os.path.exists(filepath_stock_price):
   stock_price_load = pd.read_csv(filepath_stock_price, compression='gzip')
else:
   stock_price_load = cli.get_price_range(start_dt=start_dt, end_dt=end_dt)
   stock_price_load.to_csv(filepath_stock_price, compression='gzip', index=False)

print(stock_price_load)




