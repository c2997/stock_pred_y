# 株価データダウンロード用
# stock_dataフォルダに実行時の日付でフォルダを作成しその中にダウンロードするプログラム
# jquants api v2のラッパーライブラリを使用
# 参考  https://github.com/J-Quants/jquants-api-client-python?tab=readme-ov-file

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import date, datetime, timedelta
from dateutil import tz
import jquantsapi
import os
from dotenv import load_dotenv
import warnings
warnings.simplefilter('ignore')
load_dotenv()

# API kyeの取得
my_api_key = os.getenv("JQUANTS_API_KEY")
dir_path   = 'stock_data'

# 指定されたパスが存在しない場合にのみディレクトリを作成する
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# 今日の日付のフォルダを作成する
dir_path_today = os.path.join('stock_data', str(date.today()), 'price')
os.makedirs(dir_path_today, exist_ok=True)


# 各種ファイル名セット
# 上場銘柄一覧
filepath_stock_list     = os.path.join(dir_path_today, 'stock_list.csv.gz')
# 価格データ
filepath_stock_price    = os.path.join(dir_path_today, 'stock_price.csv.gz')
# 指数四本値 TOPIX500
filepath_stock_topix500 = os.path.join(dir_path_today, 'stock_topix500.csv.gz')

# APIクライアント初期化
cli = jquantsapi.ClientV2(api_key=my_api_key)
cli.MAX_WORKERS = 3


# J-Quants API から取得するデータの期間
# スタンダードプランを使用
HISTORICAL_DATA_YEARS = 10

start_dt          = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(HISTORICAL_DATA_YEARS*365))
end_dt            = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_dt_yyyymmdd = start_dt.strftime("%Y%m%d")
end_dt_yyyymmdd   = end_dt.strftime("%Y%m%d")

print(f'{start_dt_yyyymmdd}から{end_dt_yyyymmdd}までの記録を使用します。')

# 上場銘柄一覧
if os.path.exists(filepath_stock_list):
    print("すでに上場銘柄一覧は取得済み")
else:
    stock_list_load = cli.get_eq_master()
    stock_list_load.to_csv(filepath_stock_list, compression='gzip', index=False)
    print("上場銘柄一覧取得完了")

# 株価日足（範囲指定）
if os.path.exists(filepath_stock_price):
    print("すでに価格データは取得済み")
else:
    stock_price_load = cli.get_eq_bars_daily_range(start_dt=start_dt, end_dt=end_dt)
    stock_price_load.to_csv(filepath_stock_price, compression='gzip', index=False)
    print("価格データ取得完了")

# 指数四本値 topix500
if os.path.exists(filepath_stock_topix500):
    print("すでに指数四本値 topix500は取得済み")
else:
    stock_topix500_load = cli.get_idx_bars_daily(
            code='002C',
            from_yyyymmdd=start_dt_yyyymmdd,
            to_yyyymmdd=end_dt_yyyymmdd)
    stock_topix500_load.to_csv(filepath_stock_topix500, compression='gzip', index=False)
    print("topix500取得完了")
