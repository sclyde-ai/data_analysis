import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timezone, timedelta
from typing import List
import shutil
from pandas.testing import assert_frame_equal, assert_series_equal

import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import importlib.resources

import time

from . import ListAndStr, CashTime

def get_info_data(fetch_filepath):
    if os.path.exists(fetch_filepath):
        with open(fetch_filepath, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        return info_data
    else:
        print('the file path is incorrect')
        return None

def cash_check(cash_time, last_fetch_time):
    expiration_delta = timedelta(days=cash_time.cache_days, hours=cash_time.cache_hours, minutes=cash_time.cache_minutes)
    # ▼▼▼ キャッシュが有効な場合の処理 ▼▼▼
    if datetime.now(timezone.utc) - last_fetch_time < expiration_delta:
        print(f"\n--- データはキャッシュが有効です ---")
        return True
    else:
        return False

def get_cash_data(target_dir):
    # ディレクトリ内の全ファイルを走査
    try:
        for filename in os.listdir(target_dir):
            filepath = os.path.join(target_dir, filename)
            if filename == "_fetch_info.json":
                continue
            elif filename.endswith(".csv"):
                # CSVファイルをDataFrameとして読み込む
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                data.index = pd.to_datetime(data.index, errors='coerce')
                print(f"✅ [{filename}] をDataFrameとして読み込みました。")
            elif filename.endswith(".json") and filename != "_fetch_info.json":
                # JSONファイルを辞書として読み込む (タイムスタンプファイルは除く)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ [{filename}] を辞書として読み込みました。")
        return data
    except Exception as e:
        print(f"⚠️ [{filename}] の読み込み中にエラーが発生しました: {e}")
        return None

def get_yfinance_data(ticker, attribute, parameters=None, transpose=True):
    ticker = yf.Ticker(ticker)   
    time.sleep(1)
    attr = getattr(ticker, attribute)
    try:
        if callable(attr):
            if parameters:
                data = attr(**parameters)
            else:
                data = attr()
        else:
            data = attr
        return data
    except Exception as e:
        print(f"the error occurs when getting data from yfinance: {e}")
        return None

def save_data(data, target_dir, attribute, parameters=None):
    try:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if not parameters:
                filepath = os.path.join(target_dir, f"{attribute}.csv")
            else:
                name = ""
                for key in parameters:
                    name += "_"
                    name += parameters[key]
                    print(name)
                filepath = os.path.join(target_dir, f"{attribute}{name}.csv")
            data.to_csv(filepath, encoding='utf-8-sig')
            print(f"✅ [{attribute}] -> CSVファイルとして保存しました。")
        else:
            filepath = os.path.join(target_dir, f"{attribute}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, default=str)
            print(f"✅ [{attribute}] -> JSONファイルとして保存しました。")
        return True
    except Exception as e:
        print(f"⚠️ [{attribute}] の読み込み中にエラーが発生しました: {e}")
        return False
    
def make_archive(data, target_dir, last_fetch_time):
    archive_timestamp = last_fetch_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(f"\n--- 既存データをタイムスタンプ '{archive_timestamp}' を付けてアーカイブします ---")
    try:
        for item_name in os.listdir(target_dir):
            if item_name == "_fetch_info.json":
                continue
            source_item_path = os.path.join(target_dir, item_name)

            if os.path.isfile(source_item_path):
                if source_item_path.endswith('.csv'):
                    local_data = pd.read_csv(source_item_path, index_col=0)
                    local_data.index = pd.to_datetime(local_data.index, errors='coerce')
                    try:
                        if isinstance(data, pd.DataFrame):
                            assert_frame_equal(local_data, data)
                        if isinstance(data, pd.Series):
                            local_data = local_data[local_data.columns[0]]
                            assert_series_equal(local_data, data)
                        print("df1とdf2は一致します。")
                        continue
                    except AssertionError as e:
                        print(e)
                if source_item_path.endswith('.json'):
                    # JSONファイルを読み込む
                    with open(source_item_path, 'r', encoding='utf-8') as f:
                        local_data = json.load(f)
                    if local_data == data:
                        continue
                # ファイル名と拡張子を分割
                base, ext = os.path.splitext(item_name)
                # 新しいファイル名を生成 (例: history_2025-07-06_11-46-03.csv)
                archive_filename = f"{base}_{archive_timestamp}{ext}"
                archive_dir = os.path.join(target_dir, "archive")
                os.makedirs(archive_dir, exist_ok=True)
                archive_target_path = os.path.join(archive_dir, archive_filename)
                # ファイルを新しい名前でコピー
                shutil.copy2(source_item_path, archive_target_path)
        print(f"✅ アーカイブが完了しました。")
        return True
    except Exception as e:
        print(f"⚠️ アーカイブ中にエラーが発生しました: {e}")
        return False
    
def make_fetch_data(fetch_filepath):
    try:
        fetch_timestamp = datetime.now(timezone.utc).isoformat()
        print(f"データ取得日時 (UTC): {fetch_timestamp}")
        timestamp_data = {
            'fetch_time': fetch_timestamp,
            'timezone': 'UTC'
        }
        with open(fetch_filepath, 'w', encoding='utf-8') as f:
            json.dump(timestamp_data, f, indent=4, ensure_ascii=False)
        print(f"✅ [{'fetch_info': <25}] -> JSONファイルとして保存しました。")
        return True
    except Exception as e:
        print(f"⚠️ アーカイブ中にエラーが発生しました: {e}")
        return False 


class Stock():
    def __init__(self, tickers: ListAndStr, attributes: ListAndStr):
        self.tickers = tickers
        self.attributes = attributes
    
    def get_stock(
            self,
            ticker: str,
            cash_time: CashTime = CashTime(1, 0, 0),
            parameters: dict = None,
        ):
        """
        yfinanceからデータを取得し、指定されたディレクトリに保存する。

        Args:
            ticker (str): 調査する銘柄のティッカーシンボル。
            target_directory (str): 保存先の親ディレクトリ名。
            cash_time.cache_days (int): キャッシュの有効期間（日）。
            cash_time.cache_hours (int): キャッシュの有効期間（時間）。
            cash_time.cache_minutes (int): キャッシュの有効期間（分）。
        """
        # --- 1. パスの定義 ---
        # 最新データはティッカー名のディレクトリ直下に置く
        data_directory = importlib.resources.files('financial_package')
        data_directory = os.path.join(data_directory, 'data')
        target_dir = os.path.join(data_directory, ticker)
        target_dir = os.path.join(target_dir, 'history')
        if parameters:
            for key in parameters:
                target_dir += '_'
                target_dir += parameters[key]
        os.makedirs(target_dir, exist_ok=True)

        fetch_filepath = os.path.join(target_dir, "_fetch_info.json")
        if not os.path.exists(fetch_filepath):
            print("there is not fetch data")
        else:        
            print(fetch_filepath)
            info_data = get_info_data(fetch_filepath)
            last_fetch_time = datetime.fromisoformat(info_data['fetch_time'])
            
            # --- 2. キャッシュの確認 ---
            cash_check_boolean = cash_check(cash_time, last_fetch_time)
            if cash_check_boolean:
                data = get_cash_data(target_dir)
                # print(data)
                return data

        # --- 2. データ取得処理 ---
        print(f"\n--- {ticker} のデータをAPIから取得します ---")
        data = get_yfinance_data(ticker, 'history')
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            return None
        error = save_data(data, target_dir, 'history', parameters)
        if not error:
            return None
        
        # --- 3. アーカイブ処理 ---
        # 新規取得の前に、既存の最新ファイルをファイル名に日付を付けてアーカイブする
        error = make_archive(data, target_dir, last_fetch_time)
        if not error:
            return None
        
        # --- 3. タイムスタンプの保存 ---
        error = make_fetch_data(fetch_filepath)
        if not error:
            return None
        print(f"\n--- 処理が正常に完了しました。データは '{target_dir}' に保存されています。 ---")
        return data
    
    def get_all_stock(self, cash_time: CashTime = CashTime(1, 0, 0)):
        ticker_dict = {}
        for ticker in self.tickers:
            series = self.get_stock(ticker, cash_time=cash_time)
            ticker_dict[ticker] = series
        return ticker_dict