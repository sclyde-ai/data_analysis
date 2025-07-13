import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List
import shutil
from pandas.testing import assert_frame_equal, assert_series_equal

import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import importlib.resources

import time

@dataclass
class List():
    elem: List[str]
    def __post_init__(self):
        if isinstance(self.items, list):
            return
        if isinstance(self.elem, str):
            self.elem=[self.elem]
        else:
            raise ValueError("error")

@dataclass
class HourInDay():
    hour: int
    def __post_init__(self):
        if not isinstance(self.hour, int):
            raise ValueError("type is incorrect")
        if self.hour >= 24:
            raise ValueError("hour is below 24")

@dataclass        
class MinuteInHour():
    minute: int
    def __post_init__(self):
        if not isinstance(self.minute, int):
            raise ValueError("type is incorrect")
        if self.minute >= 60:
            raise ValueError("hour is below 24")
class Company():
    def __init__(self, tickers: List, attributes: List):
        self.tickers = tickers
        self.attributes = attributes

    def get_data(
            self,
            ticker: str,
            attribute: str,
            cache_days: int = 1,
            cache_hours: HourInDay = 0,
            cache_minutes: MinuteInHour = 0,
            parameters: dict = None,
        ):
            """
            yfinanceからデータを取得し、指定されたディレクトリに保存する。

            Args:
                ticker (str): 調査する銘柄のティッカーシンボル。
                target_directory (str): 保存先の親ディレクトリ名。
                cache_days (int): キャッシュの有効期間（日）。
                cache_hours (int): キャッシュの有効期間（時間）。
                cache_minutes (int): キャッシュの有効期間（分）。
            """
            # --- 1. パスの定義 ---
            # 最新データはティッカー名のディレクトリ直下に置く
            data_directory = importlib.resources.files('financial_package')
            data_directory = os.path.join(data_directory, 'data')
            output_dir = os.path.join(data_directory, ticker)
            output_dir = os.path.join(output_dir, attribute)
            if parameters:
                for key in parameters:
                    output_dir += '_'
                    output_dir += parameters[key]
            archive_dir = os.path.join(output_dir, "archive")
            info_filepath = os.path.join(output_dir, "_fetch_info.json")
            
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(archive_dir, exist_ok=True)
            
            # --- 2. キャッシュの確認 ---
            expiration_delta = timedelta(days=cache_days, hours=cache_hours, minutes=cache_minutes)
            if os.path.exists(info_filepath):
                with open(info_filepath, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                last_fetch_time = datetime.fromisoformat(info_data['fetch_time'])
                
                # ▼▼▼ キャッシュが有効な場合の処理 ▼▼▼
                if datetime.now(timezone.utc) - last_fetch_time < expiration_delta:
                    print(f"\n--- {ticker} のデータはキャッシュが有効です ---")
                    print(f"ローカルディレクトリからデータを読み込みます: {output_dir}")

                    # ディレクトリ内の全ファイルを走査
                    for filename in os.listdir(output_dir):
                        filepath = os.path.join(output_dir, filename)
                        try:
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
                        except Exception as e:
                            print(f"⚠️ [{filename}] の読み込み中にエラーが発生しました: {e}")
                            return
                    return data

            # --- 2. データ取得処理 ---
            print(f"\n--- {ticker} のデータをAPIから取得します ---")
            
            fetch_timestamp = datetime.now(timezone.utc).isoformat()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"ディレクトリ '{output_dir}' を作成しました。")

            ticker = yf.Ticker(ticker)            
            try:
                attr = getattr(ticker, attribute)
                if callable(attr):
                    if parameters:
                        data = attr(**parameters)
                    else:
                        data = attr()
                    data = data.T
                    data.index = data.index.tz_localize(None)
                else:
                    data = attr
                
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    if not parameters:
                        filepath = os.path.join(output_dir, f"{attribute}.csv")
                    else:
                        name = ""
                        for key in parameters:
                            name += "_"
                            name += parameters[key]
                            print(name)
                        filepath = os.path.join(output_dir, f"{attribute}{name}.csv")
                    data.to_csv(filepath, encoding='utf-8-sig')
                    print(f"✅ [{attribute: <25}] -> CSVファイルとして保存しました。")
                else:
                    filepath = os.path.join(output_dir, f"{attribute}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
                    print(f"✅ [{attribute: <25}] -> JSONファイルとして保存しました。")
            except Exception as e:
                print(f"⚠️ [{attribute}] の読み込み中にエラーが発生しました: {e}")
                return
            
            # --- 3. アーカイブ処理 ---
            # 新規取得の前に、既存の最新ファイルをファイル名に日付を付けてアーカイブする
            if os.path.exists(info_filepath):
                archive_timestamp = last_fetch_time.strftime('%Y-%m-%d_%H-%M-%S')
                print(f"\n--- 既存データをタイムスタンプ '{archive_timestamp}' を付けてアーカイブします ---")
                
                try:
                    for item_name in os.listdir(output_dir):
                        if item_name == "_fetch_info.json":
                            continue
                        source_item_path = os.path.join(output_dir, item_name)

                        if os.path.isfile(source_item_path):
                            if source_item_path.endswith('.csv'):
                                local_data = pd.read_csv(source_item_path, index_col=0)
                                local_data.index = pd.to_datetime(local_data.index, errors='coerce')
                                # print(local_data.columns)
                                # print(local_data)
                                # print("type of data", type(data))
                                # print("type of local_data", type(local_data))
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
                            archive_target_path = os.path.join(archive_dir, archive_filename)
                            
                            # ファイルを新しい名前でコピー
                            shutil.copy2(source_item_path, archive_target_path)
                            
                    print(f"✅ アーカイブが完了しました。")
                except Exception as e:
                    print(f"⚠️ アーカイブ中にエラーが発生しました: {e}")
            
            # --- 3. タイムスタンプの保存 ---
            print(f"データ取得日時 (UTC): {fetch_timestamp}")
            timestamp_data = {
                'fetch_time': fetch_timestamp,
                'timezone': 'UTC'
            }
            with open(info_filepath, 'w', encoding='utf-8') as f:
                json.dump(timestamp_data, f, indent=4, ensure_ascii=False)
            print(f"✅ [{'fetch_info': <25}] -> JSONファイルとして保存しました。")

            print(f"\n--- 処理が完了しました。データは '{output_dir}' に保存されています。 ---")
            return data
    

    def get_all_data(self):
        ticker_dict = {}
        for ticker in self.tickers:
            attribute_dict = {}
            for attribute in self.attributes:
                series = self.get_data(ticker, attribute)
                attribute_dict[attribute] = series
                time.sleep(1)
            ticker_dict[ticker] = attribute_dict
        return ticker_dict

    def get_attribute(self, attribute, parameters=None):
        ticker_dict = {}
        for ticker in self.tickers:
            series = self.get_data(ticker, attribute)
            ticker_dict[ticker] = series
        return ticker_dict

    def get_ticker(self, ticker):
        attribute_dict = {}
        for attribute in self.attributes:
            series = self.get_data(ticker, attribute)
            attribute_dict[attribute] = series
        return attribute_dict

class Stock(Company):
    def __init__(self, ticker, interval='1d'):
        super().__init__(ticker)
        self.history = self.get_attribute('history', parameters={'interval': interval})
        self.dividends = self.get_attribute('dividends')
        self.splits = self.get_attribute('splits')

        # self.date = self.history.index
        # self.prices = self.history['Close']
    
    def close(self):
        self.prices = self.history['Close']
        return self
    def open(self):
        self.prices = self.history['Open']
        return self
    def high(self):
        self.prices = self.history['High']
        return self
    def low(self):
        self.prices = self.history['Low']
        return self
    
    def log(self):
        return  self.prices.apply(np.log)
    def diff(self):
        return self.prices.apply(np.diff)
    def log_diff(self):
        return self.prices.apply(np.log).apply(np.diff)
            
    def avg(self):
        return self.prices.mean().item()
    def std(self):
        return self.prices.std().item()
    
    def plot(self):
        # グラフの設定
        plt.figure(figsize=(12, 6))
        
        # 各データをプロット
        plt.plot(self.history['Close'], label='Close Price', color='blue', linewidth=2)
        plt.plot(self.history['High'], label='High Price', color='green', linestyle='--')
        plt.plot(self.history['Low'], label='Low Price', color='red', linestyle='--')
        plt.plot(self.history['Open'], label='Open Price', color='orange', linestyle=':')
        
        # グラフのタイトルとラベルを設定
        plt.title(f'{self.ticker} Stock Price (Open, High, Low, Close)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # グラフを表示
        plt.show()
    
    def candlestick(self, day=200, moving_average=[5, 10, 20 ,50, 75, 100]):
        # prepare history
        candlestick_history = self.history.copy()
        candlestick_history = candlestick_history.reset_index()
        candlestick_history['Date'] = candlestick_history['Date'].map(mdates.date2num)
        candlestick_history = candlestick_history[-day:]

        fig, ax = plt.subplots(figsize=(12, 6))
        # illustrate a candlestick
        candlestick_ohlc(ax, candlestick_history[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                        width=1, colorup='g', colordown='r')
        # add a moving average
        for ma in moving_average:
            candlestick_history[f'MA{ma}'] = candlestick_history['Close'].rolling(ma).mean()
            ax.plot(candlestick_history['Date'], candlestick_history[f'MA{ma}'], label=f'{ma} day moving average')

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.title(f'{self.ticker} candlestick')
        plt.xlabel('Date')
        plt.ylabel('price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

class Option(Stock):
    def __init__(self):
        self.options = self.get_attribute('options')

class Holder(Stock):
    def __init__(self):
        self.major_holders = self.get_attribute('major_holders')
        self.institutional_holders = self.get_attribute('institutional_holders')
        self.mutualfund_holders = self.get_attribute('mutualfund_holders')

class Finance(Company):
    def __init__(self, quarterly=False):
        self.balance_sheet = self.get_all_tickers('balance_sheet')    
        self.income_stmt = self.get_all_tickers('income_stmt')    
        self.cashflow = self.get_all_tickers('cashflow')
        if quarterly:
            self.balance_sheet = self.get_all_tickers('quarterly_balance_sheet')    
            self.income_stmt = self.get_all_tickers('quarterly_income_stmt')    
            self.cashflow = self.get_all_tickers('quarterly_cashflow')

    def get_indices(self, ticker):
        """
        財務指標を計算してDataFrameを返す関数
        :param BS_dict: 貸借対照表データの辞書
        :param PL_dict: 損益計算書データの辞書
        :param ticker: 企業ティッカー
        :return: 財務指標DataFrame
        """
        # 空のDataFrame作成
        indeces = pd.DataFrame()
        
        try:
            BS = self.balance_sheet[ticker]
            PL = self.income_stmt[ticker]
            
            # 安全にデータを取得するヘルパー関数
            def get_safe(df, col):
                return df[col] if col in df.columns else pd.Series(dtype='float64')
            
            # 必要なデータを取得
            total_revenue = get_safe(PL, 'Total Revenue')
            cost_of_revenue = get_safe(PL, 'Cost Of Revenue')
            net_income = get_safe(PL, 'Net Income')
            inventory = get_safe(BS, 'Inventory')
            stockholders_equity = get_safe(BS, 'Stockholders Equity')
            treasury_shares = get_safe(BS, 'Treasury Shares Number')
            shares_issued = get_safe(BS, 'Share Issued')
            
            # 財務指標計算 (ゼロ除算を避けるためnp.where使用)
            indeces['Gross Profit'] = total_revenue - cost_of_revenue
            
            with np.errstate(divide='ignore', invalid='ignore'):
                indeces['Cost Of Revenue Ratio'] = np.where(total_revenue != 0, cost_of_revenue / total_revenue, np.nan)
                indeces['Inventory Turnover'] = np.where(inventory != 0, cost_of_revenue / inventory, np.nan)
                indeces['ROE'] = np.where(stockholders_equity != 0, net_income / stockholders_equity, np.nan)
                indeces['Treasury Stock Ratio'] = np.where(shares_issued != 0, treasury_shares / shares_issued, np.nan)
            
            # インデックスを設定
            if not indeces.empty:
                indeces.index = PL.index if not PL.empty else BS.index
                
        except KeyError as e:
            print(f"Error: {e} not found for ticker {ticker}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
        
        return indeces

class Insider(Company):
    def __init__(self):
        self.insider_purchases = self.get_attribute('insider_purchases')    
        self.insider_roster_holders = self.get_attribute('insider_roster_holders')    
        self.insider_transactions = self.get_attribute('insider_transactions')