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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
from statsmodels.tsa.api import VAR
import time
from statsmodels.tsa.stattools import adfuller
from . import ListAndStr, CashTime

def get_info_data(fetch_filepath):
    if os.path.exists(fetch_filepath):
        with open(fetch_filepath, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        return info_data
    else:
        print('Invalid file path or file does not exist')
        return None

def cash_check(cash_time, last_fetch_time):
    expiration_delta = timedelta(days=cash_time.cache_days, 
                               hours=cash_time.cache_hours, 
                               minutes=cash_time.cache_minutes)
    print('Cache expiration period:', expiration_delta)
    if datetime.now(timezone.utc) - last_fetch_time < expiration_delta:
        print("\n--- Using cached data (still valid) ---")
        return True
    return False

def get_cash_data(target_dir):
    try:
        for filename in os.listdir(target_dir):
            filepath = os.path.join(target_dir, filename)
            if filename == "_fetch_info.json":
                continue
            elif filename.endswith(".csv"):
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                print(f"✅ Loaded [{filename}] as DataFrame")
            elif filename.endswith(".json") and filename != "_fetch_info.json":
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded [{filename}] as dictionary")
        return data
    except Exception as e:
        print(f"⚠️ Error loading [{filename}]: {e}")
        return None

def get_yfinance_data(ticker, attribute, parameters=None):
    ticker = yf.Ticker(ticker)
    time.sleep(1)
    attr = getattr(ticker, attribute)
    try:
        if callable(attr):
            return attr(**parameters) if parameters else attr()
        return attr
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")
        return None

def save_data(data, target_dir, attribute, parameters=None):
    try:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            filename = f"{attribute}.csv" if not parameters else \
                      f"{attribute}_{'_'.join(str(v) for v in parameters.values())}.csv"
            filepath = os.path.join(target_dir, filename)
            data.to_csv(filepath, encoding='utf-8-sig')
            print(f"✅ Saved [{attribute}] as CSV file")
        else:
            filepath = os.path.join(target_dir, f"{attribute}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, default=str)
            print(f"✅ Saved [{attribute}] as JSON file")
        return True
    except Exception as e:
        print(f"⚠️ Error saving [{attribute}]: {e}")
        return False

def make_archive(data, target_dir, last_fetch_time):
    archive_timestamp = last_fetch_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(f"\n--- Archiving existing data with timestamp: {archive_timestamp} ---")
    try:
        for item_name in os.listdir(target_dir):
            if item_name == "_fetch_info.json":
                continue
                
            source_path = os.path.join(target_dir, item_name)
            if not os.path.isfile(source_path):
                continue

            if source_path.endswith('.csv'):
                local_data = pd.read_csv(source_path, index_col=0)
                check_data = data.copy()
                check_data.index = check_data.index.astype(str)
                if isinstance(check_data, pd.DataFrame):
                    check_data.columns = check_data.columns.astype(str)
                try:
                    if isinstance(check_data, pd.DataFrame):
                        assert_frame_equal(local_data, check_data)
                    elif isinstance(check_data, pd.Series):
                        local_data = local_data[local_data.columns[0]]
                        assert_series_equal(local_data, check_data)
                    print("Data matches - skipping archive")
                    continue
                except AssertionError:
                    pass

            elif source_path.endswith('.json'):
                with open(source_path, 'r', encoding='utf-8') as f:
                    if json.load(f) == data:
                        continue

            base, ext = os.path.splitext(item_name)
            archive_dir = os.path.join(target_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            shutil.copy2(source_path, os.path.join(archive_dir, f"{base}_{archive_timestamp}{ext}"))
            
        print("✅ Archive completed successfully")
        return True
    except Exception as e:
        print(f"⚠️ Archive failed: {e}")
        return False

def make_fetch_data(fetch_filepath):
    try:
        fetch_timestamp = datetime.now(timezone.utc).isoformat()
        print(f"Data fetched at (UTC): {fetch_timestamp}")
        with open(fetch_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'fetch_time': fetch_timestamp,
                'timezone': 'UTC'
            }, f, indent=4, ensure_ascii=False)
        print("✅ Saved fetch timestamp")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save fetch timestamp: {e}")
        return False

class Tickers():
    def __init__(self, tickers: ListAndStr, attributes: ListAndStr=None, log=False):
        self.tickers = tickers
        self.attributes = attributes

    def get_data(
            self,
            ticker: str,
            attribute: str,
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
        target_dir = os.path.join(target_dir, attribute)
        if parameters:
            for key in parameters:
                target_dir += '_'
                target_dir += parameters[key]
        os.makedirs(target_dir, exist_ok=True)

        fetch_filepath = os.path.join(target_dir, "_fetch_info.json")
        fetch_file_exist = os.path.exists(fetch_filepath)
        if not fetch_file_exist:
            print("there is not fetch data")
        else:
            print(fetch_filepath)
            info_data = get_info_data(fetch_filepath)
            last_fetch_time = datetime.fromisoformat(info_data['fetch_time'])
            
            # --- 2. キャッシュの確認 ---
            cash_check_boolean = cash_check(cash_time, last_fetch_time)
            if cash_check_boolean:
                data = get_cash_data(target_dir)
                if isinstance(data, (pd.DataFrame, pd.Series, dict)):
                    return data

        # --- 2. データ取得処理 ---
        print(f"\n--- {ticker} のデータをAPIから取得します ---")
        data = get_yfinance_data(ticker, attribute, parameters=parameters)
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            return None
        error = save_data(data, target_dir, attribute, parameters)
        if not error:
            return None
        
        if fetch_file_exist:
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

    def get_all_data(self, cash_time: CashTime = CashTime(1, 0, 0)):
        ticker_dict = {}
        for ticker in self.tickers:
            attribute_dict = {}
            for attribute in self.attributes:
                series = self.get_data(ticker, attribute, cash_time=cash_time)
                attribute_dict[attribute] = series
            ticker_dict[ticker] = attribute_dict
        return ticker_dict

    def get_attribute(self, attribute, cash_time: CashTime = CashTime(1, 0, 0), parameters=None):
        ticker_dict = {}
        for ticker in self.tickers:
            series = self.get_data(ticker, attribute, cash_time=cash_time , parameters=parameters)
            ticker_dict[ticker] = series
        return ticker_dict

    def get_ticker(self, ticker, cash_time: CashTime = CashTime(1, 0, 0)):
        attribute_dict = {}
        for attribute in self.attributes:
            series = self.get_data(ticker, attribute, cash_time=cash_time)
            attribute_dict[attribute] = series
        return attribute_dict

class Stock(Tickers):
    def __init__(self, tickers, period='max', interval='1d'):
        super().__init__(tickers)
        self.tickers = tickers
        self.history = self.get_attribute('history', parameters={'period': period, 'interval': interval})
        self.dividends = self.get_attribute('dividends')
        self.splits = self.get_attribute('splits')
        self.data = self.history
        self.close = self.close()
        self.open = self.open()
        self.high = self.high()
        self.low = self.low()
        self.prices = self.close
    
    def log(self):
        for ticker in self.tickers:
            self.data[ticker] = self.data[ticker].apply(np.log)
        return self
    def diff(self):
        for ticker in self.tickers:
            self.data[ticker] = self.data[ticker].apply(np.diff)
        return self
    def log_diff(self):
        for ticker in self.tickers:
            self.data[ticker] = self.data[ticker].apply(np.log).apply(np.diff)
        return self

    def close(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['Close']
            series.name = ticker 
            df = df.join(series, how='outer')
        return df
    def open(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['Open']
            series.name = ticker 
            df = df.join(series, how='outer')
        return df
    def high(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['High']
            series.name = ticker 
            df = df.join(series, how='outer')
        return df
    def low(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['Low']
            series.name = ticker 
            df = df.join(series, how='outer')
        return df
            
    def avg(self):
        return self.prices.mean().item()
    def std(self):
        return self.prices.std().item()
    def maxlen(self):
        maxlen = 0
        for ticker in self.tickers:
            if maxlen < len(self.history[ticker]):
                maxlen = len(self.history[ticker])
        return maxlen

    
    def plot(self, tickers: ListAndStr=None):
        if tickers == None:
            tickers = self.tickers
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue

            plt.figure(figsize=(12, 6))
            
            plt.plot(self.close[ticker], label='Close Price', color='blue', linewidth=2)
            plt.plot(self.high[ticker], label='High Price', color='red', linestyle='--')
            plt.plot(self.low[ticker], label='Low Price', color='green', linestyle='--')
            plt.plot(self.open[ticker], label='Open Price', color='orange', linestyle=':')
            
            plt.title(f'{ticker} Stock Price (Open, High, Low, Close)')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            plt.show()
            plt.close()

    def AuotCorrelation(self, lag: int=None, tickers: list=None, save_path=None) -> pd.DataFrame:
        if tickers == None:
            tickers = self.tickers
        
        maxlen = self.maxlen()
        if lag == None or maxlen < lag:
            lag = maxlen-1
        
        autocorr_df = pd.DataFrame()
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue

            autocorr = self.prices[ticker].pct_change().dropna()
            autocorr.name = ticker

            acf_values = acf(autocorr, nlags=lag, fft=False)
            series = pd.Series(acf_values, name=ticker, index=autocorr.index[-lag-1:])
            autocorr_df = autocorr_df.join(series, how='outer')

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                autocorr_df.to_csv(f"{save_path}/autocorrelation{ticker}.csv")
        return autocorr_df
    
    def adf_df(self):
        adf_df = pd.DataFrame()
        for ticker in self.tickers:
            adf_series = pd.Series()
            adf_stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(self.prices[ticker])
            adf_series['adf_stat'] = adf_stat
            adf_series['p_value'] = p_value
            adf_series['used_lag'] = used_lag
            adf_series['n_obs'] = n_obs
            adf_series['critical_value_1%'] = critical_values['1%']
            adf_series['critical_value_5%'] = critical_values['5%']
            adf_series['critical_value_10%'] = critical_values['10%']
            adf_df[ticker] = adf_series
        return adf_df
    
    def Var(self, maxlags: int=None) -> dict:
        maxlen = len(self.prices)
        if maxlags == None or maxlen < maxlags:
            maxlags = maxlen-1
        model = VAR(self.prices)
        results = model.fit(maxlags=maxlags)
        var_dict = {}
        var_dict['params'] = results.params
        var_dict['tvalues'] = results.tvalues
        var_dict['pvalues'] = results.pvalues
        var_dict['resid'] = results.resid
        var_dict['sigma_u'] = results.sigma_u
        criteria_series = pd.Series()
        criteria_series['aic'] = results.aic
        criteria_series['bic'] = results.bic
        criteria_series['hqic'] = results.hqic
        criteria_series['fpe'] = results.fpe
        criteria_series['llf'] = results.llf
        criteria_series['detomega'] = results.detomega
        var_dict['creteria'] = criteria_series
        var_dict['adf'] = self.adf_df
        return var_dict
    
    def VarCompare(self, maxlag: int=None):
        criteria_df = pd.DataFrame()
        if maxlag == None or maxlag > len(self.prices):
            maxlag = len(self.prices)
        for lag in range(maxlag):
            try:
                model = VAR(self.prices)
                results = model.fit(maxlags=lag+1)
                criteria_series = pd.Series()
                criteria_series.name = lag+1
                criteria_series['aic'] = results.aic
                criteria_series['bic'] = results.bic
                criteria_series['hqic'] = results.hqic
                criteria_series['fpe'] = results.fpe
                criteria_series['llf'] = results.llf
                criteria_series['detomega'] = results.detomega
                # criteria_df = criteria_df.join(criteria_series, how='outer')
                criteria_df = pd.concat([criteria_df, criteria_series.to_frame().T])
            except:
                break
        return criteria_df
    
    def CandleStick(self, day=200, moving_average: ListAndStr=[5, 10, 20 ,50, 75, 100], tickers: ListAndStr=None):
        if tickers == None:
            tickers = self.tickers
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue
            # prepare history
            candlestick_history = self.history[ticker].copy()
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
            plt.title(f'{ticker} candlestick')
            plt.xlabel('Date')
            plt.ylabel('price (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.close()

class Option(Stock):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.options = self.get_attribute('options')

class Holder(Stock):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.major_holders = self.get_attribute('major_holders')
        self.institutional_holders = self.get_attribute('institutional_holders')
        self.mutualfund_holders = self.get_attribute('mutualfund_holders')

class Finance(Tickers):
    def __init__(self, tickers, quarterly=False):
        super().__init__(tickers)
        self.balance_sheet = self.get_attribute('balance_sheet')    
        self.income_stmt = self.get_attribute('income_stmt')    
        self.cashflow = self.get_attribute('cashflow')
        if quarterly:
            self.balance_sheet = self.get_attribute('quarterly_balance_sheet')    
            self.income_stmt = self.get_attribute('quarterly_income_stmt')    
            self.cashflow = self.get_attribute('quarterly_cashflow')
        for ticker in self.tickers:
            self.balance_sheet[ticker] = self.balance_sheet[ticker].T
            self.income_stmt[ticker] = self.income_stmt[ticker].T
            self.cashflow[ticker] = self.cashflow[ticker].T

    def get_indices(self, ticker):
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

class Insider(Tickers):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.insider_purchases = self.get_attribute('insider_purchases')    
        self.insider_roster_holders = self.get_attribute('insider_roster_holders')    
        self.insider_transactions = self.get_attribute('insider_transactions')