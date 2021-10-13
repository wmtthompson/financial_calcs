'''
Created on May 24, 2020

@author: william
'''

import numpy as np
import pandas as pd
from scipy import interpolate
import math
import datetime
from datetime import timedelta
date1 = datetime.date.today()
date_earlier = date1+timedelta(days=-60)
date_earlier_iso = date_earlier.isoformat()
today_iso = date1.isoformat()
from trading_calcs.trading_session import AccountInfoSession
import itertools as it
import multiprocessing
from multiprocessing import Pool
from multiprocessing.managers import SyncManager
import json
import re

class Holding(object):
    def __init__(self):
        self.cost_basis = 0
        self.gain_loss = 0
        self.description = "yada"
        self.symbol = "yada"
        self.market_value = 0
        self.price = 0
        self.purchase_price = 0
        self.quantity = 0
        self.put_call = 0
        self.option = False
    
    def set_from_holding_xml(self,h_xml):
        self.cost_basis = float(h_xml.find('costbasis').text)
        self.gain_loss = float(h_xml.find('gainloss').text)
        self.description = h_xml[3].find('desc').text
        self.symbol = h_xml[3].find('sym').text 
        self.market_value = float(h_xml.find('marketvalue').text)
        self.price = float(h_xml.find('price').text)
        self.purchase_price = float(h_xml.find('purchaseprice').text)
        self.quantity = float(h_xml.find('qty').text)
        self.put_call = int(h_xml[3].find('putcall').text)
        sectype = h_xml[3].find('sectyp').text
        if sectype == 'OPT':
            self.option = True
        else:
            self.option = False
        
class Probabilities(object):
    def __init__(self,price_data):
        if not isinstance(price_data, pd.Series):
            raise TypeError()
        price_data.name = 'price'
        self.df = pd.DataFrame(price_data)
        self.df = self.df.groupby('price')
        s = self.df['price'].agg('count')
        self.df = pd.DataFrame(s)
        self.df = self.df.rename(columns={'price':'frequency'})
        self.df['pdf'] = self.df['frequency']/self.df['frequency'].sum()
        self.df['cdf'] = self.df['pdf'].cumsum()
        self.df.reset_index()
        self.update_cdf_function()
    
    def update_cdf_function(self):
        x = self.df.index
        y = self.df['cdf']
        self.cdf = interpolate.interp1d(x,y,bounds_error=False, fill_value=(0,1))
    
    def get_prob_less_than(self, x):
        try:
            return self.cdf([x])[0]
        except ValueError:
            return 0

class StockTrade(object):
    def __init__(self, **kwargs):
        self.buy_stop = 0
        self.sell_stop = 0
        self.target_price = 0
        self.sold_call = False
        self.date_bought = 0
        self.date_sold = 0
        self.bought_price = 0
        self.sold_price = 0
        self.symbol = 'unk'
        self.holding = None
        self.strategy = None
        for key,value in kwargs.items():
            if key == 'buy_stop':
                self.buy_stop = value
            elif key == 'sell_stop':
                self.sell_stop = value
            elif key == 'target_price':
                self.target_price = value
            elif key == 'symbol':
                self.symbol = value
        if self.target_price == 0:
            self.target_price = 1.01*self.buy_stop
        
    def calc_unrealized_gain_pct(self,current_price):
        dif1 = (current_price-self.bought_price)/self.bought_price
        return dif1
    
    def calc_realized_gain_pct(self):
        dif1 = (self.bought_price - self.sold_price)/self.bought_price
        return dif1
    
    def attach_holding(self, holdings_list):
        for h in holdings_list:
            if h.symbol == self.symbol and not h.option:
                self.holding = h
                break
    
    def get_price_status(self, current_price, use_strategy=False):
        if not isinstance(current_price, (int, float)):
            raise TypeError("Current Price must be of type int or float, you passed a {0}".format(type(current_price)))
        if current_price <= 0:
            raise ValueError("Heaven forbid you have a zero or negative price, you put price = {0}".format(current_price))
        if use_strategy and self.strategy != None:
            if self.holding:
                if self.strategy.check_price(current_price) == -1:
                    return "SELL"
                else:
                    return "WAIT"
            else:
                status = self.strategy.check_price(current_price)
                if status == 0:
                    return "WAIT"
                elif status == 1:
                    return "BUY"
                elif status == -1:
                    return "SELL"
                else:
                    return "UNKNOWN"
        
        elif self.holding:
            if current_price <= self.sell_stop:
                return "SELL"
            elif current_price >= self.target_price:
                return "TARGET_SELL"
            else:
                return "WAIT"
        elif current_price >= (1.00-0.004)*self.buy_stop:
            return "BUY"
        else:
            return "WAIT"
    
    def print_data(self, current_price):
        print("Symbol = ",self.symbol)
        print("Buy Stop = ",self.buy_stop)
        print("Current Price = ",current_price)
        print("Sell Stop = ",self.sell_stop)
        print("Target Price = ",self.target_price)
        try:
            print("ATR Buy Stop (exit) = ",self.strategy.df_stops['BSTOP'].iloc[-1])
            print("ATR Sell Stop = ",self.strategy.df_stops['SSTOP'].iloc[-1])
            print("RSI = ",self.strategy.s_rsi.iloc[-1])
            print("Strategy pct return = ", self.strategy.pcr)
        except (NameError, AttributeError):
            print("Strategy Not Setup")
    
    def get_proba_target_price_or_higher(self, price_data):
        prob = Probabilities(price_data)
        prob1 = prob.get_prob_less_than(self.target_price)
        prob2 = 1-prob1
        return prob2

class PriceAction(object):
    def __init__(self, trades_list, dataframe, account_session, settings_file='None'):
        if not isinstance(trades_list, list):
            raise TypeError("trades_list should be of type list, you put in a {0}".format(type(trades_list)))
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe should be of type pandas DataFrame, you passed a {0}".format(type(dataframe)))
        if not isinstance(account_session, AccountInfoSession):
            raise TypeError("account_session should be of type trading_session.AccountInfoSession, you passed a {0}".format(type(account_session)))
        if not isinstance(settings_file, str):
            raise TypeError("settings_file should be of type str, you passed a {0}".format(type(settings_file)))
        self.now = datetime.datetime.now()
        self.trades_list = trades_list
        self.holdings_list = account_session.get_holdings_list()
        self.df = dataframe
        for tr in trades_list:
            tr.attach_holding(self.holdings_list)
        self.alarm_event = None
        self.settings_file = settings_file
        self.settings_dict = None
        

    def set_alarm_event(self, event):
        self.alarm_event = event
    
    def update_trades(self):
        for tr in self.trades_list:
            self.calc_strategy(tr)
    
    def get_price_data(self):
        for tr in self.trades_list:
            print(" ")
            print("#####################################")
            try:
                current_price = self.df['Close'][tr.symbol].iloc[-1]
            except KeyError:
                print("DataFrame does not contain the stock symbol {0}".format(tr.symbol))
                return None
            status = tr.get_price_status(current_price, use_strategy=True)
            if status == None:
                print("Status is None for some reason. Trade object is = {0}".format(tr))
            if status == "SELL" or status == "BUY" or status == "TARGET_SELL":
                try:
                    self.alarm_event()
                except :
                    print("ALARM!!")
            print("Current status = ",status)
            tr.print_data(current_price)
    
    
    def calc_strategy(self, trade):
        if not isinstance(trade, StockTrade):
            raise TypeError("trade should be of type StockTrade, you passed a {0}".format(type(trade)))
        s_symbol = trade.symbol
        strt1 = Strategy(s_symbol, self.df, self.settings_file)
        if strt1.get_settings():
            strt1.calculate_strategy()
            trade.strategy = strt1
        else:
            trade.strategy = None

def gain_loss_calc(price):
    """
    Returns a DataFrame containing the Gain and Loss from a price time series.
    Usually price is closing price.
    """
    if not isinstance(price, pd.Series):
        raise TypeError()
    chg = price.diff()
    gain = np.where(chg > 0, chg, 0)
    gain = np.abs(gain)
    gain = pd.Series(gain,index=price.index)
    loss = np.where(chg <= 0, chg, 0)
    loss = np.abs(loss)
    loss = pd.Series(loss,index=price.index)
    df1 = pd.DataFrame(data={'Gain':gain,'Loss':loss})
    return df1

def rsi_calc(gain,loss,period = 10):
    """
    Returns a Series containing the Relative Strength Index for the last period days.
    Index of the first result will be period-1
    Also assumes that gain and loss are positive.
    """
    if period < 1:
        return ValueError("Period must be greater than or equal to 1.")
    period1 = period-1
    first_avg_gain = gain.iloc[0:period+1].sum()/period
    first_avg_loss = loss.iloc[0:period+1].sum()/period
    avg_gain_arry = np.zeros(gain.shape)
    avg_loss_arry = np.zeros(loss.shape)
    for i in range(period, len(avg_gain_arry)):
        if i == period:
            avg_gain_arry[i] = first_avg_gain
            avg_loss_arry[i] = first_avg_loss
        else:
            avg_gain_arry[i] = (avg_gain_arry[i-1]*period1+gain.iloc[i])/period
            avg_loss_arry[i] = (avg_loss_arry[i-1]*period1+loss.iloc[i])/period
            
    avg_gain = pd.Series(avg_gain_arry, name='Avg Gain',index=gain.index)
    avg_loss = pd.Series(avg_loss_arry, name='Avg Loss',index=loss.index)
    RS = avg_gain/avg_loss
    RSI = 100-(100/(1+RS.abs()))
    RSI.name = 'RSI'
    return RSI

def vwap_calc(price, volume, period=100):
    """
    Returns a pd.Series of the volume weighted average price
    """
    if not isinstance(price,pd.Series) or not isinstance(volume,pd.Series):
        raise TypeError()
    if not isinstance(period,int):
        raise TypeError()
    if not period > 1:
        raise ValueError()
    series_pv = price*volume
    series_pv_sum = series_pv.rolling(period).sum()
    series_vol_sum = volume.rolling(period).sum()
    series_mvwap = series_pv_sum/series_vol_sum
    series_mvwap.name = "MVWAP"
    return series_mvwap

def std_move_calc(price, implied_volatility, calendar_days=14):
    """
    Calculate the one-standard deviation move from the current price, the implied volatility and calendar days.
    """
    if not isinstance(price,(pd.Series,int,float)):
        raise TypeError()
    std_move1 = price * implied_volatility * np.sqrt(calendar_days/365.)
    return std_move1

def std_move_high_low_calc(price, implied_volatility, calendar_days=14):
    """
    Calculate the one standard deviation move and the high and low prices based on that move.
    """ 
    std_move = std_move_calc(price, implied_volatility, calendar_days)
    upper = price+std_move
    lower = price-std_move
    return (std_move, upper, lower)

def atr_calc(high, low, close, period=14):
    """
    Returns a Pandas Series containing the average true range for the given close, high and low price and period.
    """
    if not isinstance(high,pd.Series) or not isinstance(low,pd.Series) or not isinstance(close,pd.Series):
        raise TypeError()
    if not isinstance(period,int):
        raise TypeError()
    if not period > 1:
        raise ValueError()
    high_low = high-low
    habs = high-close.shift(1)
    habs = habs.abs()# shift yesterday's closing price to compare with today's high
    labs = low-close.shift(1)
    labs = labs.abs()
    val = np.where(high_low > habs, high_low, habs)
    val = np.where(labs > val, labs, val)
    tr = pd.Series(val, index=high_low.index)
    atr = tr.ewm(span=period).mean()
    atr.name = "ATR"
    return atr

def profit_loss_calc(cost_basis, option_price, quantity, strike_price):
    """ Profit and Loss Calculation for a Covered Call """
    max_profit = (strike_price - (cost_basis-option_price))*quantity
    break_even_price = cost_basis-option_price
    return (max_profit, break_even_price)

def calc_vwap_strategy(price, volume, period):
    if not isinstance(price,pd.Series) or not isinstance(volume,pd.Series):
        raise TypeError()
    if not isinstance(period,int):
        raise TypeError()
    if not period > 1:
        raise ValueError()
    current_total = 10*price.max()
    initial_total = 10*price.max()
    shares_owned = 0
    s_vwap = vwap_calc(price, volume, period)
    for i,p in enumerate(price):
        if p < s_vwap.iloc[i].item():
            current_total -= price.iloc[i].item()
            shares_owned += 1
        if p > (1.01)*s_vwap.iloc[i].item() and shares_owned > 0:
            current_total += shares_owned*price.iloc[i].item()
            shares_owned = 0
    
    if shares_owned > 0:
        current_total += shares_owned*price.iloc[-i].item()
        shares_owned = 0
    pct = (current_total-initial_total)/initial_total
    return pct

def calc_trailing_stops(high, low, close, atr1, period = 14, mult = 2.0):
    recent_low = low.rolling(period).min()
    recent_high = high.rolling(period).max()
    patr = recent_low[period].item() + mult*atr1.iloc[period].item()
    matr = recent_high[period].item() - mult*atr1.iloc[period].item()
    v_bstop = np.zeros(recent_low.shape)
    v_sstop = np.zeros(recent_high.shape)
    loc_low = recent_low[period]
    loc_high = recent_high[period]

    for i,el in enumerate(low):
        v_bstop[i] = patr
        if i >= period:
            if (el < loc_low) or (close.iloc[i].item() > patr):
                loc_low = el
                patr = loc_low + mult*atr1.iloc[i].item()

    for i,el in enumerate(high):
        v_sstop[i] = matr
        if i >= period:
            if (el > loc_high) or (close.iloc[i].item() < matr):
                loc_high = el
                matr = loc_high - mult*atr1.iloc[i].item()
    bstop = pd.Series(v_bstop, index = close.index)
    bstop.name = "BSTOP"
    sstop = pd.Series(v_sstop, index = close.index)
    sstop.name = "SSTOP"
    df3 = pd.concat([bstop, sstop], axis=1)
    return df3

def calc_stop_position(price, bstop, sstop):
    position = np.where(price > bstop, 1,0)
    position = np.where(price < sstop, -1, position)
    s_position = pd.Series(position, index=price.index, name='STOP_POS')
    return s_position

def calc_percent_return_from_strategy(price, position):
    if isinstance(position, pd.Series):
        position = position.values
    current_total = 10*price.max()
    initial_total = 10*price.max()
    shares_owned = 0
    for i,p in enumerate(position):
        if p == 1 and current_total > price.iloc[i].item():
            current_total -= price.iloc[i].item()
            shares_owned += 1
        if p == -1 and shares_owned > 0:
            current_total += shares_owned*price.iloc[i].item()
            shares_owned = 0

    if shares_owned > 0:
        current_total += shares_owned*price.iloc[-1].item()
        shares_owned = 0
    pct = (current_total-initial_total)/initial_total
    return pct

def calc_range_position(price_close, price_low, trailing_stops, rsi, rsi_high=70, rsi_low=50):
    if isinstance(price_close, (float, int)) and isinstance(price_low, (float,int)):
        if (rsi.iloc[-1] < rsi_low) and (price_close > trailing_stops['SSTOP'].iloc[-1]):
            return 1
        elif (rsi.iloc[-1] > rsi_high) or (price_low < trailing_stops['SSTOP'].iloc[-1]) or (price_close < trailing_stops['SSTOP'].iloc[-1]):
            return -1
        else:
            return 0
    pos = np.where((rsi < rsi_low) & (price_close > trailing_stops['SSTOP']), 1, 0)
    pos = np.where((rsi > rsi_high) | (price_low < trailing_stops['SSTOP']) | (price_close < trailing_stops['SSTOP']), -1, pos)
    s_pos = pd.Series(pos, index=price_close.index)
    s_pos.name = "Position"
    return s_pos

def iterate_strategy(*args):
    rsi_high = args[0][0]
    rsi_low = args[0][1]
    rsi_periods = args[0][2]
    stops_periods = args[0][3]
    stops_mult = args[0][4]
    symbol = args[0][5]
    df1 = args[0][6]
    sh_list = args[0][7]
    atr1 = args[0][8]
    df_stops = calc_trailing_stops(df1['High'][symbol],df1['Low'][symbol],df1['Close'][symbol],atr1,stops_periods, stops_mult)
    df_gainloss = gain_loss_calc(df1['Close'][symbol])
    s_rsi = rsi_calc(df_gainloss['Gain'],df_gainloss['Loss'],rsi_periods)
    s_rsi.fillna(method='backfill',inplace=True)
    s_pos5 = calc_range_position(df1['Close'][symbol], df1['Low'][symbol], df_stops, s_rsi,rsi_high, rsi_low)
    pcr = calc_percent_return_from_strategy(df1['Close'][symbol], s_pos5)
    itm = {"percent_return":[pcr], "rsi_high":[rsi_high], "rsi_low":[rsi_low], 
           "rsi_periods":rsi_periods, "stops_periods":stops_periods, "stops_mult":stops_mult}
    sh_list.append(itm)


class Strategy(object):
    def __init__(self, stock_symbol, data_frame, settings_file):
        self.stock_symbol = stock_symbol
        self.data_frame = data_frame
        self.data_frame_manager = multiprocessing.Manager()
        self.data_frame_manager.df = self.data_frame
        try:
            self.atr = atr_calc(self.data_frame['High'][stock_symbol], self.data_frame['Low'][stock_symbol],self.data_frame['Close'][stock_symbol],14)
        except KeyError:
            print("Dataframe does not contain symbol = {0}".format(stock_symbol))
        self.data_frame_manager.atr = self.atr
        self.settings_file = settings_file
        self.best_settings = None
    
    def get_settings(self):
        try:
            with open(self.settings_file, 'r') as fp:
                self.settings_dict = json.load(fp)
                self.best_settings = self.settings_dict[self.stock_symbol]
        except FileNotFoundError:
            print("Optimal settings file not found. File path was {0}".format(self.settings_file))
            return False
        except KeyError:
            print("Stock symbol {0} was not found in settings file".format(self.stock_symbol))
            return False
        else:
            if self.best_settings != None:
                return True
    
    def optimize_strategy(self):
        rsi_high_range = range(60,100, 10)
        rsi_low_range = range(10,60, 10)
        rsi_periods = range(12,32,2)
        stops_periods = range(8,32,2)
        stops_mult = range(1,4)
        manager = SyncManager()
        manager.start()
        shared_list = manager.list()
        p = it.product(rsi_high_range, rsi_low_range, rsi_periods, stops_periods, stops_mult, 
                       [self.stock_symbol], [self.data_frame_manager.df],[shared_list],[self.data_frame_manager.atr])
        lp = list(p)
        print("")
        print("Optimizing RSI Strategy with number of rows = {0}".format(len(lp)))
        print("")
        with Pool(4) as pp:
            pp.map(iterate_strategy, lp)
        df01 = pd.concat([pd.DataFrame.from_dict(itm) for itm in shared_list], ignore_index=True)
        df02 = df01.sort_values(by=['percent_return'], ascending=False)
        manager.shutdown()
        date1 = datetime.date.today()
        today_iso = date1.isoformat()
        strategy_best_settings = dict(df02.iloc[0])
        strategy_best_settings['date_processed'] = today_iso
        self.best_settings = strategy_best_settings
    
    def calculate_strategy(self):
        if self.best_settings == None:
            print("Settings not available. No strategy calculated.")
        else:
            stops_periods = int(self.best_settings['stops_periods'])
            stops_mult = int(self.best_settings['stops_mult'])
            rsi_periods = int(self.best_settings['rsi_periods'])
            rsi_high = int(self.best_settings['rsi_high'])
            rsi_low = int(self.best_settings['rsi_low'])
            self.df_stops = calc_trailing_stops(self.data_frame['High'][self.stock_symbol],self.data_frame['Low'][self.stock_symbol],
                                                self.data_frame['Close'][self.stock_symbol],self.atr,stops_periods, stops_mult)
            self.df_gainloss = gain_loss_calc(self.data_frame['Close'][self.stock_symbol])
            self.s_rsi = rsi_calc(self.df_gainloss['Gain'],self.df_gainloss['Loss'],rsi_periods)
            self.s_rsi.fillna(method='backfill',inplace=True)
            self.s_pos = calc_range_position(self.data_frame['Close'][self.stock_symbol], self.data_frame['Low'][self.stock_symbol], 
                                             self.df_stops, self.s_rsi, rsi_high, rsi_low)
            self.pcr = calc_percent_return_from_strategy(self.data_frame['Close'][self.stock_symbol], self.s_pos)
    
    def update_settings_file(self):
        try:
            with open(self.settings_file, 'r') as fp:
                settings_dict = json.load(fp)
        except FileNotFoundError:
            with open(self.settings_file, 'w') as fp:
                settings_dict = {self.stock_symbol: self.best_settings}
                json.dump(settings_dict, fp)
        else:
            with open(self.settings_file, 'w') as fp:
                settings_dict[self.stock_symbol] = self.best_settings
                json.dump(settings_dict, fp)
    
    def check_price(self, price):
        try:
            rsi_high = int(self.best_settings['rsi_high'])
            rsi_low = int(self.best_settings['rsi_low'])
            p = calc_range_position(price, price, self.df_stops, self.s_rsi, rsi_high, rsi_low)
        except (NameError, KeyError, AttributeError):
            print("No strategy apparently, make sure calculate_strategy has been run.")
            return None
        else:
            return p
    
    def plot_strategy(self):
        df2 = pd.concat([self.data_frame['Close'][self.stock_symbol], self.df_stops], axis=1)
        plt =df2.plot(figsize=(12, 9))
        return plt

class TradeItem(object):
    def __init__(self, symbol, qty, price, date):
        if qty <= 0:
            raise ValueError("Cannot initialize with zero or negative qty.")
        self.symbol = symbol
        self.quantity = qty
        self.avg_price = price
        self.purchase_date = date
        self.cost_basis = price*qty
        self.gain_loss_pc = 0
        self.gain_loss = 0
        self.sold_total = 0
        self.sold_date = None
    
    def __repr__(self):
        str1 = """Symbol = {}, Quantity = {}, Average Price = {}, 
        Purchase Date = {}, Cost Basis = {}, Gain Loss = {}, 
        Sold Total = {}""".format(self.symbol, self.quantity,
                                                  self.avg_price, self.purchase_date.isoformat(),
                                                  self.cost_basis, self.gain_loss,
                                                  self.sold_total)
        return str1
    
    def update(self, qty, price, date):
        """ Returning true means all shares sold.
        
        """
        if qty > 0:
            self.quantity += qty
            self.cost_basis += price*qty
            return False
        if qty < 0:
            if -1*qty > self.quantity:
                raise ValueError("Quantity sold {} is more than owned {}.".format(qty, self.quantity))
            else:
                self.quantity += qty
            if self.quantity == 0:
                self.sold_total += (-1*qty * price)
                self.gain_loss = self.sold_total - self.cost_basis
                self.gain_loss_pc = (self.sold_total/self.cost_basis)-1.0
                self.sold_date = date
                return True
            elif self.quantity > 0:
                self.sold_total += (-1*qty * price)
    
    def get_record(self):
        return (self.symbol, self.purchase_date, self.sold_date, self.gain_loss, self.gain_loss_pc)

def get_date_time(str1):
    r1 = re.compile('(\d\d\d\d)-(\d\d)-(\d\d)')
#     r1 = re.compile('^\d\d\d\d')
    m = r1.match(str1)
    m1 = r1.search(m[0])
    year = int(m1.groups(0)[0])
    month = int(m1.groups(0)[1])
    day = int(m1.groups(0)[2])
    return datetime.datetime(year, month, day)                

class TradeProcessor(object):
    def __init__(self):
        self.dict_of_trades = dict()
        self.record_of_trades = list()
        self.total_list = list()
    
    def process_xml(self, xml_tree):
        for h in xml_tree.iter('transaction'):
            if h.find('activity') != None:
                activity = h.find('activity').text
                if activity == "Trade" or activity == "Assigned":
                    self.update_trades(h)
    
    def update_trades(self, h_xml):
        date = get_date_time(h_xml.find('date').text)
        hh = h_xml.find('transaction')
        hhh = hh.find('security')
        sectype = hhh.find('sectyp').text
        symbol = h_xml.find('symbol').text
        price = float(hh.find('price').text)
        qty = int(hh.find('quantity').text)
        desc = h_xml.find('desc').text
        activity = h_xml.find("activity").text
#         print("    Activity = ", activity)
#         print("    Date = ",date.isoformat())
#         print("    Description = ", desc)
#         print("    Symbol = ", symbol)
#         print("    Price = ", price)
#         print("    Qty = ", qty)
#         print("    Sectype = ", sectype)
#         print(" ")
        self.total_list.append((activity, date.isoformat(), desc, symbol, price, qty, sectype))
        if (symbol in self.dict_of_trades) and sectype == 'CS':
            trade_item = self.dict_of_trades[symbol]
            # Returning True means all shares sold
            try:
                if trade_item.update(qty, price, date):
                    self.dict_of_trades.pop(symbol)
                    self.record_of_trades.append(trade_item.get_record())
            except ValueError as e:
                print("###### Erroneous trade? =",e)
        elif sectype == 'CS':
            trade_item = TradeItem(symbol, qty, price, date)
            self.dict_of_trades[symbol] = trade_item
    
    def generate_total_trades_df(self):
        tradeframe = pd.DataFrame.from_records(self.total_list, columns=["Activity", "Date", "Description", "Symbol"
                                                                         , "Price", "Quantity", "SecType"])
        return tradeframe
    
    def generate_final_trades_df(self):
        final_trades_frame = pd.DataFrame.from_records(self.record_of_trades, 
                                                       columns=["Symbol", "Purchase Date", 
                                                                "Sold Date", "Gain", "Percent Gain"])
        return final_trades_frame
