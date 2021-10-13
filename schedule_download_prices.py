import schedule
import time
import os
# import numpy as np
# import pandas as pd
import pandas_datareader as pdr
import datetime
from datetime import timedelta
date1 = datetime.date.today()
last_60_days = date1+timedelta(days=-60)
last_60_days_iso = last_60_days.isoformat()
import json

import trading_calcs.standard as std_calcs
import trading_calcs.trading_session as trading_session

import configparser

import argparse

config = configparser.ConfigParser()

parser = argparse.ArgumentParser()
parser.add_argument("config_file_location")

last_14_days = date1+timedelta(days=-14)
last_40_days = date1+timedelta(days=-40)




def alarm_event():
    duration = 1
    freq = 440
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def price_check2():
    """Lookup prices for the last 60 Days, calculate probabilites as well."""
    fp = open("./trades.json",'r')
    trades_input_list = json.load(fp)["trades"]
    trades_list = list()
    symbol_list = list()
    for tr in trades_input_list:
        trades_list.append(std_calcs.StockTrade(**tr))
        symbol_list.append(tr['symbol'])
    df1 = pdr.get_data_yahoo(symbol_list, start=last_60_days_iso)
    df1 = round(df1,4)
    
    price_action = std_calcs.PriceAction(trades_list, df1, acct_session, 'optimal_settings_file.json')
    price_action.set_alarm_event(alarm_event)
    price_action.update_trades()
    price_action.get_price_data()

def price_check1():
    """Lookup prices for the last 14 Days, calculate probabilites as well."""
    fp = open("./trades.json",'r')
    trades_input_list = json.load(fp)["trades"]
    trades_list = list()
    symbol_list = list()
    for tr in trades_input_list:
        trades_list.append(std_calcs.StockTrade(**tr))
        symbol_list.append(tr['symbol'])
    df1 = pdr.get_data_yahoo(symbol_list, start=last_14_days)
    df1 = round(df1,4)
    
    now = datetime.datetime.now()
    holdings_list = acct_session.get_holdings_list()
    for tr in trades_list:
        tr.attach_holding(holdings_list)
    print("Current Date and Time = \n",now)
    for tr in trades_list:
        print(" ")
        print("#####################################")
        # check vwap
#         pct = std_calcs.calc_vwap_strategy(df2['Close'][tr.symbol], df2['Volume'][tr.symbol], 10)
#         s_vwap = std_calcs.vwap_calc(df2['Close'][tr.symbol], df2['Volume'][tr.symbol], 10)
#         print("VWAP strategy pct = {0}".format(pct))
#         print("Price-VWAP = {0}".format(df2['Close'][tr.symbol].iloc[-1].item()-s_vwap.iloc[-1].item()))
        current_price = df1['Close'][tr.symbol].iloc[-1]
        tr.print_data(current_price)
        status = tr.get_price_status(current_price)
        if status == "SELL" or status == "BUY" or status == "TARGET_SELL":
            alarm_event()
        print("Current status = ",status)
        pr = tr.get_proba_target_price_or_higher(df1['Close'][tr.symbol])
        print("Probability of price over target = {0:.2f} ".format(pr))
        print("#####################################")
        print(" ")
    
    
    

# def price_check():
#     df1 = pdr.get_data_yahoo(['FSM','GDX','SVXY','UGA','TEVA','SRNE'], start=date1)
#     df1 = round(df1,4)
#     now = datetime.datetime.now()
#     print("Current Date and Time = \n",now)
#     if df1['Close']['GDX'].iloc[-1] < 40:
#         print("GDX Price is < 41.5, at ",df1['Close']['GDX'].iloc[-1])
#         print("SELL YOUR GDX CALLS or Roll your Calls to a lower strike or further out in time.\n")
#         duration = 1
#         freq = 440
#         os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
#     else:
#         print("GDX Price is at ",df1['Close']['GDX'].iloc[-1])
# 
#     if df1['Close']['TEVA'].iloc[-1] > 11.24:
#         print("TEVA Price > 11.24, at ",df1['Close']['TEVA'].iloc[-1])
#         print("Sell TEVA \n")
#         duration = 1
#         freq = 440
#         os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
#     else:
#         print("TEVA Price is at ",df1['Close']['TEVA'].iloc[-1])
#     
#     if df1['Close']['SRNE'].iloc[-1] 
#     
#     
#     svxy_shares = 30
#     svxy_avg_price = 34
#     svxy_cost_basis = 34*30
#     svxy_gain = (df1['Close']['SVXY'].iloc[-1] - svxy_avg_price)*svxy_shares
#     print("SVXY Last Price = ",df1['Close']['SVXY'].iloc[-1])
#     print("SVXY Gain = ",svxy_gain)
#     print(" ")
# 
#     print("################################################\n")

def check_slv():
    slv = pdr.get_data_yahoo('SLV', start=date_earlier_iso)
    slv = slv.drop('Adj Close', axis=1)
    slv = round(slv, 4)
    gldf = std_calcs.gain_loss_calc(slv['Close'])
    rsi = std_calcs.rsi_calc(gldf['Gain'], gldf['Loss'], 14)
    print("SLV RSI = \n",rsi.tail(5))
    if rsi.iloc[-1] > 70 or rsi.iloc[-1] < 33 or slv['Close'].iloc[-1] < 16.0:
        duration = 1
        freq = 440
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    wpm = pdr.get_data_yahoo('WPM',start=date_earlier_iso)
    wpm = wpm.drop('Adj Close', axis=1)
    wpm = round(wpm, 4)
    gldf = std_calcs.gain_loss_calc(wpm['Close'])
    rsi = std_calcs.rsi_calc(gldf['Gain'], gldf['Loss'], 14)
    print("WPM RSI = \n",rsi.tail(5))
    if rsi.iloc[-1] > 60 or rsi.iloc[-1] < 33:
        duration = 1
        freq = 540
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    gdx = pdr.get_data_yahoo('GDX',start=date_earlier_iso)
    gdx = gdx.drop('Adj Close', axis=1)
    gdx = round(gdx, 4)
    gldf = std_calcs.gain_loss_calc(gdx['Close'])
    rsi = std_calcs.rsi_calc(gldf['Gain'], gldf['Loss'], 14)
    print("GDX RSI = \n",rsi.tail(5))
    if rsi.iloc[-1] > 50 or rsi.iloc[-1] < 40 or gdx['Close'].iloc[-1] > 34.5:
        duration = 1
        freq = 400
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    

def hello():
    print("Hello World!")
    duration = 1
    freq = 440
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))



if __name__ == "__main__":
    
    args = parser.parse_args()
    config_file_path = args.config_file_location
    config_file_directory = os.path.dirname(config_file_path)
    config.read(config_file_path)
    key_file_path = os.path.join(config_file_directory, config['DEFAULT']['keyfile'])
    holdings_address = config['DEFAULT']['holdings_address']
    key_data = trading_session.keydata_from_file(key_file_path)
    acct_session = trading_session.AccountInfoSession(key_data, holdings_address)
    
    price_check2()
    schedule.every(15).minutes.do(price_check2)


    while True:
        now = datetime.datetime.now()
        if now.hour >= 9 and now.hour < 16:
            schedule.run_pending()
            time.sleep(2)
        else:
            time.sleep(60*30)
