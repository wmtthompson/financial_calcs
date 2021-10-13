'''
Created on May 24, 2020

@author: william
'''
import unittest
from unittest.mock import patch, Mock
import trading_calcs.standard as std
import trading_calcs.trading_session as trading_session
import numpy as np
import pandas as pd
from builtins import isinstance
from pandas.testing import assert_frame_equal, assert_series_equal
import pickle
from xml.etree import ElementTree
import matplotlib.pyplot as plt

from datetime import timedelta
import datetime
date1 = datetime.date.today()
date_earlier = date1+timedelta(days=-60)
date_90_days_ago = date1+timedelta(days=-90)
date_90_days_ago_iso = date_90_days_ago.isoformat()
last_60_days_iso = date_earlier.isoformat()
last_60_days_iso


class TestGainLoss(unittest.TestCase):


    def setUp(self):
        price = np.random.normal(loc=20, size=(100,))
        dti = pd.date_range('2020-01-01',periods=100,freq='D')
        self.rand_price = pd.Series(price, index=dti, name='Price')
        self.dti = dti
        
    def tearDown(self):
        self.price = ''


    def testWrongType(self):
        with self.assertRaises(TypeError):
            std.gain_loss_calc(0)
    
    def testReturnType(self):
        self.assertTrue(isinstance(std.gain_loss_calc(self.rand_price),pd.DataFrame), "Did not return a DataFrame")
    
    def testIncreasingPrice(self):
        inc_price = np.linspace(20,25,100)
        inc_price = pd.Series(inc_price, index=self.dti)
        gldf = std.gain_loss_calc(inc_price)
        totgain = gldf['Gain'].sum()
        totloss = gldf['Loss'].sum()
        self.assertGreater(totgain, 0, "Total Gain was not positive for positive increasing price.")
        self.assertEqual(totloss, 0, "Total Loss was not zero for positive increasing price.")
    
    def testSmall(self):
        price = np.arange(0,3)+0.1
        dti = pd.date_range('2020-01-01',periods=len(price),freq='D')
        price = pd.Series(price, index=dti, name='Price')
        gldf = std.gain_loss_calc(price)
        totgain = gldf['Gain'].sum()
        totloss = gldf['Loss'].sum()
        self.assertGreater(totgain, 0, "Total Gain was not positive for positive increasing price.")
        self.assertEqual(totloss, 0, "Total Loss was not zero for positive increasing price.")

class TestGainLossFrame(unittest.TestCase):
    
    def setUp(self):
        self.rsi_reference = pd.read_pickle('./rsi_data.pkl')
        self.ref_gldf = self.rsi_reference[['Gain','Loss']]
        self.ref_gldf = self.ref_gldf.fillna(0)
        self.ref_gldf = self.ref_gldf.round(4)
        
    def tearDown(self):
        self.rsi_reference = 0
        self.ref_gldf = 0
    
    def testFrameEqual(self):
        gldf = std.gain_loss_calc(self.rsi_reference['Close'])
        gldf = gldf.round(4)
        result = assert_frame_equal(self.ref_gldf,gldf)
        self.assertEqual(result, None, "Gain Loss frame not equal to reference frame.")
    

class TestRSIBasic(unittest.TestCase):
    
    def setUp(self):
        price_gain = np.arange(1,4)
        price_loss = np.arange(2,0,-1)
        price = np.concatenate([price_gain, price_loss], axis=0)
        dti = pd.date_range('2020-01-01',periods=len(price),freq='D')
        self.price = pd.Series(price, index=dti, name='Price')
        self.dti = dti
        self.gldf = std.gain_loss_calc(self.price)
        
    def tearDown(self):
        self.price = ''
    
    def testPeriod1(self):
        """
        For a period of 1, the data depends on only the current day compared to previous day.
        """
        rsi = std.rsi_calc(self.gldf['Gain'], self.gldf['Loss'], 1)
        self.assertGreater(rsi.iloc[1], 99.0, "RSI value was not close to 100 for all gain.")
        self.assertLess(rsi.iloc[3], 0.01, "RSI value was not close to 0 for all loss.")

class TestRSIFrame(unittest.TestCase):
     
    def setUp(self):
        self.rsi_reference = pd.read_pickle('./rsi_data.pkl')
        self.gldf = std.gain_loss_calc(self.rsi_reference['Close'])
    
    def tearDown(self):
        self.rsi_reference = 0
        self.gldf = 0
    
    def testSeriesEqual(self):
        rsi = std.rsi_calc(self.gldf['Gain'], self.gldf['Loss'], 14)
        rsi = rsi.iloc[14:-1]
        rsi.round(4)
        rsi_ref = self.rsi_reference['RSI']
        rsi_ref = rsi_ref.iloc[14:-1]
        rsi_ref = rsi_ref.astype('float64')
        rsi_ref.round(4)
        result = assert_series_equal(rsi,rsi_ref)
        self.assertEqual(result, None, "RSI series not equal to reference series.")


class TestVWAP(unittest.TestCase):
    
    def testReturnType(self):
        dti = pd.date_range('2020-01-01',periods=100,freq='D')
        price_arry = np.random.normal(loc=20, size=(100,))
        price = pd.Series(price_arry, index=dti, name='Price')
        volume_arry = np.random.normal(loc=50000, size=(100,))
        volume = pd.Series(volume_arry, index=dti, name='Volume')
        vwap = std.vwap_calc(price, volume, 14)
        self.assertTrue(isinstance(vwap,pd.Series), "Return type of vwap_calc was not a Pandas Series.")
    
    def testWrongType(self):
        with self.assertRaises(TypeError):
            std.vwap_calc(0, 0)
    
    def testIncreasingPrice(self):
        dti = pd.date_range('2020-01-01',periods=100,freq='D')
        inc_price = np.linspace(20,25,100)
        inc_price = pd.Series(inc_price, index=dti, name='Price')
        volume_arry = np.ones((100,))*50000.0
        volume = pd.Series(volume_arry, index=dti, name='Volume')
        vwap = std.vwap_calc(inc_price, volume, 14)
        totvwap = vwap.sum()
        self.assertGreater(totvwap, 0, "Sum of VWAP was not positive for positive increasing price.")

class TestProbabilities(unittest.TestCase):
    
    def setUp(self):
        self.test_data = pd.Series([9, 5, 3, 5, 5, 4, 6, 5, 5, 8, 7], name = 'value')
    
    def testBasicProbability(self):
        self.prob = std.Probabilities(self.test_data)
        p = self.prob.get_prob_less_than(9)
        self.assertAlmostEqual(p, 1, delta=0.1, msg="Probability is not what was expected.")
    
    def testXTooLow(self):
        self.prob = std.Probabilities(self.test_data)
        p = self.prob.get_prob_less_than(2)
        self.assertAlmostEqual(p, 0, delta=0.01, msg="Should be alsmost zero.")
    
    def testXTooHigh(self):
        self.prob = std.Probabilities(self.test_data)
        p = self.prob.get_prob_less_than(50)
        self.assertAlmostEqual(p, 1, delta=0.1, msg="Should be one.")
        
class TestStockTrade(unittest.TestCase):
    
    def setUp(self):
        fp =  open('holdings_data.pkl','rb')
        self.holdings_list = pickle.load(fp)
        fp.close()
        stk = {"symbol":"GOLD","buy_stop":27.50,"sell_stop":26.5,"target_price":28.0}
        self.gold_trade = std.StockTrade(**stk)
        self.settings_file = "optimal_settings_file.json"
        
    def testBasicStockTrade(self):
        stk = {"symbol":"GDX","buy_stop":45.0,"sell_stop":40.0,"target_price":50.0}
        stk1 = std.StockTrade(**stk)
        self.assertEqual(stk1.symbol,"GDX", "Symbol not GDX")
        self.assertAlmostEqual(stk1.buy_stop, 45.0, 1)
        self.assertAlmostEqual(stk1.sell_stop, 40.0, 1)
        self.assertAlmostEqual(stk1.target_price, 50.0, 1)
    
    def testAttachHolding(self):
        self.assertEqual(self.gold_trade.symbol, "GOLD", "Symbol not GOLD")
        self.gold_trade.attach_holding(self.holdings_list)
        self.assertTrue(self.gold_trade.holding != None, "Holding object not attached.")
    
    def testPriceStatusNotBought(self):
        stk = {"symbol":"GDX","buy_stop":45.0,"sell_stop":40.0,"target_price":50.0}
        stk1 = std.StockTrade(**stk)
        self.assertTrue(stk1.get_price_status(45.0) == "BUY")
        self.assertTrue(stk1.get_price_status((1.0029)*45.0) == "BUY")
        self.assertTrue(stk1.get_price_status(39.9) == "WAIT")
        self.assertTrue(stk1.get_price_status(0.01) == "WAIT")
        self.assertTrue(stk1.get_price_status(10000) == "BUY")
    
    def testPriceStatusBought(self):
        self.gold_trade.attach_holding(self.holdings_list)
        self.assertTrue(self.gold_trade.get_price_status(27.51) == "WAIT")
        self.assertTrue(self.gold_trade.get_price_status(28.0) == "TARGET_SELL")
        self.assertTrue(self.gold_trade.get_price_status(28.1) == "TARGET_SELL")
        self.assertTrue(self.gold_trade.get_price_status(30.0) == "TARGET_SELL")
        self.assertTrue(self.gold_trade.get_price_status(26.5) == "SELL")
    
    
    def testPriceStatusBoughtUseStrategy(self):
        self.gold_trade.attach_holding(self.holdings_list)
        self.assertTrue(self.gold_trade.get_price_status(27.51, use_strategy=True) == "WAIT")
        self.assertTrue(self.gold_trade.get_price_status(27.51, use_strategy=True) == "WAIT")
        self.assertFalse(self.gold_trade.get_price_status(27.51, use_strategy=True) == "SELL")
        self.gold_trade.holding = None
        self.assertTrue(self.gold_trade.get_price_status(27.51, use_strategy=True) == "BUY")
        self.assertFalse(self.gold_trade.get_price_status(0.1, use_strategy=True) == "SELL")
        self.assertFalse(self.gold_trade.get_price_status(27.51, use_strategy=True) == "WAIT")
        

class TestHolding(unittest.TestCase):
    
    def setUp(self):
        fp = open('holding_etree_element.pkl','rb')
        self.h_xml = pickle.load(fp)
        fp.close()
    
    def testFeedXML(self):
        h = std.Holding()
        h.set_from_holding_xml(self.h_xml)
        self.assertEqual(h.symbol,"SQQQ", "Symbol not SQQQ")
        self.assertAlmostEqual(h.cost_basis, -79.45,delta=0.01)
        self.assertAlmostEqual(h.market_value, -108.00,delta=0.01)
        self.assertAlmostEqual(h.purchase_price, 0.79, delta=0.001)
        self.assertTrue(h.option)

# class TestTradingSession(unittest.TestCase):
#     
#     def setUp(self):
#         key_file = "./key_data.pkl"
#         self.key_data = trading_session.keydata_from_file(key_file)
#     
#     def testInit(self):
#         sess = trading_session.AccountInfoSession(self.key_data)
#         self.assertTrue(sess.oauth != None, "OAUTH Session not is None.")

class TestPriceAction(unittest.TestCase):
    
    def setUp(self):
        fp = open('gdx_and_slv_last_60_days.pkl','rb')
        self.df = pickle.load(fp)
        fp.close()
        trades_input_list = [{"symbol":"GDX","buy_stop":27.50,"sell_stop":26.5,"target_price":28.0},
                            {"symbol":"SLV","buy_stop":27.50,"sell_stop":26.5,"target_price":28.0}]
        self.trades_list = list()
        symbol_list = list()
        for tr in trades_input_list:
            self.trades_list.append(std.StockTrade(**tr))
            symbol_list.append(tr['symbol'])
#         key_file = "./key_data.pkl"
#         self.key_data = trading_session.keydata_from_file(key_file)
#         self.sess = trading_session.AccountInfoSession(self.key_data)
        fp =  open('holdings_data.pkl','rb')
        self.holdings_list = pickle.load(fp)
        fp.close()
        self.settings_file = "optimal_settings_file.json"
    
    @patch('calc_tests.trading_session.AccountInfoSession', spec=True)
    def testInit(self, ACI):
        sess = ACI()
#         sess_mock = Mock()
        sess.get_holdings_list().return_value = self.holdings_list
        pa = std.PriceAction(self.trades_list, self.df, sess, self.settings_file)
        self.assertTrue(pa)
    
    @patch('calc_tests.trading_session.AccountInfoSession', spec=True)
    def testCalcStrategy(self, ACI):
        sess = ACI()
        sess.get_holdings_list().return_value = self.holdings_list
        pa = std.PriceAction(self.trades_list, self.df, sess, self.settings_file)
        trade = self.trades_list[1]
        pa.calc_strategy(trade)
        trade.print_data(27.4)
    
    @patch('calc_tests.trading_session.AccountInfoSession', spec=True)
    def testPriceData(self, ACI):
        sess = ACI()
        sess.get_holdings_list().return_value = self.holdings_list
        pa = std.PriceAction(self.trades_list, self.df, sess, self.settings_file)
        trade = self.trades_list[1]
        pa.calc_strategy(trade)
        pa.get_price_data()

class TestStrategy(unittest.TestCase):
     
    def setUp(self):
        fp = open('gdx_and_slv_last_60_days.pkl','rb')
        self.df = pickle.load(fp)
        fp.close()
        self.settings_file = "optimal_settings_file.json"
     
    def testInitStrategy(self):
        st1 = std.Strategy('GDX',self.df, self.settings_file)
        self.assertTrue(st1)
        
    def testGetSettings(self):
        st1 = std.Strategy('GDX', self.df, self.settings_file)
        self.assertTrue(st1.get_settings())
    
#     def testCheckPrice(self):
#         st1 = std.Strategy('GDX', self.df, self.settings_file)
#         self.assertTrue(st1.get_settings())
#         st1.calculate_strategy()
#         self.assertTrue(st1.check_price(100) == 0)
        
#     def testPlotStrategy(self):
#         st1 = std.Strategy('GDX',self.df, self.settings_file)
#         self.assertTrue(st1)
#         st1.get_settings()
#         st1.calculate_strategy()
#         ax = st1.plot_strategy()
#         plt.show()
    
     
#     def testOptimizeStrategy(self):
#         st1 = std.Strategy('GDX',self.df, self.settings_file)
#         st1.optimize_strategy()
#         st1.update_settings_file()
#         st1.calculate_strategy()
#         print("Best settings found by optimizer = {0}".format(st1.best_settings))

# date1 = datetime.date.today()
#date_earlier

class TestTradeItem(unittest.TestCase):
       
    def testTradeItem1(self):
        self.assertRaises(ValueError, std.TradeItem,'GDX',0,20.0, date1)
        
    def testTradeItem2(self):
        TI = std.TradeItem('GDX',10,20.0, date1)
        r = TI.update(10, 21.0, date1+timedelta(days=3))
        self.assertEqual(TI.quantity, 20, "Quantity is wrong")
        self.assertFalse(r)
        r = TI.update(-10, 20.0, date1+timedelta(days=4))
        self.assertFalse(r)
        r = TI.update(-10, 20.0, date1+timedelta(days=5))
        self.assertTrue(r)
        print("Trade Item = ",TI)

    def testTradeItem3(self):
        TI = std.TradeItem('GDX',100,20.0, date1)
        r = TI.update(-100, 21.0, date1+timedelta(days=3))
        self.assertTrue(r)
        print("Trade Item = ",TI)
        
        

class TestProcessTrades(unittest.TestCase):
    
    def testStuff(self):
        pctr = std.TradeProcessor()
        with open('history_data.pkl','rb') as fp:
            hist_tree = pickle.load(fp)
        pctr.process_xml(hist_tree)
        print(pctr.record_of_trades)
        pctr.generate_total_trades_df()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()