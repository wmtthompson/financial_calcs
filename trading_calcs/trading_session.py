'''
Created on Oct 17, 2020

@author: william
'''

from xml.etree import ElementTree
from requests_oauthlib import OAuth1Session
from trading_calcs import standard as std_calcs
import pickle

def keydata_from_file(file_path):
    kdf = open(file_path, 'rb')
    keydata = pickle.load(kdf)
    kdf.close()
    return keydata

class AccountInfoSession(object):
    def __init__(self, keydata, holdings_address):
        self.keydata = keydata
        self.oauth = None
        self.account = holdings_address
        self.__start_session__()

    
    def __start_session__(self):
        self.oauth = OAuth1Session(self.keydata['client_key'], 
                      client_secret=self.keydata['client_secret'], 
                      resource_owner_key=self.keydata['fetch_response']['oauth_token'],
                      resource_owner_secret=self.keydata['fetch_response']['oauth_token_secret'])
    
    def get_holdings_list(self):
        r = self.oauth.get(self.account)
        holdings_tree = ElementTree.fromstring(r.content)
        holdings_list = list()
        for h in holdings_tree.iter('holding'):
            hld1 = std_calcs.Holding()
            hld1.set_from_holding_xml(h)
            holdings_list.append(hld1)
        return holdings_list
