#/usr/bin/python3
"""
风险因子基类

changelog:
20221225  jyxie:  init ver
"""

import sys, os
sys.path.extend(['/home/jyxie/combo'])
from utils import io, utils as ut
import numpy as np, pandas as pd, bottleneck as bn
from functools import partial
from typing import Any
from abc import ABC, abstractmethod


class RiskFactorBase(ABC):
    def __init__(self, startdate:int=None, enddate:int=None, delay:int=0):
        self.__startdate = startdate
        self.__enddate = enddate
        self.__delay = delay
    @property
    def startdate(self):
        return self.__startdate
    @property
    def enddate(self):
        return self.__enddate
    @property
    def delay(self):
        return self.__delay
    @property
    def dates(self):
        return io.load_dates(self.startdate, self.enddate)
    @property
    def tickers(self):
        return io.load_tickers()

    def f_loaddata(self, name:str, startdate:int=None, enddate:int=None, delay:int=None):
        return io.load_data(name, 
                            startdate=self.startdate if startdate is None else startdate, 
                            enddate=self.enddate if enddate is None else enddate, 
                            delay=self.__delay if delay is None else delay)

    def f_nda2dfa(self, data:Any, filled:float=np.nan):
        return ut.nda2dfa(data, index=self.dates, columns=self.tickers, filled=filled)

    def f_melt(self, dfa:pd.DataFrame, name:str):
        return ut._melt_dataframe(dfa, var_name='ticker', value_name=name).set_index(['date', 'ticker'])

    def calc_ewm_weights(self, window:int, halflife:int):
        _alpha = 1-np.exp(1)**(np.log(0.5)/halflife)
        _eww = np.array([(1 - _alpha)**i for i in range(0, window)])[::-1]
        return _eww

    def init(self):
        self.m_valid = (self.f_loaddata('BaseData.close') > 0) # Onsite
        self.m_valid2 = self.m_valid & (self.f_loaddata('BaseData.tvrvolume') >= 100) # Normal trading
        self.weight  = self.f_loaddata('BaseData.mkt_cap')
        self.weight  = ut.scale_to_one(self.weight.where(self.m_valid, np.nan))
        self.listdays= self.f_nda2dfa((self.f_loaddata('BaseData.close', startdate=0)>0).expanding().sum())

    @abstractmethod
    def run(self):
        raise NotImplemented
