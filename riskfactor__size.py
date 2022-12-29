#/usr/bin/python3
"""
市值因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.995 since 2019
"""

import sys, os
sys.path.extend(['/home/jyxie/combo'])
from utils import io, utils as ut
import numpy as np, pandas as pd, bottleneck as bn
from functools import partial
from riskfactorbase import RiskFactorBase

class diRiskFactor_SIZE(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        _startdate = 20130101 #_dates[0] if working_type == 'hist' else 20130101
        _enddate = _dates[-1]
        super(diRiskFactor_SIZE, self).__init__(
            startdate=_startdate,
            enddate=_enddate,
            delay=delay,
        )
        self.working_type = working_type
        self.init()

    def init(self):
        super().init()
        self.__runresult = None

    def run(self):
        bcap = self.f_loaddata('BaseData.mkt_cap')
        lcap = bcap.where((bcap>0)&self.m_valid, np.nan).apply(np.log)
        lcap = ut.winsorize_mad(lcap, k=5)
        self.__runresult = ut.normalize(lcap, weight=self.weight)

    @property
    def runresult(self):
        return self.__runresult

        
if __name__ == '__main__':
    diRF = diRiskFactor_SIZE(startdate=20180101, enddate=20221220, working_type='hist')
    diRF.run()
    print(diRF.runresult)