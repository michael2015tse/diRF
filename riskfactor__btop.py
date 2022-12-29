#/usr/bin/python3
"""
BTOP因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.995 since 2019
"""

import sys, os
sys.path.extend(['/home/jyxie/combo'])
from utils import io, utils as ut
import numpy as np, pandas as pd, bottleneck as bn
from functools import partial
from typing import Any
from riskfactorbase import RiskFactorBase


class diRiskFactor_BTOP(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        _startdate = 20130101 #_dates[0] if working_type == 'hist' else 20130101
        _enddate = _dates[-1]
        super(diRiskFactor_BTOP, self).__init__(
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
        mask = pd.Series(self.m_valid.index.to_numpy(), index=self.dates).diff(1) > 30
        mask = mask.where(np.array([not str(int(x%1e4)).startswith(('4')) for x in mask.index]), False)

        bookvalue = self.f_loaddata('WindDerivativeIndicator.NET_ASSETS_TODAY')
        totcapital= self.f_loaddata('WindDerivativeIndicator.S_VAL_MV')
        btop = bookvalue.fillna(method='ffill') / totcapital.where(totcapital>0, np.nan).fillna(method='ffill')

        btop = btop.where(mask, np.nan).fillna(method='ffill')
        btop = btop.where(self.m_valid, np.nan)
        btop = ut.winsorize_mad(btop, k=5)
        btop = ut.normalize(btop, self.weight)

        self.__runresult = btop

    @property
    def runresult(self):
        return self.__runresult

        
if __name__ == '__main__':
    diRF = diRiskFactor_BTOP(startdate=20130101, enddate=20221220, working_type='hist')
    diRF.run()
    print(diRF.runresult)