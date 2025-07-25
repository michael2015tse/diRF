#/usr/bin/python3
"""
非线性市值因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.995 since 2019
20230106  jyxie:  add `live` working_type to accelerate computation
"""

import sys, os
sys.path.extend(['/home/jyxie/combo'])
from utils import io, utils as ut
import numpy as np, pandas as pd, bottleneck as bn
from functools import partial
from typing import Any
import optparse
from datetime import datetime
from riskfactorbase import RiskFactorBase

class diRiskFactor_SIZENL(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        if working_type == 'hist':
            _startdate, _enddate = _dates[0],  _dates[-1]
        else:
            _startdate, _enddate = _dates[-10],  _dates[-1]
        super(diRiskFactor_SIZENL, self).__init__(
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
        size = ut.normalize(lcap, weight=self.weight)
        ## size ~ size**3 结果更精准 ##
        sinl = ut.regression_2d_weighted(b=size.to_numpy(), a=(size**3).to_numpy(), weight=self.weight, center=True)
        sinl = ut.winsorize_mad(sinl, k=5)
        sinl = ut.banorm(sinl, weight=self.weight)
        self.__runresult = self.f_nda2dfa(ut.normalize(sinl, weight=self.weight)) * -1
        
    @property
    def runresult(self):
        return self.__runresult

        
if __name__ == '__main__':
    usage = 'SizeNL factor'
    parser = optparse.OptionParser(usage)
    parser.add_option('--startdate', action='store', dest='startdate', type='int', default=20130101, help='set startdate. [default = %default]')
    parser.add_option('--enddate', action='store', dest='enddate', type='int', default=-1, help='set enddate, if enddate < 0, set enddate to today. [default = %default]')
    parser.add_option('--live', action='store_true', dest='live', default=False, help='if `live` worktype [default = %default]')
    options, args = parser.parse_args()
    if options.enddate < 0:
        options.enddate = int(datetime.today().strftime('%Y%m%d'))

    diRF = diRiskFactor_SIZENL(startdate=options.startdate, enddate=options.enddate, working_type='live' if options.live else 'hist')
    diRF.run()
    print(diRF.runresult.tail(10))

    diRF.runresult.tail(10).to_parquet(f'./results/[diRF]__[SizeNL]__[{diRF.working_type}].pqt')