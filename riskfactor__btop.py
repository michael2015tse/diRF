#/usr/bin/python3
"""
BTOP因子

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


class diRiskFactor_BTOP(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        if working_type == 'hist':
            _startdate, _enddate = _dates[0],  _dates[-1]
        else:
            _startdate, _enddate = _dates[-100],  _dates[-1]
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
    usage = 'BTop factor'
    parser = optparse.OptionParser(usage)
    parser.add_option('--startdate', action='store', dest='startdate', type='int', default=20130101, help='set startdate. [default = %default]')
    parser.add_option('--enddate', action='store', dest='enddate', type='int', default=-1, help='set enddate, if enddate < 0, set enddate to today. [default = %default]')
    parser.add_option('--live', action='store_true', dest='live', default=False, help='if `live` worktype [default = %default]')
    options, args = parser.parse_args()
    if options.enddate < 0:
        options.enddate = int(datetime.today().strftime('%Y%m%d'))

    diRF = diRiskFactor_BTOP(startdate=options.startdate, enddate=options.enddate, working_type='live' if options.live else 'hist')
    diRF.run()
    print(diRF.runresult.tail(10))

    diRF.runresult.tail(10).to_parquet(f'./results/[diRF]__[Btop]__[{diRF.working_type}].pqt')