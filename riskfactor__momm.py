#/usr/bin/python3
"""
动量因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.985 since 2019
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


class diRiskFactor_MOMENTUM(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        if working_type == 'hist':
            _startdate, _enddate = _dates[0],  _dates[-1]
        else:
            _startdate, _enddate = _dates[-600],  _dates[-1]
        super(diRiskFactor_MOMENTUM, self).__init__(
            startdate=_startdate,
            enddate=_enddate,
            delay=delay,
        )
        self.working_type = working_type
        self.init()

    def init(self):
        super().init()
        # Size, 用于回归，也可 get_data
        bcap = self.f_loaddata('BaseData.mkt_cap')
        lcap = bcap.where((bcap>0)&self.m_valid, np.nan).apply(np.log)
        lcap = ut.winsorize_mad(lcap, k=5)
        self.__size = ut.normalize(lcap, weight=self.weight)
        self.__runresult = None

    def fillna_by_regression(self, X, y):
        industry = self.f_loaddata('CITICIndustry.L2')
        industry = industry.where((industry>0)&self.m_valid, 0)
        runresults = []
        for dt in self.dates:
            _x = X.loc[dt].to_frame(name='x')
            _y = y.loc[dt].to_frame(name='y')
            _D = pd.get_dummies(industry.loc[dt])
            _drop_d = _D.sum(axis=0)<=4
            _D.loc[_D.loc[_D.loc[:, _drop_d].sum(axis=1)>0].index, 0] = 1
            _D = _D.loc[:, ~_drop_d]
            exit_code, msg, res = ut.fill_value_by_regression(y=_y, x=_x, filled_mask=None, dummy=_D, weight=None)
            runresults.append(res.rename(columns={'y': dt}))
            if exit_code < 0: continue
        return  self.f_nda2dfa(pd.concat(runresults, axis=1).T)        

    @property
    def runresult(self):
        return self.__runresult

    def run(self):
        def calc_relative_strength(rets, retb):
            def _weighted_sum(dfa, weight):
                dfa_ = dfa.copy()
                if len(dfa) < len(weight):
                    res = pd.Series(dfa_.sum(axis=0), name=dfa.index[-1])
                else:
                    res = pd.Series((dfa_ * weight).sum(axis=0), name=dfa_.index[-1])
                return res
            # 滤掉异常涨幅
            res = ((rets + 1.0).div(retb + 1.0, axis=0)).apply(np.log)
            res = res.shift(20)
            res = ut.winsorize_mad(res, k=5.0)
            runresults = []
            t_window, halflife = 504, 126
            ewm_weights = np.tile(self.calc_ewm_weights(window=t_window, halflife=halflife).reshape(-1, 1), self.m_valid.shape[1])
            for ix, r_ in enumerate(res.rolling(t_window)):
                runresults.append(_weighted_sum(r_, ewm_weights))
            res = pd.concat(runresults, axis=1).T
            return self.f_nda2dfa(res)
        def calc_descriptor_momm(dfa, wt):
            def _f_scale(ar, wt):
                ar_ = ar.copy()
                ar_ = ut.winsorize_mad(ar_, k=5)
                ar_ = ut.normalize(ar_, weight=wt)
                return self.f_nda2dfa(ar_)
            return self.f_nda2dfa(_f_scale(dfa.to_numpy(), wt))
        
        # get data
        stock_return = self.f_loaddata('BaseData.return')
        stock_return = stock_return.where(self.m_valid2, np.nan)
        stock_return = stock_return.where((abs(stock_return)<0.43) & (self.listdays>10), np.nan)
        # 无风险收益率
        rfree_return = pd.Series(1.02**(1.0/365) - 1, index=self.dates, name='riskfree_returns')
        # rfree_return = pd.read_csv('./rates.csv')[['DataDate', 'RFRate%']].set_index('DataDate').reindex(stock_return.index)
        # rfree_return = (rfree_return.where(rfree_return>0, other=np.nan).fillna(method='ffill') * 0.01 +1)**(1.0/365) - 1
        # rfree_return = rfree_return['RFRate%']

        rstr = calc_relative_strength(rets=stock_return, retb=rfree_return)
        rstr = rstr.where(self.m_valid, np.nan)
        xp_momm = calc_descriptor_momm(rstr, self.weight)
        
        # filling
        xp_momm = self.fillna_by_regression(X=self.__size, y=xp_momm.where(self.listdays>120, np.nan))
        xp_momm = ut.winsorize_mad(xp_momm, k=5)
        xp_momm = ut.normalize(xp_momm, self.weight)
        self.__runresult = xp_momm

        
if __name__ == '__main__':
    usage = 'Momentum factor'
    parser = optparse.OptionParser(usage)
    parser.add_option('--startdate', action='store', dest='startdate', type='int', default=20130101, help='set startdate. [default = %default]')
    parser.add_option('--enddate', action='store', dest='enddate', type='int', default=-1, help='set enddate, if enddate < 0, set enddate to today. [default = %default]')
    parser.add_option('--live', action='store_true', dest='live', default=False, help='if `live` worktype [default = %default]')
    options, args = parser.parse_args()
    if options.enddate < 0:
        options.enddate = int(datetime.today().strftime('%Y%m%d'))

    diRF = diRiskFactor_MOMENTUM(startdate=options.startdate, enddate=options.enddate, working_type='live' if options.live else 'hist')
    diRF.run()
    print(diRF.runresult.tail(10))

    diRF.runresult.tail(10).to_parquet(f'./results/[diRF]__[Momentum]__[{diRF.working_type}].pqt')
