#/usr/bin/python3
"""
Beta因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.98 since 2019
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


class diRiskFactor_BETA(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        if working_type == 'hist':
            _startdate, _enddate = _dates[0],  _dates[-1]
        else:
            _startdate, _enddate = _dates[-300],  _dates[-1]
        super(diRiskFactor_BETA, self).__init__(
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
        def calc_descriptor_beta(rets:pd.DataFrame, retb:pd.Series):
            def _calc_excess_ret(rets, retb):
                return rets.sub(retb, axis=0)
            def _calc_lr(y, x, w):
                _w = w**0.5
                _y, _x = y*_w, x*_w
                _y = ut.demean(_y.T).T
                _x = ut.demean(_x.T).T
                sx = bn.nansum(_x, axis=0)
                sy = bn.nansum(_y, axis=0)
                sxx = bn.ss(_x, axis=0)
                sxy = bn.nansum(_x*_y, axis=0)
                n = (_w>0).sum(axis=0)
                coef = np.where(n>4, (n*sxy - sx*sy) / (n*sxx - sx**2 + 1e-12), np.nan)
                rsid = (_y - _x * coef[None, :]) / np.where(_w>0, _w, np.nan)
                return coef, rsid
            # 处理数据
            ex_rets = _calc_excess_ret(rets, retb)
            mk_rets = (ex_rets * self.weight).sum(axis=1)
            VALID = ex_rets.notna() & mk_rets.fillna(0.0).notna().to_numpy()[:, None]
            Y = ut.winsorize_mad(ex_rets, k=5).where(VALID, 0.0)
            X = self.f_nda2dfa(np.tile(mk_rets.fillna(0.0).to_numpy().reshape(-1, 1), ex_rets.shape[1]))
            runresults = []
            t_window, halflife = 252, 63
            ewm_weights = np.tile(self.calc_ewm_weights(window=t_window, halflife=halflife).reshape(-1, 1), self.weight.shape[1])
            ewm_weights = ut.scale_to_one(ewm_weights.T).T
            for ix, r_ in enumerate(Y.rolling(t_window)):
                w_ = np.where(VALID.reindex(index=r_.index).to_numpy(), ewm_weights[-len(r_):], 0.0).reshape(-1, ewm_weights.shape[1])
                w_ = ut.fillinvalid(ut.scale_to_one(w_.T).T)
                y_ = r_.to_numpy()
                x_ = X.reindex(index=r_.index).to_numpy()
                runresults.append(pd.Series(_calc_lr(y=y_, x=x_, w=w_)[0], name=r_.index[-1], index=r_.columns))
            runresults = pd.concat(runresults, axis=1).T
            return self.f_nda2dfa(runresults)
        
        # get data
        stock_return = self.f_loaddata('BaseData.return')
        stock_return = stock_return.where(self.m_valid & (abs(stock_return)<0.42) & (self.listdays>10), np.nan)
        rfree_return = pd.Series(1.02**(1.0/365) - 1, index=self.dates, name='riskfree_returns')
        # calc 
        xp_beta = calc_descriptor_beta(stock_return, rfree_return)
        xp_beta = ut.winsorize_mad(xp_beta, k=5)
        xp_beta = ut.normalize(xp_beta, weight=self.weight)
        # filling
        xp_beta = self.fillna_by_regression(X=self.__size, y=xp_beta.where(self.listdays>252-24, np.nan))
        xp_beta = ut.winsorize_mad(xp_beta, k=5)
        xp_beta = ut.normalize(xp_beta, weight=self.weight)
        self.__runresult = xp_beta
        

if __name__ == '__main__':
    usage = 'Beta factor'
    parser = optparse.OptionParser(usage)
    parser.add_option('--startdate', action='store', dest='startdate', type='int', default=20130101, help='set startdate. [default = %default]')
    parser.add_option('--enddate', action='store', dest='enddate', type='int', default=-1, help='set enddate, if enddate < 0, set enddate to today. [default = %default]')
    parser.add_option('--live', action='store_true', dest='live', default=False, help='if `live` worktype [default = %default]')
    options, args = parser.parse_args()
    if options.enddate < 0:
        options.enddate = int(datetime.today().strftime('%Y%m%d'))

    diRF = diRiskFactor_BETA(startdate=options.startdate, enddate=options.enddate, working_type='live' if options.live else 'hist')
    diRF.run()
    print(diRF.runresult.tail(10))

    diRF.runresult.tail(10).to_parquet(f'./results/[diRF]__[Beta]__[{diRF.working_type}].pqt')
