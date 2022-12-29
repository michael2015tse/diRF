#/usr/bin/python3
"""
残差波动率因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.97 since 2019
"""

import sys, os
sys.path.extend(['/home/jyxie/combo'])
from utils import io, utils as ut
import numpy as np, pandas as pd, bottleneck as bn
from functools import partial
from typing import Any
from riskfactorbase import RiskFactorBase
from riskfactor__beta import diRiskFactor_BETA
from riskfactor__size import diRiskFactor_SIZE


class diRiskFactor_RESV(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        _startdate = 20130101 #_dates[0] if working_type == 'hist' else 20130101
        _enddate = _dates[-1]
        super(diRiskFactor_RESV, self).__init__(
            startdate=_startdate,
            enddate=_enddate,
            delay=delay,
        )
        self.working_type = working_type
        self.init()

    def init(self):
        super().init()
        # Size 和 Beta, 用于回归，也可 get_data
        _dirf_size = diRiskFactor_SIZE(self.startdate, self.enddate, self.delay, working_type=self.working_type)
        _dirf_beta = diRiskFactor_BETA(self.startdate, self.enddate, self.delay, working_type=self.working_type)
        _dirf_size.run()
        _dirf_beta.run()
        self.__size = _dirf_size.runresult.copy()
        self.__beta = _dirf_beta.runresult.copy()
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
        def calc_descriptor_dastd(rets:pd.DataFrame, retb:pd.Series):
            def _calc_excess_ret(rets, retb):
                return rets.sub(retb, axis=0)
            def _calc_wstd(x, w):
                w_ = ut.fillinvalid(ut.scale_to_one(np.where(np.isfinite(x), w, 0.0).T).T)
                x_ = np.where(np.isfinite(x), x, 0.0)
                nume = bn.nansum(w_ * (x_ - bn.nansum(x_ * w_, axis=0)[None, :])**2, axis=0)
                n_ = bn.nansum(w_>0, axis=0)
                return np.where(n_>4, (nume*n_/(n_-1))**0.5, np.nan)
            t_window, halflife = 252, 42
            ewm_weights = np.tile(self.calc_ewm_weights(window=t_window, halflife=halflife).reshape(-1, 1), self.weight.shape[1])
            ewm_weights = ut.scale_to_one(ewm_weights.T).T
            ex_rets = _calc_excess_ret(rets, retb)
            ex_rets = ut.winsorize_mad(ex_rets, k=5)
            runresults = []
            for ix, r_ in enumerate(ex_rets.rolling(t_window)):
                if len(r_) < 4: continue
                w_ = ewm_weights[-len(r_):].reshape(-1, ewm_weights.shape[1])
                runresults.append(pd.Series(_calc_wstd(x=r_.to_numpy(), w=w_), name=r_.index[-1], index=r_.columns))
            runresults = pd.concat(runresults, axis=1).T
            return self.f_nda2dfa(runresults)

        def calc_descriptor_cmra(rets:pd.DataFrame, retb:pd.Series):
            def _calc_excess_ret(rets, retb):
                return ((rets.add(1.0)).div(retb.add(1.0), axis=0)).apply(np.log)
            def _calc_cmra(x):
                _temp = []
                for i in range(1, 13):
                    _temp.append(bn.nansum(x[-i*21:, :], axis=0))
                _temp = np.array(_temp)
                # 算法在单边下跌行情会出现异常值，改用价格区间则不会有问题
                # 不区分上涨下跌，幅度大cmra大    
                res = (bn.nanmax(_temp, axis=0) + 1.0) / (bn.nanmin(_temp, axis=0) + 1.0)
                return np.where(res>0, np.log(res), np.nan)
            # 为了简单，计算 di_ex_rets，再复合
            # 这里做 winsorize_mad 会  导致一些 Zmax 和 Zmin 维持不变， cmra 计算出异常值，故不处理
            ex_rets = _calc_excess_ret(rets=rets, retb=retb)
            runresults = []
            for ix, r_ in enumerate(ex_rets.rolling(252)):
                if len(r_) < 252: continue
                runresults.append(pd.Series(_calc_cmra(x=r_.to_numpy()), name=r_.index[-1], index=r_.columns))
            runresults = pd.concat(runresults, axis=1).T
            return self.f_nda2dfa(runresults).fillna(method='ffill')

        def calc_descriptor_hsigma(rets:pd.DataFrame, retb:pd.Series):
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
            t_window, halflife = 252, 63
            ewm_weights = np.tile(self.calc_ewm_weights(window=t_window, halflife=halflife).reshape(-1, 1), self.weight.shape[1])
            ewm_weights = ut.scale_to_one(ewm_weights.T).T
            ex_rets = _calc_excess_ret(rets, retb)
            mk_rets = (ex_rets * self.weight).sum(axis=1)
            VALID = ex_rets.notna() & mk_rets.fillna(0.0).notna().to_numpy()[:, None]
            Y = ut.winsorize_mad(ex_rets, k=5).where(VALID, 0.0)
            X = self.f_nda2dfa(np.tile(mk_rets.fillna(0.0).to_numpy().reshape(-1, 1), ex_rets.shape[1]))
            runresults = []
            for ix, r_ in enumerate(Y.rolling(t_window)):
                w_ = np.where(VALID.reindex(index=r_.index).to_numpy(), ewm_weights[-len(r_):], 0.0).reshape(-1, ewm_weights.shape[1])
                w_ = ut.fillinvalid(ut.scale_to_one(w_.T).T)
                y_ = r_.to_numpy()
                x_ = X.reindex(index=r_.index).to_numpy()
                runresults.append(pd.Series(bn.nanstd(_calc_lr(y=y_, x=x_, w=w_)[1], axis=0), name=r_.index[-1], index=r_.columns))
            runresults = pd.concat(runresults, axis=1).T
            return self.f_nda2dfa(runresults)

        # get data
        stock_return = self.f_loaddata('BaseData.return')
        stock_return = stock_return.where(self.m_valid2 & (abs(stock_return)<0.42) & (self.listdays>10), np.nan)
        rfree_return = pd.Series(1.02**(1.0/365) - 1, index=self.dates, name='riskfree_returns')
        
        # calc descriptor
        dastd = calc_descriptor_dastd(stock_return, rfree_return)
        cmra  = calc_descriptor_cmra(stock_return, rfree_return)
        hsigma= calc_descriptor_hsigma(stock_return, rfree_return)

        # combine style factor
        dastd1  = self.f_melt(ut.normalize(ut.winsorize_mad(dastd.where(self.m_valid, np.nan), k=5), self.weight), 'dastd')
        cmra1   = self.f_melt(ut.normalize(ut.winsorize_mad(cmra.where(self.m_valid, np.nan), k=5), self.weight), 'cmra')
        hsigma1 = self.f_melt(ut.normalize(ut.winsorize_mad(hsigma.where(self.m_valid, np.nan), k=5), self.weight), 'hsigma')

        xp_resv = pd.concat([dastd1, cmra1, hsigma1, self.f_melt(self.m_valid, 'm_valid')], axis=1)
        xp_resv = xp_resv.loc[xp_resv.m_valid, ['dastd', 'cmra', 'hsigma']].dropna(axis=0, how='all')

        resv_wt = pd.DataFrame(np.array([0.74, 0.16, 0.1] * len(xp_resv)).reshape(-1, 3), index=xp_resv.index, columns=xp_resv.columns)
        resv_wt = ut.scale_to_one(resv_wt.where(xp_resv.notna(), other=np.nan))

        xp_resv = self.f_nda2dfa(xp_resv.mul(resv_wt).sum(axis=1).reset_index(drop=False).pivot_table(index='date', columns='ticker', values=0))
        xp_resv = xp_resv.where(self.m_valid&(self.listdays>252-24), np.nan)

        # collinearity
        xp_resv = ut.regression_matrix_weighted(a=np.moveaxis(np.array([self.__beta.to_numpy(), self.__size.to_numpy()]), source=0, destination=-1), 
                                                b=xp_resv.to_numpy()[:, :, None], 
                                                num_stock=self.weight.shape[1], num_date=self.weight.shape[0], center=True,
                                                weight=(self.weight**0.65).to_numpy()[:, :, None])
        xp_resv = self.f_nda2dfa(xp_resv)
        xp_resv = ut.winsorize2_mad(xp_resv, kp=10, kn=5)
        xp_resv = ut.normalize(xp_resv, self.weight)

        # filling
        xp_resv = self.fillna_by_regression(X=self.__size, y=xp_resv)
        xp_resv = ut.winsorize2_mad(xp_resv, kp=10, kn=5)
        xp_resv = ut.normalize(xp_resv, self.weight)
        self.__runresult = xp_resv
        
if __name__ == '__main__':
    diRF = diRiskFactor_RESV(startdate=20130101, enddate=20221219, working_type='hist')
    diRF.run()
    print(diRF.runresult)