#/usr/bin/python3
"""
流动性因子

changelog:
20221225  jyxie:  init ver. mimic BarraCNE5D, cs_corr > 0.985 since 2019
"""

import sys, os
sys.path.extend(['/home/jyxie/combo'])
from utils import io, utils as ut
import numpy as np, pandas as pd, bottleneck as bn
from functools import partial
from typing import Any
from riskfactorbase import RiskFactorBase


class diRiskFactor_LIQUIDITY(RiskFactorBase):
    def __init__(self, startdate:int, enddate:int, delay:int=0, working_type:str='hist'):
        if working_type not in ('live', 'hist'):
            raise ValueError("`working_type must be `live` or `hist`")
        _dates = io.load_dates(startdate, enddate)
        _startdate = 20130101 #_dates[0] if working_type == 'hist' else 20130101
        _enddate = _dates[-1]
        super(diRiskFactor_LIQUIDITY, self).__init__(
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

    @property
    def runresult(self):
        return self.__runresult

    def run(self):
        def calc_turnover(tvr_shares:pd.DataFrame, bas_shares:pd.DataFrame, valid:pd.DataFrame, window:int):
            tvr = bn.move_sum(tvr_shares.where(valid&(tvr_shares>=100), other=np.nan).to_numpy(), window=window, axis=0, min_count=max(1, window-5))
            tvr/= bn.move_mean(bas_shares.where(bas_shares>0, other=np.nan).fillna(method='ffill').where(valid, other=np.nan).to_numpy(), window=window, axis=0, min_count=max(1, window-5))
            return pd.DataFrame(tvr, index=tvr_shares.index, columns=tvr_shares.columns)

        def calc_descriptor_liq(dfa:pd.DataFrame, name:str, wt=None):
            def _f_scale(ar, wt):
                ar_ = ar.copy()
                ar_ = ut.winsorize_mad(ar_, k=5.0)
                ar_ = ut.normalize(ar_, weight=wt)
                return self.f_nda2dfa(ar_)
            return self.f_melt(_f_scale(np.log(dfa.where(dfa>0, other=np.nan)).to_numpy(), wt=wt), name)
        
        # get data
        b_tvrshares = self.f_loaddata('BaseData.tvrvolume')
        b_totshares = self.f_loaddata('BaseData.tot_share')
        # 取上市后35天起，到退市止
        _valid = self.m_valid & (self.listdays>35)

        # descriptor
        stom = calc_descriptor_liq(calc_turnover(b_tvrshares, b_totshares, _valid, window=21), 'stom', self.weight)
        stoq = calc_descriptor_liq(calc_turnover(b_tvrshares, b_totshares, _valid, window=63) / 3.0, 'stoq', self.weight)
        stoa = calc_descriptor_liq(calc_turnover(b_tvrshares, b_totshares, _valid, window=252) / 12.0, 'stoa', self.weight)
        xp_liqd = pd.concat([stom, stoq, stoa, self.f_melt(_valid, 'm_valid')], axis=1)
        xp_liqd = xp_liqd.loc[xp_liqd.m_valid, ['stom', 'stoq', 'stoa']].dropna(axis=0, how='all')
        
        # weights
        liqd_wt = pd.DataFrame(np.array([0.45, 0.35, 0.3] * len(xp_liqd)).reshape(-1, 3), index=xp_liqd.index, columns=xp_liqd.columns)
        liqd_wt = ut.scale_to_one(liqd_wt.where(xp_liqd.notna(), other=np.nan))
        
        # factor
        xp_liqd = self.f_nda2dfa(xp_liqd.mul(liqd_wt).sum(axis=1).reset_index(drop=False).pivot_table(index='date', columns='ticker', values=0))
        xp_liqd = xp_liqd.where(self.m_valid, np.nan)
        xp_liqd = self.f_nda2dfa(ut.regression_2d_weighted(a=self.__size.to_numpy(), b=xp_liqd.to_numpy(), weight=self.weight**0.65, center=True))
        xp_liqd = ut.normalize(xp_liqd, self.weight)
        self.__runresult = xp_liqd
        
        # filling
        

        
if __name__ == '__main__':
    diRF = diRiskFactor_LIQUIDITY(startdate=20180101, enddate=20221220, working_type='hist')
    diRF.run()
    print(diRF.runresult)