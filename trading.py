import math
import numpy as np


class Position:

    def __init__(self, di, price):
        self.type = 'long' if di > 0 else 'short'
        self.p_open = price
        self.vol = math.fabs(di)/price
        self.lifetime = 0

    def calc_profit(self, price):
        if self.type == 'long':
            return self.vol * (price - self.p_open)
        if self.type == 'short':
            return self.vol * (self.p_open - price)
        raise ValueError('Tyring to close unknown position type')

    def hold(self):
        self.lifetime += 1

    def close(self, price):
        sign = 1. if self.type == 'short' else -1.
        di = sign * self.vol * price
        self.vol = 0.
        self.p_open = 0.
        self.lifetime = -1
        return di

    def __str__(self):
        return self.type + ' ' + str(self.p_open) + ' ' + str(self.lifetime)

    def __repr__(self):
        return 'REPR: ' + self.type + ' ' + str(self.p_open) + ' ' + str(self.lifetime)

class AlgTrivial:
    """
    Описывает тривиальный алгоритм набором из 5 параметров: тренд/разность (True/False), k_p, k_i, k_d, k_dd
    """
    __H = 5000.

    def __init__(self,
                 p=[], mu=[], mu_min=0., mu_max=0., dmu=[], dt=0., _N=0, di_0=100000.,
                 idx=-1, trend_flag=True, k_p=1., k_i=1., k_d=1., k_dd=0.):
        self.p, self.mu, self.mu_min, self.mu_max, self.dmu, self.dt = p, mu, mu_min, mu_max, dmu, dt
        self.idx = idx
        self.trend_flag, self.k_p, self.k_i, self.k_d, self.k_dd = trend_flag, k_p, k_i, k_d, k_dd
        self.pos, self.des = None, 'Hold'
        self.dg = np.zeros(_N)
        self.di = np.zeros(_N)
        self.di[1] = di_0

    def calc_step(self, i, di_rec=0):
        # Производные прибыли dg для управления
        dg_diff = 0.
        dg_diff_prev = 0.
        if i > 1:
            dg_diff = (self.dg[i - 1] - self.dg[i - 2]) / self.dt
        if i > 2:
            dg_diff_prev = (self.dg[i - 2] - self.dg[i - 3]) / self.dt
        dg_diff_diff = (dg_diff - dg_diff_prev) / self.dt
        # Integral part
        integ_sum = 0.
        if i > 10:
            coef_arr = np.array([math.exp(float(-j) / 10.) for j in range(10, -1, -1)])
            for cnt in range(10, -1, -1):
                integ_sum += self.dg[i - cnt] * coef_arr[10 - cnt]
        # Double-differential part
        k_dd = 0.
        # Decision part
        if di_rec != 0:
            self.di[i] = di_rec
        if self.pos:
            self.dg[i] = self.pos.calc_profit(self.p[i])
            if self.dg[i] > 0. or self.pos.lifetime >= 2:
                self.des = 'Close ' + self.pos.type
                self.di[i] = self.pos.close(self.p[i])
                sign = 1. if self.pos.type == 'short' else -1.
                self.di[i+1] = .25 * self.__H * (sign * self.k_p * self.dg[i] +
                                                 self.k_i * integ_sum +
                                                 self.k_d * dg_diff +
                                                 self.k_dd * dg_diff_diff)
                self.pos = None
            else:
                self.des = 'Hold'
                self.dg[i] = 0.
                self.di[i], self.di[i+1] = 0., self.di[i]
                self.pos.hold()
        elif self.mu_min < math.fabs(self.mu[i]) < self.mu_max:
            if math.fabs(self.di[i]) < 1.e-5:
                print(self.idx)
                print(f'step={i}, alg={self.idx}, params={(self.trend_flag, self.k_p, self.k_i, self.k_d, self.k_dd)}',
                      f'di_cur={self.di[i]}, di_next={self.di[i+1]}, dg_cur={self.dg[i]}')
                raise ValueError(f'di = 0 (real value is {self.di[i]}')
            self.pos = Position(self.di[i], self.p[i])
            self.des = 'Open ' + self.pos.type
            self.dg[i] = 0.
            self.di[i+1] = - self.di[i]
        else:
            self.des = 'Hold'
            self.dg[i] = 0.
            self.di[i+1], self.di[i] = self.di[i], 0.
        return

    def __str__(self):
        return f'alg {self.idx}: {self.trend_flag} {self.k_p} {self.k_i} {self.k_d} {self.k_dd}'


class AlgCombined:
    """Несколько алгоритмов, торгующих в связке для получения результата"""
    def __init__(self, alg_lst=[], _N=0, di_0=100000.):
        self.algs = []
        self.number = len(alg_lst)
        for alg in alg_lst:
            self.algs.append(alg)
        self.dg = np.zeros(_N)
        self.di = np.zeros(_N)
        self.di[1] = di_0

    def calc_step(self):
        for alg in self.algs:
            alg.calc_step
