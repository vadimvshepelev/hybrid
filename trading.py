import math
import numpy as np


class Position:

    def __init__(self, di, price):
        self.type = 'long' if di > 0 else 'short'
        self.p_open = price
        self.vol = di/price
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
        di = - self.vol * price
        self.vol = 0.
        self.p_open = 0.
        self.lifetime = -1
        return di


class AlgTrivial:
    """
    Описывает тривиальный алгоритм набором из 5 параметров: тренд/разность (True/False), k_p, k_i, k_d, k_dd
    """
    __H = 5000.
    __DIFF_MIN = .1
    __DIFF_MAX = 30.

    def __init__(self,
                 p=[], mu=[], mu_min=0., mu_max=0., dmu=[], dt=0., _N=0,
                 idx=-1, trend_flag=True, k_p=1., k_i=1., k_d=1., k_dd=0.):
        self.p, self.mu, self.mu_min, self.mu_max, self.dmu, self.dt = p, mu, mu_min, mu_max, dmu, dt
        self.idx = idx
        self.trend_flag, self.k_p, self.k_i, self.k_d, self.k_dd = trend_flag, k_p, k_i, k_d, k_dd
        self.pos, self.des = None, 'Hold'
        self.dg = np.zeros(_N)

    def calc_step(self, i=-1, di_cur=0.):
        dg_cur = 0.
        di_next = 0.
        # Производные прибыли dg для управления
        dg_diff = 0.
        dg_diff_prev = 0.
        if i > 1:
            dg_diff = (self.dg[i - 1] - self.dg[i - 2]) / self.dt
        if i > 2:
            dg_diff_prev = (self.dg[i - 2] - self.dg[i - 3]) / self.dt
        dg_diff_diff = (dg_diff - dg_diff_prev) / self.dt
        dg_pos = 0.
        # Integral part
        integ_sum = 0.
        if i > 10:
            coef_arr = np.array([math.exp(float(-j) / 10.) for j in range(10, -1, -1)])
            for cnt in range(10, -1, -1):
                integ_sum += self.dg[i - cnt] * coef_arr[10 - cnt]
        # Double-differential part
        k_dd = 0.
        # Decision part
        if self.pos:
            dg_pos = self.pos.calc_profit(self.p[i])
            if dg_pos > 0. or self.pos.lifetime > 2:
                self.des = 'Close ' + self.pos.type
                dg_cur = dg_pos
                di_cur = self.pos.close(self.p[i])
                sign = 1. if self.pos.type == 'short' else -1.
                di_next = .25 * self.__H * (sign * self.k_p * dg_cur +
                                            self.k_i * integ_sum +
                                            self.k_d * dg_diff +
                                            self.k_dd * dg_diff_diff)
                pos = None
            else:
                self.des = 'Hold'
                dg_cur = 0.
                di_cur, di_next = 0., di_cur
                self.pos.hold()
        elif self.mu_min < math.fabs(self.mu[i]) < self.mu_max:
            if math.fabs(di_cur) < 1.e-5:
                print(id)
                print(f'step={i}, alg={self.idx}, params={(self.trend_flag, self.k_p, self.k_i, self.k_d, self.k_dd)}',
                      f'di_cur={di_cur}, di_next={di_next}, dg_cur={dg_cur}')
                raise ValueError(f'di = 0 (real value is {di_cur}')
            pos = Position(di_cur, self.p[i])
            self.des = 'Open ' + pos.type
            dg_cur = 0.
            di_next = - di_cur
        else:
            self.des = 'Hold'
            dg_cur = 0.
            di_next, di_cur = di_cur, 0.

        self.dg[i] = dg_cur

        return dg_cur, di_cur, di_next
