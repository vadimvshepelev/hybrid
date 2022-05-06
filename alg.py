import math
import os
import functools
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import zipfile


def load_test_data(step=5000):
    """
    Загружает из csv, лежащих в папке проекта, данные (биржевые котировки за 4 дня в феврале 2022 года)
    для тестирования алгоритмов. Возвращает кортеж из массива и тренда за каждую дату, всего 8 np.array-ев.
    """
    archive = zipfile.ZipFile('data.zip', 'r')

    col_names = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

    archive.extract('trades-16-02-22.csv', '.')
    df_stable = pd.read_csv('trades-16-02-22.csv', sep=';', names=col_names)
    os.remove('trades-16-02-22.csv')
    col3_ser_stable = df_stable['col3']
    col3_ser_stable.reset_index(drop=True, inplace=True)

    archive.extract('trades-17-02-22.csv', '.')
    df_growth = pd.read_csv('trades-17-02-22.csv', sep=';', names=col_names)
    os.remove('trades-17-02-22.csv')
    col3_ser_growth = df_growth['col3']
    col3_ser_growth.reset_index(drop=True, inplace=True)

    archive.extract('trades-si-14-02-22.csv', '.')
    df_si_14 = pd.read_csv('trades-si-14-02-22.csv', sep=';', names=col_names)
    os.remove('trades-si-14-02-22.csv')
    col3_ser_si_14 = df_si_14['col3']
    col3_ser_si_14.reset_index(drop=True, inplace=True)

    archive.extract('trades-si-21-02-22.csv', '.')
    df_si_21 = pd.read_csv('trades-si-21-02-22.csv', sep=';', names=col_names)
    os.remove('trades-si-21-02-22.csv')
    col3_ser_si_21 = df_si_21['col3']
    col3_ser_si_21.reset_index(drop=True, inplace=True)

    price_series_stable = np.array(col3_ser_stable)
    p_arr_stable = np.array([price_series_stable[i] for i in range(len(price_series_stable)) if i % step == 0])

    price_series_growth = np.array(col3_ser_growth)
    p_arr_growth = np.array([price_series_growth[i] for i in range(len(price_series_growth)) if i % step == 0])

    price_series_si_14 = np.array(col3_ser_si_14)
    p_arr_si_14 = np.array([price_series_si_14[i] for i in range(len(price_series_si_14)) if i % step == 0])

    price_series_si_21 = np.array(col3_ser_si_21)
    p_arr_si_21 = np.array([price_series_si_21[i] for i in range(len(price_series_si_21)) if i % step == 0])

    archive.extract('BTCUSDT-20210903.csv', '.')
    df_fall = pd.read_csv('BTCUSDT-20210903.csv', sep=';', names=col_names[:5])
    os.remove('BTCUSDT-20210903.csv')
    col3_ser_fall = df_fall['col3']
    col3_ser_fall.reset_index(drop=True, inplace=True)
    price_series_fall = np.array(col3_ser_fall)
    p_arr_fall = np.array([price_series_fall[i] for i in range(len(price_series_fall)) if i % step == 0])

    archive.extract('BTCUSDT-20210907.csv', '.')
    df_btc_07 = pd.read_csv('BTCUSDT-20210907.csv', sep=';', names=col_names[:5])
    os.remove('BTCUSDT-20210907.csv')
    col3_ser_btc_07 = df_btc_07['col3']
    col3_ser_btc_07.reset_index(drop=True, inplace=True)
    price_series_btc_07 = np.array(col3_ser_btc_07)
    p_arr_btc_07 = np.array([price_series_btc_07[i] for i in range(len(price_series_btc_07)) if i % step == 0])

    archive.extract('BTCUSDT-20210904.csv', '.')
    df_btc_04 = pd.read_csv('BTCUSDT-20210904.csv', sep=';', names=col_names[:5])
    os.remove('BTCUSDT-20210904.csv')
    col3_ser_btc_04 = df_btc_04['col3']
    col3_ser_btc_04.reset_index(drop=True, inplace=True)
    price_series_btc_04 = np.array(col3_ser_btc_04)
    p_arr_btc_04 = np.array([price_series_btc_04[i] for i in range(len(price_series_btc_04)) if i % step == 0])

    return p_arr_stable, p_arr_growth, p_arr_si_14, p_arr_si_21, p_arr_fall, p_arr_btc_07, p_arr_btc_04


def visualize(p_ser, diff_ser, dg_max_ser, dg_ser, profit_ser, di_ser, i_ser, k_ser, k_i_ser, k_d_ser, k_dd_ser):
    # fig, ((ax_p, ax_dg, ax_di, ax_k), (ax_mu, ax_profit, ax_i, ax_k)) = plt.subplots(figsize=(20, 3))
    n_max = len(p_ser)
    fig, ax = plt.subplots(figsize=(18, 6), nrows=2, ncols=3)
    plt.subplots_adjust(wspace=.3, hspace=.4)
    ax[0][0].set_title("Price")
    ax[0][0].set_xlabel("t")  # ось абсцисс
    ax[0][0].set_ylabel("p")  # ось ординат
    ax[0][0].plot(p_ser)
    ax[0][0].grid()

    ax[1][0].set_title("dg_max")
    ax[1][0].set_xlabel("t")  # ось абсцисс
    ax[1][0].set_ylabel("d/dt trend")  # ось ординат
    ax[1][0].plot(dg_max_ser)
    ax[1][0].grid()

    ax[0][1].set_title("Instant profit")
    ax[0][1].set_xlabel("t")  # ось абсцисс
    ax[0][1].set_ylabel("dg")  # ось ординат
    ax[0][1].plot(dg_ser)
    ax[0][1].grid()

    ax[1][1].set_title("Cumulative profit")
    ax[1][1].set_xlabel("t")  # ось абсцисс
    ax[1][1].set_ylabel("Profit")  # ось ординат
    ax[1][1].plot(profit_ser)
    ax[1][1].grid()

    ax[0][2].set_title("Instant investment")
    ax[0][2].set_xlabel("t")  # ось абсцисс
    ax[0][2].set_ylabel("dI")  # ось ординат
    ax[0][2].plot(di_ser)
    ax[0][2].grid()

    ax[1][2].set_title("Cumulative investment")
    ax[1][2].set_xlabel("t")  # ось абсцисс
    ax[1][2].set_ylabel("I")  # ось ординат
    ax[1][2].plot(i_ser)
    ax[1][2].grid()

    plt.show()


def calc_hybrid_alg(_p_arr: np.array, output_flag=True, time=14, diff_min=.1, diff_max=30.):
    """Тестирование PIDD-алгоритма на >200 различных конфигурациях параметра
    Расчет по алгоритму alg 0.5, на входе массив p и количество часов рабоыт биржи (разное для фьючерсов и биткойна)"""
    # Время для биржи с фьючерсом рубль-доллар -- 14 часов
    di_0 = 100000.
    # Для биткоина будет 24 часа
    n_max = len(_p_arr)
    t_max = time * 3600
    t_arr = np.linspace(0., t_max, n_max)
    dt = t_arr[1] - t_arr[0]
    # Настроечные параметры
    # diff_min и diff_max выношу в именные параметры функции
    # diff_min = .1
    # diff_max = 30.
    trend_min = diff_min/dt
    trend_max = diff_max/dt
    # Для интегрального члена
    dg_diff_prev = 0.
    diff_arr = np.array([_p_arr[i] - _p_arr[i-1] if i > 0 else 0. for i in range(n_max)])
    diff2_arr = np.array([diff_arr[i] - diff_arr[i-1] if i > 1 else 0. for i in range(n_max)])
    dtrend_arr = np.zeros(n_max)
    d2trend_arr = np.zeros(n_max)
    # Считаем тренды и разности для передачи в алгоритмы
    # Самую малость жульничаем с целью экономии ресурса (считаем все как будто оно уже есть),
    # но это не влияет на логику работы алгоритмов
    for i in range(5, n_max):
        # _, gdp_trend = sm.tsa.filters.hpfilter(_p_arr[:i + 1])
        # dtrend_arr[i] = (gdp_trend[i] - gdp_trend[i-1]) / dt
        # d2trend_arr[i] = (dtrend_arr[i] - dtrend_arr[i-1]) / dt
        dtrend_arr[i] = (_p_arr[i] - _p_arr[i - 1]) / dt
        d2trend_arr[i] = (dtrend_arr[i] - dtrend_arr[i - 1]) / dt
    mu_arr = np.array([])
    dmu_arr = np.array([])
    k_p_arr = np.zeros(n_max)
    k_i_arr = np.zeros(n_max)
    k_d_arr = np.zeros(n_max)
    k_dd_arr = np.zeros(n_max)
    dg_arr = np.zeros(n_max)
    dg_max_arr = np.zeros(n_max)
    di_arr = np.zeros(n_max)
    di_arr[1] = di_0
    history = []
    des_str, state = '', 'none'
    alg_id_tpl = (True, 1., 1.)
    # Набор списков для перебора тривиальных алгоритмов
    # trend_range, k_i_range, k_d_range = [True, False], np.linspace(-1., 1., 101), np.linspace(-1., 1., 101)
    """trend_range = [True, False]
    k_p_range = np.linspace(-1., 1., 10)
    k_i_range = np.linspace(-1., 1., 11)
    k_d_range = np.linspace(-1., 1., 11)
    h_range = np.array([-5000., -2000., -500., 500., 2000., 5000.])"""
    trend_range, k_p_range, k_i_range, k_d_range, h_range = [True, False], [1.], [-1., 0., 1.], [-1., 0., 1.], [5000.]
    # trend_range, k_p_range, k_i_range, k_d_range, h_range = [False], [1.], [1.], [1.], [5000.]
    #trend_range, k_i_range, k_d_range = [False], [-1.], [-1.]
    # Список словарей для окружения алгоритма
    alg_lst = []
    # Какой алгоритм считает по умолчанию
    alg_cur_id, alg_cur_dct = 0, dict()
    # Инициализация словарей
    for alg_id, (trend_flag, k_p, k_i, k_d, h) in enumerate(product(trend_range,
                                                                 k_p_range,
                                                                 k_i_range,
                                                                 k_d_range,
                                                                 h_range)):
        alg_dct = {'id': alg_id,
                   'params': (trend_flag, k_p, k_i, k_d),
                   'state': 'none',
                   'des': '',
                   'history': [],
                   'dt': dt,
                   'p': _p_arr,
                   'dg': np.copy(dg_arr),
                   'di': np.copy(di_arr),
                   'h': h,
                   'di_0': di_0
                   }
        if not trend_flag:
            alg_dct.update({'mu': diff_arr, 'dmu': diff2_arr, 'mu_min': diff_min, 'mu_max': diff_max})
        else:
            alg_dct.update({'mu': dtrend_arr, 'dmu': d2trend_arr, 'mu_min': trend_min, 'mu_max': trend_max})
        alg_lst.append(alg_dct)
    # Cоздаем для алгоритма по умолчанию отдельный словарь последним номером в списке alg_lst
    alg_cur_dct = {'id': alg_cur_id,
                   'params': (trend_flag, k_p, k_i, k_d),
                   'state': 'none',
                   'des': '',
                   'history': [],
                   'dt': dt,
                   'p': _p_arr,
                   'dg': dg_arr,
                   'di': di_arr,
                   'h': h,
                   'di_0': di_0
                   }
    if not trend_flag:
        alg_cur_dct.update({'mu': diff_arr, 'dmu': diff2_arr, 'mu_min': diff_min, 'mu_max': diff_max})
    else:
        alg_cur_dct.update({'mu': dtrend_arr, 'dmu': d2trend_arr, 'mu_min': trend_min, 'mu_max': trend_max})
    alg_lst.append(alg_cur_dct)
    # Main cycle
    num_switches = 0
    for i in range(1, n_max - 1):
        if i == n_max - 1:
            k_p_arr[i], k_i_arr[i], k_d_arr[i], k_dd_arr[i], dg_arr[i], di_arr[i] = 0., 0., 0., 0., 0., 0.
            break
        dgm = -10.e10
        rating_cur = {}
        for alg_id, alg_dct in enumerate(alg_lst):


            if i == 2 and alg_id == 0:

                qq = 1.





            calc_step(i, alg_dct)
            trend_flag, k_p, k_i, k_d = alg_dct['params']
            dg, des, state = alg_dct['dg'][i], alg_dct['des'], alg_dct['state']
            # print(f'Алг.{alg_id} с k_i={k_i}, k_d={k_d}, trend={trend_flag}: dg={dg} -> {des}')
            if state == 'none':
                rating_cur[alg_id] = dg
            if dg > dgm:
                dgm = dg
        dg_max_arr[i] = dgm
        tf_cur, kp_cur, ki_cur, kd_cur = alg_cur_dct['params']
        des_cur = alg_cur_dct['des']
        dg_cur, di_cur, dinxt_cur = alg_cur_dct['dg'][i], alg_cur_dct['di'][i], alg_cur_dct['di'][i+1]
        dg_arr[i], di_arr[i], di_arr[i+1] = dg_cur, di_cur, dinxt_cur
        alg_new_id = max(rating_cur, key=rating_cur.get) if rating_cur else -1
        if output_flag:
            # print(f'Выбираем мз алгоритмов {rating_cur.keys()} с прибылями {rating_cur.values()}')
            # print(f'Максимальная прибыль у алгоритма {alg_new_id}')
            # print(f'Текущий алгоритм {alg_cur_id}: с k = ({trend_flag}, {k_i}, {k_d}) и решением {des_cur}')
            print(f'{i}: alg{alg_cur_id} t={round(t_arr[i]/3600., 2)} p={_p_arr[i]}',
                  f'd/d2={round(diff_arr[i], 4)}/{round(diff2_arr[i], 4)}',
                  f'dtr/d2tr={round(dtrend_arr[i], 4)}/{round(d2trend_arr[i], 4)}',
                  f'dI={round(di_cur, 4)} dg={dg_cur} dg_max={round(dg_max_arr[i], 4)} -> {des_cur}')
        # Переключаемся на новый, если в том есть необходимость
        if alg_new_id >= 0 and dg_cur < 0 and rating_cur[alg_new_id] > dg_cur:
            alg_cur_id, alg_cur_dct['params'] = alg_new_id, alg_lst[alg_new_id]['params']
            h_cur = alg_cur_dct['h']
            if output_flag:
                print(f'Переключаемся на алгоритм {alg_new_id} c параметрами {alg_lst[alg_new_id]["params"]}, h={h_cur}',
                      f'который показал dg={rating_cur[alg_new_id]}')
            num_switches += 1

    dg_max_arr = np.array([max([alg['dg'][i] for alg in alg_lst]) for i in range(n_max)])

    t_ticks_arr = t_arr / 3600.
    p_ser = pd.Series(_p_arr, index=t_ticks_arr)
    diff_ser = pd.Series(diff_arr, index=t_ticks_arr)
    diff2_ser = pd.Series(diff2_arr, index=t_ticks_arr)
    dtrend_ser = pd.Series(dtrend_arr, index=t_ticks_arr)
    d2trend_ser = pd.Series(d2trend_arr, index=t_ticks_arr)
    k_p_ser = pd.Series(k_p_arr, index=t_ticks_arr)
    k_i_ser = pd.Series(k_i_arr, index=t_ticks_arr)
    k_d_ser = pd.Series(k_d_arr, index=t_ticks_arr)
    k_dd_ser = pd.Series(k_dd_arr, index=t_ticks_arr)
    dg_ser = pd.Series(dg_arr, index=t_ticks_arr)
    dg_max_ser = pd.Series(dg_max_arr, index=t_ticks_arr)
    di_ser = pd.Series(di_arr, index=t_ticks_arr)
    i_ser = pd.Series(np.cumsum(di_ser), index=t_ticks_arr)
    profit_ser = pd.Series(np.cumsum(dg_ser), index=t_ticks_arr)
    profit = profit_ser.iloc[-1]
    # if output_flag:
    print('Заработано:', profit, 'число переключений:', num_switches)
    return p_ser, diff_ser, dg_max_ser, dg_ser, profit_ser, di_ser, i_ser, k_p_ser, k_i_ser, k_d_ser, k_dd_ser


def calc_step(i, data_dct):
    """
    Один шаг алгоритма. Получаем на вход номер шага и данные.
    """
    id = data_dct['id']
    _, k_p, k_i, k_d = data_dct['params']
    p, mu, dmu, di, dg = data_dct['p'], data_dct['mu'], data_dct['dmu'], data_dct['di'], data_dct['dg']
    h, di_0, mu_min, mu_max = data_dct['h'], data_dct['di_0'], data_dct['mu_min'], data_dct['mu_max']
    state, des_str, history, dt = data_dct['state'], data_dct['des'], data_dct['history'], data_dct['dt']
    di_new, di_next = di[i], 0.
    # Производные прибыли dg для управления
    dg_diff = 0.
    dg_diff_prev = 0.


    if i==31:
        qq = 0.

    if i==2 and id==0:
        qq = 0.



    if i > 1:
        dg_diff = (dg[i-1] - dg[i-2]) / dt
    if i > 2:
        dg_diff_prev = (dg[i-2] - dg[i-3]) / dt
    dg_diff_diff = (dg_diff - dg_diff_prev) / dt
    dg_pos = 0.
    # Integral part
    integ_sum = 0.
    if i > 10:
        coef_arr = np.array([math.exp(float(-j) / 10.) for j in range(10, -1, -1)])
        for cnt in range(10, -1, -1):
            integ_sum += dg[i - cnt] * coef_arr[10 - cnt]
    # Double-differential part
    k_dd = 0.
    # Decision part
    if state == 'short opened':
        vol = math.fabs(di_new) / history[0]
        dg_pos = - (p[i] - history[0]) * vol
        if dg_pos > 0. or len(history) > 2:
            di_new = vol * p[i]
            des_str, state = 'Close short', 'none'
            k_p *= 1.
            dg_new = dg_pos
            if 0. < di_next < .001:
                di_next = di_0
            elif -.001 < di_next < 0.:
                di_next = -di_0
            else:
                di_next = .25 * h * (k_p * dg_new + k_i * integ_sum + k_d * dg_diff + k_dd * dg_diff_diff)
            history = []
        else:
            des_str, state = 'Hold', 'short opened'
            k_p *= 0.
            dg_new = 0.
            di_new, di_next = 0., di_new
            history.append(p[i])
    elif state == 'long opened':
        vol = math.fabs(di_new) / history[0]
        dg_pos = (p[i] - history[0]) * vol
        if dg_pos > 0. or len(history) > 2:
            di_new = - vol * p[i]
            des_str, state = 'Close long', 'none'
            k_p *= -1.
            dg_new = dg_pos
            if 0. < di_next < .001:
                di_next = di_0
            elif -.001 < di_next < 0.:
                di_next = -di_0
            else:
                di_next = .25 * h * (k_p * dg_new + k_i * integ_sum + k_d * dg_diff + k_dd * dg_diff_diff)
            history = []
        else:
            des_str, state = 'Hold', 'long opened'
            k_p *= 0.
            dg_new = 0.
            di_next = di_new
            di_new = 0.
            history.append(p[i])
    elif mu_min < math.fabs(mu[i]) < mu_max:
        # if di[i] * mu[i] > 0.:
        if di[i] > 0.:
            des_str, state = 'Open long', 'long opened'
            k_p *= 1.
        # elif di[i] * mu[i] < 0.:
        elif di[i] < 0.:
            des_str, state = 'Open short', 'short opened'
            k_p *= -1.
        else:
            qq = 1.
            print(id)
            print(f'step={i}, alg={id}, params={data_dct["params"]}, di[i]={di[i]}, di[i+1]={di[i+1]}, dg[i]={dg[i]}')
            print(h, data_dct['params'])
            raise ValueError('di = 0')
        dg_new = 0.
        di_next = - di_new
        history = [p[i]]
    else:
        des_str, state = 'Hold', 'none'
        k_p *= 0.
        dg_new = 0.
        di_next = di_new
        di_new = 0.
        history = []
    data_dct['des'], data_dct['state'], data_dct['history'] = des_str, state, history
    dg[i], di[i], di[i+1] = dg_new, di_new, di_next
    return


if __name__ == '__main__':
    data_tuple = load_test_data(step=15000)
    # print(data_tuple)
    # res_tpl = calc_alg5(data_tuple[0], output_flag=True)
    res_tpl = calc_hybrid_alg(data_tuple[4], time=24, output_flag=True)
    visualize(*res_tpl)

