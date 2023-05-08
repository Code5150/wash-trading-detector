import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getPrevTimeTrade(window, ts, a, b):
    _window = window[(window['a'] == b) & (window['b'] == a)]
    return _window.iloc[-1] if _window.shape[0] > 0 else None

def isDiffSmall(v1, v2):
    return (np.abs(v1 - v2) / 2) < ((v1 + v2) / 20)

def findCycle(probable_cycle: set, search_window, transaction):
    win = search_window[search_window[:, 2].astype(np.uint64) == transaction[1]]
    result = None
    for j in range(win.shape[0]):
        if isDiffSmall(transaction[3], win[j, 3]) and isDiffSmall(transaction[4], win[j, 4]):
            if win[j, 2] in probable_cycle:
                result = probable_cycle
            else:
                new_cycle = probable_cycle.copy()
                new_cycle.add(win[j, 2])
                result = findCycle(new_cycle, search_window, win[j])
            if result is not None:
                break
    return result

def prepareOrigDf(_orig, nrows):
    orig = _orig.copy(deep=True) 
    orig.drop(orig[['E', 'e', 't', 'M', 's', 'm']], axis=1, inplace=True)
    orig['p'] = MinMaxScaler().fit_transform(orig['p'].to_numpy().reshape(-1, 1))
    orig = orig.reindex(columns=['T','a','b','p','q'])
    #TODO: разобраться, почему схлопывание похожих транзагций делает датасет пустым
    #orig = orig.groupby(['T', 'a', 'b']).agg("sum").reset_index()
    return orig.iloc[:nrows]

def analyze(orig_df: pd.DataFrame, model, transaction_window_size=1000, cycle_threshold=1000 * 60 * 15, PTT_THRESHOLD=3):
    #np_dataset = np.empty((0, 5), dtype=np.int32)
    df = prepareOrigDf(orig_df, orig_df.shape[0])
    wash = np.zeros((df.shape[0]))
    val = df.values

    for i in range(df.shape[0]):
        same_traders = False
        close_time = False
        small_p_diff = False
        small_q_diff = False
        part_of_cycle = False

        start_index = i - transaction_window_size
        #TRansaction Window
        trw = val[(0 if start_index < 0 else start_index):i]
        #((trw[:, 0].astype(np.uint64) + PTT_THRESHOLD) >= row[1]) & 
        ptt = trw[(trw[:, 1].astype(np.uint64) == val[i, 2]) & (trw[:, 2].astype(np.uint64) == val[i, 1])] if trw.shape[0] > 0 else np.empty((0, 5))
        if ptt.shape[0] > 0:
            same_traders = True
            close_time = (ptt[-1, 0].astype(np.uint64) + PTT_THRESHOLD) >= val[i, 0]
            small_p_diff = isDiffSmall(val[i, 3], ptt[-1, 3])
            small_q_diff = isDiffSmall(val[i, 4], ptt[-1, 4])
        else:
            #looking For Cycles
            part_of_cycle = findCycle(set([val[i, 2]]), val[val[:, 0].astype(np.int64) >= cycle_threshold], val[i]) is not None     

        wash[i] = model.predict(np.array([same_traders, close_time, small_p_diff, small_q_diff, part_of_cycle], dtype=np.int32).reshape(1, -1))[0]

    orig_df['wash'] = wash
    return orig_df