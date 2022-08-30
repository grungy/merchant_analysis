import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
from common import process_merchant_avg_cents_hour_of_day, make_df

df = make_df()

def mostly_zero_filter(transactions, tol=0):
    # Count num zero and set threshold based on length of the time series.
    cnt_zero = np.sum(transactions <= tol)
    print("cnt_zero: ", cnt_zero)
    threshold = 0.9
    if cnt_zero >= (transactions.shape[0] * threshold):
        return True
    else:
        return False

def difference_filter(x, threshold=60):
    # Threshold in days
    print("Difference Filter")
    diffs = np.diff(x)
    mx_diff = np.max(diffs)
    if mx_diff > threshold:
        return True
    else:
        return False

def harmonic_vector(freqs, vals, fc, harms=[1, 2, 3, 4], rel_tol=0.05):
    harms = np.asarray(harms)
    fh_low = harms - rel_tol
    fh_high = harms + rel_tol
    
    vec_harmonic = np.array([])
    for hl,hh in zip(fh_low, fh_high):
        upper = (freqs >= (hl * fc))
        lower = (freqs <= (hh * fc))
        if(vals[upper & lower].shape[0] == 0):
            #FIXME set the harmonic to zero if it's below the noise floor. 
            vec_harmonic = np.append(vec_harmonic, np.array([0]))
        else:
            vec_harmonic = np.append(vec_harmonic, np.max(vals[upper & lower]))
    return vec_harmonic
    # return [np.max(vals[(freqs >= (hl * fc)) & (freqs <= (hh * fc))]) for hl,hh in zip(fh_low, fh_high)]

def process_merchant(single_merchant):
    single_merchant = single_merchant.sort_values(['time'], ascending=True)
    first_of_year = datetime( single_merchant.agg({"time": "min"})[0].year, 1, 1 )
    t = single_merchant['time'] - first_of_year
    t_secs = t.dt.total_seconds().to_numpy()
    t_days = t_secs / (3600 * 24)

    single_merchant = single_merchant.set_index(['time'])

    df_hourly = pd.DataFrame()
    df_hourly['cents'] = single_merchant['cents'].resample('1H').sum()
    df_hourly.insert(1, "merchant", id_merchant)
    df_hourly = df_hourly.reset_index('time')

    transactions = df_hourly['cents'].to_numpy()
    
    first_of_year = datetime( df.agg({"time": "min"})[0].year, 1, 1 )
    t = df_hourly['time'] - first_of_year
    t_secs = t.dt.total_seconds().to_numpy()
    t_days = t_secs / (3600 * 24)

    if t_days.shape[0] < (2 * 7 * 24):
        return None
    

    # Calculate Day Frequencies
    dt = t_secs[1] - t_secs[0]
    
    amt_to_pad = scipy.fft.next_fast_len(transactions.shape[0])
    padded_transactions = np.pad(transactions, amt_to_pad, mode='wrap')
    transactions_fft = sp.fftpack.fft(padded_transactions)
    transactions_fft_mag = np.abs(transactions_fft)
    fftfreq = sp.fftpack.fftfreq(len(transactions_fft), 1. / 24)
        
    i_mf = fftfreq.shape[0]//2
    S_f_nn = np.s_[0:i_mf-1]
    X = fftfreq[S_f_nn]
    Y = transactions_fft_mag[S_f_nn]
    fc = 1

    vec_harmonic_day = harmonic_vector(X, Y, 1)
    
    if vec_harmonic_day is not None:
        norm_day = (vec_harmonic_day / transactions_fft_mag[0])
        if np.any(np.isnan(norm_day)):
            print("vec_harmonic_day NAN")
            print(vec_harmonic_day)
            raise Exception("vech harmonic nan")
    else:
        print(vec_harmonic_day)

    # Calculate Week Frequencies
    amt_to_pad = scipy.fft.next_fast_len(transactions.shape[0])
    padded_transactions = np.pad(transactions, amt_to_pad, mode='wrap')
    transactions_fft = sp.fftpack.fft(padded_transactions)
    transactions_fft_mag = np.abs(transactions_fft)
    fftfreq = sp.fftpack.fftfreq(len(transactions_fft), 1. / 24)
            
    i_mf = fftfreq.shape[0]//2
    S_f_nn = np.s_[0:i_mf-1]
    X = fftfreq[S_f_nn]
    Y = transactions_fft_mag[S_f_nn]
    fc = 1/7

    vec_harmonic_week = harmonic_vector(X, Y, 1/7)
    if vec_harmonic_week is not None:
        norm_week = (vec_harmonic_week / transactions_fft_mag[0])
        if np.any(np.isnan(norm_week)):
            print("vec_harmonic_week NAN")
            print(vec_harmonic_day)
            raise Exception("vech harmonic nan")
    else:
        print(vec_harmonic_week)
    
    # Calculate Month Frequencies
    
    transactions_fft = sp.fftpack.fft(transactions)
    transactions_fft_mag = np.abs(transactions_fft)
    fftfreq = sp.fftpack.fftfreq(len(transactions_fft), 1. / 24)
        
    i_mf = fftfreq.shape[0]//2
    S_f_nn = np.s_[0:i_mf-1]
    X = fftfreq[S_f_nn]
    Y = transactions_fft_mag[S_f_nn]
    fc = 1/30

    vec_harmonic = np.concatenate([norm_day, norm_week])
    return vec_harmonic

def process_merchant_plot(single_merchant):

    single_merchant = single_merchant.set_index(['time'])

    df_hourly = pd.DataFrame()
    df_hourly['cents'] = single_merchant['cents'].resample('1H').sum()
    df_hourly.insert(1, "merchant", id_merchant)
    df_hourly = df_hourly.reset_index('time')

    transactions = df_hourly['cents'].to_numpy()

    
    first_of_year = datetime( df.agg({"time": "min"})[0].year, 1, 1 )
    t = df_hourly['time'] - first_of_year
    t_secs = t.dt.total_seconds().to_numpy()
    t_days = t_secs / (3600 * 24)

    plt.plot(t_days , transactions, '.-')
    plt.xlabel("Days")
    plt.ylabel("Transaction Amount (cents)")
    plt.title("Transactions of Merchant {}".format(hex(id_merchant)[2:]))
    plt.show()
    

    dt = t_secs[1] - t_secs[0]
    
    transactions_fft = sp.fftpack.fft(transactions)
    transactions_fft_mag = np.abs(transactions_fft)
    fftfreq = sp.fftpack.fftfreq(len(transactions_fft), 1. / (24))
    
    i_mf = fftfreq.shape[0]//2
    S_f_nn = np.s_[0:i_mf-1]
    X = fftfreq[S_f_nn]
    Y = transactions_fft_mag[S_f_nn]
    fc = 1

    print("harmonic_vector")
    vec_harmonic = harmonic_vector(X, Y, 1)
    print(vec_harmonic)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(X, Y, '-')
    ax.set_xlabel('Frequency (Cycles / Day)')
    ax.set_ylabel('Magnitude')
    ax.set_title("Magnitude of Merchant {}".format(hex(id_merchant)[2:]))

    for i,c in enumerate(vec_harmonic, 1):
        ax.plot(i * fc, c, 'ro')

    plt.show()
    return None

def process_to_feature_vector(harmonic_vector, num_transactions, sum_transactions, bin_width=3):
    container = np.zeros(harmonic_vector.shape[0] + 2)
    idxs_hours = grpd_merchant['hour'].to_numpy() // bin_width
    cents = grpd_merchant['cents'].to_numpy()
    container[idxs_hours] = cents
    container = np.append(container, num_transactions)
    container = np.append(container, sum_transactions)
    df = pd.DataFrame({'merchant': grpd_merchant['merchant'][0], 'vector': [container]})
    return df    


by_merchant = df.groupby('merchant')
cnt_transactions = by_merchant.size().reset_index(name='cnt').sort_values(['cnt'], ascending=False)
grp_sum_transactions = by_merchant.sum().sort_values(['cents'], ascending=False)

# Top 200 merchants by num transactions
top200 = cnt_transactions[0:200]
print("top200")
print(by_merchant)

id_merchant = cnt_transactions.iloc[0]['merchant']
id_merchant = 11804140257
id_merchant = int(0x005e8bb6fb)

single_merchant = by_merchant.get_group(id_merchant)
vec_single_harmonic = process_merchant_plot(single_merchant)