import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
from common import process_merchant_avg_cents_hour_of_day, make_df

first_day = datetime(2033, 1, 1)
last_day = datetime(2034, 12, 31)

df = make_df()
by_merchant = df.groupby("merchant")

df_churned_merchants = pd.read_pickle("churned_merchants.pkl")
print(df_churned_merchants[0:20])

def get_threshold(transactions):
    from skimage.filters import threshold_otsu, threshold_local
    transactions = np.row_stack(( transactions, np.ones(transactions.shape)))
    thresh = threshold_otsu(transactions)
    binary = transactions > thresh
    return thresh

def threshold_transactions(transactions, threshold):
    transactions[transactions <= threshold] = 0
    return transactions

def get_boolean_transactions(transactions, threshold):
    bool_array = transactions > threshold
    bool_array[-1] = True
    return bool_array

def plot_merchant(t_days, transactions, id_merchant):
    fig, ax = plt.subplots()
    ax.plot(t_days , transactions, '.-')
    plt.xlabel("Days")
    plt.ylabel("Transaction Amount (cents)")
    plt.title("Transactions of Merchant {}".format(hex(id_merchant)[2:]))

def plot_merchant_bool(t_days, transactions, bool_transactions, id_merchant):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t_days , transactions, '.-')
    ax2.plot(t_days, bool_transactions)
    ax2.set_xlabel("Days")
    ax1.set_ylabel("Transaction Amount (cents)")
    ax1.set_title("Transactions of Merchant {}".format(hex(id_merchant)[2:]))
    ax2.set_title("Are There Sales Today?")
    ax2.set_ylabel("Yes (1) / No (0)")

def process_merchant_to_time_series(single_merchant):
    id_merchant = single_merchant['merchant'].iloc[0]
    df_end = pd.DataFrame([[last_day, id_merchant, 0]], columns=['time', 'merchant', 'cents'])
    df_beg = pd.DataFrame([[first_day, id_merchant, 0]], columns=['time', 'merchant', 'cents'])
    single_merchant = single_merchant.append([df_beg, df_end], ignore_index=True)
    
    # Resample the dataframe
    single_merchant = single_merchant.set_index(['time'])

    df_hourly = pd.DataFrame()
    df_hourly['cents'] = single_merchant['cents'].resample('1D').count()
    df_hourly.insert(1, "merchant", id_merchant)
    df_hourly = df_hourly.reset_index('time')

    transactions = df_hourly['cents'].to_numpy()
    first_of_year = datetime( df.agg({"time": "min"})[0].year, 1, 1 )
    t = df_hourly['time'] - first_of_year
    t_secs = t.dt.total_seconds().to_numpy()
    t_days = t_secs / (3600 * 24)
    return t_days, transactions


id_merchant = int(653416108)
single_merchant = by_merchant.get_group(id_merchant)
churned_merchant = df_churned_merchants.set_index("merchant").loc[id_merchant]
print(churned_merchant)
t_days, transactions = process_merchant_to_time_series(single_merchant)

threshold = get_threshold(transactions)
thresholded_transactions = threshold_transactions(transactions, threshold)
bool_transactions = get_boolean_transactions(transactions, threshold)
i_non_zero = np.nonzero(bool_transactions)
i_diff_non_zero = np.diff(i_non_zero[0])

plot_merchant_bool(t_days[0:-1], transactions[0:-1], bool_transactions[0:-1], id_merchant)
plt.show()