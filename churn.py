import numpy as np
from numpy.core.numeric import ones_like
import scipy as sp
import scipy.fftpack
import pandas as pd
from datetime import date, datetime, timedelta
import random
import matplotlib.pyplot as plt
from common import process_merchant_avg_cents_hour_of_day, make_df, percentile_bins

df = make_df()
first_day = datetime(2033, 1, 1)
last_day = datetime(2034, 12, 31)

by_merchant = df.groupby('merchant')

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

def ts_to_diffs(t):
    diffs = np.diff(t)
    return diffs

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

def histogram(transactions, id_merchant, threshold=0, bins=200):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.hist(transactions, log=True, bins=bins)
    ax.set_xlabel("Number of Transactions")
    ax.set_ylabel("Counts")
    ax.set_title("Distribution of Number of Transactions for Merchant {}".format(hex(id_merchant)[2:]))
    ax.axvline(threshold, color='r')

def histogram_no_sales(transactions, id_merchant, threshold=0, bins=200):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.hist(transactions, log=True, bins=bins)
    ax.set_xlabel("Number of No Sale Days")
    ax.set_ylabel("Counts")
    ax.set_title("How Often Does Merchant {} Not Have Sales?".format(hex(id_merchant)[2:]))
    ax.axvline(threshold, color='r')

def histogram_no_sales_mult(bin_edges, vals, bins=200):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.bar(bin_edges, vals, bins=bins)
    ax.set_xlabel("Number of No Sale Days")
    ax.set_ylabel("Counts")
    ax.set_title("How Often Do Merchants Not Have Sales?")
    
    
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


# print(first_day, last_day)
# id_merchant = int(0x005e8bb6fb)
# single_merchant = by_merchant.get_group(id_merchant)
# t_days, transactions = process_merchant_to_time_series(single_merchant)

# threshold = get_threshold(transactions)
# # thresholded_transactions = threshold_transactions(transactions, threshold)
# bool_transactions = get_boolean_transactions(transactions, threshold)
# i_non_zero = np.nonzero(bool_transactions)
# i_diff_non_zero = np.diff(i_non_zero[0])  


# print(i_non_zero)
# print(i_diff_non_zero)

# plot_merchant(t_days, transactions, id_merchant)
# plot_merchant_bool(t_days, transactions, bool_transactions, id_merchant)
# histogram_no_sales(i_diff_non_zero, id_merchant)
# histogram(transactions, id_merchant, threshold=threshold)
# diffs_non_zero = i_diff_non_zero

container = []
bin_edge_container = []

num_samples = 100
# merchant_sample = pd.concat( [ by_merchant.get_group(group) for i,group in enumerate( by_merchant.groups) if i < num_samples ] ).groupby('merchant') 
merchant_ids = [0x4e0e5fd73, 0xc6ae1f908f, 0x005e8bb6fb]

for id_merchant in merchant_ids:
    single_merchant = by_merchant.get_group(id_merchant)

    single_merchant = by_merchant.get_group(id_merchant)
    t_days, transactions = process_merchant_to_time_series(single_merchant)

    threshold = get_threshold(transactions)
    thresholded_transactions = threshold_transactions(transactions, threshold)
    bool_transactions = get_boolean_transactions(transactions, threshold)
    i_non_zero = np.nonzero(bool_transactions)
    i_diff_non_zero = np.diff(i_non_zero[0])  # Pad to capture a string of zeros at the end of the array
    print(i_diff_non_zero.shape)
    
    hist, bin_edges = np.histogram(i_diff_non_zero, bins=200)
    container.append(i_diff_non_zero)
    bin_edge_container.append(bin_edges)

diffs_non_zero = np.concatenate(container)
print(diffs_non_zero.shape)
x, dx, y = percentile_bins(np.log10(diffs_non_zero + np.random.uniform(-0.01,0.01,len(diffs_non_zero))), np.ones_like(diffs_non_zero), n_bins=20)
print(x)

plt.figure()
plt.plot(x, y/dx)
plt.show()



