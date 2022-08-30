import numpy as np
import pandas as pd

def percentile_bins(x, y=None, n_bins=0):
    X = np.asarray(x)
    n_bins = int(round( n_bins or max( 0.5 * X.size ** 0.5, n_bins ) ))
    i_percentiles = np.linspace(0,len(X)-1,n_bins+1).astype(np.int) 

    ix_sorted = np.argsort(X)
    x_bin_edges = X[ix_sorted][i_percentiles]

    dx_bins = np.diff(x_bin_edges)
    xc_bins = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])

    nx_samples = np.diff(i_percentiles)

    if y is None:
        return xc_bins, dx_bins
    else:
        Y = np.asarray(y)
        y_binned = np.diff( np.cumsum(Y[ix_sorted])[i_percentiles] )
        return xc_bins, dx_bins, y_binned

def process_merchant_avg_cents_hour_of_day(single_merchant):
    
    single_merchant = single_merchant.set_index(['time'])
    id_merchant = single_merchant['merchant'][0]
    hour_3 = pd.DataFrame()
    hour_3['cents'] = single_merchant['cents'].resample('3H').sum()
    hour_3.insert(1, "merchant", id_merchant)
    hour_3 = hour_3.reset_index('time')
    hour_3['weekDay'] = hour_3['time'].dt.weekday
    hour_3['hour'] = hour_3['time'].dt.hour

    grouped_single = hour_3.groupby('merchant').apply(filter_weekday_hour)
    grouped_single = grouped_single.reset_index('hour')
    grouped_single = grouped_single.groupby('hour').agg({'cents':'mean'}).reset_index('hour')
    grouped_single.insert(0, "merchant", id_merchant)
    return grouped_single

def filter_weekday_hour(g):
    by_weekday_hour = g.groupby(['weekDay', 'hour'])
    weekday_sum = by_weekday_hour.agg({'cents':'sum'})
    return weekday_sum

def make_df():
    df = pd.read_csv('takehome_ds_written.csv')

    df['time']= pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={"amount_usd_in_cents":"cents"})
    df = df.drop(['Unnamed: 0'], axis=1)
    df['merchant'] = df['merchant'].apply(int, base=16)
    cols = ['time', 'merchant', 'cents']
    df = df[cols]
    return df