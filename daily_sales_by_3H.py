import numpy as np
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
from common import process_merchant_avg_cents_hour_of_day, make_df

def filter_weekday_hour(g):
    by_weekday_hour = g.groupby(['weekDay', 'hour'])
    weekday_sum = by_weekday_hour.agg({'cents':'sum'})
    return weekday_sum

def process_merchant(single_merchant):
    
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
    return grouped_single

def process_to_feature_vector(grpd_merchant, num_transactions, sum_transactions, bin_width=3):
    container = np.zeros([8])
    idxs_hours = grpd_merchant['hour'].to_numpy() // bin_width
    cents = grpd_merchant['cents'].to_numpy()
    container[idxs_hours] = cents
    container = np.append(container, num_transactions)
    container = np.append(container, sum_transactions)
    df = pd.DataFrame({'merchant': grpd_merchant['merchant'][0], 'vector': [container]})
    return df

# make the dataframe
df = make_df()

merchant_transactions = df.groupby('merchant')
cnt_transactions = merchant_transactions.size().reset_index(name='cnt').sort_values(['cnt'], ascending=False)
cnt_transactions = cnt_transactions.set_index('merchant')

grp_sum_transactions = merchant_transactions.sum().sort_values(['cents'], ascending=False)
print(grp_sum_transactions)
grp_feature_vectors = pd.DataFrame()

for key, grp in merchant_transactions:
    single_merchant = grp
    id_merchant = key
    num_transactions = cnt_transactions.loc[id_merchant]
    sum_transactions = grp_sum_transactions.loc[id_merchant]
    grouped_single = process_merchant_avg_cents_hour_of_day(single_merchant)
    df_feature_vector = process_to_feature_vector(grouped_single, num_transactions, sum_transactions)
    grp_feature_vectors = grp_feature_vectors.append(df_feature_vector, ignore_index=True)

print(grp_feature_vectors)
grp_feature_vectors.to_pickle('feature_vectors_daily_profit_num_transactions_sum_transactions.pkl')