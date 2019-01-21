import sys
sys.path.append("..")
from sample.CustomerDataContainer import CustomerDataContainer

split_date_b = 19970101 # Jan 1997
split_date_e = 19971001 # Oct 1997

cdc = CustomerDataContainer()
# Test load_data method with normal data
cdc.load_data('../csv/all_transactions.csv')
# test load data with empty file
# cdc.load_data('all_transactions_empty.txt')
# test load data with a string in a second column
# cdc.load_data('all_transactions_strings.txt')
# test file with 3 columns instead of 2
# cdc.load_data('all_transactions_3columns.txt')
# train validation split
df_train, df_val = cdc.Train_Val_split(split_date_b,split_date_e)
print(df_train.shape)
print(df_train.head())
# test computations of x
cdc.compute_x()
# test computation of tx and T
cdc.compute_tx_T(split_date_e)
# get the dataset back
df = cdc.get_fit_df()
print(df.shape)
print(df.head())
# save to csv
cdc.save_to_csv(filename='../csv/test_out.csv')