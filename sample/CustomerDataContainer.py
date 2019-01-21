import pandas as pd
import numpy as np
import datetime as dt
import warnings
import sys

class CustomerDataContainer:
    def __init__(self, csv = ''):
        """Constructor
        """
        if csv: self.load_data(csv)
        else: self.df = pd.DataFrame()

    
    def load_data(self, csv):
        """Load the user data into the pandas DataFrame

        Arguments:
        csv - path to the csv file or buffer (object with the read() method)
        """
        try:
            self.df = pd.read_csv(csv,names=['ID','date'])
        except NameError:
            raise NameError("File not found.")
        except pd.errors.EmptyDataError:
            print("No columns to parse from file, file is completely empty. Terminate")
            sys.exit()
        except Exception as e:
            print("Error in reading file: {}".format(csv))
            raise e
        # validate the input data
        self.__validate_data()
        # drop duplicates 
        self.__drop_duplicates()

    def __validate_data(self):
        """Check the validity of the input Data
        """
        # check for empty .csv file
        if self.df.empty: raise ValueError("Input data is empty")
        # check columns
        self.__validate_columns()
        # check for NaNs
        if self.df.isna().values.any():
            # remove the rows 
            self.df.dropna()
            # raise a warning
            warnings.warn("Input data contains NaNs. Corresponding samples are not considered")
    
    def __validate_columns(self):
        """Check the input columns
        """
        # check the number of columns
        if len(self.df.columns) != 2:
            raise ValueError("Input data should have two columns: iD and date")
        # check that type of data is int
        for col in self.df.columns:
            if self.df[col].dtype != np.int64:
                raise ValueError("Input data values should be of type int64")

    def __drop_duplicates(self):
        """Drop duplicated transactions
        """
        self.df.drop_duplicates(inplace=True)

    def Train_Val_split(self,training_date_b,training_date_e,use_train = True):
        """Splitting the df into the train and validation parts

        Arguments:
        training_date_b - the beginning of the training part YYYYMMDD
        training_date_e - the end of the training part YYYYMMDD
        use_train - store train part of the dataset (default: True) TODO: change!
        """
        # TODO: try except if the dates are not in the input data or outside of the range!!!
        # training part
        df_train = self.df[(self.df.date >= training_date_b) & (self.df.date < training_date_e)]
        # rest is our validation part
        df_val = self.df[self.df.date >= training_date_e]
        if use_train: self.df = df_train.copy()
        return df_train, df_val

    def compute_x(self):
        """Compute number of transactions done by customer: x
        """
        self.df['x'] = self.df.groupby('ID')['ID'].transform(lambda s: s.count() - 1)
        return self.df['x']

    def compute_tx_T(self,train_date_e):
        """Compute duration in weeks between customer's last and first transaction: tx
        and Duration in weeks between end of calibration period and the first customer's transaction: T

        Arguments:
        train_date_e - end of the training period
        """
        # add datetime column for calcaultions
        self.__to_datetime()
        # create temp df with first and last date columns
        df_temp = self.df.datetime.groupby(self.df['ID']).agg(['first','last'])
        # compute tx
        self.__compute_tx(df_temp)
        # compute T
        self.__compute_T(df_temp,train_date_e)
        # merge into the input df 
        self.df = pd.merge(self.df,df_temp.drop(columns=['first','last','tx_days','T_days']),on=['ID'])

    def __compute_tx(self,df_temp):
        """Compute duration in weeks between customer's last and first transaction: tx
        """
        # find the difference between the transactions in days
        df_temp['tx_days'] = (df_temp['last'] - df_temp['first'])
        # convert it to weeks and round to 2 decimals
        df_temp['tx'] = (df_temp['tx_days']/np.timedelta64(1,'W')).round(2)

    def __compute_T(self,df_temp,split_date_e):
        """Compute duration in weeks between end of calibration period and the first customer's transaction: T
        """
        df_temp['T_days'] = dt.datetime.strptime(str(split_date_e), '%Y%m%d') - df_temp['first'] - np.timedelta64(1,'D')# -1 to take the last day into account
        # convert it to weeks and round to 2 decimals
        df_temp['T'] = (df_temp['T_days'] / np.timedelta64(1,'W')).round(2)

    def get_fit_df(self,full=False,unique=True):
        """Return the DataFrame prepared for the fit.

        Arguments:
        full - whether return full dataframe or only columns needed for the fit (ID,x,tx,T)
        (default = False)
        unique - return only unique customers (default True)
        """
        # only unique customers should be stored for the future fitting
        df_out = self.df.drop_duplicates(subset=['ID']) if unique else self.df
        # return the df
        return df_out.reset_index().drop(columns=['date','datetime','index']) if not full else df_out

    def save_to_csv(self,filename='csv/summary_customers.csv',full=False,unique=True):
        """Save the df into the csv file.

        Arguments:
        filename - output filename (default: csv/summary_customers.csv)
        full - whether return full dataframe or only columns needed for the fit (ID,x,tx,T)
        (default = False)
        unique - return only unique customers (default True)
        """
        # TODO: implement try except to handle the permission denies etc
        self.get_fit_df(full,unique).to_csv(filename)

    def __to_datetime(self):
        """Convert the input YYYYMMDD to a proper datetime format
        """
        self.df['datetime'] = pd.to_datetime(self.df['date'],format='%Y%m%d')

