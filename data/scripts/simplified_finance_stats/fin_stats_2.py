# Fundamental Data
# Other financial data not included in fin_stats
import pandas as pd
import numpy as np

class fin_stats_2(object):
    """ Get fundamental data which is not included in balance,income or cashflow sheets.


        Methods:
                get_sheet()


    """

    def __init__(self,path):
        self.path = path
        self.df_fin_all = pd.read_csv(self.path)

        # Preprocess data

        # Remove rows with missing year information.
        missing_year = self.df_fin_all[self.df_fin_all['fyear'].isnull()].index.values.tolist()
        self.df_fin_all = self.df_fin_all.drop(self.df_fin_all.index[missing_year])
        self.df_fin_all = self.df_fin_all.reset_index(drop=True)

        self.df_fin_all = self.df_fin_all.set_index('fyear')
        self.df_fin_all = self.df_fin_all.fillna(0.)

        # Select financial statements variables
        self.others = ['re','wcapch','wcapc','unwcc','nim','citotal','cga',
                    'mrc1','mrc2','mrc3','mrc4','mrc5']

    def get_sheet(self,tickr):
        """ Returns data for other financials not included in balance,
        income or cash flow sheets """
        name = self.others

        # Subset company data set
        df_raw = self.df_fin_all[self.df_fin_all["tic"] == tickr]
        df_raw = df_raw[name].copy()

        # Check if tickr is not present
        if df_raw.shape[0] == 0:
            print("Tickr not found")
            return

        df_sheet = df_raw.transpose()
        df_sheet = df_sheet.reindex(name)
        df_sheet.columns = map(int,df_sheet.columns.tolist())

        return df_sheet
