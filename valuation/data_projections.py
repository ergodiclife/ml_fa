
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from input_params import get_inp_params
import sys

class data_projections(object):
    """ Builds the future projections of features like revenue, reinvestment, fcf"""

    def __init__(self,tick,finances,fin_others,mkt_data,time_period=10):
        """Builds the fundamental data"""

        # Get the fundamental data
        self.b = finances.get_sheet(tick,"balance_sheet")
        self.i = finances.get_sheet(tick,"income_sheet")
        self.c = finances.get_sheet(tick,"cashflow_sheet")
        self.o = fin_others.get_sheet(tick)
        self.mk = mkt_data.get_stock_data(tick)

        self.inp_params = get_inp_params(tick,finances,fin_others,mkt_data)

        self.time_period = time_period/2.

    def revenue_projection(hist_period = 20, time_period=5, order=2,reduction_factor=0.75):
    """Returns the projected revenue for 2*time_period years
        For 2nd order: Growth rate constant for the first half and
        decreases linearly to rf rate"""

        if order == 2:
            hist_rev  = self.i.loc['revt'].values
            hist_rev = hist_rev[-20::]

            hist_period = np.array(self.i.columns)[-20::]

            past_N = float(hist_period.shape[0])

            CAGR = ((hist_rev[-1]/hist_rev[0])**(1/past_N) - 1)*reduction_factor

            # First half
            p = range(1,time_period+1)
            factor = [(1+CAGR)**k for k in p]

            # Second half
            rf_rate = self.inp_params['rf_rate']
            grw_rate = np.linspace(CAGR,rf_rate,time_period+1)[1::]
            for r in grw_rate:
                factor.append((1+r)*factor[-1])

            rev = hist_rev[-1]*np.array(factor)
            return rev

        def oper_income(self,revenue,ebit=self.inp_params['ebit']):
            """Returns the vector of operating income data_projections
                Operating margin is reduced to the target_ebit"""

            oper_margin_ini = ebit/self.iloc['revt'].values[-1]
            oper_margin_vec = np.linspace(oper_margin_ini,
                                self.inp_params['target_ebit'],)
