
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from input_params import get_inp_params
from forecast_methods import forecast_methods
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

        self.time_period = time_period
        self.time_period_half = int(time_period/2.)

    def revenue(self,hist_period = 20, reduction_factor=0.75):
        """Returns the projected revenue for 2*time_period years
            For 2nd order: Growth rate constant for the first half and
            decreases linearly to rf rate
        """

        """
        hist_rev  = self.i.loc['revt'].values
        hist_rev = hist_rev[-1*hist_period::]

        hist_period = np.array(self.i.columns)[-1*hist_period::]

        past_N = float(hist_period.shape[0])

        CAGR = ((hist_rev[-1]/hist_rev[0])**(1/past_N) - 1)*reduction_factor

        # First half
        # Factor is the vector of multiplication factors according to growth
        # rates.
        p = range(1,self.time_period_half+1)
        factor = [(1+CAGR)**k for k in p]
        """

        hist_rev  = self.i.loc['revt'].values
        hist_rev = hist_rev[-1*hist_period::]
        series = self.i.loc['revt']

        print("Series Length:%g"%len(series))

        fm = forecast_methods(series,forecast_period=self.time_period_half,
                                hist_period=hist_period,val_period=5)
        # Run constant_model
        _,CAGR = fm.constant_model()
        # Run regression model validation
        fm.regression_model_val()
        # Run regression on all data
        fm.regression_model()
        # run LSTM model
        fm.lstm_model()
        # Select the best model for factor vector
        factor = list(fm.select_model())

        # Second half
        # Growth rate decreases from CAGR to rf_rate
        rf_rate = self.inp_params['rf_rate']
        grw_rate = np.linspace(CAGR,rf_rate,self.time_period_half+1)[1::]
        for r in grw_rate:
            factor.append((1+r)*factor[-1])

        rev = hist_rev[-1]*np.array(factor)
        return rev

    def oper_income(self,revenue):
        """Returns the vector of operating income data_projections
            Operating margin is reduced to the target_ebit"""

        ebit_ini=self.inp_params['ebit']
        oper_margin_ini = ebit_ini/self.i.loc['revt'].values[-1]
        print(oper_margin_ini)
        print(ebit_ini)
        print(self.i.loc['revt'].values[-1])
        oper_margin_vec = np.linspace(oper_margin_ini,
                            self.inp_params['target_ebit'],self.time_period+1)[1::]

        ebit = np.multiply(revenue,oper_margin_vec)

        return ebit

    def income_after_taxes(self,ebit):
        """Returns the after tax income vector"""

        tax_rate_ini = self.inp_params['eff_tax_r']
        marg_tax_r = self.inp_params['marg_tax_r']

        tax_vec = np.linspace(tax_rate_ini,marg_tax_r,self.time_period+1)[1::]

        ebit_after_tax = ebit - np.multiply(ebit,tax_vec)
        return ebit_after_tax

    def reinvestment(self,revenue,margin=1.4):
        """Returns the reinvestment vector"""
        cap_inv = self.b.loc['seq'] + self.b.loc['dltt'] - self.b.loc['che']
        delta_cap_inv = cap_inv.diff()

        delta_rev_hist = self.i.loc['revt'].diff()

        sales_to_cap_hist = delta_rev_hist/delta_cap_inv
        sales_to_cap_hist = sales_to_cap_hist.values.tolist()[-5::]
        sales_to_cap_hist = [k for k in sales_to_cap_hist if k>=0]

        sales_to_cap = np.mean(sales_to_cap_hist)/margin

        last_rev = self.i.loc['revt'].values[-1]
        revenue = np.insert(revenue,0,last_rev)
        delta_rev = np.ediff1d(revenue)

        reinvestment_vec = delta_rev/sales_to_cap

        return reinvestment_vec

    def terminal_cash_flow(self,revenue,roic):
        """Returns the terminal Cash Flow"""
        grw_r_dflt = self.inp_params['stable_growth_rate_default']
        term_rev = revenue[-1]*(1+grw_r_dflt)
        term_ebit = term_rev*self.inp_params['target_ebit']
        term_after_tax_income = term_ebit*(1-self.inp_params['marg_tax_r'])

        if roic > 0:
            term_reinv = term_after_tax_income*(grw_r_dflt/roic)
        else:
            term_reinv = 0

        terminal_cashF_flow = term_after_tax_income - term_reinv

        terminal_value = terminal_cashF_flow/ \
            (self.inp_params['stable_cc_default'] - self.inp_params['stable_growth_rate_default'])

        return terminal_value
