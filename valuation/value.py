""" Computes the valuation of the company based on the cashflows and other
    options """

import pandas as pd
import numpy as np
import sys
sys.path.append("..")

from data.scripts.simplified_finance_stats.fin_stats import fin_stats
from data.scripts.simplified_finance_stats.fin_stats_2 import fin_stats_2
from data.scripts.simplified_finance_stats.stock_stats import stock_stats
from data_projections import data_projections
from input_params import get_inp_params
from reportlab.pdfgen import canvas

class valuation(object):
    """ Computes the intrinsic value of the company.

        Uses input parameters from input_params.py.


    """

    def __init__(self,finances=None,fin_others=None,mkt_data=None):
        """ Instantiates using inp_parameters (from inp_params.py) and fcf.
        fcf is a dict containing non-terminal and terminal cashflows"""

        # If dataset if not loaded already, load the dataset
        if finances == None:

            print("Dataset not loaded, loading the dataset...")

            # Set path for data
            base_path = '../data/'
            sheets_path = 'combined_simplified/combined_all_us.csv'
            other_path = 'combined_simplified/others_all_us.csv'
            mkt_path = 'combined_simplified/stock_stats_all_us.csv'

            # setup all data
            self.finances = fin_stats(base_path + sheets_path)
            self.fin_others = fin_stats_2(base_path + other_path)
            self.mkt_data = stock_stats(base_path + mkt_path)

    def setup_data(self,tick):
        """Sets up data for one tick symbol"""
        self.inp_params = get_inp_params(tick,self.finances,self.fin_others,self.mkt_data)
        time_period = 10
        self.n_years = time_period

        self.dp = data_projections(tick,self.finances,self.fin_others,self.mkt_data,time_period)

        self.stable_cc_default = .085
        self.tick = tick
        return



    def cost_of_capital(self,terminal_cc=None):
        """ Returns the vector for cost of capital for each year in the fcf"""

        years = np.linspace(0,self.n_years-1,self.n_years)

        # cost of capital
        init_cc = self.inp_params['cost_of_capital']
        if terminal_cc == None:
            terminal_cc = self.inp_params['stable_cc_default']

        # initialize cost of capital vector
        cc_vector = np.ones(self.n_years)*init_cc

        # linearly regress from mid point to achieve target or terminal cost of capital

        n2 = int(np.floor((len(years)/2.)+1))
        cc_2_half = np.linspace(init_cc,terminal_cc,n2)
        cc_vector[-n2::] = cc_2_half

        return cc_vector


    def cum_discount_factor(self):
        """ Returns the vector of cumulative discount factors for every year"""

        cc_vector = self.cost_of_capital(self.stable_cc_default)

        # Initialize cumulative discount factor vector
        cdf = np.ones(self.n_years)

        for i in range(self.n_years):
            if i == 0:
                cdf[i] = 1./(1+cc_vector[0])
            else:
                cdf[i] = cdf[i-1]*(1./(1. + cc_vector[i]))

        return cdf


    def pv_cf(self):
        """ Returns the present value of sum of all cashflows"""

        cc_vector = self.cost_of_capital(self.stable_cc_default)
        cdf = self.cum_discount_factor()

        rev = self.dp.revenue()
        ebit = self.dp.oper_income(rev)
        ebit_at = self.dp.income_after_taxes(ebit)
        reinv = self.dp.reinvestment(rev)

        #print("Rev")
        #print(rev)
        #print("EBIT")
        #print(ebit)
        #print("EBIT_After Tax")
        #print(ebit_at)
        #print("Reinvestment")
        #print(reinv)
        #print(np.divide(reinv,rev))

        non_terminal_fcf = ebit_at - reinv

        #print(non_terminal_fcf)
        print(self.inp_params['eff_tax_r'])


        # total present value of all future cash flows excluding terminal cash flow
        pv_fcf = np.dot(non_terminal_fcf,cdf)

        # terminal Cashflow
        terminal_value = self.dp.terminal_cash_flow(rev,.14)

        pv_terminal_value = terminal_value*cdf[-1]

        # sum  of all present value
        sum_pv = pv_terminal_value + pv_fcf

        return sum_pv

    def val_eq(self):
        """ Returns the value of common equity per share and price as a % of vlaue """

        # sum of present values of all cash flows
        sum_pv = self.pv_cf()

        # Probability of failure
        fail_prob = self.inp_params['failure_prob']
        # Proceeds if firm fails
        proceeds = self.inp_params['proceeds']*self.inp_params['val_of_proceeds']

        # Value of operating assets
        val_oa = sum_pv*(1. - fail_prob)

        # Value of equity
        val_eq = val_oa - self.inp_params['bk_val_debt'] + self.inp_params['cash_eq']

        # Value of options
        #val_op = options_value()
        val_op = 643.8

        # common equity value or intrinsic value
        val_eq_common = val_eq  - val_op

        # common equity value per share
        val_per_share = val_eq_common/self.inp_params['outstanding_shares']

        # Price as a % of value
        p_to_val = self.inp_params['curr_stock_price']/val_per_share

        return val_per_share,p_to_val

    def write_pdf(self,c,img1,img2,img3,v,p):
        c.drawString(250,800,"Intrinsic Valuation Report : %s"%self.tick)
        c.drawString(50,760,"Intrinsic Value Per Share: %g"%v)
        c.drawString(50,740,"Market Value as a percentage of Intrinsic Value: %g"%p)
        c.drawString(50,700,"Intrinsic value calculation is based on Discounted Cash Flow model which requires")
        c.drawString(50,680,"revenue forecasting. The plots on revenue forecating and validation are given below.")

        c.drawInlineImage(img1,50,450, width=250,height=200)
        c.drawInlineImage(img2,300,450, width=250,height=200)
        c.drawInlineImage(img3,50,220, width=250,height=200)


if __name__ == '__main__':

    # Test
    v = valuation()

    tick_list = ['AAPL','BAC','WFC','MSFT','GOOGL']
    #tick_list = ['AAPL']
    for tick in tick_list:
        v.setup_data(tick)

        v1,p1 = v.val_eq()

        print(v1,p1)

        # Write PDF
        c = canvas.Canvas("%s Value.pdf"%tick)
        comparison_img = 'comparison.png'
        final_frcst_img = 'final_forecast.png'
        all_frcst_lstm = 'all_forecast_lstm.png'
        v.write_pdf(c,final_frcst_img,comparison_img,all_frcst_lstm,v1,p1)
        c.showPage()
        c.save()
