## Methods to forecast data
# Developed for growth rate forecast

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from lstm_build import lstm_baseclass
import matplotlib.pyplot as plt
from keras.models import load_model
from collections import defaultdict
import seaborn as sns
import sys

class forecast_methods(lstm_baseclass):
    """"Forecasting methods"""

    def __init__(self,series,forecast_period=5,hist_period=10,val_period=5):
        self.series = series
        self.forecast_period = forecast_period
        self.hist_period = hist_period
        self.val_period = val_period
        self.scores = {}
        self.val_pred = {}
        self.factors = {}

    def constant_model(self):
        """Returns the vector of multiplication factors for the forecast period
        Uses the last growth rate value for constant prediction in forecast period
        """
        hist_rev = self.series.values[-1*self.hist_period::]

        past_N = float(hist_rev.shape[0])

        CAGR = ((hist_rev[-1]/hist_rev[0])**(1/(past_N-1)) - 1)

        # build CAGR for validation and comparisons with other methods
        self.CAGR_val = [CAGR]*self.val_period
        # First half
        # Factor is the vector of multiplication factors according to growth
        # rates.
        p = range(1,self.forecast_period+1)
        factor = [(1+CAGR)**k for k in p]

        # Update the factor vector to the global dict
        self.factors['constant_model'] = factor

        return factor,CAGR

    def regression_model_val(self,regression_period=10):
        """Returns the mse of validation over validation period.
            Uses linear regression
        """
        hist_rev = self.series[-1*regression_period::]
        diff = hist_rev.diff()

        # Calculate growth rate series
        gr_series = (1.*diff)/hist_rev
        gr_series = gr_series.dropna()
        gr = gr_series.values
        gr = gr.reshape(-1,1)

        X = np.arange(len(gr))
        X = X.reshape(-1,1)
        x_val = X[-1*self.val_period::,:]
        x_train = X[0:-1*self.val_period,:]

        y_val = gr[-1*self.val_period::,:]
        y_train = gr[0:-1*self.val_period,:]

        # Create linear regression model object
        regr = linear_model.LinearRegression()

        # train the model
        regr.fit(x_train,y_train)

        # Make predictions on val set
        y_val_pred = regr.predict(x_val)

        # evaluate metrics
        regression_mse = mean_squared_error(y_val,y_val_pred)
        r_squared = r2_score(y_val,y_val_pred)

        constant_model_mse = mean_squared_error(y_val,self.CAGR_val)

        # Update validation metrics
        self.val_pred['actual'] = y_val
        self.val_pred['regression_pred'] = y_val_pred
        self.val_pred['constant_model_pred'] = self.CAGR_val

        self.scores['regression_model'] = regression_mse
        self.scores['constant_model'] = constant_model_mse

        return

    def regression_model(self,regression_period=10):
        """Returns the multplication factor for grow rate using linear regression"""

        hist_rev = self.series[-1*regression_period::]
        diff = hist_rev.diff().shift(-1)

        # Calculate growth rate series
        gr_series = (1*diff/hist_rev)
        gr_series = gr_series.dropna()
        gr_series.plot()
        gr = gr_series.values
        gr = gr.reshape(-1,1)
        X = np.arange(len(gr)+self.forecast_period)
        X = X.reshape(-1,1)
        x_train = X[0:-1*self.forecast_period,:]
        x_forecast = X[-1*self.forecast_period::,:]

        y_train = gr

        # Create linear regression model object
        regr = linear_model.LinearRegression()

        # train the model
        regr.fit(x_train,y_train)

        # Make predictions on val set
        gr_forecast = regr.predict(x_forecast)[:,0]
        # initialize factor list
        factor = [1]

        for rate in gr_forecast:
            factor.append((1+rate)*factor[-1])

        # Remove the first element which is 1
        factor.pop(0)

        # Update the factor vector to the dict
        self.factors['regression_model'] = factor

        return factor

    def lstm_model(self,look_up_period = 20):
        """Returns mse and r2 for lstm model validation"""

        hist_rev = self.series[-1*look_up_period::]
        diff = hist_rev.diff().shift(-1)

        # Calculate growth rate series
        gr_series = (1*diff/hist_rev)
        gr_series = gr_series.dropna()
        series = hist_rev

        # configure
        n_lag = 2
        #n_seq = self.forecast_period+1
        n_seq_options = range(1,self.forecast_period+1+1)
        n_test = 1
        n_epochs = 10
        n_batch = 1
        n_neurons = 1

        forecasts_dict = defaultdict(list)
        validation_forecasts = []

        for i,n_seq in enumerate(n_seq_options):

            # prepare data
            scaler, train, test,pred_X = self.prepare_data(series, n_test, n_lag, n_seq)

            # fit model
            #model = self.fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
            self.fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
            model = load_model("model.hdf5")
            # make forecasts
            curr_forecasts = self.make_forecasts(model, n_batch, train,test, n_lag, n_seq)
            test_forecasts = [self.forecast_lstm(model,pred_X,n_batch)]
            # inverse transform forecasts and test
            curr_forecasts = self.inverse_transform(series, curr_forecasts, scaler, n_test+n_seq)
            # Create a list of validation forecasts for plotting
            validation_forecasts.append(curr_forecasts[0])

            test_forecasts = self.inverse_transform(series, test_forecasts, scaler, 0)

            actual = [row[n_lag:] for row in test]
            actual = self.inverse_transform(series, actual, scaler, n_test+n_seq)
            # evaluate forecasts
            #self.evaluate_forecasts(actual, curr_forecasts, n_lag, n_seq)
            # plot forecasts
            #self.plot_forecasts(series, curr_forecasts, n_test+n_seq-1)

            for j,frcst in enumerate(test_forecasts[0]):
                forecasts_dict[j].append(frcst)

            #print(forecasts_dict)

        forecasts = [np.mean(np.array(forecasts_dict[k])) for k in forecasts_dict.keys()]

        # Plot all validation simultaneously
        self.plot_all_forecasts(series, validation_forecasts, n_test,n_seq_options)

        # Validation of growth rates
        # This method calculates the revenue forecasts while others forecast
        # growth rates

        #forecast_grw_rate = np.divide(np.ediff1d(forecasts[0]),forecasts[0][0:-1])
        forecast_grw_rate = np.divide(np.ediff1d(forecasts),forecasts[0:-1])

        lstm_mse = mean_squared_error(self.val_pred['actual'],forecast_grw_rate)

        # Update validation metrics
        self.val_pred['lstm_pred'] = forecast_grw_rate
        self.scores['lstm_model'] = lstm_mse

        # plot validation results
        plt.plot(range(len(self.val_pred['actual'])),self.val_pred['regression_pred'],label='Regression')
        plt.plot(range(len(self.val_pred['actual'])),self.val_pred['constant_model_pred'],label='Constant_model')
        plt.plot(range(len(self.val_pred['actual'])),self.val_pred['lstm_pred'],label='LSTM')
        plt.plot(range(len(self.val_pred['actual'])),self.val_pred['actual'],label='actual')
        plt.title("Validation Comparison on Growth Rate")
        plt.ylabel("Growth Rate")
        plt.xlabel("Years")

        plt.legend()
        #plt.show()
        plt.savefig("comparison.png")
        plt.clf()

        print(self.scores)

        #factor = forecasts[0]/forecasts[0][0]
        factor = forecasts/forecasts[0]
        factor = factor[1::]

        # Plot the final forecast plot
        last_rev_val = series.values[-1]
        frcstd_rev_lstm = last_rev_val*factor
        x_hist_rev = range(len(series.values))
        x_frcst_rev = range(len(series.values),len(series.values)+len(frcstd_rev_lstm))
        plt.plot(x_hist_rev,series.values,label='Historical')
        plt.plot(x_frcst_rev,frcstd_rev_lstm,label='LSTM')
        plt.plot(x_frcst_rev,last_rev_val*np.array(self.factors['regression_model']),
        label='Regression')
        plt.plot(x_frcst_rev,last_rev_val*np.array(self.factors['constant_model']),
        label='Constant Growth')
        plt.title("Forecasted Revenue")
        plt.ylabel("Millions")
        plt.xlabel("Years")
        plt.legend()
        #plt.show()
        plt.savefig("final_forecast.png")
        plt.clf()

        self.factors['lstm_model'] = factor
        print(self.factors)

        return factor

    def select_model(self):
        """Returns the factor vector for the best mse value"""
        best_model = min(self.scores,key=self.scores.get)
        return self.factors[best_model]


if __name__ == '__main__':

    df_stock = pd.read_csv("aapl.csv",header=0,index_col=0)
    series = df_stock.loc['revt'][-30::]
    print(len(series))

    fm = forecast_methods(series)
    print(fm.constant_model())
    print(fm.regression_model_val())
    print(fm.regression_model())
    fm.lstm_model()
    print(fm.select_model())
