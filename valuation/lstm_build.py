import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from math import sqrt
from matplotlib import pyplot
from numpy import array
import sys


class lstm_baseclass(object):

	# convert time series into supervised learning problem
	def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = pd.concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg

	# create a differenced series
	def difference(self,dataset, interval=1):
		diff = list()
		for i in range(interval, len(dataset)):
			value = dataset[i] - dataset[i - interval]
			diff.append(value)
		return pd.Series(diff)

	# transform series into train and test sets for supervised learning
	def prepare_data(self,series, n_test, n_lag, n_seq):
		# extract raw values
		raw_values = series.values
		# transform data to be stationary
		diff_series = self.difference(raw_values, 1)
		diff_values = diff_series.values
		diff_values = diff_values.reshape(len(diff_values), 1)
		# rescale values to -1, 1
		scaler = MinMaxScaler(feature_range=(-1, 1))
		scaled_values = scaler.fit_transform(diff_values)
		scaled_values = scaled_values.reshape(len(scaled_values), 1)
		# transform into supervised learning problem X, y
		supervised = self.series_to_supervised(scaled_values, n_lag, n_seq)
		#supervised = series_to_supervised(raw_values.reshape(len(raw_values),1), n_lag, n_seq)
		supervised_values = supervised.values

		# split into train and test sets
		train, test, pred_X = supervised_values[0:-1*n_test], supervised_values[-1*n_test:], supervised_values[-1][-1*n_lag:]
		return scaler, train, test, pred_X

	def save_best_model(self,model,record=False):
		"""Saves the best model based on min loss"""
		if record:
			filepath = "model.hdf5"
			model.save(filepath)

	# fit an LSTM network to training data
	def fit_lstm(self,train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
		# reshape training into [samples, timesteps, features]
		X, y = train[:, 0:n_lag], train[:, n_lag:]
		X = X.reshape(X.shape[0], 1, X.shape[1])
		# design network
		model = Sequential()
		model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
		#model.add(LSTM(n_neurons,return_sequences=True,batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True,kernel_regularizer=L1L2(0.0,0.0)))
		#model.add(LSTM(n_neurons, stateful=True,kernel_regularizer=L1L2(0.0,0.0)))
		model.add(Dense(y.shape[1]))
		model.compile(loss='mean_squared_error', optimizer='adam')

		hist_loss = []

		# Initialize large loss values
		min_loss = 1e12
		curr_loss = 1e12
		record = False

		# fit network
		for i in range(nb_epoch):
			checkpoint = self.save_best_model(model,record)
			history = model.fit(X, y, epochs=1, batch_size=n_batch, callbacks=checkpoint, verbose=0, shuffle=False)

			model.reset_states()
			curr_loss = history.history['loss'][0]

			# Update min Loss
			if curr_loss < min_loss:
				min_loss = curr_loss
				record= True
			else:
				record = False

			hist_loss.append(curr_loss)

		pyplot.plot(range(len(hist_loss)),hist_loss)
		pyplot.title("Training Loss History")
		pyplot.xlabel("Epochs")
		pyplot.ylabel("Loss")
		#pyplot.show()
		pyplot.savefig("train_loss_hist.png")
		pyplot.clf()
		return model

	# make one forecast with an LSTM,
	def forecast_lstm(self,model, X, n_batch):
		# reshape input pattern to [samples, timesteps, features]
		X = X.reshape(1, 1, len(X))
		# make forecast
		forecast = model.predict(X, batch_size=n_batch)
		# convert to array
		return [x for x in forecast[0, :]]

	# evaluate the persistence model
	def make_forecasts(self,model, n_batch, train, test, n_lag, n_seq):
		forecasts = list()
		for i in range(len(test)):
			X, y = test[i, 0:n_lag], test[i, n_lag:]
			# make forecast
			forecast = self.forecast_lstm(model, X, n_batch)
			# store the forecast
			forecasts.append(forecast)
		return forecasts

	# invert differenced forecast
	def inverse_difference(self,last_ob, forecast):
		# invert first forecast
		inverted = list()
		inverted.append(forecast[0] + last_ob)
		# propagate difference forecast using inverted first value
		for i in range(1, len(forecast)):
			inverted.append(forecast[i] + inverted[i-1])
		return inverted

	# inverse data transform on forecasts
	def inverse_transform(self,series, forecasts, scaler, n_test):
		inverted = list()
		for i in range(len(forecasts)):
			# create array from forecast
			forecast = array(forecasts[i])
			forecast = forecast.reshape(1, len(forecast))
			# invert scaling
			inv_scale = scaler.inverse_transform(forecast)
			inv_scale = inv_scale[0, :]
			# invert differencing
			index = len(series) - n_test + i - 1
			last_ob = series.values[index]

			if n_test==0:
				last_ob = series.values[-1]

			inv_diff = self.inverse_difference(last_ob, inv_scale)
			# store
			inverted.append(inv_diff)
		return inverted

	# evaluate the RMSE for each forecast time step
	def evaluate_forecasts(self,test, forecasts, n_lag, n_seq):
		for i in range(n_seq):
			actual = [row[i] for row in test]
			predicted = [forecast[i] for forecast in forecasts]
			rmse = sqrt(mean_squared_error(actual, predicted))
			print('t+%d RMSE: %f' % ((i+1), rmse))

	# plot the forecasts in the context of the original dataset
	def plot_forecasts(self,series, forecasts, n_test):
		# plot the entire dataset in blue
		pyplot.plot(series.values)
		# plot the forecasts in red
		for i in range(len(forecasts)):
			try:
				off_s = len(series) - n_test + i - 1
				off_e = off_s + len(forecasts[i]) + 1
				xaxis = [x for x in range(off_s, off_e)]
				yaxis = [series.values[off_s]] + forecasts[i]
				#pyplot.plot(xaxis, yaxis, color='red')
				pyplot.plot(xaxis, yaxis,label='%i'%i)
				pyplot.legend()

			except:
				pass
		# show the plot
		pyplot.title("Forecast Validation of LSTM")
		pyplot.ylabel("Revenue in Millions")
		pyplot.xlabel("Years")
		#pyplot.show()
		pyplot.savefig("forecast_val_lstm.png")
		pyplot.clf()

	# plot all forecasts in the context of the original dataset together
	def plot_all_forecasts(self,series, forecasts, n_test,n_seq_options):
		# plot the entire dataset in blue
		pyplot.plot(series.values)
		# plot the forecasts in red
		for i in range(len(forecasts)):
			n_test_tmp = n_test + n_seq_options[i] -1
			try:
				off_s = len(series) - n_test_tmp - 1
				print(off_s)
				off_e = off_s + len(forecasts[i]) + 1
				xaxis = [x for x in range(off_s, off_e)]
				yaxis = [series.values[off_s]] + forecasts[i]
				#pyplot.plot(xaxis, yaxis, color='red')
				pyplot.plot(xaxis, yaxis,label='%i'%i)
				pyplot.legend()

			except:
				pass
		# show the plot
		pyplot.grid()
		pyplot.title("Validation of LSTM Forecast")
		pyplot.ylabel("Revenue in Millions")
		pyplot.xlabel("Years")
		#pyplot.show()
		pyplot.savefig("all_forecast_lstm.png")
		pyplot.clf()
