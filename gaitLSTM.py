# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 13:46:15 2017

@author: david
"""
# load and plot dataset
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from math import sqrt
from matplotlib import pyplot
from numpy import array
from pandas import concat
from pandas import DataFrame
from pandas import read_csv
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from time import time

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	uninverted = array(new_row)
	uninverted = uninverted.reshape(1, len(uninverted))
	inverted = scaler.inverse_transform(uninverted)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1]) # reshapes it from a 2D ndarray to a 3D ndarray
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# run a repeated experiment
def experiment(repeats, series, epochs, batch_size, neurons):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    # this splits the data into sets of roughly 80%/20% of the values, rounded down to a multiple of batch size
    split_point = int(0.8*len(supervised))
    split_point = split_point - split_point % batch_size
    # define the end_point as being the end of a range from split_point where it is a multiple of batch size
    end_point = len(supervised)
    end_point = end_point - (end_point - split_point ) % batch_size
    train, test = supervised_values[0:split_point], supervised_values[split_point:end_point]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the model
        lstm_model = fit_lstm(train_scaled, batch_size, epochs, neurons)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        lstm_model.predict(train_reshaped, batch_size=batch_size)
        # forecast test dataset and reduce length of test set to a multiple of batch_size
        test_reshaped = test_scaled[:,0:-1]
        test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, 1)
        output = lstm_model.predict(test_reshaped, batch_size=batch_size)
        predictions = list()
        for i in range(len(output)):
            yhat = output[i,0]
            X = test_scaled[i, 0:-1]
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[split_point:end_point], predictions))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
        # line plot of observed vs predicted
        pyplot.plot(raw_values[split_point:end_point]) 
        pyplot.plot(predictions) 
        pyplot.show()
    return error_scores

# protocol used for first experiment on dataset
def test_multiple_resampling_rates_and_epochs(experiment_func=experiment):
    # Experiment conditions
    resampling_rates = [5, 10, 20, 30, 40]
    resampling_rates = resampling_rates[::-1] # will start with the higher/faster resampling rates, for faster feedback while developing
    epochs = [125, 250, 500, 1000, 2000, 4000]
    batch_size = 4
    neurons = 1
    repeats = 3
    
    experiment_results = {}
    overall_time_results = {}
        
    # load dataset   
    series = read_csv('david240520160001-singleLegVertForceSeries.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)#, date_parser=parser)
    for r in resampling_rates:
        print('Training and testing at resampling rate: {}'.format(r))
        # reduce sampling rate to speed training of the model
        series_resampled = series.iloc[::r]
        # line plot
        series_resampled.plot()
        pyplot.show()
        # experiment
        epoch_results = DataFrame()
        time_results = {}
        for e in epochs:
            print('Running function \"{}\" with parameters (epochs: {}; batch_size: {}; neurons: {})'.format(experiment_func.__name__, e, batch_size, neurons))
            ts = time()
            epoch_results[str(e)] = experiment_func(repeats, series_resampled, e, batch_size, neurons)
            te = time()
            runtime = te-ts
            time_results[str(e)] = runtime
            print ('Completed function \"{}\" in {} seconds'.format(experiment_func.__name__, runtime))
        # summarize results
        print(epoch_results.describe())
        experiment_results[str(r)] = epoch_results
        overall_time_results[str(r)] = time_results
        # save boxplot
        epoch_results.boxplot()
        pyplot.savefig('boxplot_epochs_resample{}.png'.format(r))
        pyplot.show()
    return experiment_results, overall_time_results

# protocol used for second experiment on dataset
def test_multiple_batch_sizes_and_neurons(experiment_func=experiment):
    # Experiment conditions
    resampling_rate = 5
    epoch_size = 250
    batch_size = [1, 2, 4, 8]
    neurons = [1, 2, 4, 8]
    repeats = 3
    
    experiment_results = {}
    overall_time_results = {}
        
    # load dataset   
    series = read_csv('david240520160001-singleLegVertForceSeries.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)#, date_parser=parser)
    # reduce sampling rate to speed training of the model
    series_resampled = series.iloc[::resampling_rate]
    for n in neurons:
        # experiment
        batch_results = DataFrame()
        time_results = {}
        for b in batch_size:
            print('Running function \"{}\" with parameters (epochs: {}; batch_size: {}; neurons: {})'.format(experiment_func.__name__, epoch_size, b, n))
            ts = time()
            batch_results[str(b)] = experiment_func(repeats, series_resampled, epoch_size, b, n)
            te = time()
            runtime = te-ts
            time_results[str(b)] = runtime
            print ('Completed function \"{}\" in {} seconds'.format(experiment_func.__name__, runtime))
        # summarize results
        print(batch_results.describe())
        experiment_results[str(n)] = batch_results
        overall_time_results[str(n)] = time_results
        # save boxplot
        batch_results.boxplot()
        pyplot.savefig('boxplot_batch_size_neurons{}.png'.format(n))
        pyplot.show()
    return experiment_results, overall_time_results
