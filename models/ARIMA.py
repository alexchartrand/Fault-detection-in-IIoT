# source: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from Motor.Motor import getMotorsData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

arima_p_d_q = (3, 1, 2)
arima_fit_size = 2.0/3.0
arima_test_size = 0.25

motors = getMotorsData()

train, validation = train_test_split(motors, test_size=arima_test_size, random_state=1234)

currantTrain = [c.steadyState().getCol("currant") for c in train]
currantVal = [c.steadyState().getCol("currant") for c in validation]


def showDifferencingOrder(data):
    # We need to show how many differencing is necessary to have a stationary serie
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

    # Original Series
    fig, axes = plt.subplots(3, 2, sharex='col')
    axes[0, 0].plot(data)
    axes[0, 0].set_title('Original Series')
    plot_acf(data, ax=axes[0, 1])

    # 1st Differencing
    data_diff_1 = np.diff(data, n=1)
    axes[1, 0].plot(data_diff_1)
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(data_diff_1, ax=axes[1, 1])

    # 2nd Differencing
    data_diff_2 = np.diff(data_diff_1, n=1)
    axes[2, 0].plot(data_diff_2)
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(data_diff_2, ax=axes[2, 1])

    plt.show()

def findARIMA_d():
    result = adfuller(currantTrain[1])
    p_value = result[1]
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    if p_value > 0.05:
        # We need differencing
        showDifferencingOrder(currantTrain[1])

def findARIMA_p(diff_order):
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})

    data_diff = np.diff(currantTrain[1], n=diff_order)

    fig, axes = plt.subplots(1, 2, sharex='col')
    axes[0].plot(data_diff)
    axes[0].set_title('{} order differencing'.format(diff_order))
    #axes[1].set(ylim=(0, 5))
    plot_pacf(data_diff, ax=axes[1])

    plt.show()

def findARIMA_q(diff_order):
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})

    data_diff = np.diff(currantTrain[1], n=diff_order)

    fig, axes = plt.subplots(1, 2, sharex='col')
    axes[0].plot(data_diff)
    axes[0].set_title('{} order differencing'.format(diff_order))
    #axes[1].set(ylim=(0, 1.2))
    plot_acf(data_diff, ax=axes[1])

    plt.show()

def evaluateARIMA():
    data = currantTrain[1]
    model = ARIMA(data, order=arima_p_d_q)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

    # Actual vs Fitted
    model_fit.plot_predict(dynamic=False)
    plt.show()

if __name__ == "__main__":
    #findARIMA_d()
    #findARIMA_p(1)
    #findARIMA_q(1)
    evaluateARIMA()


# for i in len(train):
#     X = train[i]
#     size = int(len(X) * arima_fit_size)
#     fitting_range, testing_range = X[0:size], X[size:len(X)]
#     model = ARIMA(fitting_range, order=arima_p_d_q)
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()

# model = ARIMA(currantArray[0] ,order=arima_p_d_q)
# # modelFit = model.fit(disp=0)
# # print(modelFit.summary())
# # # plot residual errors
# # residuals = pd.DataFrame(modelFit.resid)
# # residuals.plot()
# # plt.show()
# # residuals.plot(kind='kde')
# # plt.show()
# # print(residuals.describe())

