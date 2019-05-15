import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
import pickle

def main():  
    # Traing CPU Utilization Prediction Model
    train_model()

    # Forecast using CPU Utilization Prediction Model
    forecast_using_model()

def forecast_using_model():
    try:
        print("Forecasting CPU Utilization")
        # Configure the default styles for matplotlib package
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 12
        matplotlib.rcParams['ytick.labelsize'] = 12
        matplotlib.rcParams['text.color'] = 'k'

        # Read Excel File, Sort by Date in ascending order and Group it AvgCPUUtilization by Date
        df = pd.read_excel("AvgCPUUtilization.xlsx", parse_dates=['Date'])
        df.set_index('Date',inplace=True)

        # Copy the Pandas Data Frame into the variable y
        y = df['AvgCPUUtilization'].resample('24h').mean()

        # Load the CPU Utilization Prediction Model from file which has the trained model
        results = pickle.load(open("cpu_model.sav", 'rb'))        

        # Forecast
        # Forecast for 100 steps
        pred_uc = results.get_forecast(steps=100)

        # Get the confidence interval of the fitted parameters
        pred_ci = pred_uc.conf_int()    

        # Start plotting
        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Avg % CPU Utilization')
        plt.legend()
        plt.show()
    except Exception as ex:
        print("Error!",str(ex),"occured.")

def train_model():
    try:
        # Read Excel File, Sort by Date in ascending order and Group it AvgCPUUtilization by Date
        df = pd.read_excel("AvgCPUUtilization.xlsx", parse_dates=['Date'])    
        df.set_index('Date',inplace=True)

        # Copy the Pandas Data Frame into the variable y
        y = df['AvgCPUUtilization'].resample('24h').mean()

        # Forecasting using SARIMAX (Seasonal Autoregressive Integrated Moving Average with eXogenous regressors model)        
        mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 1),
                                    seasonal_order=(0, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

        # Fit the model
        print("Fitting the model")
        results = mod.fit()

        # Save the model to file
        filename = 'cpu_model.sav'
        print("Saving the model to disk: Filename -",filename)
        pickle.dump(results, open(filename, 'wb'))

        print("Model trained.")
    except Exception as ex:
        print("Error!",str(ex),"occured.")


if __name__ == "__main__":
    main()