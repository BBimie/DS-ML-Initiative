import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#for reproducibility
np.random.seed(1)


def process_data(df, commodity):
    """
    This function extracts the data on given commondity from the wfp_food_prices_nigeria dataset,
    and gets the median price of the commodity on each date.

    INPUT: 
        commodity - commodity name
    OUTPUT:
        com_df - commodity data
        price_data - median prices
    """
    com_df = df[df['cmname'] == commodity]

    com_df['price'] = com_df['price'].astype('float')
    price_data = pd.DataFrame(com_df.groupby('date')['price'].median())
    price_data.index = pd.to_datetime(price_data.index)

    return com_df, price_data


def timeseries_to_supervised(data, lag=1):
    """
    Convert timeseries data to supervised learning data by creating look back / lag features from the price
    
    INPUT:
        data - time series data
        lag - number of look back features to be created
    OUTPUT:
        df - supervised learning data with lags
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def feature_target(lagged_data):
    """
    Separates the feature and target sets in the lagged data to be used for training
    
    INPUT: 
        lagged_data - supervised learning data
    OUTPUT: 
        x, y : feature and target
    """
    array = np.array(lagged_data)
    x, y = array[:, 1:], array[:, 0:1]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    return x, y

def build_LSTM(x_train):
    """
    Builds an LSTM model using keras. 
        Model components:
        - Seqential model
        - LSTM layer with 50 units, input shape is the training data's shape
        - 20% dropout layer to prevent overfitting
        - LSTM layer with 100 units and no sequences returned
        - another 20% dropout layer to prevent overfitting
        - Dense output layer
    The function also compiles the model with a mean_squared_error loss function and rmsprop as the optimizer
    INPUT:
        x_train - training data
    OUTPUT:
        model: LSTM model
    """
    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
                              
                         
def forecast(num_prediction, model, past, look_back=1):
    """Forecasts future data
    INPUT:
        num_predictions - How many months to be predicted
        model - trained model
        past - previous prices (shape = (len(data),)
        look_back - number of lags
    OUTPUT:
        forecast - scaled forecasted prices
    """  
    forecast = past[-look_back:]
    
    for _ in range(num_prediction):
        x = forecast[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        forecast = np.append(forecast, out)
    forecast = forecast[look_back-1:]
        
    return forecast
    
def forecast_dates(df, num_prediction, frequency = 'MS'):
    """Predicts the future dates
    INPUT:
        df = price dataframe
        num_predictions - number of months
        frequency - time series frequency alias. Default value is 'MS' but this can be change to suit your task'
                    'MS' stands for month start frequency
    OUTPUT:
        forecast_dates - future dates
        """
    last_date = df.index.values[-1]
    forecast_dates = pd.date_range(last_date, periods=num_prediction+1, freq=frequency).tolist()
    return forecast_dates
                              
