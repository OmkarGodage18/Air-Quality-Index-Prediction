import pandas as pd

def preprocess(data):

    data = data.copy()

    data['CO2'] = data['CO2'].fillna(
        data['CO2'].median()
    )

    data['Date'] = pd.to_datetime(
        data['Date'],
        utc=True
    )

    data['Date'] = data['Date'].dt.tz_convert(
        'America/Sao_Paulo'
    )

    data['hour'] = data['Date'].dt.hour
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['weekday'] = data['Date'].dt.weekday

    data.drop(
        columns=['Date', 'City'],
        inplace=True
    )

    return data