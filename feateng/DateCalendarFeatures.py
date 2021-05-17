import pandas as pd

from preparation import PreparationHelper


def add_date_based_features(dataset: pd.DataFrame, seasonal_periods: int = 0):
    """
    Function adding date based features to dataset
    :param dataset: dataset for adding features
    :param seasonal_periods: seasonality used for seasonal-based features
    """
    if seasonal_periods not in [4, 12]:
        dataset['cal_date_day_of_month'] = dataset.index.day
        dataset['cal_date_weekday'] = dataset.index.weekday
    dataset['cal_date_month'] = dataset.index.month
    if seasonal_periods == 24:
        dataset['cal_date_hour'] = dataset.index.hour


def add_valentine_mothersday(dataset: pd.DataFrame):
    """
    Function adding valentine's and mother's day to public_holiday column of dataset
    :param dataset: dataset for adding valentine's and mother's day
    """
    # add valentine's day (always 14th of February)
    dataset.at[[index for index in dataset.index.date if (index.day == 14 and index.month == 2)],
               'public_holiday'] = 'Valentine'
    # add mother's day (in Germany always second sunday in May)
    dataset.at[[index for index in dataset.index.date
                if ((index.day-7) > 0 and index.day < 15 and index.weekday() == 6 and index.month == 5)],
               'public_holiday'] = 'MothersDay'


def add_public_holiday_counters(dataset: pd.DataFrame, event_lags: list, special_days: list):
    """
    Function adding counters for upcoming or past public holidays (according to event_lags)
    with own counters for those specified in special_days
    :param dataset: dataset for adding features
    :param event_lags: lags before and after holiday to add
    :param special_days: list of days with their own counter as feature
    """
    for index, row in dataset.iterrows():
        holiday = row['public_holiday']
        if holiday != 'no':
            for lag in event_lags:
                if (index+pd.Timedelta(days=lag)) in dataset.index:
                    dataset.at[index+pd.Timedelta(days=lag), 'cal_PublicHoliday_Counter'] = -lag
                    if holiday in special_days:
                        dataset.at[index+pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -lag
    PreparationHelper.drop_columns(df=dataset, columns=['public_holiday'])
    dataset[[col for col in dataset.columns if 'Counter' in col]] = \
        dataset[[col for col in dataset.columns if 'Counter' in col]].fillna(value=99)
