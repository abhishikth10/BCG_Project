import pandas as pd

def customer_frequency(data, cut_off, date_column, customer_id_column, value_column, freq='M'):
    frequency = data[data[date_column] < cut_off].copy()
    frequency.set_index(date_column, inplace=True)
    frequency.index = pd.DatetimeIndex(frequency.index)
    frequency = frequency.groupby([
    customer_id_column,
    pd.Grouper(freq=freq, level=date_column)
      ]).count()
    frequency[value_column] = 1 
    return frequency.groupby(customer_id_column).sum().reset_index().rename(columns={value_column : 'frequency'})

def customer_recency(data, cut_off, date_column, customer_id_column):
    # Get data before cut off
    recency = data[data[date_column] < cut_off].copy()
    recency[date_column] = pd.to_datetime(recency[date_column])
    # Group customers by latest transaction
    recency = recency.groupby(customer_id_column)[date_column].max()
    return ((pd.to_datetime(cut_off) - recency).dt.days).reset_index().rename(
              columns={date_column : 'recency'})

def customer_age(data, cut_off, date_column, customer_id_column):
    age = data[data[date_column] < cut_off]
    first_purchase = age.groupby(customer_id_column)[date_column].min().reset_index()
    # Get number of days between cut off and first transaction
    first_purchase['age'] = (cut_off - first_purchase[date_column]).dt.days
    return first_purchase[[customer_id_column, 'age']]

def customer_value(data, cut_off, date_column, customer_id_column, value_column):
    value = data[data[date_column] < cut_off]
    value.set_index(date_column, inplace=True)
    value.index = pd.DatetimeIndex(value.index)
    # Get mean sales amount for each customer
    return value.groupby(customer_id_column)[value_column].mean().reset_index().rename(
      columns={value_column : 'value'})

def customer_rfm(data, cut_off, date_column, customer_id_column, value_column, freq='M'):
    cut_off = pd.to_datetime(cut_off)
    # Compute Recency
    recency = customer_recency(data, cut_off, date_column, customer_id_column)
    # Compute Frequency
    frequency = customer_frequency(data, cut_off, date_column, customer_id_column, value_column, freq=freq)
    # Compute average value
    value = customer_value(data, cut_off, date_column, customer_id_column, value_column)
    # Compute age
    age = customer_age(data, cut_off, date_column, customer_id_column)
    # Merge all columns
    return recency.merge(frequency, on=customer_id_column).merge(value, on=customer_id_column).merge(age, on=customer_id_column)

def generate_churn_labels(future):
    future['DidBuy'] = 1
    return future[['client_id', 'DidBuy']]