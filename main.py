import os
import pandas as pd
from src.rfm import customer_rfm, generate_churn_labels
from src.modelling import model
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 

file_path = "./data/transaction_data.parquet"


if __name__ == "__main__":
    
    df=pd.read_parquet(file_path)
    
    df_client = df[["sales_net", "client_id"]]
    df_client = df_client.groupby('client_id')['sales_net'].sum()
    df_client = df_client.to_frame()
    df_client = df_client.sort_values(by=['sales_net'], ascending = False)
    df_client = df_client.head(17000)
    
    df2 = df.loc[df['client_id'].isin(df_client.index)]
    df2 = df2[["date_order", "client_id", "sales_net"]]
    df2 = df2.groupby(['client_id', 'date_order'])['sales_net'].sum()
    df2 = df2.to_frame()
    df2 = df2.reset_index()
    df2.date_order = pd.to_datetime(df2.date_order)
    
    cut_off = pd.datetime(2019, 7, 1)
    future = df2[(df2["date_order"] > cut_off)]
    observed = df2[df2["date_order"] < cut_off]
    
    rfm_df = customer_rfm(observed, cut_off, 'date_order', 'client_id', 'sales_net', freq='W')
    labels = generate_churn_labels(future)
    
    labels = labels.drop_duplicates()
    
    rfm_df_true = rfm_df[rfm_df.client_id.isin(labels.client_id)]
    rfm_df_true["label"] = 1
    
    rfm_df_false = rfm_df[~rfm_df.client_id.isin(labels.client_id)]
    rfm_df_false["label"] = 0
    
    frames = [rfm_df_true, rfm_df_false]
    rfm_df = pd.concat(frames)
    Customers, probs = model(rfm_df)
    
    plt.figure(figsize=(12, 6))
    plt.hist(probs, bins = int(180/6))
    plt.title('Probability Distribution of Retention')
    plt.xlabel('Retention Probability')
    plt.ylabel('# Customers')
    x = plt.show()
    
    Customers['Retention Probability'] = probs
    Customers['R_rank'] = Customers['recency'].rank(ascending=False)
    Customers['F_rank'] = Customers['frequency'].rank(ascending=True)
    Customers['M_rank'] = Customers['value'].rank(ascending=True)
 
    # normalizing the rank of the customers
    Customers['R_rank_norm'] = (Customers['R_rank']/Customers['R_rank'].max())*100
    Customers['F_rank_norm'] = (Customers['F_rank']/Customers['F_rank'].max())*100
    Customers['M_rank_norm'] = (Customers['F_rank']/Customers['M_rank'].max())*100
 
    Customers.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
    Customers['RFM_Score'] = (0.15*Customers['R_rank_norm'])+(0.28*Customers['F_rank_norm'])+(0.57*Customers['M_rank_norm'])
    Customers['RFM_Score'] *= 0.05
    Customers.drop(columns=['R_rank_norm', 'F_rank_norm', 'M_rank_norm', 'recency', 'frequency', 'value', 'age'], inplace=True)
    
    Retain_Customers = Customers.loc[Customers["Retention Probability"]<0.5]
    Retain_Customers = Retain_Customers.loc[Retain_Customers["RFM_Score"] > 1.6]
    