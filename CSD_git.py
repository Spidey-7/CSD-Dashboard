# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:47:00 2024

@author: devhs
"""

# Importing Libraries

import time
import datetime

import numpy as np
import pandas as pd
import talib
import calendar

import concurrent.futures

from copy import deepcopy
import matplotlib.pyplot as plt

from functools import lru_cache

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import seaborn as sns

import os

import warnings
warnings.filterwarnings("ignore")
#st.set_option('deprecation.showPyplotGlobalUse', False)

import subprocess

# Importing Data

#data = pd.read_csv(r"C:\Users\devhs\OneDrive\Desktop\Stock Data\NIFTY\NSE_ADANIENT, 1D.csv")

# Data pre-processing

#data['date'] = data['time'].str[:10]
#data['date'] = pd.to_datetime(data['date'])

#data = data.set_index('date')

# Define Candle Stick Chart Creator
    
def create_chart(df):
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    
    # Add markers for bullish engulfing signals
    signal_df = df[df['signal'] == 1]
    fig.add_trace(go.Scatter(x=signal_df['date'], 
                             y=signal_df['low'], 
                             mode='markers', 
                             marker=dict(symbol='triangle-up', color='green', size=10),
                             name='Bullish Engulfing Signal'))
    
    fig.update_layout(title='Candlestick Chart with Bullish Engulfing Signals',
                      xaxis_title='Date',
                      yaxis_title='Price')
    
    st.plotly_chart(fig)
    
# Defining Strategies

def bullish_engulfing(df):
    
    df['signal'] = 0
    
    for i in range(2, len(df['open'])):
        
        if df['high'][i-1] < df['high'][i]:
            if df['low'][i-1] > df['low'][i]:
                if df['open'][i-1] > df['close'][i-1]:
                    if df['open'][i] < df['close'][i]:
                        if df['open'][i-1] < df['close'][i]:
                            if df['close'][i-1] > df['open'][i]:
                                df['signal'][i] = 1
                            
    return df

# Backtesting Strategies

def backtest(df, profit_target, stop_loss, capital, trade_size):
    open_trades = []
    closed_trades = []
    current_capital = capital

    for i in range(len(df)):
        # Check for open trades and exit conditions
        for trade in open_trades[:]:  # Iterate over a copy to modify list in place
            entry_price = trade['entry_price']
            exit_price = df['close'][i]
            profit_target_price = entry_price * (1 + trade['profit_target'])
            stop_loss_price = entry_price * (1 - trade['stop_loss'])
            
            # Check for profit target hit
            if df['high'][i] >= profit_target_price:
                trade['exit_price'] = profit_target_price
                trade['exit_date'] = df['date'][i]
                trade['pnl'] = (profit_target_price - entry_price) / entry_price
                trade['exit_reason'] = 'Profit Target'
                closed_trades.append(trade)
                open_trades.remove(trade)
                current_capital += trade['pnl'] * trade['trade_value']
            
            # Check for stop loss hit
            elif df['low'][i] <= stop_loss_price:
                trade['exit_price'] = stop_loss_price
                trade['exit_date'] = df['date'][i]
                trade['pnl'] = (stop_loss_price - entry_price) / entry_price
                trade['exit_reason'] = 'Stop Loss'
                closed_trades.append(trade)
                open_trades.remove(trade)
                current_capital += trade['pnl'] * trade['trade_value']

        # Enter new trade if signal is 1
        if df['signal'][i] == 1:
            trade_value = current_capital * trade_size
            open_trades.append({
                'entry_date': df['date'][i],
                'entry_price': df['close'][i],
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'trade_value': trade_value
            })
    
    # Convert closed trades to DataFrame
    closed_trades_df = pd.DataFrame(closed_trades, columns=['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl', 'exit_reason', 'trade_value'])
    
    closed_trades_df['points'] = closed_trades_df['exit_price'] - closed_trades_df['entry_price']
    
    # Convert open trades to DataFrame (trades that remain open)
    open_trades_df = pd.DataFrame(open_trades, columns=['entry_date', 'entry_price', 'profit_target', 'stop_loss', 'trade_value'])
    
    return closed_trades_df, open_trades_df, current_capital

# Stats of Strategies

def statssheet_trial(tradesheet, capital, lot_size):
    
    df = deepcopy(tradesheet)
    
    df['capital'] = capital
    
    df.fillna(0, axis = 1, inplace = True)
                           
    df['Sell_Points'] = df['points'].copy()

    df['Lot_Size'] = lot_size

    df['Number_of_Lots'] = 1
    
    df['Total_Abs_Return'] = (df['points'] * df['Lot_Size'] * df['Number_of_Lots']) - 100
    
    df.fillna(0, axis = 1, inplace = True)
    
    df['Daily_Return'] = (df['Total_Abs_Return'] / df['capital']) * 100
    
    df['Cumm_Returns'] = df['Daily_Return'].cumsum()
    
    df['Cumm_Abs_Returns'] = df['Total_Abs_Return'].cumsum() + df['capital']
    
    df['Profitable_Trades'] = df.loc[df['Total_Abs_Return'] > 0]['Total_Abs_Return'].count()
    
    df['Loss_Trades'] = df.loc[df['Total_Abs_Return'] <= 0]['Total_Abs_Return'].count()
    
    df['Avg_Profit_Per_Trade'] = df.loc[df['Total_Abs_Return'] > 0]['Total_Abs_Return'].mean()
    
    df['Avg_Loss_Per_Trade'] = df.loc[df['Total_Abs_Return'] < 0]['Total_Abs_Return'].mean()
    
    df['Cumm_max_Returns'] = df['Cumm_Abs_Returns'].cummax()
    
    df['Total_Daily_Drawdown'] = ((df['Cumm_Abs_Returns'] - df['Cumm_max_Returns'])/df['capital']) 
    
    df['Max_Daily_Drawdown'] = df['Total_Daily_Drawdown'].cummin()
    
    df['Hit_Ratio'] = df['Profitable_Trades'] / (df['Profitable_Trades'] + df['Loss_Trades'])
    
    df['Winning_Streak'] = 0.0
    
    df['Loosing_Streak'] = 0.0
    
    x = 0 
    
    for i in range(1 , len(df.Daily_Return) - 1):
        
        if df['Daily_Return'].iloc[i] > 0 and df['Daily_Return'].iloc[i-1] > 0:
            
            x = x + 1
            df['Winning_Streak'].iloc[i] = x
            
        elif df['Daily_Return'].iloc[i] <= 0 and df['Daily_Return'].iloc[i-1] <= 0 :
            
            x = x-1
            df['Loosing_Streak'].iloc[i] = x
        
        else:
            
            x = 0
    
    df['Recovery_Period'] = 0.0
    
    recovery = 0.0
    
    for i in range(1, len(df.Daily_Return) - 1):
        
        if df['Total_Daily_Drawdown'].iloc[i] < 0 and df['Total_Daily_Drawdown'].iloc[i-1] < 0 :
            
            recovery = recovery + 1
        
        elif df['Total_Daily_Drawdown'].iloc[i] == 0:
            
            df['Recovery_Period'].iloc[i] = recovery + 1
        
        else:
            
            recovery = 0 
            
    return df

# Optimization Params

def optimization(df, tp_range, sl_range, capital, trade_size):
    
    results = []
    
    for tp in tp_range:
        for sl in sl_range:
            
            print(tp, sl)
            
            tp = tp/100
            sl = sl/100
            
            tradesheet, x, y = backtest(df, tp, sl, capital, trade_size)
            stats_sheet = statssheet_trial(tradesheet, capital, trade_size)
            
            #profit = stats_sheet['Cumm_Returns'].iloc[-1] #:.2f
            if 'Cumm_Returns' in stats_sheet.columns and not stats_sheet['Cumm_Returns'].empty:
                profit = stats_sheet['Cumm_Returns'].iloc[-1]
            else:
                st.warning("Cumm_Returns column is empty or not present.")
                profit = 0 
            
            # Check if 'Max_Daily_Drawdown' column is present and not empty
            if 'Max_Daily_Drawdown' in stats_sheet.columns and not stats_sheet['Max_Daily_Drawdown'].empty:
                max_drawdown = stats_sheet['Max_Daily_Drawdown'].iloc[-1] * 100
            else:
                st.warning("Max_Daily_Drawdown column is empty or not present.")
                max_drawdown = 0  # or another default value
            
            # Check if 'Hit_Ratio' column is present and has at least one row
            if 'Hit_Ratio' in stats_sheet.columns and not stats_sheet['Hit_Ratio'].empty:
                hit_ratio = stats_sheet['Hit_Ratio'].iloc[0] * 100
            else:
                st.warning("Hit_Ratio column is empty or not present.")
                hit_ratio = 0  # or another default value
            
            rounded_tp = round(tp * 100)
            rounded_sl = round(sl * 100)
            
            tp = tp*100
            sl = sl*100
            
            print(tp, sl)
            
            results.append((rounded_tp, rounded_sl, profit, max_drawdown, hit_ratio))
            
    return results

def report(stock, tp_range, sl_range, capital, trade_size):
    
    new_file_content = f"""
# This Python file was created based on user input

def initialize(context):
    context.run_once = False
    #context.security = symbol('CASH, EUR, USD')
    context.security = symbol('{stock}')
    
    
def handle_data(context, data):
    dt = data.history(context.security, ['open', 'high', 'low', 'close', 'volume'], 1500, '1d')
    dt.to_csv('stock_data.csv')
    end()
"""

    new_file_name = "data_source.py"
    full_file_path = f"{save_directory}/{new_file_name}"

    # Create and write to the new file in the specified directory
    with open(full_file_path, "w") as new_file:
        new_file.write(new_file_content)
        
    #st.write(f"New Python file '{new_file_name}' has been created.")
    
    python_file = r"C:\Users\devhs\OneDrive\Desktop\Ibpy_new\RUN_ME.py"

    subprocess.run(["python", python_file])
    
    #st.write('New Stock File created')
    
    data = pd.read_csv(r"C:\Users\devhs\stock_data.csv")

    data['date'] = data['timestamp'].str[:10]
    data['date'] = pd.to_datetime(data['date'])

    test = bullish_engulfing(data)
    #df_ts, open_trades_df, final_capital = backtest(test, profit_target= profit_target, stop_loss= stop_loss, capital=capital, trade_size= trade_size)
    #stats_sheet = statssheet_trial(df_ts, capital, trade_size)
    
    tp_values = np.arange(tp_range[0], tp_range[1] + 1)
    sl_values = np.arange(sl_range[0], sl_range[1] + 1)
    
    results = optimization(test, tp_values, sl_values, capital, trade_size)
    
    return results
    

# Dashboard

st.title("Candle Stick - Savla Enterprises")

st.write('## Select Strategy and Stock')

stock_lst = [
    ('STK,RELIANCE, INR'),
    ('STK,TCS, INR'),
    ('STK,HDFCBANK,INR'),
    ('STK,BHARTIART,INR'),
    ('STK,ICICIBANK, INR'),
    ('STK,INFY, INR'),
    ('STK,SBIN, INR'),
    ('STK,HINDUNILVR, INR'),
    ('STK,ITC, INR'),
    ('STK,LT, INR'),
    ('STK,HCLTECH, INR'),
    ('STK,ONGC, INR'),
    ('STK,SUNPHARMA, INR'),
    ('STK,BAJFINANCE, INR'),
    ('STK,TATAMOTOR, INR'),
    ('STK,NTPC, INR'),
    ('STK,MARUTI, INR'),
    ('STK,AXISBANK, INR'),
    ('STK,KOTAKBANK, INR'),
    ('STK,ADANIENT, INR'),
    ('STK,M_M, INR'),
    ('STK,ULTRACEMCO, INR'),
    ('STK,ADANIPORTS, INR'),
    ('STK,COALINDIA, INR'),
    ('STK,POWERGRID, INR'),
    ('STK,TITAN, INR'),
    ('STK,ASIANPAINT, INR'),
    ('STK,BAJAJ_AUTO, INR'),
    ('STK,WIPRO, INR'),
    ('STK,BAJAJFINSV, INR'),
    ('STK,NESTLEIND, INR'),
    ('STK,JSWSTEEL, INR'),
    ('STK,TATASTEEL, INR'),
    ('STK,GRASIM, INR'),
    ('STK,LTIM, INR'),
    ('STK,SBILIFE, INR'),
    ('STK,TECHM, INR'),
    ('STK,BPCL, INR'),
    ('STK,HINDALCO, INR'),
    ('STK,HDFCLIFE, INR'),
    ('STK,BRITANNIA, INR'),
    ('STK,EICHERMOT, INR'),
    ('STK,CIPLA, INR'),
    ('STK,DIVISLAB, INR'),
    ('STK,TATACONSUM, INR'),
    ('STK,SHRIRAMFIN, INR'),
    ('STK,DRREDDY, INR'),
    ('STK,INDUSINDBK, INR'),
    ('STK,HEROMOTOCO, INR'),
    ('STK,APOLLOHOSP, INR')
]

stock = st.selectbox('Stock', stock_lst)

save_directory = r"C:\Users\devhs\OneDrive\Desktop\Ibpy_new\Strategies"

if st.button('Create File'):
    
    new_file_content = f"""
# This Python file was created based on user input

def initialize(context):
    context.run_once = False
    #context.security = symbol('CASH, EUR, USD')
    context.security = symbol('{stock}')
    
    
def handle_data(context, data):
    dt = data.history(context.security, ['open', 'high', 'low', 'close', 'volume'], 1500, '1d')
    dt.to_csv('stock_data.csv')
    end()
"""

    new_file_name = "data_source.py"
    full_file_path = f"{save_directory}/{new_file_name}"

    # Create and write to the new file in the specified directory
    with open(full_file_path, "w") as new_file:
        new_file.write(new_file_content)
        
    st.write(f"New Python file '{new_file_name}' has been created.")
    
    python_file = r"C:\Users\devhs\OneDrive\Desktop\Ibpy_new\RUN_ME.py"

    subprocess.run(["python", python_file])
    
    st.write('New Stock File created')
    

st.write('## Input Data')

col1, col2 = st.columns(2)

capital = col1.number_input('Capital', min_value = 1000000)
trade_size = col2.number_input('Number of Stocks', min_value = 25)
profit_target = col1.number_input("Target Profit (in %)", min_value = 1)/100
stop_loss = col2.number_input('Stop Loss (in %)', min_value = 1)/100

data = pd.read_csv(r"C:\Users\devhs\stock_data.csv")

data['date'] = data['timestamp'].str[:10]
data['date'] = pd.to_datetime(data['date'])

test = bullish_engulfing(data)
df_ts, open_trades_df, final_capital = backtest(test, profit_target= profit_target, stop_loss= stop_loss, capital=capital, trade_size= trade_size)
stats_sheet = statssheet_trial(df_ts, capital, trade_size)

st.write('## Strategy Results')
col1, col2, col3 = st.columns(3)
col1.metric(label = "Total Returns", value = f"{stats_sheet['Cumm_Returns'].iloc[-1]:.2f}%")
col2.metric(label = 'Max Drawdown', value = f"{stats_sheet['Max_Daily_Drawdown'].iloc[-1] * 100:.2f}%")
col3.metric(label = "Hit Ratio %", value = f"{(stats_sheet['Hit_Ratio'][0]) * 100}%")


st.write('## Return Plot')
st.line_chart(stats_sheet['Cumm_Returns'])

st.write('## Drawdown Plot')
st.line_chart(stats_sheet['Max_Daily_Drawdown'])

check = st.checkbox("Click Here for Candle Stick Graph of Strategy Signals")

if check:
    create_chart(test)
#st.write(final_capital)
#st.write(stats_sheet['Cumm_Abs_Returns'])

st.write('## Optimization')

st.write('NOTE: Optimization would be done based on the same number of stock selected and will take time.')

tp_range = st.slider("Take Profit Range", 0, 100, (10, 50))
sl_range = st.slider("Stop Loss Range", 0, 100, (10, 50))

optimize = st.button("Optimize Settings")
show_heatmap = st.checkbox("Show Heatmap")

if optimize:
    tp_values = np.arange(tp_range[0], tp_range[1] + 1)
    sl_values = np.arange(sl_range[0], sl_range[1] + 1)
    
    results = optimization(test, tp_values, sl_values, capital, trade_size)
    
    # Find optimal settings
    best_result = max(results, key=lambda x: (x[2], -x[3], x[4]))  # Optimize for profit, min drawdown, max hit ratio
    st.write(f"Optimal Take Profit: {best_result[0]}, Optimal Stop Loss: {best_result[1]}")
    
    # Create matrices for profit, drawdown, and hit ratio
    profit_matrix = np.zeros((len(tp_values), len(sl_values)))
    drawdown_matrix = np.zeros((len(tp_values), len(sl_values)))
    hit_ratio_matrix = np.zeros((len(tp_values), len(sl_values)))
    
    # Populate the matrices
    for result in results:
        tp_value = result[0]
        sl_value = result[1]
        
        if tp_value in tp_values and sl_value in sl_values:
            tp_idx = np.where(tp_values == tp_value)[0][0]
            sl_idx = np.where(sl_values == sl_value)[0][0]
            
            profit_matrix[tp_idx, sl_idx] = result[2]  # Assuming result[2] is profit
            drawdown_matrix[tp_idx, sl_idx] = result[3]  # Assuming result[3] is max_drawdown
            hit_ratio_matrix[tp_idx, sl_idx] = result[4]  # Assuming result[4] is hit_ratio
        else:
            st.warning(f"Take Profit {tp_value} or Stop Loss {sl_value} is not found in the ranges.")
    
    # Plot profit heatmap
    fig, ax = plt.subplots()
    sns.heatmap(profit_matrix, annot=True, fmt=".1f", xticklabels=sl_values, yticklabels=tp_values, ax=ax)
    plt.title("Profit Heatmap")
    st.pyplot(fig)
    
    # Plot drawdown heatmap
    fig, ax = plt.subplots()
    sns.heatmap(drawdown_matrix, annot=True, fmt=".1f", xticklabels=sl_values, yticklabels=tp_values, ax=ax)
    plt.title("Drawdown Heatmap")
    st.pyplot(fig)
    
    # Plot hit ratio heatmap
    fig, ax = plt.subplots()
    sns.heatmap(hit_ratio_matrix, annot=True, fmt=".1f", xticklabels=sl_values, yticklabels=tp_values, ax=ax)
    plt.title("Hit Ratio Heatmap")
    st.pyplot(fig)
    
st.write('## Report (Multi Stock Backtesting)')
st.write('Note: Generating the report will take time please be patient after clicking the button.')

report_type = st.selectbox('Select Report Type', ["Tomorrow's Strategy Hits", 'Top Stocks for This Strategy'])

col1_report, col2_report = st.columns(2)

capital_report = col1_report.number_input('Capital (Report)', min_value = 1000000)
trade_size_report = col2_report.number_input('Number of Stocks (Report)', min_value = 25)

report_button = st.button('Generate Report')

if report_button:
    
    # Define the columns for the final DataFrame
    columns = ['Stock', 'Take_Profit', 'Stop_Loss', 'Profit', 'Max_Drawdown', 'Hit_Ratio']
    
    # Initialize an empty DataFrame to store the results for all stocks
    all_results_df = pd.DataFrame(columns=columns)
    
    for stock in stock_lst:
        
       # Call the report function to get results for the current stock
        results = report(stock, tp_range, sl_range, capital_report, trade_size_report)
        
        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results, columns=columns[1:])
        
        # Find the row with the maximum profit
        max_profit_row = results_df.loc[results_df['Profit'].idxmax()]
        
        # Find the row with the minimum drawdown
        min_drawdown_row = results_df.loc[results_df['Max_Drawdown'].idxmin()]
        
        # Find the row with the maximum hit ratio
        max_hit_ratio_row = results_df.loc[results_df['Hit_Ratio'].idxmax()]
        
        # Create a temporary DataFrame to store these rows
        temp_df = pd.DataFrame([max_profit_row, min_drawdown_row, max_hit_ratio_row])
        
        # Add the stock name to the DataFrame
        temp_df['Stock'] = stock
        
        # Reorder the columns to have 'Stock' as the first column
        temp_df = temp_df[['Stock'] + columns[1:]]
        
        # Append the important rows of the current stock to the final DataFrame
        all_results_df = all_results_df.append(temp_df, ignore_index=True)
    
    # Display the complete DataFrame with all stocks' results
    #print(all_results_df)
    
    st.dataframe(all_results_df)
#optimize = st.selectbox('Optimize Parameter')