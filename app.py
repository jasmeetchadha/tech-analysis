import streamlit as st
import yfinance as yf  # Import yfinance
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import riskfolio as rp
import matplotlib.dates as mdates
import datetime


warnings.filterwarnings("ignore")

def generate_stock_analysis(asset, start_date, end_date):
    """
    Generates stock analysis charts and signals.

    Args:
        asset: Stock ticker symbol (e.g., 'AAPL').
        start_date: Start date for data retrieval (e.g., '2023-01-01').
        end_date: End date for data retrieval (e.g., '2023-12-31').

    Returns:
        A dictionary containing chart figures and signals data.
    """
    all_signals_df = pd.DataFrame(columns=['Symbol', 'Signal Type', 'Date', 'Price'])
    fig = None  # Initialize fig to None



    try:
        # Download historical data
        data = yf.download(asset, start=start_date, end=end_date)

        # Check if the DataFrame is empty
        if data.empty:
            print(f"No data found for {asset} between {start_date} and {end_date}.")
            return {"chart_figure": None, "signals_data": None}

        # Calculations
        data['ValueWeightedPrice'] = (data['Close'] * data['Volume']).rolling(window=30).sum() / data['Volume'].rolling(window=30).sum()
        outstanding_shares = yf.Ticker(asset).info.get('sharesOutstanding')
        if outstanding_shares is None or outstanding_shares == 0:
            print(f"Could not find outstanding shares for {asset}. Skipping...")
            return {"chart_figure": None, "signals_data": None}
        
        rsi_rolling_window = 14
        
        data['VolumeRatio'] = data['Volume'].rolling(window=30).sum() / outstanding_shares
        data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).apply(lambda x: max(x, 0)).rolling(window=rsi_rolling_window).sum() / data['Close'].diff(1).apply(lambda x: abs(min(x, 0))).rolling(window=rsi_rolling_window).sum())))
        data['SMA200'] = data['Close'].rolling(window=200).mean()        
        data['Signal'] = ((data['Close'] < data['ValueWeightedPrice']) & (data['VolumeRatio'] > 0.5) & (data['RSI'] < 40)).astype(int)
        
        # 1. Adjust the lookback period (14 in your code) for price and RSI comparisons:
        lookback_period = 14  # Experiment with different values

        data['PriceLower'] = data['Close'] < data['Close'].shift(lookback_period)
        data['RSIHiger'] = data['RSI'] > data['RSI'].shift(lookback_period)
        data['PositiveRSIDivergence'] = data['PriceLower'] & data['RSIHiger']

        data['PriceHigher'] = data['Close'] > data['Close'].shift(lookback_period)
        data['RSILower'] = data['RSI'] < data['RSI'].shift(lookback_period)
        data['NegativeRSIDivergence'] = data['PriceHigher'] & data['RSILower']  # Corrected condition

        # 2. Add a threshold for price and RSI differences:
        price_threshold = 0.04  # 5% price difference
        rsi_threshold = 5  # 5 points RSI difference

        data['PositiveRSIDivergence'] = data['PositiveRSIDivergence'] & \
                                        ((data['Close'].shift(lookback_period) - data['Close']) / data['Close'].shift(lookback_period) > price_threshold) & \
                                        (data['RSI'] - data['RSI'].shift(lookback_period) > rsi_threshold)

        data['NegativeRSIDivergence'] = data['NegativeRSIDivergence'] & \
                                        ((data['Close'] - data['Close'].shift(lookback_period)) / data['Close'].shift(lookback_period) > price_threshold) & \
                                        (data['RSI'].shift(lookback_period) - data['RSI'] > rsi_threshold)

        # 3. Filter divergences based on trend:
        # (Example: Only consider positive divergences during a downtrend)
        data['Downtrend'] = data['SMA200'] < data['SMA200'].shift(20)  # Example downtrend condition
        data['PositiveRSIDivergence'] = data['PositiveRSIDivergence'] & data['Downtrend'] 
        # Add the opposite condition for NegativeRSIDivergence (Uptrend):
        data['Uptrend'] = data['SMA200'] > data['SMA200'].shift(20)  # Example uptrend condition
        data['NegativeRSIDivergence'] = data['NegativeRSIDivergence'] & data['Uptrend']



        # Calculate drawdown
        mu_prices = data['Adj Close']
        peak = mu_prices.cummax()
        drawdown = (mu_prices - peak) / peak





        # Plotting
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})


        # Plot Close Price, SMA200, and Value Weighted Price on the same axis
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax1.plot(data.index, data['SMA200'], label='SMA200', color='lightblue')
        ax1.plot(data.index, data['ValueWeightedPrice'], label='Value Weighted Price', color='red')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params('y', labelcolor='blue')

        # Plot Volume Ratio (on a secondary axis)
        ax2 = ax1.twinx()
        ax2.plot(data.index, data['VolumeRatio'], label='Volume Ratio', color='gray')
        ax2.set_ylabel('Volume Ratio', color='gray')
        ax2.tick_params('y', labelcolor='gray')
        ax2.set_ylim([0, 1])

        
        # Plot RSI on ax3 (below the main chart)
        ax3.plot(data.index, data['RSI'], label='RSI', color='purple')
        ax3.set_ylabel('RSI', color='purple')
        ax3.tick_params('y', labelcolor='purple')
        ax3.axhline(y=30, color='gray', linestyle='--')  # Add horizontal line at RSI 30
        ax3.axhline(y=70, color='gray', linestyle='--')  # Add horizontal line at RSI 70


        # Shade yellow for Positive RSI Divergence
        for i in range(len(data)):
            if data['PositiveRSIDivergence'][i]:
                ax3.axvspan(data.index[i], data.index[i], color='yellow', alpha=0.5)

        # Shade orange for Negative RSI Divergence
        for i in range(len(data)):
            if data['NegativeRSIDivergence'][i]:
                ax3.axvspan(data.index[i], data.index[i], color='orange', alpha=0.5)

        # Shade Green for Positive Signals
        for i in range(len(data)):
            if data['Signal'][i] == 1:
                ax1.axvspan(data.index[i], data.index[i], color='green', alpha=0.1)

        # Formatting
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.title(f'{asset} Chart')

        # Combine legends from all axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        fig.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc=(0.1, 0.8), ncol=1, fancybox=True, shadow=True)

        plt.tight_layout()
        #plt.close(fig)  # Ensure the figure is closed


        # Get last 5 RSI Divergence signals
        rsi_divergence_signals = data[data['PositiveRSIDivergence'] == True].tail(10).index.tolist()

        # Get last 5 Positive signals
        positive_signals = data[data['Signal'] == 1].tail(10).index.tolist()

        # Add signals to DataFrame, including the price at the signal date
        for signal_date in rsi_divergence_signals:
            signal_price = data.loc[signal_date, 'Close']
            all_signals_df = pd.concat([all_signals_df, pd.DataFrame({'Date': [signal_date], 'Ticker': [asset], 'Price': [signal_price], 'RSI Divergence': [1], 'Positive Signal': [0]})], ignore_index=True)
        for signal_date in positive_signals:
            signal_price = data.loc[signal_date, 'Close']
            all_signals_df = pd.concat([all_signals_df, pd.DataFrame({'Date': [signal_date], 'Ticker': [asset], 'Price': [signal_price], 'RSI Divergence': [0], 'Positive Signal': [1]})], ignore_index=True)

        # Group by 'Date' and combine signals into a single row
        all_signals_df = all_signals_df.groupby('Date').agg({'Ticker': 'first', 'Price': 'first', 'RSI Divergence': 'sum', 'Positive Signal': 'sum'}).reset_index()

        # Sort the DataFrame by 'Date' in descending order (latest first)
        all_signals_df = all_signals_df.sort_values('Date', ascending=False)

        # Highlight rows where both signals are on
        def highlight_both_signals(row):
            if row['RSI Divergence'] == 1 and row['Positive Signal'] == 1:
                return ['background-color: green'] * len(row)
            return [''] * len(row)

        all_signals_df = all_signals_df.style.apply(highlight_both_signals, axis=1)

        # Convert the 'Date' column to the desired format
        all_signals_df['Date'] = pd.to_datetime(all_signals_df['Date']).dt.strftime('%Y-%M-%D')

        # plt.tight_layout() # just added 
        # plt.close(fig) # just added 

        



    except Exception as e:
        print(f"Error processing {asset}: {e}")
        fig = plt.figure()  # Create an empty figure as a fallback
        plt.text(0.5, 0.5, f"Error: {e}", fontsize=12, ha='center')
        all_signals_df = pd.DataFrame(columns=['Symbol', 'Signal Type', 'Date', 'Price'])


    return {"chart_figure": fig, "signals_data": all_signals_df}



def drawdown_analysis(asset, start_date, end_date):
    """Generates and displays a drawdown chart for the given asset."""

    try:
        data = yf.download(asset, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for {asset} between {start_date} and {end_date}.")
            return None  # Return None to indicate no chart

        # Calculate drawdown
        mu_prices = data['Adj Close']
        peak = mu_prices.cummax()
        drawdown = (mu_prices - peak) / peak

        # Plot the drawdown
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(drawdown)
        ax.set_title(f'{asset} Drawdown from Previous High')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')

        return fig  # Return the figure

    except Exception as e:
        print(f"Error processing {asset} in drawdown analysis: {e}")
        return None  # Return None to indicate no chart



st.title("Technical Analysis")



# Input fields for ticker, start date, and end date
asset = st.text_input("Enter Asset Ticker (e.g., AAPL):", "AAPL")
default_start_date = datetime.date.today() - datetime.timedelta(days=1095) 
start_date = st.date_input("Start Date:", value=default_start_date)
end_date = st.date_input("End Date:")

# Button to trigger analysis
if st.button("Generate Analysis"):
    # Call your function to perform the analysis
    result = generate_stock_analysis(asset, start_date, end_date)
    drawdown_fig = drawdown_analysis(asset, start_date, end_date) # Call the new function


    # Display the chart
    # if result["chart_figure"] is not None:
    #     st.pyplot(result["chart_figure"])
    if result["chart_figure"]:
        st.pyplot(result["chart_figure"])
    else:
        st.error("Failed to generate chart.")

    #st.write(result["signals_data"])  # Display signals data


    # Display the drawdown chart
    if drawdown_fig is not None:
        st.pyplot(drawdown_fig)  # Display using st.pyplot
    
    
    # Display the signals data
    if result["signals_data"] is not None:
        st.write("Signals:")
        st.dataframe(result["signals_data"])