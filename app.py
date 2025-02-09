import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.title("Tech Analysis")

# User inputs
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL").upper()
period = st.selectbox("Select Time Period", ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"), index=3)
num_bins = st.slider("Number of Volume Profile Bins", 5, 50, 40)
volume_profile_width = 20 #st.slider("Volume Profile Width (%)", 1, 50, 20)  # in percent
up_down_volume_days = st.slider("Up/Down Volume Days (x)", 10, 60, 30)

@st.cache_data
def load_data(ticker, period):
    data = yf.download(tickers=ticker, period=period, group_by='Ticker')
    if data.empty:
        st.error(f"No data found for {ticker} with the period {period}.")
        return None
    if len(data.columns.levels) > 1:
        data = data[ticker]
    return data

data = load_data(ticker, period)

def calculate_volume_by_price(data, num_bins=20):
    """Calculates volume by price. Returns a Pandas Series."""
    min_price = data['Low'].min()
    max_price = data['High'].max()
    price_range = max_price - min_price
    bin_size = price_range / num_bins
    volume_by_price = {}
    for i in range(num_bins):
        lower_bound = min_price + i * bin_size
        upper_bound = min_price + (i + 1) * bin_size
        mask = (data['Close'] >= lower_bound) & (data['Close'] < upper_bound)
        volume_by_price[(lower_bound + upper_bound) / 2] = data.loc[mask, 'Volume'].sum()
    return pd.Series(volume_by_price)

if data is not None:
    volume_profile = calculate_volume_by_price(data, num_bins)

    def plot_stock_with_all_signals(data, symbol, volume_profile=None):
        try:
            if 'Close' not in data.columns:
                st.error(f"Error: 'Close' column not found in data for {symbol}.")
                return None

            data['Close'] = data['Close'].astype(float)

            # --- Calculations ---
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            data['ValueWeightedPrice'] = (data['Close'] * data['Volume']).rolling(window=30).sum() / data['Volume'].rolling(window=30).sum()

            # Volume Ratio
            try:
                shares_outstanding = yf.Ticker(symbol).info.get('sharesOutstanding')
                if shares_outstanding is not None:
                    data['VolumeRatio'] = data['Volume'].rolling(window=30).sum() / shares_outstanding
                else:
                    st.warning(f"Shares outstanding not found for {symbol}. Using volume mean.")
                    data['VolumeRatio'] = data['Volume'].rolling(window=30).mean()  # Fallback
            except Exception as e:
                st.warning(f"Error getting shares outstanding: {e}. Using volume mean.")
                data['VolumeRatio'] = data['Volume'].rolling(window=30).mean() # Fallback

            # RSI
            def rsi(close, length=14):
                delta = close.diff()
                up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
                roll_up1 = up.ewm(alpha=1/length).mean()
                roll_down1 = down.ewm(alpha=1/length).mean()
                RS = roll_up1 / roll_down1
                RSI = 100.0 - (100.0 / (1.0 + RS))
                return RSI

            data['RSI'] = rsi(data['Close'], length=14)

            # MACD
            def macd(close, fast=20, slow=40, signal=20):
                EMAfast = close.ewm(span=fast, min_periods=fast).mean()
                EMAslow = close.ewm(span=slow, min_periods=slow).mean()
                MACD = EMAfast - EMAslow
                MACD_signal = MACD.ewm(span=signal, min_periods=signal).mean()
                MACD_histogram = MACD - MACD_signal
                return MACD, MACD_signal, MACD_histogram

            data['MACD_20_40_20'], data['MACDs_20_40_20'], data['MACDh_20_40_20'] = macd(data['Close'], fast=20, slow=40, signal=20)

            # RSI Divergence
            def calculate_rsi_divergence(close, rsi, lookback, price_thresh, rsi_thresh):
                pos_div = pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
                neg_div = pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
                for i in range(lookback, len(close)):
                    if (close.iloc[i] is None or close.iloc[i - lookback] is None or rsi.iloc[i] is None or rsi.iloc[i - lookback] is None):
                        continue
                    price_lower, rsi_higher = close.iloc[i] < close.iloc[i - lookback], rsi.iloc[i] > rsi.iloc[i - lookback]
                    price_higher, rsi_lower = close.iloc[i] > close.iloc[i - lookback], rsi.iloc[i] < rsi.iloc[i - lookback]
                    if close.iloc[i - lookback] == 0: continue
                    price_change_pos = (close.iloc[i - lookback] - close.iloc[i]) / close.iloc[i - lookback]
                    price_change_neg = (close.iloc[i] - close.iloc[i-lookback]) / close.iloc[i - lookback]
                    rsi_change_pos = rsi.iloc[i] - rsi.iloc[i - lookback]
                    rsi_change_neg = rsi.iloc[i - lookback] - rsi.iloc[i]
                    if price_lower and rsi_higher and price_change_pos > price_thresh and rsi_change_pos > rsi_thresh:
                        pos_div.iloc[i] = True
                    if price_higher and rsi_lower and price_change_neg > price_thresh and rsi_change_neg > rsi_thresh:
                        neg_div.iloc[i] = True
                return pos_div, neg_div

            lookback_period, price_threshold, rsi_threshold = 14, 0.03, 5
            data['PositiveRSIDivergence'], data['NegativeRSIDivergence'] = calculate_rsi_divergence(data['Close'], data['RSI'], lookback_period, price_threshold, rsi_threshold)

            # MACD Signals
            data['PositiveMACDSignal'] = data['MACD_20_40_20'] <= (-0.04 * data['Close'])
            data['NegativeMACDSignal'] = data['MACD_20_40_20'] >= (0.04 * data['Close'])

            # Main Signal
            volume_ratio_threshold = max(0.5, data['VolumeRatio'].mean() + data['VolumeRatio'].std())
            data['Signal'] = (data['Close'] < data['ValueWeightedPrice']) & (data['VolumeRatio'] > volume_ratio_threshold) & (data['RSI'] < 40)
            data['Signal'] = data['Signal'].astype(int)

            # Up/Down Volume Calculation
            if shares_outstanding is not None:
                data['VolumePercent'] = data['Volume'] / shares_outstanding
                avg_volume_percent = data['VolumePercent'].mean()
                std_volume_percent = data['VolumePercent'].std()
                data['StdevVolume'] = (data['VolumePercent'] - avg_volume_percent) / std_volume_percent
            else:
                avg_volume = data['Volume'].mean()
                std_volume = data['Volume'].std()
                data['StdevVolume'] = (data['Volume'] - avg_volume) / std_volume
                st.warning("Shares outstanding not found, using raw volume instead.")

            data['PriceChange'] = data['Close'].diff()
            data['UpDownVolume'] = np.where(data['PriceChange'] > 0, data['StdevVolume'], -data['StdevVolume'])
            data['UpDownVolume'] = data['UpDownVolume'].rolling(window=up_down_volume_days).mean()

            data['PositiveUpDownVolumeSignal'] = (data['UpDownVolume'] > 0.5).astype(int)
            data['NegativeUpDownVolumeSignal'] = (data['UpDownVolume'] < -0.5).astype(int)

            data.ffill(inplace=True)
            data.bfill(inplace=True)


            # --- Plotting ---
            fig, (ax1, ax_up_down, ax2, ax3) = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

            # -- Top plot (Price, Indicators, and Volume Profile) --
            ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
            ax1.plot(data.index, data['SMA200'], label='200-day SMA', color='red')
            ax1.plot(data.index, data['ValueWeightedPrice'], label='30-day VWAP', color='green')
            ax1.set_ylabel('Price')
            ax1.set_title(f'{symbol} - Price, Indicators, and Signals')
            ax1.grid(True)

            # -- Volume Profile (Corrected) --
            if volume_profile is not None:
                ax1v = ax1.twiny()  # Create a twin axis that shares the y-axis
                # Normalize the volume profile for consistent scaling
                normalized_volume = volume_profile / volume_profile.max()
                # Calculate the maximum x-position for the bars based on the price range *and user input*
                price_range = data['High'].max() - data['Low'].min()
                max_volume_x = price_range * (volume_profile_width / 100)  # Use user input

                # Correctly plot the horizontal bars on the twin axis
                ax1v.barh(volume_profile.index, normalized_volume * max_volume_x, color='purple', alpha=0.3,
                          height=(price_range / num_bins)*0.8)  # Consistent height

                ax1v.set_xlim(0, max_volume_x) # set x limit
                ax1v.invert_xaxis() # bars go left
                ax1v.spines[['top', 'bottom', 'right']].set_visible(False) #remove extra axis lines
                ax1v.tick_params(axis='x', colors='purple') # set tick color
                ax1v.set_xlabel("Volume", color='purple')  # Optional x-axis label
                ax1v.set_xticks([]) # remove xticks


            # -- Volume Ratio --
            ax1_2 = ax1.twinx()  # Create another twin axis for Volume Ratio
            ax1_2.plot(data.index, data['VolumeRatio'], label='30-day Volume Ratio', color='gray')
            ax1_2.set_ylabel('Volume Ratio', color='gray')
            ax1_2.tick_params(axis='y', labelcolor='gray')
            ax1_2.set_ylim(0, 1)  # Volume ratio is typically between 0 and 1

            # Combine legends from both ax1 and ax1_2
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines1_2, labels1_2 = ax1_2.get_legend_handles_labels()
            ax1.legend(lines1 + lines1_2, labels1 + labels1_2, loc='upper left') # combined legend


            # Highlight Main Signal
            for i, row in data.iterrows():
                if row['Signal'] == 1:
                    ax1.axvspan(i, i, color='green', alpha=0.3)

            # -- Up/Down Volume Plot --
            ax_up_down.plot(data.index, data['UpDownVolume'], label='Up/Down Volume', color='orange')
            ax_up_down.axhline(0.5, color='green', linestyle='--', label='+0.5 Threshold')
            ax_up_down.axhline(-0.5, color='red', linestyle='--', label='-0.5 Threshold')
            ax_up_down.set_ylabel('Up/Down Vol')
            ax_up_down.grid(True)
            ax_up_down.legend()

            # Highlight Up/Down Volume Signals
            for i, row in data.iterrows():
                if row['PositiveUpDownVolumeSignal'] == 1:
                    ax_up_down.axvspan(i, i, color='green', alpha=0.3)
                if row['NegativeUpDownVolumeSignal'] == 1:
                    ax_up_down.axvspan(i, i, color='red', alpha=0.3)


            # -- RSI Plot --
            ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
            ax2.set_ylabel('RSI')
            ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            ax2.legend()
            ax2.grid(True)

            # Highlight RSI Divergences
            for i, row in data.iterrows():
                if row['PositiveRSIDivergence']:
                    ax2.axvspan(i, i, color='green', alpha=0.3)
                if row['NegativeRSIDivergence']:
                    ax2.axvspan(i, i, color='red', alpha=0.3)

            # -- MACD Plot --
            ax3.plot(data.index, data['MACD_20_40_20'], label='MACD', color='blue')
            ax3.plot(data.index, data['MACDs_20_40_20'], label='Signal Line', color='red')
            ax3.bar(data.index, data['MACDh_20_40_20'], label='Histogram', color='gray')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True)

            # Highlight MACD Signals
            for i, row in data.iterrows():
                if row['PositiveMACDSignal']:
                    ax3.axvspan(i, i, color='green', alpha=0.3)
                if row['NegativeMACDSignal']:
                    ax3.axvspan(i, i, color='red', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # --- Combined Signal Table ---
            data['MainSignal'] = data['Signal'].astype(int)
            data['PositiveRSI'] = data['PositiveRSIDivergence'].astype(int)
            data['NegativeRSI'] = data['NegativeRSIDivergence'].astype(int)
            data['PositiveMACD'] = data['PositiveMACDSignal'].astype(int)
            data['NegativeMACD'] = data['NegativeMACDSignal'].astype(int)

            signal_cols = ['Close', 'MainSignal', 'PositiveRSI', 'NegativeRSI', 'PositiveMACD', 'NegativeMACD', 'PositiveUpDownVolumeSignal', 'NegativeUpDownVolumeSignal']
            signal_data = data[(data['MainSignal'] == 1) | (data['PositiveRSI'] == 1) | (data['NegativeRSI'] == 1) |
                               (data['PositiveMACD'] == 1) | (data['NegativeMACD'] == 1) |
                               (data['PositiveUpDownVolumeSignal'] == 1) | (data['NegativeUpDownVolumeSignal'] == 1)][signal_cols].tail(10).copy()

            def highlight_positive_signals(row):
                highlight = 'background-color: yellow;'
                default = ''
                if row['MainSignal'] == 1 and row['PositiveRSI'] == 1 and row['PositiveMACD'] == 1:
                    return [highlight] * len(row)
                else:
                    return [default] * len(row)

            if not signal_data.empty:
                signal_data.reset_index(inplace=True)
                signal_data.rename(columns={'Date': 'Date', 'Close': 'Price'}, inplace=True)
                signal_data['Date'] = pd.to_datetime(signal_data['Date']).dt.strftime('%Y-%m-%d')
                signal_data['Ticker'] = symbol
                signal_cols = ['Date', 'Ticker', 'Price', 'MainSignal', 'PositiveRSI', 'NegativeRSI', 'PositiveMACD', 'NegativeMACD', 'PositiveUpDownVolumeSignal', 'NegativeUpDownVolumeSignal']
                signal_data = signal_data[signal_cols]
                signal_data = signal_data.sort_values(by='Date', ascending=False)
                st.write("\nLast 10 Signals (All Types):")
                styled_df = signal_data.style.apply(highlight_positive_signals, axis=1)
                st.dataframe(styled_df)
            else:
                st.info("\nNo signals generated.")
            return data

        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")
            return None

    processed_data = plot_stock_with_all_signals(data, ticker, volume_profile)
else:
    st.stop()

# Explanation:
# This Streamlit app takes a ticker symbol and a time period as input from the user.
# It uses yfinance to download the historical stock data for the given ticker and period.
# The downloaded data is then displayed as a Pandas DataFrame.
# The `plot_stock_with_all_signals` function is called which:
#   - calculates additional technical indicators like SMA, VWAP, RSI, and MACD.
#   - identifies potential buy/sell signals based on these indicators.
#   - generates a plot showing the stock price, indicators, and signals.
#   - creates a table summarizing the latest signals.
# The resulting plot and signal table are displayed in the Streamlit app.
# A caching mechanism is used to improve performance by storing downloaded data.
# Basic error handling is implemented to inform the user if there are issues with the data or ticker.
# If the user provides invalid inputs or if there are errors during data processing, the app displays an error message.
# The app uses a twin axis plot which can be difficult to understand.