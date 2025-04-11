import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # Needed for percentage formatting
import pandas as pd
import numpy as np
import datetime # Needed for drawdown potential date handling (though less critical here)

# st.title("Tech Analysis:")

# User inputs
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL").upper()
period = st.selectbox("Select Time Period", ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"), index=3)
num_bins = st.slider("Number of Volume Profile Bins", 5, 50, 40)
volume_profile_width = 20 #st.slider("Volume Profile Width (%)", 1, 50, 20)  # in percent
up_down_volume_days = st.slider("Up/Down Volume Days (x)", 10, 60, 30)

@st.cache_data
def load_data(ticker, period):
    """Downloads stock data using yfinance."""
    try:
        # For 'max' period, yfinance doesn't need start/end
        if period == 'max':
             data = yf.download(tickers=ticker, period=period, group_by='Ticker')
        else:
            # For fixed periods like '1y', '6mo', yfinance handles the date range
             data = yf.download(tickers=ticker, period=period, group_by='Ticker')

        if data.empty:
            st.error(f"No data found for {ticker} with the period {period}.")
            return None
        # Handle potential MultiIndex columns if multiple tickers were downloaded (though unlikely here)
        if isinstance(data.columns, pd.MultiIndex):
            data = data[ticker]
            # Or if the first level is the ticker: data.columns = data.columns.droplevel(0)
        # Ensure columns are standard (sometimes yfinance returns lowercase)
        data.columns = [col.capitalize() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return None


data = load_data(ticker, period)

def calculate_volume_by_price(data, num_bins=20):
    """Calculates volume by price. Returns a Pandas Series."""
    if data is None or 'Low' not in data.columns or 'High' not in data.columns or data[['Low', 'High']].isnull().values.any():
        st.warning("Could not calculate volume profile due to missing price data.")
        return None
    min_price = data['Low'].min()
    max_price = data['High'].max()
    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        st.warning("Could not calculate volume profile due to invalid price range.")
        return None
    price_range = max_price - min_price
    bin_size = price_range / num_bins
    if bin_size <= 0:
         st.warning("Could not calculate volume profile due to zero bin size.")
         return None

    volume_by_price = {}
    for i in range(num_bins):
        lower_bound = min_price + i * bin_size
        upper_bound = min_price + (i + 1) * bin_size
        # Ensure we include the highest price in the last bin
        if i == num_bins - 1:
            mask = (data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)
        else:
            mask = (data['Close'] >= lower_bound) & (data['Close'] < upper_bound)
        volume_by_price[(lower_bound + upper_bound) / 2] = data.loc[mask, 'Volume'].sum()
    return pd.Series(volume_by_price)


if data is not None:
    volume_profile = calculate_volume_by_price(data, num_bins)

    def plot_stock_with_all_signals(data, symbol, volume_profile=None):
        """Calculates indicators, plots charts (including drawdown), and shows signals."""
        try:
            # --- Data Validation ---
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                st.error(f"Error: Missing one or more required columns ({', '.join(required_cols)}) in data for {symbol}.")
                return None
            if data['Close'].isnull().all():
                st.error(f"Error: 'Close' column contains only NaN values for {symbol}.")
                return None

            # Ensure Close is float, handling potential errors
            try:
                data['Close'] = data['Close'].astype(float)
                data['Volume'] = data['Volume'].astype(float) # Also ensure Volume is float
            except ValueError as e:
                 st.error(f"Error converting 'Close' or 'Volume' to numeric: {e}")
                 return None

            # Handle potential initial NaNs from rolling calculations
            data = data.copy() # Avoid SettingWithCopyWarning


            # --- Calculations ---
            data['SMA200'] = data['Close'].rolling(window=200, min_periods=1).mean() # Use min_periods
            data['ValueWeightedPrice'] = (data['Close'] * data['Volume']).rolling(window=30, min_periods=1).sum() / data['Volume'].rolling(window=30, min_periods=1).sum()

            # Volume Ratio
            shares_outstanding = None # Initialize
            try:
                ticker_info = yf.Ticker(symbol).info
                shares_outstanding = ticker_info.get('sharesOutstanding')
                if shares_outstanding is None or shares_outstanding == 0:
                    st.warning(f"Shares outstanding not found or is zero for {symbol}. Using volume mean for ratio.")
                    shares_outstanding = None # Force fallback
                    data['VolumeRatio'] = data['Volume'].rolling(window=30, min_periods=1).mean() / data['Volume'].mean() # Relative mean
                else:
                     # Calculate rolling sum of volume / shares outstanding
                    rolling_vol_sum = data['Volume'].rolling(window=30, min_periods=1).sum()
                    data['VolumeRatio'] = rolling_vol_sum / shares_outstanding
            except Exception as e:
                st.warning(f"Error getting shares outstanding for {symbol}: {e}. Using relative volume mean.")
                data['VolumeRatio'] = data['Volume'].rolling(window=30, min_periods=1).mean() / data['Volume'].mean() # Fallback

            # --- Drawdown Calculation (Integrated) ---
            peak = data['Close'].cummax()
            data['Drawdown'] = (data['Close'] - peak) / peak
            # Replace potential division by zero (if peak is 0) or inf results with 0
            data['Drawdown'].replace([np.inf, -np.inf], np.nan, inplace=True)
            data['Drawdown'].fillna(0, inplace=True) # Fill any NaNs (e.g., from division by zero) with 0 drawdown


            # RSI
            def rsi(close, length=14):
                delta = close.diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                # Use Wilder's smoothing (equivalent to EMA with alpha = 1/length)
                roll_up = up.ewm(alpha=1/length, adjust=False).mean()
                roll_down = down.ewm(alpha=1/length, adjust=False).mean()

                # Calculate RS and RSI
                rs = roll_up / roll_down
                rsi = 100.0 - (100.0 / (1.0 + rs))
                 # Handle potential division by zero if roll_down is 0
                rsi = rsi.replace([np.inf, -np.inf], 100).fillna(50) # Assign 100 if down=0 (strong uptrend), 50 if NaN
                return rsi

            data['RSI'] = rsi(data['Close'], length=14)

            # MACD
            def macd(close, fast=20, slow=40, signal=20):
                ema_fast = close.ewm(span=fast, adjust=False).mean()
                ema_slow = close.ewm(span=slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal, adjust=False).mean()
                histogram = macd_line - signal_line
                return macd_line, signal_line, histogram

            data['MACD_20_40_20'], data['MACDs_20_40_20'], data['MACDh_20_40_20'] = macd(data['Close'], fast=20, slow=40, signal=20)

            # RSI Divergence
            def calculate_rsi_divergence(close, rsi, lookback, price_thresh, rsi_thresh):
                pos_div = pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
                neg_div = pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
                # Ensure inputs are Series
                close = pd.Series(close)
                rsi = pd.Series(rsi)
                for i in range(lookback, len(close)):
                    # Skip if data is missing
                    if pd.isna(close.iloc[i]) or pd.isna(close.iloc[i - lookback]) or pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - lookback]):
                        continue
                    # Avoid division by zero
                    if close.iloc[i - lookback] == 0: continue

                    price_lower = close.iloc[i] < close.iloc[i - lookback]
                    rsi_higher = rsi.iloc[i] > rsi.iloc[i - lookback]
                    price_higher = close.iloc[i] > close.iloc[i - lookback]
                    rsi_lower = rsi.iloc[i] < rsi.iloc[i - lookback]

                    # Calculate percentage/absolute changes
                    price_change_pos = (close.iloc[i - lookback] - close.iloc[i]) / close.iloc[i - lookback] # Positive divergence: price fell
                    price_change_neg = (close.iloc[i] - close.iloc[i-lookback]) / close.iloc[i - lookback] # Negative divergence: price rose
                    rsi_change_pos = rsi.iloc[i] - rsi.iloc[i - lookback] # Positive divergence: RSI rose
                    rsi_change_neg = rsi.iloc[i - lookback] - rsi.iloc[i] # Negative divergence: RSI fell

                    # Check thresholds
                    if price_lower and rsi_higher and price_change_pos > price_thresh and rsi_change_pos > rsi_thresh:
                        pos_div.iloc[i] = True
                    if price_higher and rsi_lower and price_change_neg > price_thresh and rsi_change_neg > rsi_thresh:
                        neg_div.iloc[i] = True
                return pos_div, neg_div

            lookback_period, price_threshold, rsi_threshold = 14, 0.03, 5 # 3% price change, 5 points RSI change
            data['PositiveRSIDivergence'], data['NegativeRSIDivergence'] = calculate_rsi_divergence(data['Close'], data['RSI'], lookback_period, price_threshold, rsi_threshold)

            # MACD Signals
            data['PositiveMACDSignal'] = data['MACD_20_40_20'] <= (-0.04 * data['Close'])
            data['NegativeMACDSignal'] = data['MACD_20_40_20'] >= (0.04 * data['Close'])

            # Main Signal
            # Define threshold dynamically: mean + 1 std dev, but at least 0.005 (0.5%) if shares outstanding known, or relative if not
            if shares_outstanding:
                vol_ratio_mean = data['VolumeRatio'].mean()
                vol_ratio_std = data['VolumeRatio'].std()
                volume_ratio_threshold = max(0.005, vol_ratio_mean + vol_ratio_std) # At least 0.5% turnover
            else:
                # Use relative threshold if ratio is based on relative mean
                vol_ratio_mean = data['VolumeRatio'].mean()
                vol_ratio_std = data['VolumeRatio'].std()
                volume_ratio_threshold = vol_ratio_mean + vol_ratio_std # Compare relative value to its own stats

            data['Signal'] = (data['Close'] < data['ValueWeightedPrice']) & (data['VolumeRatio'] > volume_ratio_threshold) & (data['RSI'] < 40)
            data['Signal'] = data['Signal'].astype(int)

            # Up/Down Volume Calculation
            data['PriceChange'] = data['Close'].diff()
            # Calculate StdevVolume based on availability of shares outstanding
            if shares_outstanding is not None and shares_outstanding > 0:
                data['VolumePercent'] = data['Volume'] / shares_outstanding
                avg_volume_percent = data['VolumePercent'].mean()
                std_volume_percent = data['VolumePercent'].std()
                # Avoid division by zero if std dev is 0
                if std_volume_percent > 0:
                     data['StdevVolume'] = (data['VolumePercent'] - avg_volume_percent) / std_volume_percent
                else:
                     data['StdevVolume'] = 0 # Or some other default if volume never changes
            else:
                avg_volume = data['Volume'].mean()
                std_volume = data['Volume'].std()
                if not shares_outstanding: # Only warn if we expected it but failed
                     st.warning("Shares outstanding not found, using raw volume for Stdev Calc.")
                if std_volume > 0:
                    data['StdevVolume'] = (data['Volume'] - avg_volume) / std_volume
                else:
                    data['StdevVolume'] = 0

            data['UpDownVolumeRaw'] = np.where(data['PriceChange'] > 0, data['StdevVolume'], np.where(data['PriceChange'] < 0, -data['StdevVolume'], 0))
            data['UpDownVolume'] = data['UpDownVolumeRaw'].rolling(window=up_down_volume_days, min_periods=1).mean()

            data['PositiveUpDownVolumeSignal'] = (data['UpDownVolume'] > 0.5).astype(int)
            data['NegativeUpDownVolumeSignal'] = (data['UpDownVolume'] < -0.5).astype(int)

            # Final fillna after all calculations
            # Use ffill first then bfill to handle NaNs at the beginning
            data.ffill(inplace=True)
            data.bfill(inplace=True)


            # --- Plotting ---
            # INCREASE number of subplots to 5
            fig, (ax1, ax_drawdown, ax_up_down, ax2, ax3) = plt.subplots(
                5, 1, figsize=(14, 20), sharex=True, # Increased height
                gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]} # Give more height to main price chart
            )
            plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style


            # -- Panel 1: Price, Indicators, Volume Profile, Volume Ratio --
            ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
            ax1.plot(data.index, data['SMA200'], label='SMA 200', color='red', linestyle='--', linewidth=1)
            ax1.plot(data.index, data['ValueWeightedPrice'], label=f'VWAP {30}-day', color='green', linestyle=':', linewidth=1)
            ax1.set_ylabel('Price ($)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title(f'{symbol} Technical Analysis ({data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")})', fontsize=14)
            #ax1.grid(True) # Grid is handled by style

            # Highlight Main Signal
            main_signal_dates = data.index[data['Signal'] == 1]
            if not main_signal_dates.empty:
                # Use small markers instead of vspan for clarity if many signals
                ax1.plot(main_signal_dates, data.loc[main_signal_dates, 'Close'], '^', color='lime', markersize=6, label='Main Signal')
                # Add alpha transparency to avoid obscuring lines if using vspan:
                #for date in main_signal_dates:
                #    ax1.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=0.5), color='green', alpha=0.2)

            # Volume Profile
            if volume_profile is not None and not volume_profile.empty:
                ax1v = ax1.twiny()
                normalized_volume = volume_profile / volume_profile.max()
                # Determine width based on price range and percentage
                price_min, price_max = ax1.get_ylim() # Get current y limits of price axis
                plot_width = (price_max - price_min) * (volume_profile_width / 100.0)

                ax1v.barh(volume_profile.index, normalized_volume * plot_width, color='purple', alpha=0.3,
                          height=(price_max - price_min) / num_bins * 0.8) # Adjust height based on bins

                ax1v.set_xlim(plot_width, 0) # Invert x-axis to plot on the left
                ax1v.set_xticks([]) # Hide volume profile x-ticks
                ax1v.set_xlabel("Vol Profile", color='purple', alpha=0.7)
                ax1v.tick_params(axis='x', colors='purple')
                ax1v.spines[['top', 'bottom', 'left','right']].set_visible(False) # Cleaner look
            else:
                ax1.text(0.02, 0.95, "Volume Profile Unavailable", transform=ax1.transAxes, color='red', alpha=0.7, ha='left', va='top')


            # Volume Ratio
            ax1_2 = ax1.twinx()
            ax1_2.plot(data.index, data['VolumeRatio'], label=f'Vol Ratio {30}-day', color='gray', linestyle='-', linewidth=1, alpha=0.7)
            ax1_2.set_ylabel('Volume Ratio', color='gray', alpha=0.8)
            ax1_2.tick_params(axis='y', labelcolor='gray', colors='gray')
            ax1_2.spines['right'].set_color('gray') # Color the axis spine
            # Set ylim based on ratio type for better scale
            if shares_outstanding:
                ax1_2.set_ylim(bottom=0, top=max(0.1, data['VolumeRatio'].max() * 1.1)) # Sensible upper limit > 0
                ax1_2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)) # Format as percentage if absolute ratio
            else:
                 ax1_2.set_ylim(bottom=0) # Relative ratio, just start at 0

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines1_2, labels1_2 = ax1_2.get_legend_handles_labels()
            ax1.legend(lines1 + lines1_2, labels1 + labels1_2, loc='upper left', fontsize=8)


            # -- Panel 2: Drawdown Plot (NEW) --
            ax_drawdown.plot(data.index, data['Drawdown'], label='Drawdown', color='cornflowerblue', linewidth=1)
            ax_drawdown.set_ylabel('Drawdown', color='cornflowerblue')
            ax_drawdown.tick_params(axis='y', labelcolor='cornflowerblue')
            ax_drawdown.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)) # Format as percentage
            ax_drawdown.grid(True, linestyle=':', alpha=0.6)
            ax_drawdown.set_title('Price Drawdown from Peak', loc='left', fontsize=9, alpha=0.8)
            # Optional: Annotate last drawdown point
            last_drawdown_value = data['Drawdown'].iloc[-1]
            last_date = data.index[-1]
            ax_drawdown.plot(last_date, last_drawdown_value, 'ro', markersize=4)
            ax_drawdown.text(last_date, last_drawdown_value, f' {last_drawdown_value:.1%}',
                              verticalalignment='center', horizontalalignment='left', fontsize=8, color='red')


            # -- Panel 3: Up/Down Volume Plot --
            ax_up_down.plot(data.index, data['UpDownVolume'], label=f'Up/Down Vol ({up_down_volume_days}d avg)', color='orange', linewidth=1)
            ax_up_down.axhline(0.5, color='green', linestyle='--', linewidth=0.8, label='+0.5 Threshold')
            ax_up_down.axhline(-0.5, color='red', linestyle='--', linewidth=0.8, label='-0.5 Threshold')
            ax_up_down.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5) # Zero line
            ax_up_down.set_ylabel('Up/Down Vol', color='orange')
            ax_up_down.tick_params(axis='y', labelcolor='orange')
            ax_up_down.grid(True, linestyle=':', alpha=0.6)
            ax_up_down.set_title('Up/Down Volume Momentum', loc='left', fontsize=9, alpha=0.8)
            # Highlight Up/Down Volume Signals (Using markers for less clutter)
            pos_udv_dates = data.index[data['PositiveUpDownVolumeSignal'] == 1]
            neg_udv_dates = data.index[data['NegativeUpDownVolumeSignal'] == 1]
            if not pos_udv_dates.empty:
                ax_up_down.plot(pos_udv_dates, data.loc[pos_udv_dates, 'UpDownVolume'], 'go', markersize=4, alpha=0.7, label='Pos Signal')
            if not neg_udv_dates.empty:
                ax_up_down.plot(neg_udv_dates, data.loc[neg_udv_dates, 'UpDownVolume'], 'ro', markersize=4, alpha=0.7, label='Neg Signal')
            ax_up_down.legend(fontsize=8, loc='upper left')


            # -- Panel 4: RSI Plot --
            ax2.plot(data.index, data['RSI'], label='RSI (14)', color='purple', linewidth=1)
            ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='Overbought (70)')
            ax2.axhline(50, color='gray', linestyle=':', linewidth=0.8)
            ax2.axhline(30, color='green', linestyle='--', linewidth=0.8, label='Oversold (30)')
            ax2.set_ylabel('RSI', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.set_ylim(0, 100) # RSI range
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.set_title('Relative Strength Index (RSI)', loc='left', fontsize=9, alpha=0.8)
            # Highlight RSI Divergences (Using markers)
            pos_rsi_div_dates = data.index[data['PositiveRSIDivergence']]
            neg_rsi_div_dates = data.index[data['NegativeRSIDivergence']]
            if not pos_rsi_div_dates.empty:
                ax2.plot(pos_rsi_div_dates, data.loc[pos_rsi_div_dates, 'RSI'], 'g^', markersize=5, alpha=0.8, label='Pos Div')
            if not neg_rsi_div_dates.empty:
                 ax2.plot(neg_rsi_div_dates, data.loc[neg_rsi_div_dates, 'RSI'], 'rv', markersize=5, alpha=0.8, label='Neg Div')
            ax2.legend(fontsize=8, loc='upper left')


            # -- Panel 5: MACD Plot --
            ax3.plot(data.index, data['MACD_20_40_20'], label='MACD (20,40)', color='blue', linewidth=1)
            ax3.plot(data.index, data['MACDs_20_40_20'], label='Signal (20)', color='red', linestyle=':', linewidth=1)
            # Use step plot for histogram for better visualization
            colors = ['green' if x >= 0 else 'red' for x in data['MACDh_20_40_20']]
            ax3.bar(data.index, data['MACDh_20_40_20'], label='Histogram (20)', color=colors, width=1.0, alpha=0.5) # Adjust width as needed
            ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5) # Zero line
            ax3.set_xlabel('Date')
            ax3.set_ylabel('MACD', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            ax3.grid(True, linestyle=':', alpha=0.6)
            ax3.set_title('MACD (20, 40, 20)', loc='left', fontsize=9, alpha=0.8)
            # Highlight MACD Signals (Using markers)
            pos_macd_sig_dates = data.index[data['PositiveMACDSignal']]
            neg_macd_sig_dates = data.index[data['NegativeMACDSignal']]
            if not pos_macd_sig_dates.empty:
                 ax3.plot(pos_macd_sig_dates, data.loc[pos_macd_sig_dates, 'MACD_20_40_20'], 'g^', markersize=5, alpha=0.8, label='Pos Signal')
            if not neg_macd_sig_dates.empty:
                 ax3.plot(neg_macd_sig_dates, data.loc[neg_macd_sig_dates, 'MACD_20_40_20'], 'rv', markersize=5, alpha=0.8, label='Neg Signal')
            ax3.legend(fontsize=8, loc='upper left')


            # --- Final Plot Adjustments ---
            plt.tight_layout(h_pad=1.5) # Add some vertical padding between plots
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory

            # --- Combined Signal Table ---
            data['MainSignal'] = data['Signal'].astype(int) # Already calculated
            data['PositiveRSI'] = data['PositiveRSIDivergence'].astype(int)
            data['NegativeRSI'] = data['NegativeRSIDivergence'].astype(int)
            data['PositiveMACD'] = data['PositiveMACDSignal'].astype(int)
            data['NegativeMACD'] = data['NegativeMACDSignal'].astype(int)
            # Up/Down Volume signals already calculated

            signal_cols_to_check = ['MainSignal', 'PositiveRSI', 'NegativeRSI', 'PositiveMACD', 'NegativeMACD', 'PositiveUpDownVolumeSignal', 'NegativeUpDownVolumeSignal']
            signal_cols_display = ['Close'] + signal_cols_to_check # Columns to show in table

            # Filter rows where *any* signal is active
            signal_active_mask = data[signal_cols_to_check].any(axis=1)
            signal_data = data.loc[signal_active_mask, signal_cols_display].tail(15).copy() # Show last 15 signals

            def highlight_signals(row):
                styles = [''] * len(row) # Default no style
                # Highlight positive signals green, negative red
                if row['MainSignal'] == 1: styles[row.index.get_loc('MainSignal')] = 'background-color: lightgreen; color: black;'
                if row['PositiveRSI'] == 1: styles[row.index.get_loc('PositiveRSI')] = 'background-color: lightgreen; color: black;'
                if row['PositiveMACD'] == 1: styles[row.index.get_loc('PositiveMACD')] = 'background-color: lightgreen; color: black;'
                if row['PositiveUpDownVolumeSignal'] == 1: styles[row.index.get_loc('PositiveUpDownVolumeSignal')] = 'background-color: lightgreen; color: black;'

                if row['NegativeRSI'] == 1: styles[row.index.get_loc('NegativeRSI')] = 'background-color: salmon; color: black;'
                if row['NegativeMACD'] == 1: styles[row.index.get_loc('NegativeMACD')] = 'background-color: salmon; color: black;'
                if row['NegativeUpDownVolumeSignal'] == 1: styles[row.index.get_loc('NegativeUpDownVolumeSignal')] = 'background-color: salmon; color: black;'
                return styles

            if not signal_data.empty:
                signal_data.reset_index(inplace=True) # Make Date a column
                signal_data.rename(columns={'Date': 'Date', 'Close': 'Price'}, inplace=True)
                signal_data['Date'] = pd.to_datetime(signal_data['Date']).dt.strftime('%Y-%m-%d')
                signal_data['Ticker'] = symbol
                # Reorder columns for the table
                display_order = ['Date', 'Ticker', 'Price', 'MainSignal', 'PositiveUpDownVolumeSignal', 'PositiveRSI', 'PositiveMACD', 'NegativeUpDownVolumeSignal', 'NegativeRSI', 'NegativeMACD']
                signal_data = signal_data[display_order]
                signal_data = signal_data.sort_values(by='Date', ascending=False)

                st.write(f"\nLast {len(signal_data)} Signal Occurrences:")
                # Apply styling and formatting
                styled_df = signal_data.style.apply(highlight_signals, axis=1).format({'Price': '${:,.2f}'}) # Format price
                st.dataframe(styled_df, hide_index=True) # Hide the default index
            else:
                st.info(f"No signals generated for {symbol} in the selected period.")

            return data # Return processed data if needed elsewhere

        except Exception as e:
            st.error(f"An error occurred during analysis or plotting for {symbol}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}") # More detailed error for debugging
            return None

    # --- Run the Analysis and Plotting ---
    processed_data = plot_stock_with_all_signals(data, ticker, volume_profile)

else:
    st.warning("Data could not be loaded. Please check the ticker symbol and period.")
    st.stop()


# Explanation Block (Optional - you can keep or remove this)
# st.markdown("""
# ---
# ### Explanation
# *   **Top Panel:** Shows Close Price, 200-day SMA, 30-day VWAP-like average, Volume Ratio (vs shares outstanding if available), and Volume Profile bars on the left.
# *   **Drawdown Panel:** Shows the percentage decline from the highest price reached previously within the selected period.
# *   **Up/Down Vol Panel:** Shows the rolling average of standardized volume weighted by daily price change direction. Helps gauge buying/selling pressure intensity.
# *   **RSI Panel:** Standard 14-period RSI with overbought/oversold levels and potential divergence markers.
# *   **MACD Panel:** Standard MACD (20, 40, 20 settings) with line, signal, histogram, and custom threshold signal markers.
# *   **Signal Table:** Lists the most recent dates where any of the defined signals (Main, Up/Down Vol, RSI Div, MACD Threshold) occurred. Positive signals are highlighted green, negative signals red.
# """)




# import streamlit as st
# import yfinance as yf
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # st.title("Tech Analysis:")

# # User inputs
# ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL").upper()
# period = st.selectbox("Select Time Period", ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"), index=3)
# num_bins = st.slider("Number of Volume Profile Bins", 5, 50, 40)
# volume_profile_width = 20 #st.slider("Volume Profile Width (%)", 1, 50, 20)  # in percent
# up_down_volume_days = st.slider("Up/Down Volume Days (x)", 10, 60, 30)

# @st.cache_data
# def load_data(ticker, period):
#     data = yf.download(tickers=ticker, period=period, group_by='Ticker')
#     if data.empty:
#         st.error(f"No data found for {ticker} with the period {period}.")
#         return None
#     if len(data.columns.levels) > 1:
#         data = data[ticker]
#     return data

# data = load_data(ticker, period)

# def calculate_volume_by_price(data, num_bins=20):
#     """Calculates volume by price. Returns a Pandas Series."""
#     min_price = data['Low'].min()
#     max_price = data['High'].max()
#     price_range = max_price - min_price
#     bin_size = price_range / num_bins
#     volume_by_price = {}
#     for i in range(num_bins):
#         lower_bound = min_price + i * bin_size
#         upper_bound = min_price + (i + 1) * bin_size
#         mask = (data['Close'] >= lower_bound) & (data['Close'] < upper_bound)
#         volume_by_price[(lower_bound + upper_bound) / 2] = data.loc[mask, 'Volume'].sum()
#     return pd.Series(volume_by_price)

# if data is not None:
#     volume_profile = calculate_volume_by_price(data, num_bins)

#     def plot_stock_with_all_signals(data, symbol, volume_profile=None):
#         try:
#             if 'Close' not in data.columns:
#                 st.error(f"Error: 'Close' column not found in data for {symbol}.")
#                 return None

#             data['Close'] = data['Close'].astype(float)

#             # --- Calculations ---
#             data['SMA200'] = data['Close'].rolling(window=200).mean()
#             data['ValueWeightedPrice'] = (data['Close'] * data['Volume']).rolling(window=30).sum() / data['Volume'].rolling(window=30).sum()

#             # Volume Ratio
#             try:
#                 shares_outstanding = yf.Ticker(symbol).info.get('sharesOutstanding')
#                 if shares_outstanding is not None:
#                     data['VolumeRatio'] = data['Volume'].rolling(window=30).sum() / shares_outstanding
#                 else:
#                     st.warning(f"Shares outstanding not found for {symbol}. Using volume mean.")
#                     data['VolumeRatio'] = data['Volume'].rolling(window=30).mean()  # Fallback
#             except Exception as e:
#                 st.warning(f"Error getting shares outstanding: {e}. Using volume mean.")
#                 data['VolumeRatio'] = data['Volume'].rolling(window=30).mean() # Fallback

#             # RSI
#             def rsi(close, length=14):
#                 delta = close.diff()
#                 up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
#                 roll_up1 = up.ewm(alpha=1/length).mean()
#                 roll_down1 = down.ewm(alpha=1/length).mean()
#                 RS = roll_up1 / roll_down1
#                 RSI = 100.0 - (100.0 / (1.0 + RS))
#                 return RSI

#             data['RSI'] = rsi(data['Close'], length=14)

#             # MACD
#             def macd(close, fast=20, slow=40, signal=20):
#                 EMAfast = close.ewm(span=fast, min_periods=fast).mean()
#                 EMAslow = close.ewm(span=slow, min_periods=slow).mean()
#                 MACD = EMAfast - EMAslow
#                 MACD_signal = MACD.ewm(span=signal, min_periods=signal).mean()
#                 MACD_histogram = MACD - MACD_signal
#                 return MACD, MACD_signal, MACD_histogram

#             data['MACD_20_40_20'], data['MACDs_20_40_20'], data['MACDh_20_40_20'] = macd(data['Close'], fast=20, slow=40, signal=20)

#             # RSI Divergence
#             def calculate_rsi_divergence(close, rsi, lookback, price_thresh, rsi_thresh):
#                 pos_div = pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
#                 neg_div = pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
#                 for i in range(lookback, len(close)):
#                     if (close.iloc[i] is None or close.iloc[i - lookback] is None or rsi.iloc[i] is None or rsi.iloc[i - lookback] is None):
#                         continue
#                     price_lower, rsi_higher = close.iloc[i] < close.iloc[i - lookback], rsi.iloc[i] > rsi.iloc[i - lookback]
#                     price_higher, rsi_lower = close.iloc[i] > close.iloc[i - lookback], rsi.iloc[i] < rsi.iloc[i - lookback]
#                     if close.iloc[i - lookback] == 0: continue
#                     price_change_pos = (close.iloc[i - lookback] - close.iloc[i]) / close.iloc[i - lookback]
#                     price_change_neg = (close.iloc[i] - close.iloc[i-lookback]) / close.iloc[i - lookback]
#                     rsi_change_pos = rsi.iloc[i] - rsi.iloc[i - lookback]
#                     rsi_change_neg = rsi.iloc[i - lookback] - rsi.iloc[i]
#                     if price_lower and rsi_higher and price_change_pos > price_thresh and rsi_change_pos > rsi_thresh:
#                         pos_div.iloc[i] = True
#                     if price_higher and rsi_lower and price_change_neg > price_thresh and rsi_change_neg > rsi_thresh:
#                         neg_div.iloc[i] = True
#                 return pos_div, neg_div

#             lookback_period, price_threshold, rsi_threshold = 14, 0.03, 5
#             data['PositiveRSIDivergence'], data['NegativeRSIDivergence'] = calculate_rsi_divergence(data['Close'], data['RSI'], lookback_period, price_threshold, rsi_threshold)

#             # MACD Signals
#             data['PositiveMACDSignal'] = data['MACD_20_40_20'] <= (-0.04 * data['Close'])
#             data['NegativeMACDSignal'] = data['MACD_20_40_20'] >= (0.04 * data['Close'])

#             # Main Signal
#             volume_ratio_threshold = max(0.5, data['VolumeRatio'].mean() + data['VolumeRatio'].std())
#             data['Signal'] = (data['Close'] < data['ValueWeightedPrice']) & (data['VolumeRatio'] > volume_ratio_threshold) & (data['RSI'] < 40)
#             data['Signal'] = data['Signal'].astype(int)

#             # Up/Down Volume Calculation
#             if shares_outstanding is not None:
#                 data['VolumePercent'] = data['Volume'] / shares_outstanding
#                 avg_volume_percent = data['VolumePercent'].mean()
#                 std_volume_percent = data['VolumePercent'].std()
#                 data['StdevVolume'] = (data['VolumePercent'] - avg_volume_percent) / std_volume_percent
#             else:
#                 avg_volume = data['Volume'].mean()
#                 std_volume = data['Volume'].std()
#                 data['StdevVolume'] = (data['Volume'] - avg_volume) / std_volume
#                 st.warning("Shares outstanding not found, using raw volume instead.")

#             data['PriceChange'] = data['Close'].diff()
#             data['UpDownVolume'] = np.where(data['PriceChange'] > 0, data['StdevVolume'], -data['StdevVolume'])
#             data['UpDownVolume'] = data['UpDownVolume'].rolling(window=up_down_volume_days).mean()

#             data['PositiveUpDownVolumeSignal'] = (data['UpDownVolume'] > 0.5).astype(int)
#             data['NegativeUpDownVolumeSignal'] = (data['UpDownVolume'] < -0.5).astype(int)

#             data.ffill(inplace=True)
#             data.bfill(inplace=True)


#             # --- Plotting ---
#             fig, (ax1, ax_up_down, ax2, ax3) = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

#             # -- Top plot (Price, Indicators, and Volume Profile) --
#             ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
#             ax1.plot(data.index, data['SMA200'], label='200-day SMA', color='red')
#             ax1.plot(data.index, data['ValueWeightedPrice'], label='30-day VWAP', color='green')
#             ax1.set_ylabel('Price')
#             ax1.set_title(f'{symbol} - Price, Indicators, and Signals')
#             ax1.grid(True)

#             # -- Volume Profile (Corrected) --
#             if volume_profile is not None:
#                 ax1v = ax1.twiny()  # Create a twin axis that shares the y-axis
#                 # Normalize the volume profile for consistent scaling
#                 normalized_volume = volume_profile / volume_profile.max()
#                 # Calculate the maximum x-position for the bars based on the price range *and user input*
#                 price_range = data['High'].max() - data['Low'].min()
#                 max_volume_x = price_range * (volume_profile_width / 100)  # Use user input

#                 # Correctly plot the horizontal bars on the twin axis
#                 ax1v.barh(volume_profile.index, normalized_volume * max_volume_x, color='purple', alpha=0.3,
#                           height=(price_range / num_bins)*0.8)  # Consistent height

#                 ax1v.set_xlim(0, max_volume_x) # set x limit
#                 ax1v.invert_xaxis() # bars go left
#                 ax1v.spines[['top', 'bottom', 'right']].set_visible(False) #remove extra axis lines
#                 ax1v.tick_params(axis='x', colors='purple') # set tick color
#                 ax1v.set_xlabel("Volume", color='purple')  # Optional x-axis label
#                 ax1v.set_xticks([]) # remove xticks


#             # -- Volume Ratio --
#             ax1_2 = ax1.twinx()  # Create another twin axis for Volume Ratio
#             ax1_2.plot(data.index, data['VolumeRatio'], label='30-day Volume Ratio', color='gray')
#             ax1_2.set_ylabel('Volume Ratio', color='gray')
#             ax1_2.tick_params(axis='y', labelcolor='gray')
#             ax1_2.set_ylim(0, 1)  # Volume ratio is typically between 0 and 1

#             # Combine legends from both ax1 and ax1_2
#             lines1, labels1 = ax1.get_legend_handles_labels()
#             lines1_2, labels1_2 = ax1_2.get_legend_handles_labels()
#             ax1.legend(lines1 + lines1_2, labels1 + labels1_2, loc='upper left') # combined legend


#             # Highlight Main Signal
#             for i, row in data.iterrows():
#                 if row['Signal'] == 1:
#                     ax1.axvspan(i, i, color='green', alpha=0.3)

#             # -- Up/Down Volume Plot --
#             ax_up_down.plot(data.index, data['UpDownVolume'], label='Up/Down Volume', color='orange')
#             ax_up_down.axhline(0.5, color='green', linestyle='--', label='+0.5 Threshold')
#             ax_up_down.axhline(-0.5, color='red', linestyle='--', label='-0.5 Threshold')
#             ax_up_down.set_ylabel('Up/Down Vol')
#             ax_up_down.grid(True)
#             ax_up_down.legend()

#             # Highlight Up/Down Volume Signals
#             for i, row in data.iterrows():
#                 if row['PositiveUpDownVolumeSignal'] == 1:
#                     ax_up_down.axvspan(i, i, color='green', alpha=0.3)
#                 if row['NegativeUpDownVolumeSignal'] == 1:
#                     ax_up_down.axvspan(i, i, color='red', alpha=0.3)


#             # -- RSI Plot --
#             ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
#             ax2.set_ylabel('RSI')
#             ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
#             ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
#             ax2.legend()
#             ax2.grid(True)

#             # Highlight RSI Divergences
#             for i, row in data.iterrows():
#                 if row['PositiveRSIDivergence']:
#                     ax2.axvspan(i, i, color='green', alpha=0.3)
#                 if row['NegativeRSIDivergence']:
#                     ax2.axvspan(i, i, color='red', alpha=0.3)

#             # -- MACD Plot --
#             ax3.plot(data.index, data['MACD_20_40_20'], label='MACD', color='blue')
#             ax3.plot(data.index, data['MACDs_20_40_20'], label='Signal Line', color='red')
#             ax3.bar(data.index, data['MACDh_20_40_20'], label='Histogram', color='gray')
#             ax3.set_xlabel('Date')
#             ax3.set_ylabel('MACD')
#             ax3.legend()
#             ax3.grid(True)

#             # Highlight MACD Signals
#             for i, row in data.iterrows():
#                 if row['PositiveMACDSignal']:
#                     ax3.axvspan(i, i, color='green', alpha=0.3)
#                 if row['NegativeMACDSignal']:
#                     ax3.axvspan(i, i, color='red', alpha=0.3)

#             plt.tight_layout()
#             st.pyplot(fig)
#             plt.close(fig)

#             # --- Combined Signal Table ---
#             data['MainSignal'] = data['Signal'].astype(int)
#             data['PositiveRSI'] = data['PositiveRSIDivergence'].astype(int)
#             data['NegativeRSI'] = data['NegativeRSIDivergence'].astype(int)
#             data['PositiveMACD'] = data['PositiveMACDSignal'].astype(int)
#             data['NegativeMACD'] = data['NegativeMACDSignal'].astype(int)

#             signal_cols = ['Close', 'MainSignal', 'PositiveRSI', 'NegativeRSI', 'PositiveMACD', 'NegativeMACD', 'PositiveUpDownVolumeSignal', 'NegativeUpDownVolumeSignal']
#             signal_data = data[(data['MainSignal'] == 1) | (data['PositiveRSI'] == 1) | (data['NegativeRSI'] == 1) |
#                                (data['PositiveMACD'] == 1) | (data['NegativeMACD'] == 1) |
#                                (data['PositiveUpDownVolumeSignal'] == 1) | (data['NegativeUpDownVolumeSignal'] == 1)][signal_cols].tail(10).copy()

#             def highlight_positive_signals(row):
#                 highlight = 'background-color: yellow;'
#                 default = ''
#                 if row['MainSignal'] == 1 and row['PositiveRSI'] == 1 and row['PositiveMACD'] == 1:
#                     return [highlight] * len(row)
#                 else:
#                     return [default] * len(row)

#             if not signal_data.empty:
#                 signal_data.reset_index(inplace=True)
#                 signal_data.rename(columns={'Date': 'Date', 'Close': 'Price'}, inplace=True)
#                 signal_data['Date'] = pd.to_datetime(signal_data['Date']).dt.strftime('%Y-%m-%d')
#                 signal_data['Ticker'] = symbol
#                 signal_cols = ['Date', 'Ticker', 'Price', 'MainSignal', 'PositiveRSI', 'NegativeRSI', 'PositiveMACD', 'NegativeMACD', 'PositiveUpDownVolumeSignal', 'NegativeUpDownVolumeSignal']
#                 signal_data = signal_data[signal_cols]
#                 signal_data = signal_data.sort_values(by='Date', ascending=False)
#                 st.write("\nLast 10 Signals (All Types):")
#                 styled_df = signal_data.style.apply(highlight_positive_signals, axis=1)
#                 st.dataframe(styled_df)
#             else:
#                 st.info("\nNo signals generated.")
#             return data

#         except Exception as e:
#             st.error(f"Error processing {symbol}: {e}")
#             return None

#     processed_data = plot_stock_with_all_signals(data, ticker, volume_profile)
# else:
#     st.stop()

# # Explanation:
# # This Streamlit app takes a ticker symbol and a time period as input from the user.
# # It uses yfinance to download the historical stock data for the given ticker and period.
# # The downloaded data is then displayed as a Pandas DataFrame.
# # The `plot_stock_with_all_signals` function is called which:
# #   - calculates additional technical indicators like SMA, VWAP, RSI, and MACD.
# #   - identifies potential buy/sell signals based on these indicators.
# #   - generates a plot showing the stock price, indicators, and signals.
# #   - creates a table summarizing the latest signals.
# # The resulting plot and signal table are displayed in the Streamlit app.
# # A caching mechanism is used to improve performance by storing downloaded data.
# # Basic error handling is implemented to inform the user if there are issues with the data or ticker.
# # If the user provides invalid inputs or if there are errors during data processing, the app displays an error message.
# # The app uses a twin axis plot which can be difficult to understand.