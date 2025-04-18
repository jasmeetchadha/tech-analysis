# --- [Imports and previous code remain the same] ---
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # Needed for percentage formatting
import pandas as pd
import numpy as np
import datetime

# st.title("Tech Analysis:")

# --- [User inputs, load_data, calculate_volume_by_price remain the same] ---
# User inputs
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL").upper()
period = st.selectbox("Select Time Period", ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"), index=3)
num_bins = st.slider("Number of Volume Profile Bins", 5, 50, 40)
volume_profile_width = 20 #st.slider("Volume Profile Width (%)", 1, 50, 20)  # in percent
up_down_volume_days = st.slider("Up/Down Volume Days (x)", 10, 60, 30)

@st.cache_data
def load_data(ticker, period):
    # ... [load_data function content - ensure it's the full robust version] ...
    """Downloads stock data using yfinance."""
    try:
        if period == 'max': data = yf.download(tickers=ticker, period=period, group_by='Ticker', progress=False)
        else: data = yf.download(tickers=ticker, period=period, group_by='Ticker', progress=False)
        if data.empty: st.error(f"No data found for {ticker} with the period {period}."); return None
        if isinstance(data.columns, pd.MultiIndex):
            try: data = data[ticker]
            except KeyError:
                if ticker in data.columns.levels[0]: data.columns = data.columns.droplevel(0)
                else: st.warning(f"Could not flatten MultiIndex for {ticker}."); data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.columns = [col.capitalize() for col in data.columns]
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols): st.error(f"Missing required columns: {', '.join([c for c in required_cols if c not in data.columns])}"); return None
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
             if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
        if data['Close'].isnull().all(): st.error("'Close' column is all NaN."); return None
        data.dropna(subset=['Close'], inplace=True)
        if data.empty: st.error("No data after removing missing Close prices."); return None
        return data
    except Exception as e: st.error(f"Error downloading/processing data for {ticker}: {e}"); import traceback; st.error(f"Traceback: {traceback.format_exc()}"); return None

data = load_data(ticker, period)

def calculate_volume_by_price(data, num_bins=20):
    # ... [calculate_volume_by_price function content - ensure it's the full robust version] ...
    """Calculates volume by price. Returns a Pandas Series."""
    if data is None or not all(c in data.columns for c in ['Low','High','Close','Volume']): st.warning("Volume profile requires Low, High, Close, Volume."); return None
    calc_data = data[['Low', 'High', 'Close', 'Volume']].dropna();
    if calc_data.empty: st.warning("No valid data for volume profile calc."); return None
    min_price = calc_data['Low'].min(); max_price = calc_data['High'].max()
    if pd.isna(min_price) or pd.isna(max_price) or min_price >= max_price: st.warning(f"Invalid price range for Vol Profile (Min: {min_price}, Max: {max_price})."); return None
    price_range = max_price - min_price;
    if num_bins <= 0: st.warning("Vol Profile bins must be positive."); return None
    bin_size = price_range / num_bins
    if bin_size <= 0: st.warning("Vol Profile bin size is non-positive."); return None
    volume_by_price = {}; bin_edges = np.linspace(min_price, max_price, num_bins + 1)
    for i in range(num_bins):
        lower_bound = bin_edges[i]; upper_bound = bin_edges[i+1]; mid_point = (lower_bound + upper_bound) / 2
        if i == num_bins - 1: mask = (calc_data['Close'] >= lower_bound) & (calc_data['Close'] <= upper_bound * (1 + 1e-9))
        else: mask = (calc_data['Close'] >= lower_bound) & (calc_data['Close'] < upper_bound)
        volume_by_price[mid_point] = calc_data.loc[mask, 'Volume'].sum()
    if not volume_by_price: st.warning("No volume matched price bins for Vol Profile."); return None
    return pd.Series(volume_by_price).sort_index()

# --- Plotting Function ---
if data is not None:
    volume_profile = calculate_volume_by_price(data, num_bins)

    def plot_stock_with_all_signals(data, symbol, volume_profile=None):
        """Calculates indicators, plots charts (including drawdown), and shows signals."""
        try:
            # --- Styles and Font Sizes ---
            plt.rcdefaults()
            plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 13,
                                 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
                                 'figure.titlesize': 18, 'grid.color': 'grey', 'grid.linestyle': ':',
                                 'grid.linewidth': 0.6, 'grid.alpha': 0.7})

            # --- Data Validation and Calculations (Use the full robust versions) ---
            data = data.copy() # Avoid SettingWithCopyWarning first
            # Ensure required columns exist and attempt numeric conversion
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'];
            if not all(col in data.columns for col in required_cols): st.error(f"Missing required columns."); return None
            try:
                for col in required_cols: data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e: st.error(f"Error converting columns to numeric: {e}"); return None
            data.dropna(subset=['Close'], inplace=True);
            if data.empty: st.error("No data after removing missing Close prices."); return None

            # SMA200
            data['SMA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
            # ValueWeightedPrice
            close_vol = data['Close'] * data['Volume']; rolling_close_vol_sum = close_vol.rolling(window=30, min_periods=1).sum(); rolling_vol_sum_vwap = data['Volume'].rolling(window=30, min_periods=1).sum()
            data['ValueWeightedPrice'] = rolling_close_vol_sum.divide(rolling_vol_sum_vwap).replace([np.inf, -np.inf], np.nan)
            # Volume Ratio (Full logic)
            shares_outstanding = None;
            try:
                ticker_info = yf.Ticker(symbol).info; shares_outstanding = ticker_info.get('sharesOutstanding')
                if shares_outstanding is None or shares_outstanding == 0: shares_outstanding = None; raise ValueError("Not found or zero")
                rolling_vol_sum_ratio = data['Volume'].rolling(window=30, min_periods=1).sum(); data['VolumeRatio'] = rolling_vol_sum_ratio / shares_outstanding
            except Exception as e:
                # Don't show warning if it was expected (None)
                # if shares_outstanding is not None: st.warning(f"Shares outstanding error ({e}). Using relative volume.", icon="⚠️")
                # else: st.info(f"Shares outstanding unavailable for {symbol}. Using relative volume.", icon="ℹ️") # Use info
                shares_outstanding = None; valid_volume = data['Volume'].dropna()
                if not valid_volume.empty and valid_volume.mean() != 0: data['VolumeRatio'] = data['Volume'].rolling(window=30, min_periods=1).mean() / valid_volume.mean()
                else: data['VolumeRatio'] = 0
            data['VolumeRatio'] = data['VolumeRatio'].fillna(0)
            # Drawdown
            peak = data['Close'].cummax(); data['Drawdown'] = (data['Close'] - peak).divide(peak).replace([np.inf, -np.inf], np.nan).fillna(0)
            # RSI (Full logic)
            def rsi(close, length=14):
                close_series = pd.Series(close); delta = close_series.diff(); up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
                roll_up = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean(); roll_down = down.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
                rs = roll_up / roll_down; rsi_calc = 100.0 - (100.0 / (1.0 + rs))
                rsi_calc[roll_down == 0] = 100; rsi_calc[roll_up == 0] = 0; rsi_calc = rsi_calc.fillna(50)
                return rsi_calc
            data['RSI'] = rsi(data['Close'], length=14)
            # MACD (Full logic)
            def macd(close, fast=20, slow=40, signal=20):
                close_series=pd.Series(close); ema_fast=close_series.ewm(span=fast, adjust=False, min_periods=fast).mean(); ema_slow=close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
                macd_line=ema_fast-ema_slow; signal_line=macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean(); histogram=macd_line-signal_line
                return macd_line, signal_line, histogram
            data['MACD_20_40_20'], data['MACDs_20_40_20'], data['MACDh_20_40_20'] = macd(data['Close'], fast=20, slow=40, signal=20)
            # RSI Divergence (Full logic)
            def calculate_rsi_divergence(close, rsi, lookback, price_thresh, rsi_thresh):
                pos_div=pd.Series(np.zeros(len(close), dtype=bool), index=close.index); neg_div=pd.Series(np.zeros(len(close), dtype=bool), index=close.index)
                close=pd.Series(close); rsi=pd.Series(rsi)
                # Ensure lookback doesn't exceed available data length
                if len(close) <= lookback: return pos_div, neg_div
                for i in range(lookback, len(close)):
                    if pd.isna(close.iloc[i]) or pd.isna(close.iloc[i-lookback]) or pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-lookback]): continue
                    if close.iloc[i-lookback] == 0: continue
                    price_lower=close.iloc[i]<close.iloc[i-lookback]; rsi_higher=rsi.iloc[i]>rsi.iloc[i-lookback]; price_higher=close.iloc[i]>close.iloc[i-lookback]; rsi_lower=rsi.iloc[i]<rsi.iloc[i-lookback]
                    price_change_pos=(close.iloc[i-lookback]-close.iloc[i])/close.iloc[i-lookback]; price_change_neg=(close.iloc[i]-close.iloc[i-lookback])/close.iloc[i-lookback]
                    rsi_change_pos=rsi.iloc[i]-rsi.iloc[i-lookback]; rsi_change_neg=rsi.iloc[i-lookback]-rsi.iloc[i]
                    if price_lower and rsi_higher and price_change_pos>price_thresh and rsi_change_pos>rsi_thresh: pos_div.iloc[i]=True
                    if price_higher and rsi_lower and price_change_neg>price_thresh and rsi_change_neg>rsi_thresh: neg_div.iloc[i]=True
                return pos_div, neg_div
            lookback_period, price_threshold, rsi_threshold = 14, 0.03, 5
            data['PositiveRSIDivergence'], data['NegativeRSIDivergence'] = calculate_rsi_divergence(data['Close'], data['RSI'], lookback_period, price_threshold, rsi_threshold)
            # MACD Signals
            data['PositiveMACDSignal'] = data['MACD_20_40_20'] <= (-0.04*data['Close']); data['NegativeMACDSignal'] = data['MACD_20_40_20'] >= (0.04*data['Close'])
            # Main Signal (Full logic)
            valid_volume_ratio = data['VolumeRatio'].dropna()
            if not valid_volume_ratio.empty:
                vol_ratio_mean=valid_volume_ratio.mean(); vol_ratio_std=valid_volume_ratio.std()
                if shares_outstanding: volume_ratio_threshold = max(0.5, vol_ratio_mean + vol_ratio_std)
                else: volume_ratio_threshold = vol_ratio_mean + vol_ratio_std
            else: volume_ratio_threshold = 0
            data['Signal'] = (data['Close']<data['ValueWeightedPrice']) & (data['VolumeRatio']>volume_ratio_threshold) & (data['RSI']<40)
            data['Signal'] = data['Signal'].astype(int)
            # Up/Down Volume (Full logic)
            data['PriceChange'] = data['Close'].diff()
            valid_volume = data['Volume'].dropna()
            if not valid_volume.empty:
                avg_volume=valid_volume.mean(); std_volume=valid_volume.std();
                if std_volume==0: std_volume=1
                if shares_outstanding is not None and shares_outstanding>0:
                    data['VolumePercent']=data['Volume']/shares_outstanding; valid_volume_percent=data['VolumePercent'].dropna()
                    avg_volume_percent=valid_volume_percent.mean() if not valid_volume_percent.empty else 0; std_volume_percent=valid_volume_percent.std() if not valid_volume_percent.empty else 0
                    if std_volume_percent==0: std_volume_percent=1
                    data['StdevVolume']=(data['VolumePercent']-avg_volume_percent)/std_volume_percent
                else: data['StdevVolume']=(data['Volume']-avg_volume)/std_volume
            else: data['StdevVolume']=0
            data['StdevVolume'] = data['StdevVolume'].fillna(0)
            data['UpDownVolumeRaw'] = np.where(data['PriceChange']>0, data['StdevVolume'], np.where(data['PriceChange']<0, -data['StdevVolume'], 0))
            data['UpDownVolume'] = data['UpDownVolumeRaw'].rolling(window=up_down_volume_days, min_periods=1).mean()
            data['PositiveUpDownVolumeSignal'] = (data['UpDownVolume']>0.5).astype(int); data['NegativeUpDownVolumeSignal'] = (data['UpDownVolume']<-0.5).astype(int)
            # Final fillna
            cols_to_fill = ['SMA200', 'ValueWeightedPrice', 'VolumeRatio', 'Drawdown', 'RSI', 'MACD_20_40_20', 'MACDs_20_40_20', 'MACDh_20_40_20', 'UpDownVolume']
            for col in cols_to_fill:
                 if col in data.columns: data[col] = data[col].ffill().bfill()
            # --- End Calculations ---


            # --- Plotting ---
            fig, (ax1, ax_drawdown, ax_up_down, ax2, ax3) = plt.subplots(
                5, 1, figsize=(15, 24), sharex=True,
                gridspec_kw={'height_ratios': [5, 1.5, 1.5, 1.5, 1.5]}
            )

            # -- Panel 1: Price, Indicators, Volume Profile, Volume Ratio --
            ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5, zorder=3)
            ax1.plot(data.index, data['SMA200'], label='SMA 200', color='red', linestyle='-', linewidth=1.3, zorder=2)
            ax1.plot(data.index, data['ValueWeightedPrice'], label=f'VWAP {30}-day', color='green', linestyle=':', linewidth=1.2, zorder=2)
            ax1.set_ylabel('Price ($)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title(f'{symbol} Technical Analysis ({data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")})')
            ax1.grid(True, zorder=0)

            # Main Signal Highlighting (REVERTED TO SIMPLER axvspan loop)
            main_signal_dates = data.index[data['Signal'] == 1]
            if not main_signal_dates.empty:
                # Add a dummy line for the legend entry
                ax1.plot([], [], color='lightgreen', linewidth=10, label='Main Signal', alpha=0.6)
                # Loop through each date where the signal is active
                for date in main_signal_dates:
                    # Span approximately 1 day centered on the date
                    start_date = date - pd.Timedelta(days=0.5)
                    end_date = date + pd.Timedelta(days=0.5)
                    ax1.axvspan(start_date, end_date,
                                color='lightgreen', alpha=0.4, zorder=1) # zorder=1 puts it behind lines but above grid


            # Volume Profile
            if volume_profile is not None and not volume_profile.empty:
                ax1v = ax1.twiny()
                normalized_volume = volume_profile / volume_profile.max()
                price_min, price_max = ax1.get_ylim()
                # Prevent division by zero if ylims are the same
                if price_max > price_min:
                    plot_width = (price_max - price_min) * (volume_profile_width / 100.0)
                    bin_height = ((price_max - price_min) / num_bins * 0.8) if num_bins > 0 else 1
                else:
                    plot_width = 1 # Default width if range is zero
                    bin_height = 1 # Default height

                ax1v.barh(volume_profile.index, normalized_volume * plot_width, color='purple', alpha=0.35, height=bin_height)
                ax1v.set_xlim(plot_width, 0); ax1v.set_xticks([])
                ax1v.set_xlabel("Vol Profile", color='purple', alpha=0.8)
                ax1v.tick_params(axis='x', colors='purple')
                ax1v.spines[['top', 'bottom', 'left','right']].set_visible(False)
            else:
                 ax1.text(0.02, 0.95, "Vol Profile Unavailable", transform=ax1.transAxes, color='red', alpha=0.7, ha='left', va='top', fontsize=11)

            # Volume Ratio (with fixed 100% max y-axis)
            ax1_2 = ax1.twinx()
            ax1_2.plot(data.index, data['VolumeRatio'], label=f'Vol Ratio {30}-day', color='dimgray', linestyle='-', linewidth=1.2, alpha=0.75)
            ax1_2.set_ylabel('Volume Ratio', color='dimgray', alpha=0.9)
            ax1_2.tick_params(axis='y', labelcolor='dimgray', colors='dimgray')
            ax1_2.spines['right'].set_color('dimgray')
            ax1_2.set_ylim(bottom=0, top=1.0) # Set max to 100%
            ax1_2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines1_2, labels1_2 = ax1_2.get_legend_handles_labels()
            valid_lines1 = [line for line in lines1 if hasattr(line, 'get_label') and line.get_label() != '_nolegend_']
            valid_labels1 = [label for line, label in zip(lines1, labels1) if hasattr(line, 'get_label') and line.get_label() != '_nolegend_']
            valid_lines1_2 = [line for line in lines1_2 if hasattr(line, 'get_label') and line.get_label() != '_nolegend_']
            valid_labels1_2 = [label for line, label in zip(lines1_2, labels1_2) if hasattr(line, 'get_label') and line.get_label() != '_nolegend_']
            # Only show legend if there are valid items
            if valid_lines1 or valid_lines1_2:
                 ax1.legend(valid_lines1 + valid_lines1_2, valid_labels1 + valid_labels1_2, loc='upper left')


            # -- Panel 2: Drawdown Plot --
            ax_drawdown.plot(data.index, data['Drawdown'], label='Drawdown', color='navy', linewidth=1.2); ax_drawdown.set_ylabel('Drawdown', color='navy'); ax_drawdown.tick_params(axis='y', labelcolor='navy'); ax_drawdown.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)); ax_drawdown.grid(True); ax_drawdown.set_title('Price Drawdown from Peak', loc='left', alpha=0.9); last_drawdown_value = data['Drawdown'].iloc[-1]; last_date = data.index[-1]; ax_drawdown.plot(last_date, last_drawdown_value, 'ro', markersize=5); ax_drawdown.text(last_date, last_drawdown_value, f' {last_drawdown_value:.1%}', verticalalignment='center', horizontalalignment='left', color='red', fontsize=12)


            # -- Panel 3: Up/Down Volume Plot (CORRECTED - MARKERS) --
            ax_up_down.plot(data.index, data['UpDownVolume'], label=f'Up/Down Vol ({up_down_volume_days}d avg)', color='darkorange', linewidth=1.2)
            ax_up_down.axhline(0.5, color='green', linestyle='--', linewidth=1, label='+0.5 Threshold')
            ax_up_down.axhline(-0.5, color='red', linestyle='--', linewidth=1, label='-0.5 Threshold')
            ax_up_down.axhline(0, color='black', linestyle='-', linewidth=0.6, alpha=0.6)
            ax_up_down.set_ylabel('Up/Down Vol', color='darkorange')
            ax_up_down.tick_params(axis='y', labelcolor='darkorange')
            ax_up_down.grid(True)
            ax_up_down.set_title('Up/Down Volume Momentum', loc='left', alpha=0.9)
            # Use markers for signals on this plot
            pos_udv_dates = data.index[data['PositiveUpDownVolumeSignal'] == 1]
            neg_udv_dates = data.index[data['NegativeUpDownVolumeSignal'] == 1]
            udv_plots = [] # Store plot objects for legend
            if not pos_udv_dates.empty:
                p1, = ax_up_down.plot(pos_udv_dates, data.loc[pos_udv_dates, 'UpDownVolume'], 'go', markersize=6, alpha=0.8, label='Pos Signal', linestyle='None')
                udv_plots.append(p1)
            if not neg_udv_dates.empty:
                p2, = ax_up_down.plot(neg_udv_dates, data.loc[neg_udv_dates, 'UpDownVolume'], 'ro', markersize=6, alpha=0.8, label='Neg Signal', linestyle='None')
                udv_plots.append(p2)
            # Add main line and threshold lines to legend handles
            line_handles, line_labels = ax_up_down.get_legend_handles_labels()
            # Ensure marker labels are not duplicated if they exist in line_labels
            marker_labels = [p.get_label() for p in udv_plots]
            final_handles = line_handles + [p for p, label in zip(udv_plots, marker_labels) if label not in line_labels]
            final_labels = line_labels + [label for label in marker_labels if label not in line_labels]
            if final_handles: # Only show legend if there's something to show
                ax_up_down.legend(handles=final_handles, labels=final_labels)


            # -- Panel 4: RSI Plot --
            ax2.plot(data.index, data['RSI'], label='RSI (14)', color='purple', linewidth=1.2); ax2.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)'); ax2.axhline(50, color='gray', linestyle=':', linewidth=0.8, alpha=0.7); ax2.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)'); ax2.set_ylabel('RSI', color='purple'); ax2.tick_params(axis='y', labelcolor='purple'); ax2.set_ylim(0, 100); ax2.grid(True); ax2.set_title('Relative Strength Index (RSI)', loc='left', alpha=0.9); pos_rsi_div_dates = data.index[data['PositiveRSIDivergence']]; neg_rsi_div_dates = data.index[data['NegativeRSIDivergence']];
            rsi_plots = []
            if not pos_rsi_div_dates.empty: p1, = ax2.plot(pos_rsi_div_dates, data.loc[pos_rsi_div_dates, 'RSI'], 'g^', markersize=7, alpha=0.85, label='Pos Div', linestyle='None'); rsi_plots.append(p1)
            if not neg_rsi_div_dates.empty: p2, = ax2.plot(neg_rsi_div_dates, data.loc[neg_rsi_div_dates, 'RSI'], 'rv', markersize=7, alpha=0.85, label='Neg Div', linestyle='None'); rsi_plots.append(p2)
            line_handles, line_labels = ax2.get_legend_handles_labels(); marker_labels = [p.get_label() for p in rsi_plots]; final_handles = line_handles + [p for p, label in zip(rsi_plots, marker_labels) if label not in line_labels]; final_labels = line_labels + [label for label in marker_labels if label not in line_labels];
            if final_handles: ax2.legend(handles=final_handles, labels=final_labels)


            # -- Panel 5: MACD Plot --
            ax3.plot(data.index, data['MACD_20_40_20'], label='MACD (20,40)', color='blue', linewidth=1.2); ax3.plot(data.index, data['MACDs_20_40_20'], label='Signal (20)', color='red', linestyle=':', linewidth=1.2); colors = ['forestgreen' if x >= 0 else 'salmon' for x in data['MACDh_20_40_20']]; num_days = (data.index.max() - data.index.min()).days; bar_width = max(0.5, min(2.0, num_days / 252.0 * 0.8 if num_days > 0 else 1.0)); ax3.bar(data.index, data['MACDh_20_40_20'], label='Histogram (20)', color=colors, width=bar_width, alpha=0.6); ax3.axhline(0, color='black', linestyle='-', linewidth=0.6, alpha=0.6); ax3.set_xlabel('Date'); ax3.set_ylabel('MACD', color='blue'); ax3.tick_params(axis='y', labelcolor='blue'); ax3.tick_params(axis='x'); ax3.grid(True); ax3.set_title('MACD (20, 40, 20)', loc='left', alpha=0.9); pos_macd_sig_dates = data.index[data['PositiveMACDSignal']]; neg_macd_sig_dates = data.index[data['NegativeMACDSignal']];
            macd_plots = []
            if not pos_macd_sig_dates.empty: p1, = ax3.plot(pos_macd_sig_dates, data.loc[pos_macd_sig_dates, 'MACD_20_40_20'], 'g^', markersize=7, alpha=0.85, label='Pos Signal', linestyle='None'); macd_plots.append(p1)
            if not neg_macd_sig_dates.empty: p2, = ax3.plot(neg_macd_sig_dates, data.loc[neg_macd_sig_dates, 'MACD_20_40_20'], 'rv', markersize=7, alpha=0.85, label='Neg Signal', linestyle='None'); macd_plots.append(p2)
            line_handles, line_labels = ax3.get_legend_handles_labels(); marker_labels = [p.get_label() for p in macd_plots]; final_handles = line_handles + [p for p, label in zip(macd_plots, marker_labels) if label not in line_labels]; final_labels = line_labels + [label for label in marker_labels if label not in line_labels];
            if final_handles: ax3.legend(handles=final_handles, labels=final_labels)


            # --- Final Plot Adjustments ---
            plt.tight_layout(h_pad=2.5)
            st.pyplot(fig)
            plt.close(fig)

            # --- [Combined Signal Table remains the same] ---
            # ... (Signal Table code) ...
            data['MainSignal'] = data['Signal'].astype(int); data['PositiveRSI'] = data['PositiveRSIDivergence'].astype(int); data['NegativeRSI'] = data['NegativeRSIDivergence'].astype(int); data['PositiveMACD'] = data['PositiveMACDSignal'].astype(int); data['NegativeMACD'] = data['NegativeMACDSignal'].astype(int)
            signal_cols_to_check = ['MainSignal', 'PositiveRSI', 'NegativeRSI', 'PositiveMACD', 'NegativeMACD', 'PositiveUpDownVolumeSignal', 'NegativeUpDownVolumeSignal']; signal_cols_display = ['Close'] + signal_cols_to_check
            signal_active_mask = data[signal_cols_to_check].any(axis=1); signal_data = data.loc[signal_active_mask, signal_cols_display].tail(15).copy()
            def highlight_signals(row):
                styles = [''] * len(row)
                if row['MainSignal'] == 1: styles[row.index.get_loc('MainSignal')] = 'background-color: lightgreen; color: black;';
                if row['PositiveRSI'] == 1: styles[row.index.get_loc('PositiveRSI')] = 'background-color: lightgreen; color: black;';
                if row['PositiveMACD'] == 1: styles[row.index.get_loc('PositiveMACD')] = 'background-color: lightgreen; color: black;';
                if row['PositiveUpDownVolumeSignal'] == 1: styles[row.index.get_loc('PositiveUpDownVolumeSignal')] = 'background-color: lightgreen; color: black;';
                if row['NegativeRSI'] == 1: styles[row.index.get_loc('NegativeRSI')] = 'background-color: salmon; color: black;';
                if row['NegativeMACD'] == 1: styles[row.index.get_loc('NegativeMACD')] = 'background-color: salmon; color: black;';
                if row['NegativeUpDownVolumeSignal'] == 1: styles[row.index.get_loc('NegativeUpDownVolumeSignal')] = 'background-color: salmon; color: black;';
                return styles
            if not signal_data.empty:
                signal_data.reset_index(inplace=True); signal_data.rename(columns={'index': 'Date', 'Close': 'Price'}, inplace=True); signal_data['Date'] = pd.to_datetime(signal_data['Date']).dt.strftime('%Y-%m-%d'); signal_data['Ticker'] = symbol
                display_order = ['Date', 'Ticker', 'Price', 'MainSignal', 'PositiveUpDownVolumeSignal', 'PositiveRSI', 'PositiveMACD', 'NegativeUpDownVolumeSignal', 'NegativeRSI', 'NegativeMACD']
                display_order = [col for col in display_order if col in signal_data.columns]; signal_data = signal_data[display_order]; signal_data = signal_data.sort_values(by='Date', ascending=False)
                st.write(f"\nLast {len(signal_data)} Signal Occurrences:"); styled_df = signal_data.style.apply(highlight_signals, axis=1).format({'Price': '${:,.2f}'}); st.dataframe(styled_df, hide_index=True)
            else: st.info(f"No signals generated for {symbol} in the selected period.")


            return data

        except Exception as e:
            st.error(f"An error occurred during analysis or plotting for {symbol}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
             plt.rcdefaults() # Reset Matplotlib settings

    # --- Run the Analysis and Plotting ---
    processed_data = plot_stock_with_all_signals(data, ticker, volume_profile)

else:
    st.warning("Data could not be loaded. Please check the ticker symbol and period.")
    st.stop()

# --- [Optional Explanation Block remains the same] ---

# --- [Optional Explanation Block remains the same] ---
# --- [Optional Explanation Block remains the same] ---



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


###################################################

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