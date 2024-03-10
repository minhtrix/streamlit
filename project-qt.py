import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Libraries
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go
import plotly.express as px

from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from scipy.fft import fft



#Create 7 pages 
st.title("Predictive Analytics for Stock Market Trends: A Machine Learning Approach:")
st.sidebar.title("Navigation")
pages=["Introduction", "Dataset", "EDA", "Feature Engineering", "Modelling 1", "Modelling 2", "Conclusion"]
page=st.sidebar.radio("Steps", pages)

# Page 1: Project Introduction
if page == pages[0] : 
  st.write("### The Project") 
  st.write('The aim of this project is to predict the outcome of stock prices based on historical price data on Yahoo Finance with various technical market indicators. At its core, the model automatically learns the patterns and trends of the stock market and predicts future price movements using a range of machine learning methods and a rich dataset of historical stock price data.', help=None)
  st.image("https://s.aolcdn.com/membership/omp-static/biblio/plus/finance/img/features/Analytics_portfolio.jpg")
  st.image("https://upload.wikimedia.org/wikipedia/commons/8/8f/Yahoo%21_Finance_logo_2021.png", width = 100)

# Page 2: Dataset
if page == pages[1] : 
  st.write("### Dataset") 
  st.write("")
  st.write('Select time period for stock prices')

  col1, col2 = st.columns(2)

  with col1:
        symbols = ["^GSPC", "^IXIC", "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA"]
        symbol = st.selectbox('Select a ticker', symbols)
        start_date = st.date_input('Start date', value=pd.to_datetime('1950-01-01'))
        end_date = st.date_input('End date', value=pd.to_datetime('2024-01-01'))
        
        if start_date and end_date:
           sp_data = yf.download(symbol, start=start_date, end=end_date)
           st.session_state['sp_data'] = sp_data
        
        st.image("https://www.researchgate.net/publication/352017050/figure/fig1/AS:1029759769268225@1622525269849/e-OHLC-candlestick-data-in-chart-and-comma-separated-value-CSV-representations.ppm")


        
  with col2:
        #Dataset
        st.dataframe(sp_data.head(20)) 
        st.write(sp_data.shape) 
        
  #Charts  
  data_reset = sp_data.reset_index()
  fig = px.line(data_reset, x='Date', y=['Open', 'High', 'Low', 'Close'], title=f'Express & Graph Objects Candlestick Chart for {symbol}')
  st.write("OHLC is a format in financial analysis that summarizes the four key data points in a specific time window of a financial instrument: opening price (Open), maximum price (High), low price (Low) and closing price (Close). The OHLC format is often represented in candlestick charts, which provide a quick and effective method to visualize and analyze the price behavior of a security over a specific period of time. Such data is fundamental to technical analysis as it provides insight into market trends and potential reversal points.")
  st.plotly_chart(fig)
  
  fig = go.Figure(data=[go.Candlestick(
    x=sp_data.index,
    open=sp_data['Open'],
    high=sp_data['High'],
    low=sp_data['Low'],
    close=sp_data['Close'],)])
  st.plotly_chart(fig)  
  

# Page 3: EDA / DataViz
if page == pages[2] : 
  st.write("### Exploratory data analysis") # Untertitel

  if 'sp_data' in st.session_state:
    model_data = st.session_state['sp_data']
    
    model_data['Daily_Return'] = model_data['Close'].pct_change() * 100
    model_data['Std_Dev'] = model_data['Daily_Return'].rolling(window=20).std()

    #Volume
    st.write("# Volume over time")
    st.write("Historically, the consistent growth in market volume shows that interest in stocks has increased over time. This may be due to several reasons, such as the dissemination of information through digital media, increased availability of trading platforms that make investing more accessible, as well as a growing awareness of the importance of the stock market as a tool for wealth creation. It could also indicate a growing economy as companies expand and raise more capital by issuing shares. This observation could also be a reflection of the increasing integration of the global economy, where global liquidity and investments flow across national borders, further driving market volumes.")
    fig_volume = px.bar(model_data, x=model_data.index, y='Volume')
    fig_volume.update_layout(xaxis_title='Date')
    st.plotly_chart(fig_volume)

    #Daily Return
    st.write("# Daily Return %")
    st.write("The daily return visualization shows no discernible patterns or trends and indicates high fluctuations over the course of the data. This volatility makes it difficult to determine a specific trend in the stock's price behavior. The randomness of daily returns could indicate an efficient market where all known information is already factored into the stock's price, so only new information or events cause price changes. This highlights the challenge of predicting price movements in the short term and may lead investors to focus on long-term strategies or exploiting volatility as part of their trading approaches.")
    fig_daily_return = px.bar(model_data, x=model_data.index, y='Daily_Return')
    fig_daily_return.update_layout(xaxis_title='Date')
    st.plotly_chart(fig_daily_return)
    
    # StDev
    st.write("# Standard Deviation (1 Month)")
    st.write("The volatility visualization shows that rising prices and market activity often coincide with higher volatility. Particularly noticeable spikes in volatility occurred during well-known economic events: the market experienced significant swings during the dot-com bubble in 2001, the housing bubble in 2008 and the coronavirus pandemic in 2020. Such periods of high volatility often reflect a mix of investor uncertainty and speculative trading activity as the market reacts to new information. These times can serve as turning points for market sentiment and present both risks and opportunities; Investors who take high risks face potentially high returns, while cautious investors look for safe havens.")
    fig_std_dev = px.line(model_data, x=model_data.index, y='Std_Dev')
    fig_std_dev.update_layout(xaxis_title='Date', yaxis_title='Standard Deviation')
    st.plotly_chart(fig_std_dev)


    # Autocorrelation
    st.write("# Autocorrelation")
    st.write("We can see that historical price values ​​alone would not be a really good indicator of future prices as they decrease significantly and remain below the significance threshold")
    plt.figure(figsize=(10, 4))
    pd.plotting.autocorrelation_plot(model_data['Close'])
    st.pyplot(plt)

    # FFT
    st.write("# Fast Fourier Transformation (FFT)")
    st.image("https://www.researchgate.net/publication/323281289/figure/fig6/AS:701217198059529@1544194616847/Fast-Fourier-Transformation-2-Fast-Fourier-Transformation-To-increase-the-performance.ppm")
    st.write("Applying Fast Fourier Transform (FFT) to time series data allows identification of the dominant frequencies, which in turn can help discover the optimal periodicity of seasonal patterns. By decomposing the time series into its frequency components, FFT provides valuable insights into the underlying behavior of the data set. This allows us to filter out those frequencies that reflect significant seasonal trends and cycles in the data. These insights are particularly useful for analyzing stock markets because they can help identify regular patterns that can be used to time investment decisions or develop trading strategies. For example, certain frequencies could represent weekly or monthly fluctuations that provide investors with guidance on ideal buying and selling times.")

    fft_result = fft(model_data['Close'].values)
    amplitudes = np.abs(fft_result)
    sample_spacing = 1
    frequencies = np.fft.fftfreq(len(model_data['Close']), d=sample_spacing)
    fft_df = pd.DataFrame({'Frequency': frequencies, 'Amplitude': amplitudes})
    fft_df = fft_df[fft_df['Frequency'] > 0].sort_values('Amplitude', ascending=False)
    fig_fft = px.bar(fft_df.head(20), x='Frequency', y='Amplitude', title='FFT - Top 20 Frequencies')
    st.plotly_chart(fig_fft)

    # Seasonal Decompose
    st.write("# Seasonal Decompose")
    st.write("The seasonal decomposition of the time series indicates a general upward trend, with the seasonal component showing regular fluctuations. These patterns appear to repeat approximately every five years, which could indicate long-term economic or market cycles. In addition, even in shorter time periods, such as quarters, patterns can be identified that may be related to the publication of company results. Such quarterly fluctuations could be due to market reaction to new earnings information or other significant company-specific events. The residuals, i.e. the parts of the time series that are not explained by the trend and the seasonal component, show variations over time. This could indicate that the model did not fully capture some patterns. There may be hidden or unpredictable influences not captured by the seasonal decomposition that could be important for a complete analysis of market dynamics.")
    res = seasonal_decompose(model_data['Close'], model='multiplicative', period=2617)
    res.plot()
    st.pyplot(plt)

    # ACF und PACF
    st.write("# ACF und PACF")
    st.write("ACF: In the present case, the ACF shows that the correlation gradually decreases, indicating a non-stationary time series. This means that the statistical properties of the time series, such as the mean or variance, are not constant over time. Furthermore, the Pearson correlation coefficient lies outside the confidence interval at some lags, indicating statistical significance. This means that the observed correlations for these specific lags are likely not random and that there is an inherent lagged relationship in the data. This can be important for time series modeling, such as forecasting future values, and should be taken into account when developing predictive models to better capture the underlying dynamics of the time series.")
    st.write("PACF: The Partial Autocorrelation Function (PACF) provides insight into the relationship between a time series and its lagged values ​​after removing the effects due to intermediate lags. The analysis shows that the partial autocorrelation is very high at the first lag and then quickly falls to zero. This suggests that there is a strong direct correlation between a value and its immediate predecessor. Such a pattern is typical of time series where the current value is strongly influenced by its previous value. For an ARIMA (autoregressive integrated moving average) model, this pattern indicates an AR(1) model because only the first lag has significant partial autocorrelation. This means that the current value of the time series is best predicted by the immediately previous value, and further past values ​​provide little to no additional predictive value. This observation is consistent with the Markov chain principle of -memorylessness-, in which the next state depends only on the current state and not on the complete history of the time series. In practice, this means that future values ​​of the time series can be modeled based on their last observed value, thereby simplifying the model in both its construction and interpretation.")
    plt.figure(figsize=(20,7))
    ax1 = plt.subplot(121)
    plot_acf(model_data["Close"], lags=300, ax=ax1)
    ax2 = plt.subplot(122)
    plot_pacf(model_data["Close"], lags=300, ax=ax2)
    st.pyplot(plt)

    # Augmented Dickey-Fuller-Test
    st.write("# Augmented Dickey-Fuller-Test")
    st.write("The Augmented Dickey-Fuller (ADF) test is a common statistical procedure used to check the stationarity of a time series. The first time we ran the ADF test on the original time series, we found that the data was not stationary. Non-stationary behavior, particularly in the form of a random walk, is not uncommon in financial time series. A random walk implies that the future values ​​of the series are unpredictable and only deviate from the previous value by a random step. This property presents a challenge for predicting future values ​​because traditional statistical models often assume that the underlying data is stationary.")
    results = adfuller(model_data["Close"])
    st.write(f"ADF Statistic: {results[0]}")
    st.write(f"p-value: {results[1]}")
    st.write("Critical Values:")
    for key, value in results[4].items():
        st.write(f"{key}: {value}")
    
    # Stationarity Results
    if results[0] < results[4]['5%']:
       st.write('Reject Null Hypothesis - Time Series is Stationary')
    else:
       st.write('Failed to Reject Null Hypothesis - Time Series is Non-Stationary')
    
    # Differentiation
    st.write("# Differentiation")
    st.write("To achieve stationarity in the time series, first-order differentiation was applied. This process, in which each value in the time series is subtracted from its previous value, aims to remove the temporal dependencies in the data and thereby achieve stationarity. Differentiation attempts to remove trend and seasonality from the time series, making the series more like a stationary process. In many cases, particularly in financial time series, differentiation is seen as an effective way to prepare a stationary time series for further analysis.")
    daily_diff = model_data['Close'] - model_data['Close'].shift(1)
    plt.figure(figsize=(10, 5))
    daily_diff[1:].plot(c='grey')
    daily_diff[1:].rolling(20).mean().plot(label='Rolling Mean', c='orange')
    daily_diff[1:].rolling(20).std().plot(label='Rolling STD', c='yellow')
    plt.legend()
    st.pyplot(plt)

    # Augmented Dickey-Fuller-Test 2
    st.write("# Augmented Dickey-Fuller-Re-Test")
    st.write("After differentiation, the ADF test was applied again to the transformed time series. This time the test result showed that the differentiated time series is stationary. Achieving stationarity is a critical step in analyzing time series data as it enables the applicability of many time series models. In particular, models such as ARIMA (Autoregressive Integrated Moving Average), which assume stationarity, can now be usefully applied. The finding that the data is stationary after the first differentiation suggests that the underlying data has an integrated structure and that this can be adequately modeled by the applied transformation. This finding provides a solid foundation for further analysis and modeling of the time series data.")
    results2 = adfuller(daily_diff[1:])
    st.write(f"ADF Statistic: {results2[0]}")
    st.write(f"p-value: {results2[1]}")
    st.write("Critical Values:")
    for key, value in results2[4].items():
        st.write(f"{key}: {value}")

    # Stationarity Results
    if results2[0] < results2[4]['5%']:
       st.write('Reject Null Hypothesis - Time Series is Stationary')
    else:
       st.write('Failed to Reject Null Hypothesis - Time Series is Non-Stationary')
    

# Page 4: Feature Engineering
if page == pages[3] : 
  st.write("### Feature Engeering")
  if 'sp_data' in st.session_state:
    sp_feat = st.session_state['sp_data']
    # Reduce Noise through weighted Price
    sp_feat['Median'] = (sp_feat['High'] + sp_feat['Low']) / 2
    
    # Technical Analysis - Trend Series Mean with Fibonacci Sequences as Windows
    sp_feat['MA9'] = sp_feat['Median'].rolling(window=9).mean() # Fast MA
    sp_feat['MA21'] = sp_feat['Median'].rolling(window=21).mean() # Fast MA
    sp_feat['MA34'] = sp_feat['Median'].rolling(window=34).mean() # Fast MA
    sp_feat['MA55'] = sp_feat['Median'].rolling(window=55).mean() # Slow MA
    sp_feat['MA89'] = sp_feat['Median'].rolling(window=89).mean() # SLow MA
    
    # Technical Analysis - Momentum Series
    momentumLen1 = 5 # 1 Week
    momentumLen2 = 20 # 4 Week
    sp_feat['Daily_Return'] = sp_feat['Median'].pct_change() # Percentage Change
    sp_feat['Momentum'] = sp_feat['Median'] - sp_feat['Median'].shift(momentumLen1) # Momentum (Absolute Change within 1 Week)
    sp_feat['Momentum2'] = sp_feat['Median'] - sp_feat['Median'].shift(momentumLen2) # Momentum (Absolute Change within 4 Week)
    
    # Technical Analysis - Volatility Series
    maxminLen = 5 
    maxminLen2 = 20 
    devLen = 5
    devLen2 = 20
    num_std_dev = 2 # Std.Dev Multiplier (Bollinger Bands)
    
    sp_feat['std_dev5'] = sp_feat['Median'].rolling(window=devLen).std() # Standard Deviation 1 Week
    sp_feat['std_dev20'] = sp_feat['Median'].rolling(window=devLen2).std() # Standard Deviation 4 Weeks
    sp_feat['Upper Std.Dev'] = sp_feat['MA21'] + (sp_feat['std_dev20'] * num_std_dev) # Upper Dev
    sp_feat['Lower Std.Dev'] = sp_feat['MA21'] - (sp_feat['std_dev20'] * num_std_dev) # Lower Dev
    sp_feat['MAX'] = sp_feat['High'].rolling(window=maxminLen).max() # MAX Highest High 1 Weeks
    sp_feat['MAX2'] = sp_feat['High'].rolling(window=maxminLen2).max() # MAX Highest High 4 Weeks
    sp_feat['MIN'] = sp_feat['Low'].rolling(window=maxminLen).min() # MIN Lowest Low 1 Weeks
    sp_feat['MIN2'] = sp_feat['Low'].rolling(window=maxminLen2).min() # MIN Lowest Low 4 Weeks
    
    #Technical Analysis - Volume Series
    sp_feat['PVT'] = (sp_feat['Volume'] * ((sp_feat['Median'] - sp_feat['Median'].shift(1)) / sp_feat['Median'].shift(1))).cumsum() # Price Volume Trend
    clv = ((sp_feat['Median'] - sp_feat['Median'].min()) - (sp_feat['Median'].max() - sp_feat['Median']) / (sp_feat['Median'].max() - sp_feat['Median'].min())) * sp_feat['Volume'] # Accumulation/Distribution
    sp_feat['AD'] = clv.cumsum()  # Accumulation/Distribution Final

    st.dataframe(sp_feat.tail(20)) 
    st.write(sp_feat.shape) 
    
    st.session_state['sp_feat'] = sp_feat
    sp_feat_last = sp_feat.last('300D')

    # Gleitende Durchschnitte
    st.write("# Moving Averages")
    st.image("https://forextraininggroup.com/wp-content/uploads/2016/05/Moving-Average-Crossover-1024x460.png")
    st.write("The moving average is a fundamental tool in technical analysis and is used to smooth price data by calculating average prices over a specific period of time. Moving averages are often used as indicators of market trends. If the price is above the moving average, it could be an uptrend, while if the price is below the moving average, it could indicate a downtrend. Moving averages can also serve as support or resistance levels and are often used to identify buy or sell signals, especially when two different moving averages cross each other (so-called 'crossover'). Despite their usefulness, moving averages are backward-looking indicators; they are based on past prices and therefore do not provide any direct indication of future market movements.")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['Median'], name='Median'))
    fig_ma.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MA9'], name='MA 9'))
    fig_ma.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MA21'], name='MA 21'))
    fig_ma.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MA34'], name='MA 34'))
    fig_ma.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MA55'], name='MA 55'))
    fig_ma.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MA89'], name='MA 89'))
    fig_ma.update_layout(xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_ma)

    # Accumulation/Distribution
    st.write("# Accumulation/Distribution Line")
    st.image("https://www.fidelity.com/bin-public/600_Fidelity_Com_English/images/migration/article/content_12-lc/AD_Confirm602x345.png")
    st.write("The Accumulation/Distribution Line (A/D Line) is a financial indicator used to determine the cash flow volume of a security. It is designed to identify whether a security is being accumulated (bought) or dumped (sold) by combining price and volume data. The A/D Line is calculated by weighting volume based on the closing price relative to the high-low price range for the day. If the A/D line is rising, it indicates that the security is being predominantly accumulated as the closing prices are closer to the daily high. If the line falls, the security is more likely to be sold off because the closing prices are closer to the daily low.")
    fig_ad = go.Figure()
    fig_ad.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['AD']))
    fig_ad.update_layout(xaxis_title='Date', yaxis_title='AD')
    st.plotly_chart(fig_ad)

    # Donchian Channel
    st.write("# Donchian Channel")
    st.image("https://indicatorspot.com/wp-content/uploads/2021/05/2-7.png")
    st.write("The Donchian Channel is a technical indicator developed by Richard Donchian. It consists of three lines that are formed based on the highest and lowest prices of a security over a certain period of time. The upper line of the channel indicates the maximum of high prices within the selected period, while the lower line represents the minimum of low prices. The middle line is often the average of the two, i.e. the middle band. The Donchian Channel is primarily used to visualize a security's volatility and identify breakouts. If the price of a security breaks above the upper band of the Donchian Channel, it could be interpreted as a buy signal, indicating an upward trend. Conversely, a price falling below the lower band could be seen as a sell signal and an indication of a downward trend.")
    fig_donchian = go.Figure()
    fig_donchian.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MAX'], name='Upper Bound'))
    fig_donchian.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MIN'], name='Lower Bound'))
    fig_donchian.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['Median'], name='Median', fill='tonexty'))
    fig_donchian.update_layout(xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_donchian)

    # Bollinger Bands
    st.write("# Bollinger Bands")
    st.image("https://www.incrediblecharts.com/images/png-charts/bollinger-bands-trading-signals.png")
    st.write("Bollinger Bands are a popular technical indicator developed by John Bollinger in the 1980s. They consist of three lines placed around a security's price: a middle band, which is usually a simple moving average (SMA), and an upper and lower band, each a certain number of standard deviations (typically two ) lie above and below the middle band. These bands automatically adjust to market volatility: they widen as volatility increases and narrow as volatility decreases.")
    fig_bollinger = go.Figure()
    fig_bollinger.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['Upper Std.Dev'], name='Upper Band'))
    fig_bollinger.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['Lower Std.Dev'], name='Lower Band'))
    fig_bollinger.add_trace(go.Scatter(x=sp_feat_last.index, y=sp_feat_last['MA21'], name='MA 21', fill='tonexty'))
    fig_bollinger.update_layout(xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_bollinger)

import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX # for seasonality and ex. variables


# Page 5:
if page == pages[4] : 
  st.write("### Modelling Forecast")

  choice = ['SARIMAX', 'PROPHET', 'LSTM']
  option = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :', option)
  
  if 'sp_data' in st.session_state:
    sp_data = st.session_state['sp_data']
    # SARIMAX Order
    p = 1 # AR
    d = 1 # Differencing
    q = 1 # MA
    # SARIMAX Seasonal Order
    P = 2
    D = 2
    Q = 0
    s = 12 # Seasonality
    # trend: n = none, c = constant trend, t = linear trend, ct = constant + linear
    
    # SARIMAX Model
    mod_sp_close = SARIMAX(sp_data['Close'], trend='n',  order=(p,d,q), seasonal_order=(P,D,Q,s))
    res_sp_close = mod_sp_close.fit()

    # Forecast and MAE
    pred_sp_close = res_sp_close.predict()
    mae_sp_close = np.mean(np.abs(pred_sp_close - sp_data['Close']))

    # Forecast next n days

    st.title('Forecast settings')
    n_steps_sp = st.number_input('Number of days for forecast', min_value=1, max_value=365, value=10)
    end_of_series_sp = len(sp_data['Close']) - 1
    forecast_next_10_sp = res_sp_close.get_prediction(start=end_of_series_sp + 1, end=end_of_series_sp + n_steps_sp)
    forecast_values_sp = forecast_next_10_sp.predicted_mean
    conf_int_sp = forecast_next_10_sp.conf_int()

    # Forecast DataFrame
    forecast_values_sp.index = pd.date_range(start=sp_data['Close'].index[-1] + pd.Timedelta(days=1), periods=n_steps_sp, freq='D')
    conf_int_sp.index = forecast_values_sp.index

    # View visualizations and metrics in Streamlit
    st.title('SARIMAX Model Analysis')
    st.subheader('SARIMAX Summary')
    st.text(res_sp_close.summary().as_text())
    st.subheader('Mean Absolute Error')
    st.write(f'SARIMAX MAE (sp_data[\'Close\']): {mae_sp_close:.6f}')
    st.subheader('Historical data vs. SARIMAX forecast')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sp_data['Close'], label='Historical data')
    ax.plot(pred_sp_close, label='SARIMAX Forecast', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.set_title('Historical data vs. SARIMAX forecast')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader('n-day-forecast')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sp_data['Close'], label='Aktuelle Daten')
    ax.plot(forecast_values_sp, label='n-day-forecast', linestyle='--')
    ax.fill_between(forecast_values_sp.index, conf_int_sp.iloc[:, 0], conf_int_sp.iloc[:, 1], color='orange', alpha=0.3, label='Confidence interval')
    ax.set_title('current data and n-day-forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

    st.subheader('n-day-forecast table')
    st.dataframe(forecast_values_sp)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Page 6: 
if page == pages[5] : 
  st.write("### Modelling")
  #if 'sp_feat' in st.session_state:
  sp_feat = st.session_state['sp_feat']
  # Drop NaNs due to MA Creation
  sp_feat.dropna(inplace=True)
  # Drop Open/High/Low/Close/AdjClose -> Median
  sp_set = sp_feat.drop(['Open', 'High', 'Low', 'Close', 'Adj Close'], axis=1)
  # New DF for binary classification
  sp_feat_BC = sp_feat
  # Will the Median be higher tomorrow than today?  Target := if median(0) < median(-1) ? 1 : 0
  sp_feat_BC['Buy_Target'] = (sp_feat['Median'] < sp_feat['Median'].shift(-1)).astype(int)
  # Drop NaN , due to target creation
  sp_feat_BC.dropna(subset=['Buy_Target'], inplace=True)
  # Drop Columns again
  sp_set_BC = sp_feat_BC.drop(['Open', 'High', 'Low', 'Close', 'Adj Close'], axis=1)
  # Drop Median, because we replaced it with Buy Target
  sp_set_BC.drop('Median', axis=1, inplace = True)
  model_choice = st.selectbox("Choose a model:", ["Dense", "LSTM", "XGBoost"])
  
  # Features
  X = sp_set_BC.drop(['Buy_Target'], axis=1)
  y = sp_set_BC['Buy_Target']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  if model_choice == "Dense":
    # Add Layers DNN-Modell
    model_dnn = Sequential()
    model_dnn.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model_dnn.add(Dense(64, activation='relu'))
    model_dnn.add(Dense(1, activation='sigmoid'))
    # Compile
    model_dnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # Train
    history_dnn = model_dnn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluation
    y_pred_proba_dnn = model_dnn.predict(X_test_scaled).ravel()
    y_pred_dnn = (y_pred_proba_dnn > 0.5).astype('int32')
    # Metrics
    conf_matrix_dnn = confusion_matrix(y_test, y_pred_dnn)
    accuracy_dnn = accuracy_score(y_test, y_pred_dnn)
    roc_auc_dnn = roc_auc_score(y_test, y_pred_proba_dnn)
    
    # ROC-Curve
    fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(y_test, y_pred_proba_dnn)
    roc_auc_dnn = auc(fpr_dnn, tpr_dnn)

    # Results
    st.write(f"Accuracy: {accuracy_dnn}")
    st.write(f"Confusion Matrix:\n{conf_matrix_dnn}")
    # ROC-Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_dnn, tpr_dnn, color='darkorange', lw=2, label=f'{model_choice} ROC curve (area = {roc_auc_dnn:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    st.pyplot(plt)

  elif model_choice == "LSTM":
    # Restructuring for LSTM [Samples, Time Steps, Features]
    X_train_scaled_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    # Add Layers LSTM-Model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled_lstm.shape[2])))
    model_lstm.add(Dense(1, activation='sigmoid'))
    # Compile
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # Train
    history_lstm = model_lstm.fit(X_train_scaled_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluation
    y_pred_proba_lstm = model_lstm.predict(X_test_scaled_lstm).ravel()
    y_pred_lstm = (y_pred_proba_lstm > 0.5).astype('int32')
    # Metrics
    conf_matrix_lstm = confusion_matrix(y_test, y_pred_lstm)
    accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
    roc_auc_lstm = roc_auc_score(y_test, y_pred_proba_lstm)

    # ROC-Kurve
    fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_test, y_pred_proba_lstm)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
 
    # Results
    st.write(f"Accuracy: {accuracy_lstm}")
    st.write(f"Confusion Matrix:\n{conf_matrix_lstm}")
    # ROC-Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_lstm, tpr_lstm, color='darkorange', lw=2, label=f'{model_choice} ROC curve (area = {roc_auc_lstm:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    st.pyplot(plt)

  elif model_choice == "XGBoost":
    # XGBoost Modell
    model_xgb = XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1, max_depth=10, alpha=10)
    model_xgb.fit(X_train_scaled, y_train)

    y_pred_proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    y_pred_xgb = (y_pred_proba_xgb > 0.5).astype('int32')
    # Metriken für XGBoost
    conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

    # ROC-Curve für XGBoost
    fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_pred_proba_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

    # Results
    st.write(f"Accuracy: {accuracy_xgb}")
    st.write(f"Confusion Matrix:\n{conf_matrix_xgb}")

    # ROC-Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'{model_choice} ROC curve (area = {roc_auc_xgb:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    st.pyplot(plt)





