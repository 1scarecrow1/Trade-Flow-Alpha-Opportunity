import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#import h5py

# =============================================================================
# Reading Data
# =============================================================================


def load_files(pair: str, datatype: str) -> pd.DataFrame:
    path = "Crypto_export\\2023"

    df = pd.read_csv(f'{path}\\{datatype}_narrow_{pair}_2023.delim', delimiter = '\t')

    df.timestamp_utc_nanoseconds = pd.to_datetime(df.timestamp_utc_nanoseconds, unit='ns')
    df.drop('received_utc_nanoseconds', axis=1, inplace=True)

    df.set_index('timestamp_utc_nanoseconds', inplace=True)
    df = df.sort_index(ascending=True)

    if datatype == 'trades': # correct incorrectly recorded sides
        df.Side = np.where(df.Side>=1,1,-1)

    if datatype == 'book': # drop all other columns
        df = df.Mid 

    return df


"""
def load_h5_files(type: str, path: str):
    trades_dict = {}
    for file in os.listdir(path):
        if file.endswith(f"{type}.h5"):
            with h5py.File(path+file, 'r') as f:
                key = list(f.keys())[0]
                dataset = file[f'{key}/{key}']
                data = dataset[:]
                df = pd.DataFrame(data)
                trades_dict[key] = df
    return trades_dict
"""

# =============================================================================
# Computations
# =============================================================================

def split_data(pair: str, df: pd.DataFrame, test_size=0.6):

    i = int((1 - test_size) * len(df)) + 1

    train_data, test_data = np.split(df, [i])

    train_data.to_csv(f'output\\train_sets\\{pair}_train', index=False)

    test_data.to_csv(f'output\\test_sets\\{pair}_test', index=False)

    return train_data, test_data


def sample_irregular_timestamps(df, tau):

    timestamps = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{tau}S')

    timestamps.name = 'timestamp_utc_nanoseconds'

    df.index.name = 'timestamp_utc_nanoseconds'

    joined_index = df.index.union(timestamps).sort_values()

    joined_index_df = pd.DataFrame(index=joined_index).reset_index()

    df_reset = df.reset_index()

    merged_df = pd.merge_asof(joined_index_df, df_reset, on='timestamp_utc_nanoseconds', direction='forward')

    merged_df.set_index('timestamp_utc_nanoseconds', inplace=True)

    return merged_df



def compute_trade_flow(df, tau):

    merged_df = sample_irregular_timestamps(df, tau)

    merged_df['CumSize'] = merged_df['SizeBillionths'] * merged_df['Side']

    merged_df['TradeFlow'] = merged_df['CumSize'].rolling(window=str(tau)+'s', closed='left').sum()

    return merged_df.fillna(0)


def compute_forward_returns(df, tau, T):
    """
    Calculates T-second forward returns based on top of book data
    """
    
    merged_df = sample_irregular_timestamps(df, tau)

    shifted_price = df['PriceMillionths'].shift(-T)

    forward_returns_df = pd.merge_asof(merged_df, shifted_price.reset_index(), on='timestamp_utc_nanoseconds', direction='forward')

    forward_returns_df['ForwardReturns'] = (forward_returns_df['PriceMillionths_y'] / forward_returns_df['PriceMillionths_x']) - 1

    forward_returns_df.drop(columns=['PriceMillionths_y'], inplace=True)

    forward_returns_df.rename(columns={'PriceMillionths_x': 'PriceMillionths'}, inplace=True)

    forward_returns_df.set_index('timestamp_utc_nanoseconds', inplace=True)

    return forward_returns_df['ForwardReturns']


def regressionBeta(x,y):

    x, y = x.fillna(0), y.fillna(0)
    beta = sm.OLS(y,x).fit().params[0]
    
    return beta

def scaledregressionBeta(x,y):

    scaler = StandardScaler()

    x, y = x.to_frame(), y.to_frame()

    x, y = x.fillna(0), y.fillna(0)

    X_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    model = sm.OLS(y_scaled,X_scaled).fit()
    beta = model.params[0]

    return beta

def cv_beta(x, y, n_folds, pair, tau, T, simple_reg_beta):

    kf = KFold(n_splits=n_folds, shuffle=False, random_state=None)

    betas, rmse_scores, preds = [], [], {}

    x, y = x.fillna(0), y.fillna(0)

    for train_index, test_index in kf.split(x):

        X_train, X_test = x[train_index], x[test_index]

        y_train, y_test = y[train_index], y[test_index]

        model = sm.OLS(y_train, X_train).fit()

        betas.append(model.params[0])

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred.fillna(0)))

        rmse_scores.append(rmse)

        preds[f'Fold {len(rmse_scores)}'] = y_pred

    # plot betas
    markers = ['o', 'x', '^', 's', 'd']  
    plt.figure(figsize=(8, 6))
    for i, beta in enumerate(betas):
        plt.scatter([i], [beta], marker=markers[i % len(markers)], label=f'CV Fold {i+1}')

    plt.scatter([n_folds], [simple_reg_beta], marker='*', color='black', s=300, label='Simple Regression Beta')
    plt.xlabel('Fold Number')
    plt.ylabel('Beta Value')
    plt.title('Beta Values Across Folds and Simple Regression')
    plt.xticks(range(n_folds + 1), [f'Fold {i+1}' for i in range(n_folds)] + ['Simple Reg.'])
    plt.legend()
    plt.grid(True)  
    plt.savefig(f'output\\imgs\\{pair}\\{(tau, T)}\\cv_betas.png')
    plt.close()
    

    return betas, rmse_scores, preds


        
# =============================================================================
# Strategy
# =============================================================================

def trade_signals(test_set, regressionbeta, r_hat, pair, tau, T, price_col_name='PriceMillionths', init_cash=1000000, trading_cost=0.001):

    results_dict = {}
    participation_rate_dict = {}

    # define thresholds based on percentiles of r_hat
    j_values = np.percentile(abs(r_hat), np.arange(15, 65, 10)) 

    if regressionbeta != 0:

        for j in j_values:

            df = test_set.copy()
            condition_reach_threshold = (j < abs(r_hat))

            df['signal'] = np.zeros(len(df))

            if regressionbeta < 0:
                df['signal'] = np.where(condition_reach_threshold, 1*df['Side'], 0)
            elif regressionbeta > 0:
                df['signal'] = np.where(condition_reach_threshold, -1*df['Side'], 0)


            df['position'] = df['signal'] * df['SizeBillionths']
            df['trading_cost'] = np.where(df['signal'] != 0, np.abs(df['position']) * trading_cost, 0)
            df['position_value'] = df[price_col_name] * df['position'] - df['trading_cost']
            df['cash'] = np.cumsum(np.concatenate(([init_cash], -1 * (df.signal * df[price_col_name]).values[1:])))
            df['total_value'] = df['cash'] + df['position_value']
            df['accum_quantity'] = df['position'].cumsum()
            df['PnL'] = df['position_value'].diff().fillna(0)
            df['PnL_cumulative'] = df['PnL'].cumsum()

            sharpe_ratio = df['PnL'].mean() / df['PnL'].std() if df['PnL'].std() != 0 else np.nan
            rolling_max = df['PnL_cumulative'].cummax()
            drawdowns = df['PnL_cumulative'] - rolling_max
            max_drawdown = drawdowns.min()
            negative_volatility = df.loc[df['PnL'] < 0, 'PnL'].std()
            sortino_ratio = df['PnL'].mean() / negative_volatility if negative_volatility != 0 else np.nan
            
            metrics = {'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'sortino_ratio': sortino_ratio}
            results_dict[j] = {'df': df, 'metrics': metrics}

            # compute trade participation rate
            num_trades = df['signal'].abs().sum()
            total_possible_trades = len(df)
            participation_rate = num_trades / total_possible_trades if total_possible_trades != 0 else np.nan
            participation_rate_dict[j] = participation_rate
            
        

        max_part_j = max(participation_rate_dict, key=participation_rate_dict.get)
        df_max_j = results_dict[max_part_j]['df']
        savepath = f'output\\imgs\\{pair}\\{(tau, T)}\\{trading_cost}'
        plot_signals_against_variable(df_max_j, savepath, max_part_j, price_col_name)
        plot_signals_against_variable(df_max_j, savepath, max_part_j, "TradeFlow")
        plot_cumulative_pnl(df_max_j[:-100], savepath, title='Cumulative PnL Over Time', 
                                xlabel='Time', ylabel='Cumulative PnL', 
                                output_filename='cumulative_pnl_plot.png')
        plot_trade_participation(j_values, savepath, participation_rate_dict)

        return results_dict, participation_rate_dict


    else:
        print(f"Regression Beta is 0 for {pair} with {(tau, T)}. There is no return predictability.")


def calcTradeStats(table, price_col_name, init_cash,rolling_window=50000):
    if price_col_name not in table.columns: raise ValueError(f'{price_col_name} must be a table column.')
    if 'position' not in table.columns: raise ValueError('position must be a table column.')
    table['signal'] = table.position.diff().fillna(0).astype(np.int16)
    table['signal'].iloc[0] = table.position.iloc[0]
    table['position_value'] = table[price_col_name] * table.position
    table['cash'] = np.cumsum(np.concatenate(([init_cash], -1 * (table.signal * table[price_col_name]).values[1:])))
    table['total_value'] = table['position_value'] + table['cash']
    table['PnL_daily'] = table['total_value'].pct_change().fillna(0)
    table['PnL_cumulative'] = table['total_value'] / init_cash - 1

    risk_free_rate = 0.00
    daily_returns = table['PnL_daily']
    excess_returns = daily_returns - risk_free_rate / 252

    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    table['sharpe_ratio'] = sharpe_ratio

    cumulative_returns = (1 + daily_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    table['max_drawdown'] = max_drawdown

    negative_volatility = excess_returns[excess_returns < 0].std()
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / negative_volatility
    table['sortino_ratio'] = sortino_ratio

    return table

# =============================================================================
# Charts
# =============================================================================

def plot_buy_sell_volumes(df, pair, freq='1min', subset=False, num_obs=100):
    
    df_resampled = df.groupby('Side').resample(freq).sum()
    
    buy_volume = df_resampled.loc[1, 'SizeBillionths']
    sell_volume = -df_resampled.loc[-1, 'SizeBillionths']

    if subset:
        buy_volume = df_resampled.loc[1, 'SizeBillionths'][:num_obs]
        sell_volume = -df_resampled.loc[-1, 'SizeBillionths'][:num_obs]
    
    plt.figure(figsize=(10, 6))
    buy_volume.plot(label='Buy Volume')
    sell_volume.plot(label='Sell Volume')
    plt.title(f'Buy and Sell Volume in {freq} intervals on 24-25 Jan 2023 for {pair}')
    plt.xlabel('Time')
    plt.ylabel('SizeBillionths')
    plt.legend()
    plt.show()


def plot_average_price(df, pair, freq='5min'):
    """
    Plot the average price in 5-minute intervals for a given crypto pair DataFrame.
    
    Parameters:
    - df: DataFrame containing the price data and timestamps.
    - pair_name: A string representing the name of the crypto pair (e.g., 'BTC-USD').
    """

    df_price = df.groupby("Side").resample(freq).mean()[["PriceMillionths"]]
    
    plt.figure(figsize=(10, 6))
    df_price.reset_index().plot(
        x='timestamp_utc_nanoseconds',
        y='PriceMillionths',
        title=f"{pair}: Average Price in {freq} intervals"
    )
    plt.xlabel('Time')
    plt.ylabel('PriceMillionths')
    plt.show()


def plot_average_size(df, pair, freq='5min'):
    """
    Plot the average size traded in 5-minute intervals for a given crypto pair DataFrame.
    
    Parameters:
    - df: DataFrame containing the size data and timestamps.
    - pair_name: A string representing the name of the crypto pair (e.g., 'BTC-USD').
    """

    df_price = df.groupby("Side").resample(freq).mean()[["SizeBillionths"]]
    
    plt.figure(figsize=(10, 6))
    df_price.reset_index().plot(
        x='timestamp_utc_nanoseconds',
        y='SizeBillionths',
        title=f"{pair}: Average Size Traded in {freq} intervals"
    )
    plt.xlabel('Time')
    plt.ylabel('SizeBillionths')
    plt.show()


def plot_price_changes(df, pair, num_obs=20):

    subset_df = df.head(num_obs)

    plt.figure(figsize=(10, 6))

    plt.plot(subset_df.index, subset_df['PriceMillionths'], drawstyle='steps-post', color='red')

    plt.xlabel('Time')
    plt.ylabel('PriceMillionths')
    plt.title(f'Price Changes Over Time in {pair}')

    plt.gcf().autofmt_xdate()

    plt.show()

def plot_signals_against_variable(df, savepath, j, x_variable, signal_variable='signal'):
    """
    Plots a given variable against buy/sell signals.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_variable: The name of the column to plot.
    - signal_variable: The name of the column containing trade signals.
    - title: The title of the plot.
    - xlabel: The label for the x-axis.
    - ylabel: The label for the y-axis.
    - output_filename: The filename for saving the plot.
    """
    sample_df = df[:100]

    plt.figure(figsize=(10, 6))

    plt.plot(sample_df.index, 
             sample_df[x_variable], 
             label=x_variable, 
             color='blue', 
             linewidth=1)    
  
    plt.scatter(sample_df.index[sample_df[signal_variable] == 1], 
                sample_df[x_variable][sample_df[signal_variable] == 1], 
                color='green', 
                label='Buy Signal', 
                marker='o', 
                alpha=0.7)
    
    plt.scatter(sample_df.index[sample_df[signal_variable] == -1], 
                sample_df[x_variable][sample_df[signal_variable] == -1], 
                color='red', 
                label='Sell Signal', 
                marker='o', 
                alpha=0.7)
    
    plt.title(f'{x_variable} and Trade Signals for j={j} (Sampled)')
    plt.xlabel('Time')
    ylabel = 'SizeBillionths' if x_variable == 'TradeFlow' else 'PriceMillionths'
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f'{savepath}\\{x_variable}_{j}_signals.png')
    plt.close()



def plot_trade_participation(j_values, savepath, participation_rate_dict):
    # Plot trade participation rate for each j

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(j_values)), 
            list(participation_rate_dict.values()), 
            tick_label=[f'{j:.4f}' for j in j_values])
    plt.xlabel('Threshold j')
    plt.ylabel('Trade Participation Rate')
    plt.title('Trade Participation Rate by Threshold j')
    plt.xticks(rotation=45)
    plt.savefig(f'{savepath}\\participation_rate.png')
    plt.close()

def plot_cumulative_pnl(df, savepath, 
                        title='Cumulative PnL Over Time', 
                        xlabel='Time', 
                        ylabel='Cumulative PnL', 
                        output_filename='cumulative_pnl.png'):
    """
    Plots the cumulative P&L from a DataFrame.

    Parameters:
    - df: DataFrame containing the cumulative P&L data.
    - title: The title of the plot.
    - xlabel: The label for the x-axis.
    - ylabel: The label for the y-axis.
    - output_filename: The filename for saving the plot.

    """
    sub_df = df[:-50]
    plt.figure(figsize=(10, 6))
    plt.plot(sub_df.index, sub_df['PnL_cumulative'], label='Cumulative PnL', color='blue', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True) 
    plt.legend()
    plt.tight_layout()  
    plt.savefig(f'{savepath}\\{output_filename}')
    plt.close()



