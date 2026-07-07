import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_weights(weights, tickers, title):
    """
    Plot a pie chart of portfolio weights.
    Args:
        weights (list): List of portfolio weights.
        tickers (list): List of stock ticker symbols.
        title (str): Title for the plot.
    Returns:
        None
    """
    filtered_data = [(w, t) for w, t in zip(weights, tickers) if w > 0.0001]
    filtered_weights, filtered_tickers = zip(*filtered_data)

    plt.figure(figsize=[10, 10])
    plt.pie(filtered_weights, labels=filtered_tickers, autopct='%1.2f%%',
            startangle=140, colors=sns.color_palette(palette='Blues'))
    plt.title(f'Pesos Óptimos del Portafolio\n {title}.\n\n')
    plt.axis('equal')
    plt.show()


def plot_backtesting(history: pd.DataFrame):
    """
    Plot the backtesting results.
    Args:
        history (pd.DataFrame): DataFrame containing the capital history for each optimizer.
    Returns:
        None
    """
    colors = ['gray', '#08306b', '#08519c', '#9ecae1', 'black']
    ls = ['--', '-.', '-', ':', '-.', '-', '-']

    plt.figure(figsize=[16, 8])

    for i, column in enumerate(history.columns):
        plt.plot(history.index, history[column], label=column, color=colors[i], linestyle=ls[i])

    plt.title('Backtesting Results')
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(ls='--', alpha=0.4)
    plt.show()