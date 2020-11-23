import pandas as pd
import matplotlib.pyplot as plt

def save_data(stats,name):
    df = pd.DataFrame().from_records(stats)
    df.to_csv('data/{}.csv'.format(name))



def plot_mpd_graph(stats, title, ylabel, stat_col):
    df_stat = pd.DataFrame.from_records(stats)
    plt.close()
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel(ylabel)
    lw = 2
    plt.plot(df_stat.reset_index()['index'], 
             df_stat[stat_col], 
             label=stat_col,
             color="darkorange", 
             lw=lw)
    """
    plt.plot(param_range, test_scores_mean, label="3-Fold Cross-validation score",
                 color="navy", lw=lw)
    """
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/{}_{}.png'.format(title, stat_col))
    plt.show()
    

