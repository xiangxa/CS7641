import numpy as np
from hiive.mdptoolbox import mdp
#from util import plot_mpd_graph
from generate_frozen_lake import generate_FrozenLake 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    plt.savefig('frozen_lake_plots/{}_{}.png'.format(title, stat_col))
    plt.show()



def main():

    
    print("Create a frozen lake of Size 10x10")
    p=generate_FrozenLake(size=40)
    num_states=len(p)
    num_actions=len(p[0])
    print("Num of States:",num_states)
    print("Num of Actions:",num_actions)
    P = np.zeros((num_actions,num_states,num_states))
    R = np.zeros((num_actions,num_states,num_states))
   
    for i in range(num_states):
        for j in range(num_actions):
            sum=0
            for prob, next_state, rewards, done in p[i][j]:
                P[j][i][next_state] +=prob
                R[j][i][next_state] = rewards
                sum+=prob

    # VI 
    for gamma in [.9,0.6,0.3]:
        vi = mdp.ValueIteration(transitions=P,
                                reward=R,
                                gamma=gamma,
                                epsilon=0.000001,
                                max_iter=5000)
        stats_data = vi.run()
        plot_mpd_graph(stats_data,
                       'VI Frozen_Lake(40x40), Gamma={}, Error plot'.format(gamma),
                       'Reward',
                       'Reward')
        
        plot_mpd_graph(stats_data,
                       'VI Frozen_Lake(40x40), Gamma={}, Time PLot'.format(gamma),
                       'Time(seconds)',
                       'Time')
        
    # PI
    for gamma in [.9,0.6, 0.3]:
        print('PI {}'.format(gamma))
        pi = mdp.PolicyIteration(transitions=P,
                                 reward=R,
                                 gamma=gamma,
                                 max_iter=5000,
                                 eval_type=1)
        stats_data = pi.run()
        plot_mpd_graph(stats_data,
                       'PI Frozen_Lake(40x40), Gamma={}, Error plot'.format(gamma),
                       'Error',
                       'Error')
        
        plot_mpd_graph(stats_data,
                       'PI Frozen_Lake(40x40), Gamma={}, Time PLot'.format(gamma),
                       'Time(seconds)',
                       'Time')



    
    # QLearning
    for gamma in [0.9, 0.6, 0.3]:
        qlearn = mdp.QLearning(transitions=P,
                           reward=R,
                           gamma=gamma,
                           alpha=0.1,
                           alpha_decay=0.1,
                           alpha_min=0.0001,
                           epsilon=0.9,
                           epsilon_min=0.9,
                           epsilon_decay=0,
                           n_iter=10000)
        stats_data = qlearn.run()
        plot_mpd_graph(stats_data,
                       'Qlearning Frozen_Lake(40x40),  Gamma={}, Error plot'.format(gamma),
                       'Error',
                       'Error')
        
        plot_mpd_graph(stats_data,
               'Qlearning Frozen_Lake(40x40),  Gamma={}, Reward plot'.format(gamma),
               'Reward',
               'Reward')
        
        plot_mpd_graph(stats_data,
                       'Qlearning Frozen_Lake(40x40),  Gamma={}, Time PLot'.format(gamma),
                       'Time(seconds)',
                       'Time')


if __name__ == '__main__':
    main()