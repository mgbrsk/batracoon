import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
from random import choice
import click


@click.command()
@click.option("--csv_name", default='net_log1.csv', help="sasi")
def main(csv_name):
    
    names = ['w' + str(i) for i in range(1,7)] + ['in1', 'in2'] + ['out_true', 'out_pred']
    df = pd.read_csv(csv_name, names=names)
    
    fig, ax = plt.subplots(5, 1)
    fig.set_size_inches((20, 30))
    plt.rc('font', size=16, weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=15)


    linestyles = ['-', '--', '-.', ':']
    colors = ['b', 'r', 'g', 'c', 'm', 'olive', 'k', 'tab:orange', 'tab:cyan', 'darkred']

    for i in range(1,7):
        ax[0].plot(df['w'+str(i)], label='w'+str(i), linestyle=choice(linestyles), color=colors[i-1])

    ax[1].plot(df['in1'], label='in1', linestyle=choice(linestyles), color=colors[i])
    ax[2].plot(df['in2'], label='in2', linestyle=choice(linestyles), color=colors[i+1])
    ax[3].plot(df['out_true'], label='out_true', linestyle=choice(linestyles), color=colors[i+2])
    ax[4].plot(df['out_pred'], label='out_pred', linestyle=choice(linestyles), color=colors[i+3])

    for j in ax:
        j.legend()

    plt.savefig('graphs.png', dpi=144)
    plt.close(fig)


if __name__ == '__main__':
    main()
