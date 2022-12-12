import pandas as pd
from matplotlib import pyplot as plt
from os.path import join


def draw_and_save(df:pd.DataFrame, epoch, plot_interval, fig_dir, *keys):
    df = df.reset_index()
    if (epoch+1) % plot_interval == 0:
        for k in keys:
            plt.plot(df[k], label=k)
            plt.grid(which='both')
            plt.legend()
            plt.savefig(join(fig_dir, f'{k}-epoch={epoch+1}.png'))
            plt.close()