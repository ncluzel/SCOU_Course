import matplotlib.pyplot as plt
import scienceplots
import datetime
from statsmodels.tsa.stattools import acf
from scipy.stats import spearmanr, gaussian_kde
from sklearn.linear_model import LinearRegression
from SCOU_extended_3_1 import *

def model_fit(data, obs_column, lod_column, NB_CHAINS=3, TUNING_ITERS=3000, SAMPLING_ITERS=2000, target_accept=0.95, n_sites=1, n_replicates=1):

    n_steps = data.shape[0]

    observation_matrix = data.loc[::, obs_column].values
    observation_matrix = observation_matrix.reshape(n_steps, n_sites, n_replicates)

    lod_matrix = data.loc[::, lod_column].values
    lod_matrix = lod_matrix.reshape(n_steps, n_sites, n_replicates)
    loq_matrix = None

    scou = SCOU(observation_matrix, lod_matrix, loq_matrix, tuning_iters=TUNING_ITERS, sampling_iters=SAMPLING_ITERS,
                                 export_name=None, 
                                 p_out_frozen=False,
                                 nb_chains=NB_CHAINS, export_chains=False,
                                 RW_order=1,
                                 target_accept=target_accept)
    scou.fit()

    return scou

def get_results(data, scou, remove_those=[], NB_CHAINS=3):
    # getting results from the model and storing it
    selected_chains = np.arange(NB_CHAINS).tolist()
    for i in remove_those:
        selected_chains.remove(i)

    scou.visualize_latents(selected_chains)
    scou.predict(selected_chains)
    scou.compute_pointwise_outlier_probabilities(selected_chains)
    data['muX'] = scou.muX
    data['ICL'] = scou.CIL
    data['ICU'] = scou.CIU
    data['pout'] = scou.pointwise_pout[:, 0, 0]

def plot_signals(ww_data, clinical_cases, usetex=False, use_scienceplots=False, use_log_scale=False, use_bounds=False, left_bound=None, right_bound=None, plot_ghost=False, lag=None, plot_scatter=False):

    if use_bounds:
        ww_data = ww_data.loc[(ww_data.dateStart>=left_bound)&(ww_data.dateStart<=right_bound)]
        clinical_cases = clinical_cases.loc[(clinical_cases.dateStart>=left_bound)&(clinical_cases.dateStart<=right_bound)]

    if use_scienceplots:
        context = ['science', 'notebook', 'grid']
    else:
        context = []

    with plt.style.context(context):
        LABEL_SIZE = 30
        TICK_SIZE = 30
        TITLE_SIZE = 38
        LEGEND_SIZE = 30
        DATES_SIZE = 18
        figsize = (28, 14) 
        
        plt.rc('axes', labelsize=LABEL_SIZE)
        plt.rc('xtick', labelsize=TICK_SIZE)   
        plt.rc('ytick', labelsize=TICK_SIZE)
        plt.rc('figure', titlesize=TITLE_SIZE)
        plt.rc('legend', fontsize=LEGEND_SIZE)

        xlabel = 'Sampling date'
        ww_ylabel = 'Concentration (GU/L)'
        cases_ylabel = 'Incidence rate (positive cases per 100,000)'
        title = 'Viral evolution'

        if usetex:
            plt.rcParams['text.usetex'] = True
            ylabel = 'Concentration (GU/L) - $\log_{10}$ scale'
            
        
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        if plot_scatter:
            ax_dict = fig.subplot_mosaic(
                """
                AA
                BC
                """
            )
        else:
            ax_dict = fig.subplot_mosaic(
                """
                A
                """
            )
        twinx = ax_dict['A'].twinx()
        
        ax_dict['A'].plot(ww_data.dateStart.values, ww_data.X_t.values, label='WW signal', color='darkorange', linewidth=10, zorder=3)
        ax_dict['A'].plot(ww_data.dateStart.values, ww_data.X_t.values, color='black', linewidth=3, zorder=3)   

        twinx.plot(clinical_cases.dateStart.values, clinical_cases.cases.values, label='Clinical cases', color='deepskyblue', linewidth=10, zorder=3)
        twinx.plot(clinical_cases.dateStart.values, clinical_cases.cases.values, color='black', linewidth=3, zorder=3)   

        if plot_ghost:
            margin = datetime.timedelta(days=lag)
            ghost_cases = clinical_cases.copy()
            ghost_cases['dateStart'] += margin
            twinx.plot(ghost_cases.dateStart.values, ghost_cases.cases.values, label='Clinical cases', color='deepskyblue', linewidth=10, zorder=3, alpha=0.3)
            twinx.plot(ghost_cases.dateStart.values, ghost_cases.cases.values, color='black', linewidth=3, zorder=3, alpha=0.3)   

        
        ax_dict['A'].set_ylabel(ww_ylabel)
        twinx.set_ylabel(cases_ylabel)
        ax_dict['A'].set_xlabel(xlabel)
        ax_dict['A'].tick_params(axis='x', labelsize=TICK_SIZE, rotation=45)
        ax_dict['A'].tick_params(axis='y', labelsize=TICK_SIZE)
        ax_dict['A'].grid(linewidth=1, color='black', alpha=0.8)
        ax_dict['A'].set_title(title, size=TITLE_SIZE)

        if use_log_scale:
            ax_dict['A'].set_yscale('log', base=10)
            twinx.set_yscale('log', base=10)

        ### Pairwise plots:
        if plot_scatter:
            ax_dict['B'].scatter(ww_data.X_t.values, clinical_cases.cases.values, 
                                     color='darkorange', edgecolor='black', s=360, zorder=3,
                                     linewidths=1.5, alpha=0.9, vmin=0, vmax=1)

            lr = LinearRegression()
            lr.fit(ww_data.X_t.values.reshape(-1,1), clinical_cases.cases.values)
            lr_pred = lr.predict(ww_data.X_t.values.reshape(-1,1))
            ax_dict['B'].plot([np.min(ww_data.X_t.values), np.max(ww_data.X_t.values)], [np.min(clinical_cases.cases.values), np.max(clinical_cases.cases.values)], color='red', linewidth=6, zorder=3)
            ax_dict['B'].plot(ww_data.X_t.values, lr_pred, color='dodgerblue', linewidth=6, zorder=3)

            common_timestamps = np.intersect1d(ww_data.dateStart.values, ghost_cases.dateStart.values)
            ax_dict['C'].scatter(ww_data.loc[ww_data.dateStart.isin(common_timestamps), 'X_t'].values,
                                 ghost_cases.loc[ghost_cases.dateStart.isin(common_timestamps), 'cases'].values, 
                                     color='darkorange', edgecolor='black', s=360, zorder=3,
                                     linewidths=1.5, alpha=0.9, vmin=0, vmax=1)

            lr = LinearRegression()
            lr.fit(ww_data.X_t.values.reshape(-1,1), ghost_cases.cases.values)
            lr_pred = lr.predict(ww_data.X_t.values.reshape(-1,1))

            ax_dict['C'].plot([np.min(ww_data.X_t.values), np.max(ww_data.X_t.values)], [np.min(ghost_cases.cases.values), np.max(ghost_cases.cases.values)], color='red', linewidth=6, zorder=3)
            ax_dict['C'].plot(ww_data.X_t.values, lr_pred, color='dodgerblue', linewidth=6, zorder=3)

        # Main legend
        plt.rcParams['text.usetex'] = False
        h1, l1 = ax_dict['A'].get_legend_handles_labels()
        fig.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=4)
        
        plt.show()

def plot_hist(ax, key, distribution, xlabel, color):
    percentile_min = np.percentile(distribution, 2.5)
    percentile_max = np.percentile(distribution, 97.5)
    
    kde = gaussian_kde(distribution)
    x_kde = np.linspace(min(distribution), max(distribution), 1000)
    y_kde = kde(x_kde)

    ax[key].plot(x_kde, y_kde, linewidth=5, color=color)
    ax[key].plot(x_kde, y_kde, linewidth=1, color='black') 
    ax[key].axvline(x=percentile_min, color=color, linewidth=5, zorder=1)
    ax[key].axvline(x=percentile_max, color=color, linewidth=5, zorder=1)
    ax[key].axvspan(xmin=percentile_min, xmax=percentile_max, color=color, alpha=0.5)                
    ax[key].hist(distribution, density=True, color=color, edgecolor='black', linewidth=6)
    ax[key].set_xlabel(xlabel)

def plot_single_wave(tcorrs, tlags, usetex=False, use_scienceplots=False):

    if use_scienceplots:
        context = ['science', 'notebook', 'grid']
    else:
        context = []

    with plt.style.context(context):
        LABEL_SIZE = 30
        TICK_SIZE = 30
        TITLE_SIZE = 38
        LEGEND_SIZE = 30
        DATES_SIZE = 18
        figsize = (28, 7) 
        
        plt.rc('axes', labelsize=LABEL_SIZE)
        plt.rc('xtick', labelsize=TICK_SIZE)   
        plt.rc('ytick', labelsize=TICK_SIZE)
        plt.rc('figure', titlesize=TITLE_SIZE)
        plt.rc('legend', fontsize=LEGEND_SIZE)

        xlabel = 'Sampling date'
        ylabel = 'Concentration (GU/L) - log scale'
        title = 'Viral evolution'

        if usetex:
            plt.rcParams['text.usetex'] = True
            
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        ax_dict = fig.subplot_mosaic(
            """
            AB
            """
        )

        key = 'A'
        plot_hist(ax_dict, key, tcorrs, 'Correlation', 'royalblue')
        
        key = 'B'
        plot_hist(ax_dict, key, tlags, 'Lag', 'darkorange')

        print(f'Correlation results: Median: {np.median(tcorrs):.3f} | 95% CIs: [{np.percentile(tcorrs, 2.5):.3f}, {np.percentile(tcorrs, 97.5):.3f}]')
        print(f'Lag         results: Median: {np.median(tlags):.3f} | 95% CIs: [{np.percentile(tlags, 2.5):.3f}, {np.percentile(tlags, 97.5):.3f}]')
        
        plt.show()

def plot_three_waves(tcorrs, tlags, usetex=False, use_scienceplots=False):

    if use_scienceplots:
        context = ['science', 'notebook', 'grid']
    else:
        context = []

    with plt.style.context(context):
        LABEL_SIZE = 30
        TICK_SIZE = 30
        TITLE_SIZE = 38
        LEGEND_SIZE = 30
        DATES_SIZE = 18
        figsize = (28, 7) 
        
        plt.rc('axes', labelsize=LABEL_SIZE)
        plt.rc('xtick', labelsize=TICK_SIZE)   
        plt.rc('ytick', labelsize=TICK_SIZE)
        plt.rc('figure', titlesize=TITLE_SIZE)
        plt.rc('legend', fontsize=LEGEND_SIZE)

        xlabel = 'Sampling date'
        ylabel = 'Concentration (GU/L) - log scale'
        title = 'Viral evolution'

        if usetex:
            plt.rcParams['text.usetex'] = True
            
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        ax_dict = fig.subplot_mosaic(
            """
            AB
            """
        )

        key = 'A'
        plot_hist(ax_dict, key, tcorrs, 'Correlation', 'royalblue')
        
        key = 'B'
        plot_hist(ax_dict, key, tlags, 'Lag', 'darkorange')
        
        plt.show()