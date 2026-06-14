import matplotlib.pyplot as plt
import scienceplots
from SCOU_extended_3_1 import *

def model_fit(data, n_sites=1, n_replicates=1):

    # signal smoothing | model's hyperparameters
    NB_CHAINS = 3
    TUNING_ITERS = 3000
    SAMPLING_ITERS = 2000
    target_accept = 0.95

    n_steps = data.shape[0]

    observation_matrix = data.loc[::, 'obs'].values
    observation_matrix = observation_matrix.reshape(n_steps, n_sites, n_replicates)

    lod_matrix = data.loc[::, 'lod'].values
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

def plot_results(data, plot_inference=False, savefile=False, filename=None, usetex=True, use_scienceplots=True, identify_outliers=False, outliers_indexes=None, colortheme='orange'):

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
        ylabel = 'Concentration (GU/L) - log scale'
        title = 'Viral evolution'

        if usetex:
            plt.rcParams['text.usetex'] = True
            ylabel = 'Concentration (GU/L) - $\log_{10}$ scale'
            
        
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        ax_dict = fig.subplot_mosaic(
            """
            A
            """
        )

        if colortheme=='orange':
            reconstructed_color = 'limegreen'
            true_color = 'darkorange'
            meas_color = 'orange'
            CI_color = 'forestgreen'

        if not plot_inference:
            scatter_points = ax_dict['A'].scatter(data.dateStart.values, data['obs'].values, label='Measurements', 
                                 color=meas_color, edgecolor='black', s=360, zorder=3,
                                 linewidths=1.5, alpha=0.9, vmin=0, vmax=1)

        lod_points = ax_dict['A'].scatter(data.loc[data['obs']<=data['lod']].dateStart.values,
                                             data.loc[data['obs']<=data['lod'], 'obs'].values, label='Signal below LoD',
                                             color='none', edgecolor='red', s=520, zorder=2, linewidth=5)

        if identify_outliers:
            outliers_points = ax_dict['A'].scatter(data.loc[outliers_indexes, 'dateStart'].values,
                                         data.loc[outliers_indexes, 'obs'].values, label='True outliers',
                                         color='none', edgecolor='darkorchid', s=640, zorder=2, linewidth=10)

        ax_dict['A'].plot(data.dateStart.values, data.X_t.values, label='True latent signal', color=true_color, linewidth=10, zorder=3)
        ax_dict['A'].plot(data.dateStart.values, data.X_t.values, color='black', linewidth=3, zorder=3)   

        if plot_inference:

            scatter_points = ax_dict['A'].scatter(data.dateStart.values, data.obs.values, label='Measurements', 
                                 c=data.pout.values, cmap='bwr', edgecolor='black', s=360, zorder=3,
                                 linewidths=1.5, alpha=0.9, vmin=0, vmax=1)

            ax_dict['A'].plot(data.dateStart.values, data.muX.values, label='Reconstructed signal', color=reconstructed_color, linewidth=10, zorder=3)
            ax_dict['A'].plot(data.dateStart.values, data.muX.values, color='black', linewidth=3, zorder=3)               
            ax_dict['A'].fill_between(data.dateStart.values, data.ICL.values, data.ICU.values, alpha=.3, color=CI_color)


        
        ax_dict['A'].set_ylabel(ylabel)
        ax_dict['A'].set_xlabel(xlabel)
        ax_dict['A'].tick_params(axis='x', labelsize=TICK_SIZE, rotation=45)
        ax_dict['A'].tick_params(axis='y', labelsize=TICK_SIZE)
        ax_dict['A'].grid(linewidth=1, color='black', alpha=0.8)
        ax_dict['A'].set_title(title, size=TITLE_SIZE)

        # Main legend
        plt.rcParams['text.usetex'] = False
        h1, l1 = ax_dict['A'].get_legend_handles_labels()
        fig.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=4)
        if savefile:
            plt.savefig(f'../outputs/figs/{filename}.pdf', bbox_inches = 'tight')
        
        plt.show()