import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# modif par rapport à 2_9 : le prior du premier pas de temps est affiné pour être plus pertinent

## TODO : 
## - rajouter un export de l'objet SCOU en pickle plutôt que des traces 
## - remplacer l'approx de la cdf de Siyun par la fonction scipy 
## - finir les commentaires de la classe, notamment des exemples d'utilisation
## - trigger la sélection automatique des chaînes si l'un des deux critères suivants n'est pas respectés :
## - ESS >= 700 pour toutes les variables
## - Rhat > 1.01 pour toutes les variables
## - passer ces deux variables en hyperparamètres du modèles
## - permettre de fixer tous les hyperparamètres, et simplifier le code en conséquence
## - inclure un "fast mode" avec des tirages plus raisonnables (~500/500) pour avoir une première idée du résultat (toggled off par défaut)
## - améliorer la fonction visualize_latents pour rendre la visu plus jolie 
## (idéalement, une épaisseur différente par chaîne pour voir comment elles se recoupent, ça peut être un hyperparamètre de la fonction)
## - eps_pcr a une shape (self.n_replicates) par défaut, mais il faudrait prévoir une option si m labos différents traitent plusieurs sites
## et assigner à chaque site son labo 
## - modifier T_ronde pour gérer le fait que des calendriers d'échantillonnage peuvent être différents entre les sites
## - vérifier que l'intégration de la LoQ a bien été gérée dans le modèle multivarié
## - implémenter la pout sampling
## - renommer certaines variables pour améliorer la clarté et la concision du code
## - repenser la classe dans un esprit sklearn -> fit(X1, X2, X3=None) où X1 est le tenseur des observations, X2 celui des LoD et X3 celui de LoQ

def get_95CI(signal, alpha=5.0):
    """Computes the (default to 95%) confidence intervals of a sequential signal.

    Parameters
    ----------
    signal : {array-like} of shape (n_samples, n_timesteps).
    The matrix of samples. Each row represents a sample, while each column is associated to a timestep.

    alpha : float.
    1 - CI_range, where CI_range represents the range of the confidence interval.

    Returns
    -------
    CI95_lower : {array} of shape (n_timesteps, ).
    The lower bound of the (1 - alpha) confidence interval.

    CI95_upper : {array} of shape (n_timesteps, ).
    The upper bound of the (1 - alpha) confidence interval.
    """
    CI95_lower = []
    CI95_upper = []

    for timestep in range(signal.shape[1]):
        
        drawn_at_time_t = signal[:,timestep] # Gather the samples at time t
        lower_p = alpha / 2.0 # Computes the lower bound
        lower = np.percentile(drawn_at_time_t, lower_p) # Retrieves the observation at the lower percentile index
        upper_p = (100 - alpha) + (alpha / 2.0) # Computes the upper bound
        upper = np.percentile(drawn_at_time_t, upper_p) # Retrieves the observation at the upper percentile index

        CI95_lower.append(lower)
        CI95_upper.append(upper)
        
    return np.array(CI95_lower), np.array(CI95_upper)


def normal_distribution_pdf(x, loc=0, scale=1): 
    """Computes the normal probability density function (PDF) for each value in the input vector x.
    NB : Empirically much faster than scipy's, probably related to sample size I guess.

    Parameters
    ----------
    x : {array} of shape (n_samples, ).
    The array upon which the PDF is going to be applied.

    loc : float.
    The mean of the normal distribution.

    scale : float.
    The standard deviation of the normal distribution.

    Returns
    -------
    pdf : {array} of shape (n_samples, ).
    The PDF related to the input vector x.
    """
    A = 1 / (scale * np.sqrt(2 * np.pi))
    B = - (1/2) * ((x - loc)/ scale) ** 2
    
    return A * np.exp(B)

def approx_standard_normal_cdf_sw(x, loc=0, scale=1):
    """Computes the normal cumulative distribution function (CDF) for each value in the input vector x,
    using Page's approximation formula.

    Parameters
    ----------
    x : {array} of shape (n_samples, ).
    The array upon which the CDF is going to be applied.

    loc : float.
    The mean of the normal distribution.

    scale : float.
    The standard deviation of the normal distribution.

    Returns
    -------
    cdf : {array} of shape (n_samples, ).
    The CDF related to the input vector x.
    """
    xx = (x - loc) / scale
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (xx + 0.044715 * xx**3)))

class SCOU():
    """
    Bayesian implementation of the SCOU algorithm.

    SCOU is an extended Kalman Smoother, taking into consideration
    left-censored values as well as outliers.

    Parameters
    ----------
    p_out_frozen : bool, default=False
        Whether to estimate the a priori outlier probability. If set
        to False, a deterministic value will be used.

    p_out_deterministic : float, default=None
        The value to be used for p_out if p_out_frozen is set to True.

    tuning_iters : int, default=4000
        The number of tuning iterations used for the NUTS MCMC sampler.

    sampling_iters : int, default=2000
        The number of sampling iterations used for the NUTS MCMC sampler.

    nb_chains : int, default=3
        The number of Markov chains used for the NUTS MCMC sampler.

    export_chains : bool, default=False
        Whether to export the chains of parameters in a *.nc file.

    export_name : string, default='default.nc'
        The name of the export file if export_chains is set to True.

    RW_order : {1, 2}, default=1
        The order of the gaussian random walk of the underlying process.

    Attributes
    ----------
    latent_posterior_distribution : array of shape (n_samples, n_steps) 
        The posterior distribution of the latent variable.

    muX : array of shape (n_steps, )
        The average signal of the latent variable, performed over all samples.

    CIU : array of shape (n_steps, )
        The upper bound of the 95% CI of the latent variable.

    CIL : array of shape (n_steps, )
        The lower bound of the 95% CI of the latent variable.

    pointwise_pout : array of shape (n_steps, )
        The posterior probability of each observation to be an outlier.

    SCOU_model : 

    SCOU_traces : 

    T_ronde :

    borne_inf :

    borne_sup : 

    unobserved_indexes : 

    

    See Also
    --------

    Notes
    -----
    This algorithm was tailor-made to meet the expectations of the Obepine research consortium
    during the Covid-19 pandemic in terms of microbiological data processing.

    References
    ----------

    .. [1] M. Courbariaux et al., "A Flexible Smoother Adapted to Censored Data
           With Outliers and Its Application to SARS-CoV-2 Monitoring in Wastewater",
           Frontiers in Applied Mathematics and Statistics, 2022. 
           https://doi.org/10.3389/fams.2022.836349

    Examples
    --------
    >>> TODO
    """

    def __init__(self, observations,
                 censoring_threshold_lod_vect=np.array([]),
                 censoring_threshold_loq_vect=None,
                 p_out_frozen=False,
                 p_out_sampling_deterministic=None,
                 p_out_pcr_deterministic=None,
                 tuning_iters=4000,
                 sampling_iters=2000,
                 nb_chains=3,
                 export_chains=False,
                 export_name='default.nc',
                 RW_order=1,
                 target_accept=0.8):

        self.observations = observations
        self.censoring_threshold_lod_vect = censoring_threshold_lod_vect
        self.censoring_threshold_loq_vect = censoring_threshold_loq_vect
        self.n_steps = self.observations.shape[0]
        self.n_sites = self.observations.shape[1]
        self.n_replicates = self.observations.shape[2]
        self.rng = np.random.default_rng(666)
        self.tuning_iters = tuning_iters
        self.sampling_iters = sampling_iters
        self.nb_chains = nb_chains
        self.export_name = export_name
        self.export_chains = export_chains
        self.p_out_frozen = p_out_frozen
        self.p_out_sampling_deterministic = p_out_sampling_deterministic
        self.p_out_pcr_deterministic = p_out_pcr_deterministic
        self.RW_order = RW_order
        self.target_accept = target_accept


    def obs_discrimination(self):
        """Defines the set of observations, whether they are censored or not, which timesteps are not observed
        and the lower and upper bounds of the uniform distribution.
        
        """
        # TODO for pointwise comp
        #self.unobserved_indexes = np.where(np.isnan(self.observations[:,0]))[0] 
        self.T_ronde = np.unique(np.where(~np.isnan(self.observations).all(axis=1))[0])

        # premiere version, mais améliorable avec des masques vectorisés à mon avis
        self.observations_below_LoD, self.observations_between_LoQD, self.observations_above_LoQ = {}, {}, {}
        self.borne_inf, self.borne_sup = {}, {}

        for j in range(self.n_sites):
            for k in range(self.n_replicates):
                key = str(j) + '_' + str(k)

                temp_observations_below_LoD = np.where(self.observations[:,j,k]<=self.censoring_threshold_lod_vect[:,j,k])[0]

                if self.censoring_threshold_loq_vect is None:
                    temp_observations_above_LoQ = np.where(self.observations[:,j,k]>self.censoring_threshold_lod_vect[:,j,k])[0]
                    temp_observations_between_LoQD = None
                else:
                    temp_observations_above_LoQ = np.where(self.observations[:,j,k]>self.censoring_threshold_loq_vect[:,j,k])[0]
                    temp_observations_between_LoQD = np.where((self.observations[:,j,k]>self.censoring_threshold_lod_vect[:,j,k]) & (self.observations[:,j,k]<=self.censoring_threshold_loq_vect[:,j,k]))[0]
                self.observations_below_LoD[key] = temp_observations_below_LoD
                self.observations_above_LoQ[key] = temp_observations_above_LoQ
                self.observations_between_LoQD[key] = temp_observations_between_LoQD

                temp_std = np.nanstd(self.observations[:,j,k], axis=0)
                temp_min, temp_max = np.nanmin(self.observations[:,j,k], axis=0), np.nanmax(self.observations[:,j,k], axis=0)
                self.borne_inf[key], self.borne_sup[key] = temp_min - 2*temp_std, temp_max + 2*temp_std

        self.uncensored_T_ronde = {}

        for j in range(self.n_sites):
            observed_j = np.where(~np.isnan(self.observations[:,j,:]).all(axis=1))[0]
            above_lod_j = np.zeros(len(observed_j), dtype=bool)
            for k in range(self.n_replicates):
                key = f'{j}_{k}'
                above_lod_j |= np.isin(observed_j, self.observations_above_LoQ[key])
            self.uncensored_T_ronde[j] = observed_j[above_lod_j]

    def model_definition(self):
        """Defines the model parameters and observation in pyMC's framework.
        
        """
        self.obs_discrimination()

        if self.n_sites > 1 or self.n_replicates > 1:
            print('Applying multidimensional model...')
            with pm.Model() as self.SCOU_model:
                ### ----- Priors definition ----- ###
                sig = pm.InverseGamma('sig', alpha=5, beta=1) # Latent process innovation's drift
                eps_sampling = pm.InverseGamma('eps_sampling', alpha=3, beta=1, shape=self.n_sites) 
                eps_pcr = pm.InverseGamma('eps_pcr', alpha=3, beta=1, shape=self.n_replicates)

                if self.p_out_frozen:
                     p_out_sampling = self.p_out_sampling_deterministic
                     p_out_pcr = self.p_out_pcr_deterministic

                else:
                    p_out_sampling_logit = pm.Normal('p_out_sampling_logit', mu=-3, sigma=0.5, shape=self.n_sites)
                    p_out_sampling = pm.Deterministic('p_out_sampling', pm.math.invlogit(p_out_sampling_logit))
                    p_out_pcr_logit = pm.Normal('p_out_pcr_logit', mu=-3, sigma=0.5, shape=self.n_replicates)
                    p_out_pcr = pm.Deterministic('p_out_pcr', pm.math.invlogit(p_out_pcr_logit))  


                ### ----- Latent definition ----- ###
                init_mean = np.nanmean(self.observations[0])
                init_std = 5
                init_dist = pm.Normal.dist(init_mean, init_std, shape=self.n_steps)
                if self.RW_order==1:
                    latent = pm.AR("latent", rho=np.array([1]), sigma=sig, shape=self.n_steps, init_dist=init_dist) # Latent process X[t] defined as (AR(1))
                elif self.RW_order==2:
                    latent = pm.AR("latent", rho=np.array([2, -1]), sigma=sig, shape=self.n_steps, init_dist=init_dist) # Latent process X[t] defined as (AR(2))

                ### ----- Intermediate latent and observations layers  ----- ###
                for j in range(self.n_sites):
                    key = str(j) + '_' + str(0)
                    ### ----- Intermediate latent layer ----- ###
                    normal_component_j = pm.Normal.dist(mu=latent[self.uncensored_T_ronde[j]], sigma=eps_sampling[j])
                    uniform_component_j = pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key])

                    intermediate_latent_j = pm.Mixture(f'intermediate_latent_{j}',
                                                        w=[1-p_out_sampling[j], p_out_sampling[j]],
                                                        comp_dists=[normal_component_j, uniform_component_j],
                                                        transform=None
                                                        )

                    for k in range(self.n_replicates):
                        key = str(j) + '_' + str(k)
                        ### ----- Obs layer : uncensored part ----- ###
                        indexes_uncensored_j_k = np.searchsorted(self.uncensored_T_ronde[j], self.observations_above_LoQ[key])
                        ll_xhat_normal_j_k = pm.logp(pm.Normal.dist(mu=intermediate_latent_j[indexes_uncensored_j_k], sigma=eps_pcr[k]), self.observations[self.observations_above_LoQ[key], j, k])                
                        ll_xhat_uniform_j_k = pm.logp(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[self.observations_above_LoQ[key], j, k])
                        ll_xhat_mixture_j_k = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k]) + ll_xhat_normal_j_k, pm.math.log(p_out_pcr[k]) + ll_xhat_uniform_j_k)
                        pm.Potential('xhat_likelihood_uncensored_' + key, pm.math.sum(ll_xhat_mixture_j_k))

                        if self.censoring_threshold_loq_vect is not None:
                            ### ----- Obs layer : censored part - LoQ ----- ###
                            indexes_censored_loq_j_k = np.searchsorted(self.uncensored_T_ronde[j], self.observations_between_LoQD[key])
                            ll_xhat_normal_j_k = pm.logcdf(pm.Normal.dist(mu=intermediate_latent_j[indexes_censored_loq_j_k], sigma=eps_pcr[k]), self.observations[self.observations_between_LoQD[key], j, k])
                            ll_xhat_uniform_j_k = pm.logcdf(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[self.observations_between_LoQD[key], j, k])
                            ll_xhat_mixture_j_k = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k]) + ll_xhat_normal_j_k, pm.math.log(p_out_pcr[k]) + ll_xhat_uniform_j_k)
                            pm.Potential('xhat_likelihood_LoQD_' + key, pm.math.sum(ll_xhat_mixture_j_k))

                        ### ----- Obs layer : censored part - LoD ----- ###
                        #indexes_censored_j_k = np.searchsorted(self.T_ronde, self.observations_below_LoD[key])
                        #ll_xhat_normal_j_k = pm.logcdf(pm.Normal.dist(mu=intermediate_latent_j[indexes_censored_j_k], sigma=eps_pcr[k]), self.observations[self.observations_below_LoD[key], j, k])                
                        #ll_xhat_uniform_j_k = pm.logcdf(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[self.observations_below_LoD[key], j, k])
                        #ll_xhat_mixture_j_k = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k]) + ll_xhat_normal_j_k, pm.math.log(p_out_pcr[k]) + ll_xhat_uniform_j_k)
                        #pm.Potential('xhat_likelihood_censored_' + key, pm.math.sum(ll_xhat_mixture_j_k))
                        agg_std_j_k = np.sqrt(eps_sampling[j]**2 + eps_pcr[k]**2)
                        ll_xhat_normal_j_k = pm.logcdf(pm.Normal.dist(mu=latent[self.observations_below_LoD[key]], sigma=agg_std_j_k), self.observations[self.observations_below_LoD[key], j, k])                
                        ll_xhat_uniform_j_k = pm.logcdf(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[self.observations_below_LoD[key], j, k])
                        ll_xhat_mixture_j_k = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k] - p_out_sampling[j]) + ll_xhat_normal_j_k, pm.math.log(p_out_pcr[k]) + ll_xhat_uniform_j_k, pm.math.log(p_out_sampling[j]) + ll_xhat_uniform_j_k)
                        pm.Potential('xhat_likelihood_censored_' + key, pm.math.sum(ll_xhat_mixture_j_k))

        else:
            print('Applying unidimensional model...')
            with pm.Model() as self.SCOU_model:
                ### ----- Priors definition ----- ###
                sig = pm.InverseGamma('sig', alpha=5, beta=1) # Latent process innovation's drift
                eps_pcr = pm.InverseGamma('eps_pcr', alpha=3, beta=1, shape=self.n_replicates)

                if self.p_out_frozen:
                     p_out_pcr = self.p_out_pcr_deterministic

                else: 
                    p_out_pcr_logit = pm.Normal('p_out_pcr_logit', mu=-3, sigma=0.5, shape=self.n_replicates)
                    p_out_pcr = pm.Deterministic('p_out_pcr', pm.math.invlogit(p_out_pcr_logit))  


                ### ----- Latent definition ----- ###
                init_mean = np.nanmean(self.observations[0])
                init_std = 5
                init_dist = pm.Normal.dist(init_mean, init_std, shape=self.n_steps)
                if self.RW_order==1:
                    latent = pm.AR("latent", rho=np.array([1]), sigma=sig, shape=self.n_steps, init_dist=init_dist) # Latent process X[t] defined as (AR(1))
                elif self.RW_order==2:
                    latent = pm.AR("latent", rho=np.array([2, -1]), sigma=sig, shape=self.n_steps, init_dist=init_dist) # Latent process X[t] defined as (AR(2))

                j, k = 0, 0
                key = str(j) + '_' + str(k)
                ### ----- Obs layer : uncensored part ----- ###
                indexes_uncensored_j_k = self.observations_above_LoQ[key]
                ll_xhat_normal_j_k = pm.logp(pm.Normal.dist(mu=latent[indexes_uncensored_j_k], sigma=eps_pcr[k]), self.observations[indexes_uncensored_j_k, j, k])                
                ll_xhat_uniform_j_k = pm.logp(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[indexes_uncensored_j_k, j, k])
                ll_xhat_mixture_j_k = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k]) + ll_xhat_normal_j_k, pm.math.log(p_out_pcr[k]) + ll_xhat_uniform_j_k)
                pm.Potential('xhat_likelihood_uncensored_' + key, pm.math.sum(ll_xhat_mixture_j_k))

                ### ----- Obs layer : censored part - LoD ----- ###
                indexes_censored_j_k = self.observations_below_LoD[key]
                ll_xhat_normal_j_k_LoD = pm.logcdf(pm.Normal.dist(mu=latent[indexes_censored_j_k], sigma=eps_pcr[k]), self.observations[indexes_censored_j_k, j, k])                
                ll_xhat_uniform_j_k_LoD = pm.logcdf(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[indexes_censored_j_k, j, k])
                ll_xhat_mixture_j_k_LoD = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k]) + ll_xhat_normal_j_k_LoD, pm.math.log(p_out_pcr[k]) + ll_xhat_uniform_j_k_LoD)
                pm.Potential('xhat_likelihood_censored_' + key, pm.math.sum(ll_xhat_mixture_j_k_LoD))

                if self.censoring_threshold_loq_vect is not None:
                    ### ----- Obs layer : censored part - LoQ ----- ###
                    indexes_censored_loq_j_k = self.observations_between_LoQD[key]
                    ll_xhat_normal_j_k_LoQ = pm.logcdf(pm.Normal.dist(mu=latent[indexes_censored_loq_j_k], sigma=eps_pcr[k]), self.observations[indexes_censored_loq_j_k, j, k])
                    ll_xhat_uniform_j_k_LoQ = pm.logcdf(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.observations[indexes_censored_loq_j_k, j, k])
                    
                    ll_xhat_normal_j_k_LoD_LoQ = pm.logcdf(pm.Normal.dist(mu=latent[indexes_censored_loq_j_k], sigma=eps_pcr[k]), self.censoring_threshold_lod_vect[indexes_censored_loq_j_k, j, k])
                    ll_xhat_uniform_j_k_LoD_LoQ = pm.logcdf(pm.Uniform.dist(lower=self.borne_inf[key], upper=self.borne_sup[key]), self.censoring_threshold_lod_vect[indexes_censored_loq_j_k, j, k])

                    contrib_normal = pm.math.log(pm.math.exp(ll_xhat_normal_j_k_LoQ) - pm.math.exp(ll_xhat_normal_j_k_LoD_LoQ))
                    contrib_uniform = pm.math.log(pm.math.exp(ll_xhat_uniform_j_k_LoQ) - pm.math.exp(ll_xhat_uniform_j_k_LoD_LoQ))

                    ll_xhat_mixture_j_k_LoQ = pm.math.logaddexp(pm.math.log(1 - p_out_pcr[k]) + contrib_normal, pm.math.log(p_out_pcr[k]) + contrib_uniform)
                    pm.Potential('xhat_likelihood_LoQD_' + key, pm.math.sum(ll_xhat_mixture_j_k_LoQ))

    def fit(self):
        """Computes the MCMC estimation of the model parameters using the NUTS sampler.

        """
        self.model_definition()

        # Inférence
        with self.SCOU_model:
            self.SCOU_traces = pm.sample(self.sampling_iters, tune=self.tuning_iters, 
                                    chains=self.nb_chains, 
                                    return_inferencedata=True, 
                                    random_seed=self.rng,
                                    target_accept=self.target_accept)

        if self.n_sites > 1 or self.n_replicates > 1:
            self.params = ['sig', 'eps_sampling', 'eps_pcr'] 
            if not self.p_out_frozen:
                self.params = ['sig', 'eps_sampling', 'p_out_sampling', 'eps_pcr', 'p_out_pcr'] 

        else:
            self.params = ['sig', 'eps_pcr'] 
            if not self.p_out_frozen:
                self.params = ['sig', 'eps_pcr', 'p_out_pcr'] 

        print("Raw summary:")
        print(az.summary(self.SCOU_traces, var_names=self.params))
        self.params_summary = az.summary(self.SCOU_traces, var_names=self.params)

        if self.export_chains:
            self.SCOU_traces.to_netcdf(self.export_name)

    def predict(self, selected_chains):
        """Computes the latent distribution, as well as its mean and 95% confidence intervals and pointwise outlier probabilities
        for a subset of selected chains.

        Parameters
        ----------
        selected_chains : {array} of shape (n_selected_chains, ).
        The array of indexes of the selected chains, ranging from 0 to self.nb_chains.
        """
        self.n_samples = len(selected_chains)*self.sampling_iters
        self.latent_posterior_distribution = self.SCOU_traces['posterior']['latent'].values[selected_chains].reshape(self.n_samples, -1)
        self.muX = self.latent_posterior_distribution.mean(axis=0)
        self.CIU, self.CIL = get_95CI(self.latent_posterior_distribution)

        if self.n_sites > 1 or self.n_replicates > 1:
            self.intermediate_latent_distributions = {}
            self.muY = {}
            for j in range(self.n_sites):
                self.intermediate_latent_distributions[j] = self.SCOU_traces['posterior']['intermediate_latent_' + str(j)].values[selected_chains].reshape(self.n_samples, -1)
                self.muY[j] = self.intermediate_latent_distributions[j].mean(axis=0)
        self.compute_pointwise_outlier_probabilities(selected_chains)

        print("Best chain combination summary:")
        print(az.summary(self.SCOU_traces.sel(chain=selected_chains), var_names=self.params))

    def compute_pointwise_outlier_probabilities_pcr(self, selected_chains):
        """Computes the pointwise outlier probabilities for a subset of selected chains.

        Parameters
        ----------
        selected_chains : {array} of shape (n_selected_chains, ).
        The array of indexes of the selected chains, ranging from 0 to self.nb_chains.
        """
        nb_draws = self.sampling_iters * len(selected_chains)

        # Output shapes : (n_steps, n_sites, n_replicates), (n, W, R)
        self.pointwise_pout = np.ones((self.n_steps, self.n_sites, self.n_replicates)) * np.nan
        self.pointwise_pout_dist = np.ones((self.n_steps, self.n_sites, self.n_replicates, nb_draws)) * np.nan
        



        #else:
        for j in range(self.n_sites):
            for k in range(self.n_replicates):
                key = f'{j}_{k}'#str(j) + '_' + str(k)
                
                # compute g_outlier(\hat{X}_{t,j,k}, a_{j,k}, b_{j,k}):
                a_jk = self.borne_inf[key]
                b_jk = self.borne_sup[key]
                g_out = np.ones(self.n_steps) * (1 / (b_jk - a_jk))
                idx_lod = self.observations_below_LoD[key]
                g_out[idx_lod] = ((self.censoring_threshold_lod_vect[idx_lod, j, k] - a_jk) / (b_jk - a_jk))
                if self.censoring_threshold_loq_vect is not None:
                    idx_loq = self.observations_between_LoQD[key]
                    g_out[idx_loq] = ((self.censoring_threshold_loq_vect[idx_loq, j, k] - self.censoring_threshold_lod_vect[idx_loq, j, k]) / (b_jk - a_jk))

                # gather hyperparameters MCMC's distributions:
                eps_k = self.SCOU_traces['posterior']['eps_pcr'].values[selected_chains, :, k].reshape(len(selected_chains) * self.sampling_iters, )  # (nb_draws,)
                if self.n_sites>1:
                    y_all = self.SCOU_traces['posterior'][f'intermediate_latent_{j}'].values[selected_chains].reshape(len(selected_chains) * self.sampling_iters, -1)  # (nb_draws, n_steps)

                x_all = self.SCOU_traces['posterior']['latent'].values[selected_chains].reshape(len(selected_chains) * self.sampling_iters, -1)  # (nb_draws, n_steps)

                if self.p_out_frozen:
                    pout_k = self.p_out_pcr_deterministic[k]
                else:
                    pout_k = self.SCOU_traces['posterior']['p_out_pcr'].values[selected_chains, :, k].reshape(len(selected_chains) * self.sampling_iters, )  # (nb_draws,)

                # compute g_norm:
                for this_timestep in self.T_ronde:
                    xhat_tjk = self.observations[this_timestep, j, k]
                    
                    if self.n_sites>1:
                        if this_timestep in self.uncensored_T_ronde[j]:
                            ts_idx = np.where(self.uncensored_T_ronde[j]==this_timestep)[0][0]
                            x_t = y_all[:, ts_idx]   # (nb_draws,)
                        else:
                            x_t = x_all[:, this_timestep] # if both measures are censored, a specific case has to be made as Y_t is not drawn so we take x as an approximation

                    else:
                        x_t = x_all[:, this_timestep] # if both measures are censored, a specific case has to be made as Y_t is not drawn so we take x as an approximation

                    num = pout_k * g_out[this_timestep]

                    if this_timestep in idx_lod:
                        denom_not_outlier = (1 - pout_k) * approx_standard_normal_cdf_sw(xhat_tjk, x_t, eps_k)
                    elif (self.censoring_threshold_loq_vect is not None and this_timestep in idx_loq):
                        cdf_loq = approx_standard_normal_cdf_sw(xhat_tjk, x_t, eps_k)
                        cdf_lod = approx_standard_normal_cdf_sw(self.censoring_threshold_lod_vect[this_timestep, j, k], x_t, eps_k)
                        denom_not_outlier = (1 - pout_k) * (cdf_loq - cdf_lod)
                    else: 
                        denom_not_outlier = (1 - pout_k) * normal_distribution_pdf(xhat_tjk, x_t, eps_k)

                    denom = denom_not_outlier + num

                    self.pointwise_pout_dist[this_timestep, j, k] = num / denom
                    self.pointwise_pout[this_timestep, j, k] = np.mean(self.pointwise_pout_dist[this_timestep, j, k])


    


    def compute_pointwise_outlier_probabilities_sampling(self, selected_chains):
        """
        """

    def compute_pointwise_outlier_probabilities(self, selected_chains):
        """
        """
        self.compute_pointwise_outlier_probabilities_pcr(selected_chains)
        #self.compute_pointwise_outlier_probabilities_sampling(selected_chains)

    def visualize_latents(self, selected_chains):
        """Plots the mean of the distributions of the latent variable for each chain on a first figure.
        Plots the same distribution only for a subset of selected chains on a second figure.

        Parameters
        ----------
        selected_chains : {array} of shape (n_selected_chains, ).
        The array of indexes of the selected chains, ranging from 0 to self.nb_chains.
        """
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
        
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        ax_dict = fig.subplot_mosaic(
            """
            AB
            """
        )
        for i in range(self.nb_chains):
            ax_dict['A'].plot(self.SCOU_traces['posterior']['latent'][i].mean(axis=0), label=i, linewidth=3)
        ax_dict['A'].set_title('Raw chains', size=TITLE_SIZE)
        ax_dict['A'].grid(linewidth=1, color='black', alpha=0.8)
        ax_dict['A'].legend()
        
        for i in selected_chains:
            ax_dict['B'].plot(self.SCOU_traces['posterior']['latent'][i].mean(axis=0), label=i, linewidth=3)

        ax_dict['B'].set_title('Optimized chains', size=TITLE_SIZE)
        ax_dict['B'].grid(linewidth=1, color='black', alpha=0.8)
        ax_dict['B'].legend()

        plt.show()