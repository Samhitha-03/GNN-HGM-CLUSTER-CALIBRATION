# standard Python imports
import numpy as np
import os, sys
import argparse
import yaml

# PyTorch imports
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import MaxAbsScaler

#######################################################################
### plot / histogram settings ###
try:
    try:
        plt.style.use('/afs/cern.ch/user/i/issainzs/private/nn-topocluster-calibration/plotting-routine/plotting.mplstyle')
    except Exception:
        plt.style.use('default')
except Exception:
    plt.style.use('default')
figsize_double = (2.953, 3.2221)
subs           = (0.2, 0.4, 0.6, 0.8, 1.0)
rect_single    = (0.13, 0.12, 0.97, 0.98) # left, bottom, right, top
rect_double    = (0.14, 0.08, 1.00, 0.97) # left, bottom, right, top
color_truth    = 'steelblue' # 
color_model    = 'indigo' # 
num_bins       = 50   # number of histogram bins



def standardize(x):
    '''
    Standardization rescales the data to have a mean of 0
    and a standard deviation of 1 (unit variance).
    '''
    mean, std = np.mean(x), np.std(x)
    out = (x-mean)/std
    return out, mean, std

def apply_save_log(x):
    epsilon = 1e-10
    minimum = np.min(x)
    if minimum <= 0:
        x = x-minimum+epsilon
    else:
        minimum = 0
        epsilon = 0
    return np.log10(x), minimum, epsilon




class ClusterCalibrationDataset(Dataset):

    def __init__(self, device, config_file, input_path, output_path, make_plots):

        super(ClusterCalibrationDataset, self).__init__()
        self.device      = device      # device (GPU or CPU)
        self.config_file = config_file # config file
        self.input_path  = input_path  # input path (datasets)
        self.output_path = output_path # output path (for plots and logfile)

        with open(self.config_file) as info:
            config = yaml.safe_load(info)

        self.mc_release  = config['model']['data_args']['mc_release']
        data_size_trn    = config['model']['data_args']['data_size_trn']
        data_size_val    = config['model']['data_args']['data_size_val']
        data_size_tst    = config['model']['data_args']['data_size_tst']
        self.batch_idxs  = config['model']['data_args']['batch_idxs']

        # define the path and filenames where the train/validation/test datasets are stored
        self.num_clus = [data_size_trn, data_size_val, data_size_tst]
        num_clus_tot = int(np.sum(self.num_clus))
        num_files    = int(np.ceil(num_clus_tot/500e3))

        print(f'[info] starting dataset preprocessing')
        print(f'[info] using folder : {self.input_path}')
        print(f'[info] number of initial topo-cluster trn/val/tst : {self.num_clus}', flush=True)
        print(f'[info] (randomly) sampled batch numbers   : {self.batch_idxs}', flush=True)
        
        
        if len(self.batch_idxs) > 0:
            print(f'[debug] total number of files and length of batch list: {num_files} and {len(self.batch_idxs)}')
            assert num_files == len(self.batch_idxs), 'total number of jets does not match length of batch list'
        else: 
            lst = os.listdir(self.input_path) # read number of files in directory
            max_idx = len(lst)
            print(f'[debug] number of files in input directory: {max_idx}')
            self.batch_idxs = np.random.choice(np.arange(1, max_idx+1), size=num_files, replace=False)

        if self.mc_release == 'mc16':
            filenames = [f'clustercalib_bnn_500k_batch{i:03d}.npy' for i in self.batch_idxs]
            
        else:
            filenames = [f'clustercalib_nn_500k_batch{i:03d}.npy' for i in self.batch_idxs]

        

        print(f'[info] starting loading dataset batches', flush=True)
        data = [] # load and combine dataset batches (trn/val/tst combined)
        for file in filenames: data.append(np.load(f'{self.input_path}/{file}'))
        data = np.concatenate(data, axis=0)
        print(f'[info] loading dataset batches done', flush=True)

        print(f'[info] split combined dataset into trn/val/tst data', flush=True)
        self.data_trn_before = data[0:self.num_clus[0],:].copy()
        self.data_val_before = data[self.num_clus[0]:int(self.num_clus[0]+self.num_clus[1]),:].copy()
        self.data_tst_before = data[int(self.num_clus[0]+self.num_clus[1]):,:].copy()

        self.feature_range = { # dictionary containing all variables and their corresponding index, value range, binning and log-scaling
                #'r_ems': [0,  +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_truth or r_target (120 bins if x-axis)
                'r_ems': [0,  +1.000e-01, +6.900e+00, 1200, True, +1.000e-05, +1.000e+03], # r_truth or r_target (120 bins if x-axis)
                'e_ems': [1,  +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # clusterE (cluster signal)
                'y_ems': [2,  -2.750e+00, +2.750e+00,   55, False, -2.400e+00, +2.400e+00], # clusterEta (cluster rapidity)
                'm_sig': [3,  +1.259e-03, +7.943e+02,   57,  True, +8.000e-02, +7.943e+02], # cluster_SIGNIFICANCE (signal significance)
                't_ems': [4,  -1.230e+02, +1.230e+02,  123, False, -5.800e+01, +4.400e+01], # cluster_time (cluster time)
                'm_tim': [5,  -1.000e+01, +1.990e+02,   50, False, +0.000e+00, +1.990e+02], # cluster_SECOND_TIME (cell-time variance)
                'm_lam': [6,  +0.000e+00, +6.000e+03,  100, False, +4.900e-03, +4.100e+03], # cluster_CENTER_LAMBDA (depth in calorimeter)
                'm_cog': [7,  +1.200e+03, +7.800e+03,  100, False, +1.200e+03, +6.600e+03], # cluster_CENTER_MAG (distance from vertex)
                #'m_fem': [8,  -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_ENG_FRAC_EM (signal fraction in EMC)
                'm_fem': [8,  -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_ENG_FRAC_EM_INCL (signal ratio in EMC)
                'm_rho': [9,  +1.000e-09, +1.000e-03,   60,  True, +1.000e-09, +4.400e-04], # cluster_FIRST_ENG_DENS (signal density)
                'm_lon': [10, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_LONGITUDINAL (longitudinal signal dispersion)
                'm_lat': [11, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_LATERAL (lateral signal dispersion)
                'm_ptd': [12, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_PTD (signal compactness)
                'm_iso': [13, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_ISOLATION (cluster isolation)
                'p_npv': [14, -2.500e-01, +9.575e+01,   48, False, -5.000e-02, +5.850e+01], # nPrimVtx (number of reconstructed primary vertices)
                #'p_mmu': [15, -2.500e-01, +9.575e+01,   48, False, -5.000e-02, +8.050e+01], # avgMu (number of pile-up interactions)
                'p_mmu': [15, -2.500e-01, +11.000e+01,  44, False, -5.000e-02, +8.050e+01], # avgMu (number of pile-up interactions)
                'e_dep': [16, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_truth or cluster_ENG_CALIB_TOT
                'e_hgm': [17, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_model from deterministic NN (HGM)
                'e_bnn': [18, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_model from Bayesian NN (BNN)
                'r_hgm': [19, +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_model from deterministic NN (120 bins if x-axis)
                'r_bnn': [20, +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_model from Bayesian NN (120 bins if x-axis)
                'c_hgm': [21, -1.050e+00, +6.950e+00, 1600, False, -6.500e-01, +0.950e+00], # closure or prediction power: r_hgm/r_ems-1 (160 bins if x-axis)
                'c_bnn': [22, -1.050e+00, +6.950e+00, 1600, False, -6.500e-01, +0.950e+00], # closure or prediction power: r_bnn/r_ems-1 (160 bins if x-axis)
                'l_ems': [23, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_ems/e_dep-1 = clusterE/cluster_ENG_CALIB_TOT-1
                'l_had': [24, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_had/e_dep-1 = clusterECalib/cluster_ENG_CALIB_TOT-1
                'l_hgm': [25, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_hgm/e_dep-1 = e_hgm/cluster_ENG_CALIB_TOT-1
                'l_bnn': [26, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_bnn/e_dep-1 = e_bnn/cluster_ENG_CALIB_TOT-1
            }
        self.feature_names = { # dictionary containing all variables and their corresponding LaTeX description (and unit if needed)
                'r_ems': [0,  'target response', '$\mathcal{R}_{\mathrm{clus}}^{\mathrm{EM}}$'],
                'e_ems': [1,  'cluster signal', '$E_{\mathrm{clus}}^{\mathrm{EM}}$', '[$\mathrm{GeV}$]'],
                'y_ems': [2,  'cluster rapidity', '$y_{\mathrm{clus}}^{\mathrm{EM}}$'],
                'm_sig': [3,  'signal significance', '$\zeta_{\mathrm{clus}}^{\mathrm{EM}}$'],
                't_ems': [4,  'cluster time', '$t_{\mathrm{clus}}$', '[$\mathrm{ns}$]'],
                'm_tim': [5,  'cell-time variance', '$\mathrm{Var}_{\mathrm{clus}}(t_{\mathrm{cell}})$', '[$\mathrm{ns}^{2}$]'],
                'm_lam': [6,  'depth in calorimeter', '$\lambda_{\mathrm{clus}}$', '[$\mathrm{mm}$]'],
                'm_cog': [7,  'distance from vertex', '$\\vert\\vec{c}_{\mathrm{clus}}\\vert$', '[$\mathrm{mm}$]'],
                #'m_fem': [8,  'signal fraction in EMC', '$f_{\mathrm{emc}}$'],
                'm_fem': [8,  'signal ratio in EMC', '$f_{\mathrm{emc}}^{\mathrm{incl}}$'],
                'm_rho': [9,  'signal density', '$\\rho_{\mathrm{clus}}=\langle\\rho_{\mathrm{cell}}\\rangle$', '[$\mathrm{GeV}/\mathrm{mm}^{3}$]'],
                'm_lon': [10, 'longitudinal signal dispersion', '$\langle\mathfrak{m}_{\mathrm{long}}^{2}\\rangle$'],
                'm_lat': [11, 'lateral signal dispersion', '$\langle\mathfrak{m}_{\mathrm{lat}}^{2}\\rangle$'],
                'm_ptd': [12, 'signal compactness', '$p_{\mathrm{T}}D$'],
                'm_iso': [13, 'cluster isolation', '$f_{\mathrm{iso}}$'],
                'p_npv': [14, 'number of reconstructed primary vertices', '$N_{\mathrm{PV}}$'],
                'p_mmu': [15, 'number of pile-up interactions', '$\mu$'],
                'e_dep': [16, 'deposited energy', '$E_{\mathrm{clus}}^{\mathrm{dep}}$', '[$\mathrm{GeV}$]'],
                'e_hgm': [17, 'HGM-learned energy', '$E_{\mathrm{clus}}^{\mathrm{HGM}}$', '[$\mathrm{GeV}$]'],
                'e_bnn': [18, 'BNN-learned energy', '$E_{\mathrm{clus}}^{\mathrm{BNN}}$', '[$\mathrm{GeV}$]'],
                'r_hgm': [19, 'HGM-learned response', '$\mathcal{R}_{\mathrm{clus}}^{\mathrm{HGM}}$'],
                'r_bnn': [20, 'BNN-learned response', '$\mathcal{R}_{\mathrm{clus}}^{\mathrm{BNN}}$'],
                'c_hgm': [21, 'prediciton power', '$\Delta_{\mathcal{R}}^{\mathrm{HGM}}$'],
                'c_bnn': [22, 'prediciton power', '$\Delta_{\mathcal{R}}^{\mathrm{BNN}}$'],
                'l_ems': [23, 'signal linearity', '$\Delta_{E}^{\mathrm{EM}}$'],
                'l_had': [24, 'signal linearity', '$\Delta_{E}^{\mathrm{had}}$'],
                'l_hgm': [25, 'signal linearity', '$\Delta_{E}^{\mathrm{HGM}}$'],
                'l_bnn': [26, 'signal linearity', '$\Delta_{E}^{\mathrm{BNN}}$'],
            }

        # select target and input-feature keys from directory (response and 15 topo-cluster observables)
        self.keys_feats = [key for key in self.feature_names if (self.feature_names[key][0] >= 0) and (self.feature_names[key][0] <= 15)]

        if make_plots:
            self.make_plots_before(data_type='trn')
            self.make_plots_before(data_type='val')
            self.make_plots_before(data_type='tst')

       
        print(f'[info] starting dataset preprocessing', flush=True)

        # preprocess the training dataset (and save the corresponding scales)
        self.data_trn_after, self.scales_trn = self.preprocessing(config, self.data_trn_before, self.feature_names)

        # apply training-dataset scales to the validation and test datasets
        self.data_val_after = self.apply_scales(config, self.data_val_before, self.feature_names, self.scales_trn)
        self.data_tst_after = self.apply_scales(config, self.data_tst_before, self.feature_names, self.scales_trn)

        if make_plots:
            self.make_plots_after(data_type='trn')
            self.make_plots_after(data_type='val')
            self.make_plots_after(data_type='tst')


        # convert NumPy arrays to PyTorch tensors and move everything to the GPU
        self.data_trn_after = torch.from_numpy(self.data_trn_after).to(self.device)
        self.data_val_after = torch.from_numpy(self.data_val_after).to(self.device)
        self.data_tst_after = torch.from_numpy(self.data_tst_after).to(self.device)


        print(f'[info] dataset preprocessing done', flush=True)




    def preprocessing(self, config, data_before, columns):

        scales = {}
        data_after = data_before.copy()

        # preprocessing: log10-preprocessing
        fields = ['r_ems']
        for field in fields:
            x = np.log10(data_after[:,columns[field][0]])
            data_after[:,columns[field][0]] = x

        # preprocessing: standardization
        features = ['clusterEta', 'cluster_CENTER_MAG', 'nPrimVtx', 'avgMu', 'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_ISOLATION' ] #'cluster_ENG_FRAC_EM_INCL'
        #fields = ['y_ems', 'm_cog', 'p_npv', 'p_mmu', 'm_lat', 'm_lon', 'm_ptd', 'm_iso'] #'m_fem'
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
                x = data_after[:,columns[field][0]]
                x, mean, std = standardize(x)
                #config['data']['features'][feature]['preprocessing']['parameters'] = [mean, std]
                scales[field] = ('standardization', mean, std)
                data_after[:,columns[field][0]] = x
        
        # preprocessing: log10-preprocessing and standardization
        features = ['clusterE', 'cluster_CENTER_LAMBDA', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE' ]
        #fields = ['e_ems', 'm_lam', 'm_rho', 'm_tim', 'm_sig']
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
                x, minimum, epsilon = apply_save_log(data_after[:,columns[field][0]])
                x, mean, std = standardize(x)
                #config['data']['features'][feature]['preprocessing']['parameters'] = [minimum, epsilon, mean, std]
                scales[field] = ('log10-preprocessing and standardization', minimum, epsilon, mean, std)
                
                data_after[:,columns[field][0]] = x

        # preprocessing: special preprocessing
        features = ['cluster_time']
        #fields = ['t_ems'] #m_fem
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
                x = data_after[:,columns[field][0]]
                #x = np.sign(x)*np.abs(x)**(1./3.)
                #x, mean, std = standardize(x)
                #scales[field] = ('sqrt3 and standardization', mean, std)
                #xmin = np.min(x)
                #xmax = np.max(x)
                #scales[field] = ('min-max normalization', xmin, xmax)
                #data_after[:,columns[field][0]] = -4+8*(x-xmin)/(xmax-xmin)
                if field=='m_fem':
                    x_clipped = np.clip(x, np.percentile(x, 0.1), np.percentile(x, 99.9))
                    x = x_clipped
                
                scale = max(abs(x))
                #config['data']['features'][feature]['preprocessing']['parameters'] = [scale]
                scales[field] = ('maxabs-scaling', scale)
                self.Maxabsscaler = MaxAbsScaler()
                data_after[:,columns[field][0]] = np.squeeze(self.Maxabsscaler.fit_transform(x[:,None]))

        # preprocessing: Minmax scaling
        features = ['cluster_ENG_FRAC_EM_INCL']
        #fields = ['m_fem']
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
               x = data_after[:,columns[field][0]]
               xmin = np.min(x)
               xmax = np.max(x)
               #config['data']['features'][feature]['preprocessing']['parameters'] = [xmin, xmax]
               scales[field] = ('min-max normalization', xmin, xmax)
               data_after[:,columns[field][0]] = (x-xmin)/(xmax-xmin)


        """
        #Overwritting the yaml file currently destroys the structure and deletes comments 
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=True)
        """

        return data_after, scales

    def apply_scales(self, config, data_before, columns, scales):

        data_after = data_before.copy()

        # preprocessing: log10-preprocessing
        fields = ['r_ems']
        for field in fields:
            x = np.log10(data_after[:,columns[field][0]])
            data_after[:,columns[field][0]] = x

        # preprocessing: standardization
        features = ['clusterEta', 'cluster_CENTER_MAG', 'nPrimVtx', 'avgMu', 'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_ISOLATION' ] #'cluster_ENG_FRAC_EM_INCL'
        #fields = ['y_ems', 'm_cog', 'p_npv', 'p_mmu', 'm_lat', 'm_lon', 'm_ptd', 'm_iso'] #'m_fem'
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
                x = data_after[:,columns[field][0]]
                #loading from config file
                params = config['data']['features'][feature]['preprocessing']['parameters']
                mean, std = params[0], params[1]
                #loading from preprocessing function
                #mean, std = scales[field][-2], scales[field][-1]

                data_after[:,columns[field][0]] = (x-mean)/std
        
        # preprocessing: log10-preprocessing and standardization
        features = ['clusterE', 'cluster_CENTER_LAMBDA', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE' ]
        #fields = ['e_ems', 'm_lam', 'm_rho', 'm_tim', 'm_sig']
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
                #loading from config file
                params = config['data']['features'][feature]['preprocessing']['parameters']
                mean, std = params[2], params[3]
                #loading from preprocessing function
                #mean, std = scales[field][-2], scales[field][-1]
                x, minimum, epsilon = apply_save_log(data_after[:,columns[field][0]])

                data_after[:,columns[field][0]] = (x-mean)/std

        # preprocessing: special preprocessing, max abs scaling
        features = ['cluster_time']
        #fields = ['t_ems'] #m_fem
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
                x = data_after[:,columns[field][0]]
                #loading from config file
                params = config['data']['features'][feature]['preprocessing']['parameters']
                xmax = params[0]
                #loading from preprocessing function
                #xmax = scales[field][-1]
                #data_after[:,columns[field][0]] = np.squeeze(self.Maxabsscaler.transform(x[:,None]))

                data_after[:,columns[field][0]] = x/xmax

        # preprocessing: Minmax scaling
        features = ['cluster_ENG_FRAC_EM_INCL']
        #fields = ['m_fem']
        for feature in features:
            field = config['data']['features'][feature]['trained_name']
            if field in list(columns.keys()):
               x = data_after[:,columns[field][0]]
               #loading from config file
               params = config['data']['features'][feature]['preprocessing']['parameters']
               xmin, xmax = params[0], params[1]
               #loading from preprocessing function
               #xmin, xmax = scales[field][-2], scales[field][-1]
               data_after[:,columns[field][0]] = (x-xmin)/(xmax-xmin)
              

        return data_after
    



    def make_plots_before(self, data_type):

        filename = f'{self.output_path}/dataset-plots/plots_{data_type}_dataset_before.pdf'
        with PdfPages(filename) as pdf:
            for key in self.keys_feats:
                if data_type == 'trn': data = self.data_trn_before[:,self.feature_names[key][0]]
                if data_type == 'val': data = self.data_val_before[:,self.feature_names[key][0]]
                if data_type == 'tst': data = self.data_tst_before[:,self.feature_names[key][0]]

                if key == 'r_ems':

                    fig, ax = plt.subplots(1, 1, figsize=(1.40*2.953, 1.20*2.568))
                    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
                    bins_log = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), num_bins)
                    ax.hist(data, bins=bins_log, histtype='step', density=False, color=color_truth)
                    ax.set_xlabel(' '.join(self.feature_names[key][1:]))
                    ax.set_ylabel('Topo-clusters')
                    ax.set_xlim((np.min(data), np.max(data)))
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                    pdf.savefig(fig)
                    plt.close(fig)

                fig, ax = plt.subplots(1, 1, figsize=(1.40*2.953, 1.20*2.568))
                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
                ax.hist(data, bins=num_bins, histtype='step', density=False, color=color_truth)
                ax.set_xlabel(' '.join(self.feature_names[key][1:]))
                ax.set_ylabel('Topo-clusters')
                ax.set_xlim((np.min(data), np.max(data)))
                ax.set_yscale('log')
                ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                pdf.savefig(fig)
                plt.close(fig)
                
    def make_plots_after(self, data_type):

        filename = f'{self.output_path}/dataset-plots/plots_{data_type}_dataset_after.pdf'
        with PdfPages(filename) as pdf:
            for key in self.keys_feats:
                if data_type == 'trn': data = self.data_trn_after[:,self.feature_names[key][0]]
                if data_type == 'val': data = self.data_val_after[:,self.feature_names[key][0]]
                if data_type == 'tst': data = self.data_tst_after[:,self.feature_names[key][0]]

                fig, ax = plt.subplots(1, 1, figsize=(1.40*2.953, 1.20*2.568))
                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
                ax.hist(data, bins=num_bins, histtype='step', density=True, color=color_truth)
                ax.set_xlabel(' '.join(self.feature_names[key][1:]))
                ax.set_ylabel('Fraction of topo-clusters')
                ax.set_xlim((np.min(data), np.max(data)))
                ax.set_yscale('log')
                ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                pdf.savefig(fig)
                plt.close(fig)
