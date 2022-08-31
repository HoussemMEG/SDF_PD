import json
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import mne
import numpy as np

import preprocessing
import reader
import utils
from featuregen import DFG
from plots import Plotter
from utils import print_c


# Verbosity
mne.set_log_level(verbose='WARNING')  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
reader.reader_verbose = False
preprocessing.preprocessing_verbose = True
data_scale = 1e6  # to convert from µV to V


# Paths initialisation
data_dir = "1-30[Hz] + EB ICA removed"  # read data from this directory, name should be changed in the MATLAB file Step2
data_path = os.path.join('./data/processed', data_dir).replace("\\", "/")
save_to = "preprocessing info folder"  # info of the preprocessing are kept in this folder for reproductivity purpose


# Session creation, each generated features are time stamped + their generation parameters are stored in the same folder
date = datetime.now().strftime("%Y-%m-%d %H;%M")
session_folder = os.path.join(os.getcwd(), 'generated SDF', date)
backup_folder = os.path.join(os.getcwd(), 'generated SDF backup', date)


# Reading parameters
subj_info_keys = ('category', 'state', 'gender', 'age')  # Subject information we should look for in the info folder


# Parameters
subj_session = ['on_medication',      # 0 to select on_medication sessions with CTL
                'off_medication'][1]  # 1 to select off_medication sessions with CTL

stim_types = [[['standard'], ['novel'], ['target']],  # 0 to generate the SDF for all the stim separately
              [['standard'], ['novel'], ['target'], ['standard', 'novel', 'target']],  # 1 same as 0 but also for the sum of all of the stims
              [['target']]  # -1 to generate the SDF of only the target stim
              ][-1]
features_dict = dict([('_'.join(stim), {}) for stim in stim_types])

case = ['evoked',       # 0 to first compute ERP then generate SDF on the resulting ERP
        'epochs'][0]    # 1 to generate SDF for each epoch, (we tried that but we did not put it in the paper)

channel_picks = [['CP1', 'CPz', 'CP2'],     # 0  to pick the selected channels (the order is not important)
                 ['CPz', 'FCz'],            # 1
                 'all',                     # -1 to pick all channels
                 ][0]


# File reading class
my_reader = reader.Reader(path=data_path,
                          info_path='./data/info',
                          subj_info_keys=subj_info_keys,
                          use_eog=True)

# PreProcessing class
preprocess = preprocessing.PreProcessing(path=data_path,
                                         save=False, save_path=save_to, save_precision='double', overwrite=True,
                                         shift=False, t_shift=0.0,
                                         sampling_freq=None,
                                         filter=[None, None], n_jobs=-1,
                                         rejection_type=None, reject_t_min=-0.1, reject_t_max=0.6,
                                         ref_channel=None,
                                         crop=False, crop_t_min=0.0, crop_t_max=0.5, include_t_max=False,
                                         baseline=[None, None])

# Channel extractor parameters, to transform the data into PCA / ICA domain instead of the channel ones
# preferably don't use it and keep it as it is
ch_extractor_param = {'method': None,  # 'PCA', 'ICA', None
                      'n_components': None,
                      'whiten': True}

# Dynamical Feature Generator class, mainly this is the class you need to play with, see documentation of the class
feature_gen = DFG(method='LARS',
                  f_sampling=500,
                  version=1,
                  alpha=8e-4,
                  normalize=True,
                  omit=None,  # omit either 'x0' or 'u' from y_hat computation
                  model_freq=list(np.concatenate((np.linspace(1, 15, 20, endpoint=False), np.linspace(15, 30, 20)))),
                  damping=None,  # (under-damped 0.008 / over-damped 0.09)
                  fit_path=True, ols_fit=True,
                  fast=True,
                  selection=np.arange(0.02, 1.02, 0.02), selection_alpha=None,
                  plot=False, show=True, fig_name="fig name", save_fig=False)

# Plotting class of the EEG signal
plotter = Plotter(disable_plot=True,             # if True disable all plots
                  plot_data=False,               # plot all the data (epochs / evoked)
                  plot_psd=False,                # plot power spectral density (epochs)
                  plot_sensors=False,            # sensor location plot (epochs / evoked)
                  plot_image=False, split=True,  # plot epochs image and ERP (epochs)
                  plot_psd_topomap=False,        # plot power spectral density and topomap (epochs)
                  plot_topo_image=False,         # plot epochs image on topomap (epochs)
                  plot_topo_map=False,           # plot ERPs on topomap (evoked)
                  plot_evoked_joint=False,       # plot ERPs data and the topomap on peaks time (evoked)
                  show=True,                     # to whether show the plotted figures or to save them directly
                  save_fig=False,
                  save_path=os.path.join(os.getcwd(), 'figures'))


# Main
for data in my_reader.read_data():  # Data reading
    subj_desc = json.loads(data.info['description'])
    if subj_desc['subject_info']['state'] != subj_session and subj_desc['subject_info']['state'] is not None:
        print('\tSubject not in <{:}> session'.format(subj_session))  # skip if not in the right session
        continue

    preprocess.set_data(data, epoch_drop_idx=None, epoch_drop_reason='USER', channel_drop=None)
    data = preprocess.process()

    # Channel extraction (perform PCA / ICA if not None, else skip)
    proj_data, x, decomp = utils.virtual_ch_extractor(data=data, **ch_extractor_param, plot_data=False,
                                                      plot_evoked=False, save_fig=False, show=True)
    utils.save_args({'ch_extractor': ch_extractor_param},
                    path=preprocess.save_path, save_name='info\\preprocessing_parameters')
    # my_preprocess.save(data=proj_data, save_format='-PCA-epo.fif')

    # Dynamical Feature Extraction
    for i, stim in enumerate(stim_types):
        print_c('\tStim type: <{:}>  {:}/{:}'.format('_'.join(stim), i + 1, len(stim_types)), 'green')
        picks = [ch for ch in data.info['ch_names'] if ch in channel_picks] if not channel_picks == 'all' \
            else data.info['ch_names']  # channel ordering should be kept like this or the classifier wont work

        # Evoked
        if case == 'evoked':
            evoked = data[stim].average().pick_channels(picks)
            evoked.apply_baseline((None, None))  # remove average
            data_ = evoked.data

        # Epochs
        if case == 'epochs':
            epochs = data[stim].pick_channels(picks)
            epochs.apply_baseline((None, None))  # remove average
            data_ = epochs.get_data()
            data_ = data_[np.random.choice(data_.shape[0], 20, replace=False), 0, :]  # draw randomly 20 epochs

        # SDF generation
        features, x0 = feature_gen.generate(data_.T * data_scale)
        features, x0 = utils.compress(features), utils.ndarray_to_list(x0)

        # Appending the new subject calculated SDF to the SDF container (memory)
        temp = {subj_desc['file_name']: {'features': features,
                                         'x0': x0,
                                         'subject_info': subj_desc}}
        features_dict['_'.join(stim)].update(temp)
    plt.show(block=True)


# Saving the generated SDF and the working parameters
utils.save_args(features_dict, path=session_folder, save_name='generated_features', verbose=True)
utils.save_args(preprocess._saved_args, verbose=True, path=session_folder, save_name='preprocessing_parameters')
utils.save_args({**feature_gen.parameters, **{'channel_picks': picks}, **{'data_case': case}},
                path=session_folder, save_name='DFG_parameters', verbose=True)
shutil.copytree(session_folder, backup_folder)


# Plotting
plotter.plot(data)
plotter.plot(data.average())
plt.show(block=True)
