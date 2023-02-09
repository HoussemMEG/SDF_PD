import json
import os
import sys
import time
from typing import Union, List

import mne
import mne.preprocessing
import numpy as np
import pandas as pd
import termcolor
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from plots import Plotter

stdout = sys.stdout


# Disable print
def disable_print():
    sys.stdout = open(os.devnull, 'w')


# Restore print
def enable_print():
    sys.stdout = stdout


# Timing decorator, use @execution_time before a function to time it
def execution_time(function):
    def my_function(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        if time.time() - start_time > 1e-2:
            text = "Execution time of " + function.__name__ + ": {0:.3f} (s)".format(time.time() - start_time)
        else:
            text = "Execution time of " + function.__name__ + ": {0:.3f} (ms)".format(1000 * (time.time() - start_time))
        print_c(text, 'blue')
        return result

    return my_function


# Safely find sample boundaries
def time_mask(times, tmin=None, tmax=None, sfreq=None, raise_error=True, include_tmax=True):
    orig_tmin = tmin
    orig_tmax = tmax
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    if not np.isfinite(tmin):
        tmin = times[0]
    if not np.isfinite(tmax):
        tmax = times[-1]
        include_tmax = True  # ignore this param when tmax is infinite
    if sfreq is not None:
        # Push to a bit past the nearest sample boundary first
        sfreq = float(sfreq)
        tmin = int(round(tmin * sfreq)) / sfreq - 0.5 / sfreq
        tmax = int(round(tmax * sfreq)) / sfreq
        tmax += (0.5 if include_tmax else -0.5) / sfreq
    else:
        assert include_tmax  # can only be used when sfreq is known
    if raise_error and tmin > tmax:
        raise ValueError('tmin (%s) must be less than or equal to tmax (%s)' % (orig_tmin, orig_tmax))
    mask = (times >= tmin)
    mask &= (times <= tmax)
    if raise_error and not mask.any():
        extra = '' if include_tmax else 'when include_tmax=False '
        raise ValueError('No samples remain when using tmin=%s and tmax=%s %s'
                         '(original time bounds are [%s, %s])' % (orig_tmin, orig_tmax, extra, times[0], times[-1]))
    return mask


# Sort an array and return the initial indexes
def index_sort(array):
    array = np.asarray(array)
    index = np.unravel_index(np.argsort(array, axis=None), array.shape)
    sorted_array = array[index]
    return index, sorted_array


# Compress the features to take less space (since they are really sparse)
def compress(features: List[np.ndarray]) -> dict:
    idx = []  # shape: (n_target, (n_non_zero, n_non_zero))
    val = []  # shape: (n_target, n_non_zero)
    n_path = []  # shape: (n_target)
    for target_idx in range(len(features)):
        idx_i, idx_j = np.nonzero(features[target_idx])
        idx.append([idx_i.tolist(), idx_j.tolist()])
        val.append(features[target_idx][np.nonzero(features[target_idx])].tolist())
        n_path.append(features[target_idx].shape[1])
    compressed_features = {'idx': idx, 'val': val, 'n_path': n_path}
    return compressed_features


# Decompress the features from the compressed format
def decompress(compressed_features: dict, n_features) -> List[np.ndarray]:
    decompressed_features = []
    n_target = len(compressed_features['idx'])
    for target_idx in range(n_target):
        idx = compressed_features['idx'][target_idx]  # shape: (n_target, (n_non_zero, n_non_zero))
        val = compressed_features['val'][target_idx]  # shape: (n_target, n_non_zero)
        n_path = compressed_features['n_path'][target_idx]  # shape: (n_target)
        features = np.zeros((n_features, n_path))  # shape: (n_features, n_path)
        features[tuple(idx)] = val
        decompressed_features.append(features)
    return decompressed_features


# Save JSON file safely
def save_args(args: dict, path, save_name, verbose=False):
    """
    :param args: the dictionary to save
    :param path: path where to save the JSON file
    :param save_name: name of the JSON file
    :param verbose: if False, no printing else print that the file is saved in which location
    :return:
    """
    if not isinstance(args, dict):
        raise TypeError('Parameters to be saved in {:} as {:}.json must be dict type'.format(path, save_name))

    save_path = os.path.join(path, save_name + '.json')
    # Some parsing
    if 'self' in args:
        del args['self']
    # If path doesn't exist create it
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(save_path):  # if file exists, update it
        with open(save_path, 'r') as file:
            data: dict = json.load(file)
        if not args.items() <= data.items():
            data.update(args)
            args = data
            with open(save_path, 'w') as fp:
                json.dump(args, fp, indent=4)
                if verbose:
                    print_c('JSON saved: <{: <23} \t at: {:}>'.format(save_name, path), bold=True)
    else:  # if file does not exists, create a new file
        with open(save_path, 'w') as fp:
            json.dump(args, fp, indent=4)
            if verbose:
                print_c('JSON saved: <{: <23} \t at: {:}>'.format(save_name, path), bold=True)


# Transform from channels domain to ICA or PCA domain
def virtual_ch_extractor(data, n_components,
                         method=None, whiten: bool = False,
                         plot_data: bool = False, plot_evoked: bool = True, show: bool = True, save_fig: bool = False,
                         verbose: bool = False) -> Union[mne.BaseEpochs, mne.io.BaseRaw, any]:
    """
    :param data: mne data instance to transform
    :param n_components: number of desired component to keep, if (int) keep n_components, if 0 < (float) < 1 keep
                          x components to have n_components of pvaf (percentage of variance accounted for). See
                          sk-learn PCA documentation
    :param method: None will not apply any projection to the data, 'PCA' will project the data into the principal
                    components. 'ICA', 'FASTICA', 'INFOMAX' and 'PICARD' will project the data into the independent
                    component domain, the only difference is in the computation algorithm.
    :param whiten: When True (False by default) the components_ vectors are multiplied by the square root of n_samples
                    and then divided by the singular values to ensure uncorrelated outputs with unit component-wise
                    variances.
                   Whitening will remove some information from the transformed signal (the relative variance scales of
                    the components) but can sometime improve the predictive accuracy of the downstream estimators by
                    making their data respect some hard-wired assumptions.
    :param plot_data: if True plot the projected data (in the new domain) in epoch or continuous form.
    :param plot_evoked: if True plot the projected data (in the new domain) in evoked form.
    :param show: True in order to show the plotted figures. Ignored if plot_data=False and plot_evoked=False
    :param save_fig: True in order to save the figures. Ignored if plot_data=False and plot_evoked=False
    :param verbose: manage the verbosity of the 'ICA' projection methods. Ignored for 'PCA' transform
    :return: Projected mne instance (epochs or continuous). Channel names are changed to "ICx y %" and "PCx y %"
              x is the component number and y is the pvaf (percentage of variance accounted for)
    """
    if method is None:
        return data, data.get_data(), None

    if data.info['description']['decomposition'] is not None:
        raise ValueError('Data already have a decomposition')

    plotter = Plotter()
    method = str(method).upper()
    picks = ['eeg']

    print("\tVirtual channel extraction:")
    print("\t\tMethod: {:}".format(method))
    print("\t\tNumber of component: {:}".format(n_components))
    print("\t\tWhitening: {:}".format(whiten))

    # Get only data we are interested in
    picks_idx = mne.io.pick.channel_indices_by_type(info=data.info, picks=picks)
    pick_idx = []
    for key, value in picks_idx.items():
        pick_idx += value

    x = data.get_data()
    if isinstance(data, mne.BaseEpochs):
        x = x[:, pick_idx, :]
        if x.shape[0] < 5:  # case where all windows get EB rejected
            print(data.info['description']['file_name'])
            data.info['description']['file_name'] = 'BAD-' + data.info['description']['file_name']
            return data, data.get_data(), None

    elif isinstance(data, mne.io.RawArray):
        x = x[pick_idx, :]

    if method == 'PCA':
        if isinstance(data, mne.BaseEpochs):  # evoked only for epoched data
            pca = PCA(n_components=n_components, whiten=whiten,
                      svd_solver='auto')  # svd_solver ='auto' 'full' 'arpack' 'randomized'
            pca_data: np.ndarray = UnsupervisedSpatialFilter(pca, average=False).fit_transform(x)
            pca_data = pca_data / np.std(pca_data[:, 0, :]) * 10e-6
            pvaf = pca.explained_variance_ratio_

            # Info creation
            ch_names = ['PC {:}|{:4.2f}%'.format(i + 1, value * 100) for i, value in enumerate(pvaf)]
            pca_info = mne.create_info(ch_names=ch_names, sfreq=data.info['sfreq'], ch_types='eeg')
            pca_info['description'] = data.copy().info['description']
            pca_info['description']['decomposition'] = 'PCA'

            # PCA epochs
            pca_epo = mne.EpochsArray(data=pca_data, info=pca_info,
                                      events=data.events, tmin=data.tmin, event_id=data.event_id,
                                      verbose=False, on_missing='warn')
            if plot_data:
                plotter.plot(pca_epo, **{'plot_data': True, 'show': show, 'save_fig': save_fig})
            if plot_evoked:
                plotter.plot(pca_epo.average(), **{'plot_data': True, 'show': show, 'save_fig': save_fig})
            return pca_epo, pca_data, pca

        elif isinstance(data, mne.io.BaseRaw):
            pca = PCA(n_components=n_components, whiten=whiten,
                      svd_solver='auto')
            pca_data = pca.fit_transform(x.T)
            pca_data = pca_data.T
            pvaf = pca.explained_variance_ratio_

            # Info creation
            ch_names = ['PC {:}|{:4.2f}%'.format(i + 1, value * 100) for i, value in enumerate(pvaf)]
            pca_info = mne.create_info(ch_names=ch_names, sfreq=data.info['sfreq'], ch_types='eeg')
            pca_info['description'] = data.copy().info['description']
            pca_info['description']['decomposition'] = 'PCA'

            pca_raw = mne.io.RawArray(data=pca_data, info=pca_info, first_samp=0, verbose=False)
            events = mne.find_events(data, stim_channel='STIM', output='onset', consecutive='increasing',
                                     min_duration=0, shortest_event=2, mask=None, uint_cast=False, mask_type='and',
                                     initial_event=False, verbose=False)

            # Adding 'STIM' channel
            stim_data = np.zeros((1, len(data.times)))
            stim_info = mne.create_info(ch_names=['STIM'], sfreq=pca_raw.info['sfreq'], ch_types=['stim'])
            stim_raw = mne.io.RawArray(stim_data, stim_info, verbose=False)
            pca_raw = pca_raw.add_channels([stim_raw], force_update_info=True)
            pca_raw.add_events(events, stim_channel=['STIM'], replace=True)

            if plot_data:
                plotter.plot(pca_raw, **{'plot_data': True, 'show': show, 'save_fig': save_fig})
            return pca_raw, pca_data, pca

    elif method == 'ICA' or method == 'FASTICA' or method == 'INFOMAX' or method == 'PICARD':
        max_iter = 200
        method = 'fastica' if method == 'ICA' else method.lower()  # method = 'fastica', 'infomax', 'picard'

        ica = ICA(n_components=n_components, noise_cov=None, method=method, max_iter=max_iter, verbose=verbose)
        # random_state=97 for same seed and get always the same results
        ica.fit(data, picks=picks)
        if ica.n_iter_ == max_iter:
            print('/!\\ ICA may not converged')
        # ica to exclude do it better latter (by subject and fixing the seed)
        ica.exclude = []
        pvaf = mne.preprocessing.ica._ica_explained_variance(ica, data, normalize=True)

        # since apply change data in place, we will use a copy of it
        data_copy = data.copy()
        ica.apply(data_copy, verbose=verbose)
        ica_data = ica.get_sources(inst=data).get_data()
        ica_data = ica_data / np.std(ica_data[:, 0, :]) * 10e-6  # scale the data according to the first component only

        if isinstance(data, mne.BaseEpochs):
            # Info creation
            ch_names = ['IC {:}|{:4.2f}%'.format(i + 1, value * 100) for i, value in enumerate(pvaf)]
            ica_info = mne.create_info(ch_names=ch_names, sfreq=data.info['sfreq'], ch_types='eeg')
            ica_info['description'] = data.copy().info['description']
            ica_info['description']['decomposition'] = 'ICA'

            ica_epo = mne.EpochsArray(data=ica_data, info=ica_info,
                                      events=data.events, tmin=data.tmin, event_id=data.event_id,
                                      verbose=False, on_missing='warn')

            if plot_data:
                plotter.plot(ica_epo, **{'plot_data': True, 'show': show, 'save_fig': save_fig})
            if plot_evoked:
                plotter.plot(ica_epo.average(), **{'plot_data': True, 'show': show, 'save_fig': save_fig})

            return ica_epo, ica_data, ica

        elif isinstance(data, mne.io.RawArray):
            # Info creation
            ch_names = ['IC {:}'.format(i + 1) for i in range(ica_data.shape[0])]
            ica_info = mne.create_info(ch_names=ch_names, sfreq=data.info['sfreq'], ch_types='eeg')
            ica_info['description'] = data.copy().info['description']
            ica_info['description']['decomposition'] = 'ICA'

            ica_raw = mne.io.RawArray(data=ica_data, info=ica_info, first_samp=0, verbose=False)
            events = mne.find_events(data, stim_channel='STIM', output='onset', consecutive='increasing',
                                     min_duration=0, shortest_event=2, mask=None, uint_cast=False, mask_type='and',
                                     initial_event=False, verbose=False)

            # Adding 'STIM' channel
            stim_data = np.zeros((1, len(data.times)))
            stim_info = mne.create_info(ch_names=['STIM'], sfreq=ica_raw.info['sfreq'], ch_types=['stim'])
            stim_raw = mne.io.RawArray(stim_data, stim_info, verbose=False)
            ica_raw = ica_raw.add_channels([stim_raw], force_update_info=True)
            ica_raw.add_events(events, stim_channel=['STIM'], replace=True)

            if plot_data:
                plotter.plot(ica_raw, **{'plot_data': True, 'show': show, 'save_fig': save_fig})

            return ica_raw, ica_data, ica
    else:
        raise NotImplementedError("The supported methods are : 'PCA', 'ICA', 'fastica', 'infomax' and 'picard'"
                                  " but got instead {:}".format(method))


# Print in color and bold
def print_c(text, color=None, highlight=None, bold=False):
    if bold:
        if color is None:
            print(termcolor.colored(text, attrs=['bold']))
        else:
            print(termcolor.colored(text, color, attrs=['bold']))
    else:
        if highlight:
            print(termcolor.colored(text + '\033[1m{:}\033[0m'.format(highlight), color))
        else:
            print(termcolor.colored(text, color))


# Transform safely ndarray to list for JSON saving
def ndarray_to_list(arr: List[np.ndarray]):
    for target_idx in range(len(arr)):
        if arr[target_idx] is not None:
            arr[target_idx] = arr[target_idx].tolist()
    return arr


# Apply the "function" to "x" in order to extract features
def features(x: np.ndarray, function, **kwargs):
    res = np.zeros(x.shape[0])
    for i in range(len(x)):
        res[i] = function(x[i, :], **kwargs)
    return np.array(res)


def mean(lst):
    return sum(lst) / len(lst)


def select_data(grouped, stim, feature, pars, channel):
    pars = float("{:.2f}".format(pars))
    if len(channel) > 1:
        if len(feature) > 1:
            pass
            data = np.hstack(
                [grouped.get_group((stim, f, pars))[channel].to_numpy() for f in feature])
        else:  # k_feat == 1
            data = grouped.get_group((stim, feature[0], pars))[channel].to_numpy()

    else:  # k_chan == 1
        if len(feature) > 1:
            data = np.array(
                [grouped.get_group((stim, f, pars))[channel[0]].to_numpy() for f in feature]).T
        else:  # k_feat == 1
            data = grouped.get_group((stim, feature[0], pars))[channel[0]].to_numpy()[:, np.newaxis]
    return data


def reset_outer_eval(n_subj, parsimony):
    return {'train': np.zeros((n_subj, len(parsimony))),
            'validation': np.zeros((n_subj, len(parsimony))),
            'test': [],
            'test_category': np.zeros((n_subj,)),
            'learning': [],
            'learning_category': [],
            'selected validation': []}


def read_parameters(session, param_files):
    # Reading the JSON files
    parameters = {}
    for file in os.listdir(session):
        if file in param_files:
            parameters_path = os.path.join(session, file)
            with open(parameters_path) as f:
                parameters.update(json.load(f))

    # Parameters reading
    model_freq = np.array(parameters['model_freq'])
    n_freq = parameters['n_freq']
    n_point = parameters['n_point']
    n_features = parameters['n_features']
    pars = parameters['selection'] if parameters['selection'] is not None else parameters.get('selection_alpha', None)
    data_case = parameters.get('data_case', 'evoked')
    alpha = parameters['alpha']
    version = parameters.get('version', 0)

    # Channel reading
    channels = parameters['channel_picks']  # Used channels

    # Printing
    print_c('\nSessions: {:}'.format(session.split('\\')[-1]), 'blue', bold=True)
    print_c(' Data case: ', highlight=data_case)
    print_c(' Version: ', highlight=str(version))
    print_c(' Alpha: ', highlight=alpha)
    print_c(' Channels: ', highlight=channels)
    print(' Model frequencies: {:}'.format(model_freq))
    print_c(' N_freq = ', highlight=n_freq)
    print_c(' N_point = ', highlight=n_point)
    print_c(' Beta_dim = ', highlight=n_features)
    print(' Parsimony: {:}\n'.format(np.array(pars)))
    return parameters


def feature_extraction(x):
    dic = {'my_argmin': np.argmin(x, axis=1) // 40 * 2,
           'my_mean': np.sum(x[:, 40 * 90:40 * 249], axis=1),
           'energy': np.sum(x ** 2, axis=1),
           'count_non_zero': np.count_nonzero(x, axis=1),
           # 'mean': np.mean(x, axis=1),
           'max': np.max(x, axis=1),
           'min': np.min(x, axis=1),
           'pk-pk': np.max(x, axis=1) - np.min(x, axis=1),
           # 'argmin': np.argmin(x, axis=1),
           'argmax': np.argmax(x, axis=1),
           'argmax-argmin': np.argmax(x, axis=1) - np.argmin(x, axis=1),
           'sum abs': np.sum(np.abs(x), axis=1),
           'var': np.var(x, axis=1),
           'std': np.std(x, axis=1),
           'kurtosis': kurtosis(x, axis=1),
           'skew': skew(x, axis=1),
           'count above mean': np.array([np.count_nonzero(row[np.where(row >= np.mean(np.abs(row)))]) for row in x]),
           'count below mean': np.array([np.count_nonzero(row[np.where(row <= np.mean(np.abs(row)))]) for row in x]),
           'max abs': np.max(np.abs(x), axis=1),
           'argmax abs': np.argmax(np.abs(x), axis=1),
           }

    for key, value in dic.items():
        if value.ndim > 1:
            raise ValueError("Feature {:} not extracted properly, has the dimension {:}".format(key, value.shape))
        if value.shape != (x.shape[0],):
            raise ValueError("Feature not corresponding to the right dimensions")
    return dic


@execution_time
def read_data_(session, use_x0, param):
    for file in os.listdir(session):
        if file != "generated_features.json":  # read only json file containing features not parameters
            continue

        file_path = os.path.join(session, file)
        with open(file_path) as f:
            data: dict = json.load(f)

        dic = []
        for stim in data.keys():
            for subj, subj_data in tqdm(data[stim].items(), position=0, leave=True):
                if use_x0:
                    VMS = np.array(subj_data['x0'])  # pre np array shape: List(n_channels)(n_features, n_path)
                else:
                    # subj_feat pre np array shape: List(n_channels)(n_features, n_path)
                    VMS = decompress(subj_data['features'], n_features=param['n_features'])

                # stim: stim
                # subject ID: subj
                category = subj_data['subject_info']['subject_info']['category']
                category = 0 if category == 'PD' else 1
                channels = param['channel_picks']
                if 'VEOG' in channels:
                    channels.remove('VEOG')
                parsimony = np.array(param['selection'] if param['selection'] is not None else param.get('selection_alpha', None))

                n_freq = param['n_freq']
                n_point = param['n_point']
                n_features = param['n_features']

                # VMS[:, :, pars_idx] shape: (n_channel, n_features, n_path) per subject
                for i, ch_val in enumerate(VMS):
                    if ch_val.shape[-1] != len(parsimony):
                        repeat = len(parsimony) - ch_val.shape[-1]
                        for _ in range(repeat):
                            ch_val = np.append(ch_val, ch_val[:, [-1]], axis=1)
                        VMS[i] = ch_val
                VMS = np.array(VMS)

                for pars_idx, pars in enumerate(parsimony):
                    features_dict = feature_extraction(VMS[:, :, pars_idx])
                    for feature_name, features in features_dict.items():
                        subj_dict = {'stim': str(stim),
                                     'subject': str(subj),
                                     'category': category,
                                     'parsimony': float("{:.2f}".format(pars)),
                                     'feature': feature_name}
                        for ch_idx, channel in enumerate(channels):
                            subj_dict.update({channel: features[ch_idx]})
                        dic.append(subj_dict)

        # columns = ['stim', 'subject', 'category', 'pasimony', *param['channel_picks']]
        data_df = pd.DataFrame(data=dic, index=None)
        return data_df


def smoothing_scores(score, smoothing=True):
    if not smoothing:
        return score
    temp = [(score[i - 1] + 4 * score[i] + score[i + 1]) / 6 for i in range(1, len(score) - 1)]
    return np.array([(score[0] + score[1]) / 2, *temp, (score[-2] + score[-1]) / 2])


def update_table(table, parsimony, outer_memory, highlight_above):
    for pars_idx, pars in enumerate(parsimony):
        train_acc_pars = outer_memory['train'][:, pars_idx].mean()
        val_acc_pars = outer_memory['validation'][:, pars_idx].mean()
        if outer_memory['train'][:, pars_idx].mean() >= highlight_above:
            table[0].append('\033[92m{:}\033[0m'.format(int(pars * 100)))
            table[1].append('\033[92m{:.1f}\033[0m'.format(train_acc_pars))
            table[2].append('\033[92m{:.1f}\033[0m'.format(val_acc_pars))
            table[-1].append('\033[92m OK\033[0m')
        else:
            table[0].append('{:}'.format(int(pars * 100)))
            table[1].append('{:.1f}'.format(train_acc_pars))
            table[2].append('{:.1f}'.format(val_acc_pars))
            table[-1].append('-')


def prints(i, j, k, stim, select_stim, channel, select_channel, feature, select_feature, table, outer_mem, timed, highlight_above):
    print_c('Stimuli: {:}     <{:}/{:}>'.format(stim, i + 1, len(select_stim)), 'yellow', bold=True)
    print_c('\tChannel: {:}   <{:}/{:}>'.format(" / ".join(list(channel)), j + 1, len(select_channel)), 'magenta', bold=True)
    print_c('\t\tFeature: {:<20}     <{:}/{:}>\t\t{:.1f}s/it'.format(" / ".join(list(feature)), k + 1, len(select_feature), timed), 'blue', bold=True)
    print(tabulate(table, headers='firstrow', tablefmt="rounded_outline"))

    learning = "\t\t Learning: {:.1f} %".format(mean(outer_mem['learning']))
    if mean(outer_mem['learning']) > highlight_above:
        learning = "\t\t\033[92m Learning: {:.1f} %\033[0m".format(mean(outer_mem['learning']))

    validation = "\t\t Validation: {:.1f} %".format(mean(outer_mem['selected validation']))
    if mean(outer_mem['selected validation']) > highlight_above:
        validation = "\t\t\033[92m Validation: {:.1f} %\033[0m".format(mean(outer_mem['selected validation']))

    testing = "\t\t Test: {:.1f} %\033[0m".format(mean(outer_mem['test']))
    if mean(outer_mem['test']) > highlight_above:
        testing = "\t\t\033[92m Test: {:.1f} %\033[0m".format(mean(outer_mem['test']))

    print('\t\tAccuracy: {:.1f} % ± {:.2f} %'.format(outer_mem['train'].mean(), outer_mem['train'].std()),
          learning, validation, testing, '\n')


def update_experience_values(experience_values, outer_memory, channel, stim):
    selection_validation = mean(outer_memory['selected validation'])
    if selection_validation > min(experience_values['validation_acc'][:]):
        i_min = np.argmin(experience_values['validation_acc'])
        experience_values['validation_acc'][i_min] = selection_validation
        experience_values['learning_category'][i_min] = np.array(outer_memory['learning_category'])
        experience_values['predicted_category'][i_min] = outer_memory['test_category']
        if selection_validation >= max(experience_values['validation_acc']):
            experience_values['channel'] = channel
            experience_values['stim'] = stim
            experience_values['test'] = mean(outer_memory['test'])


def read_data(session, path_session, read_only, param, never_use):
    file = os.path.join('extracted features', session + '.csv')
    if read_only:
        if os.path.exists(file):
            data_df = pd.read_csv(file, index_col=[0])
        else:
            print_c('File not found, read_only has been set to <False>\n', 'red', bold=True)
            read_only = False
    if not read_only:
        data_df = read_data_(path_session, use_x0=False, param=param)
        data_df.to_csv(file)
        print_c('File saved at {:}'.format(file), bold=True)
    data_df.replace(np.nan, 0, inplace=True)
    data_df.drop(data_df[data_df['subject'].isin(never_use)].index, inplace=True)  # remove outlier / test subjects
    data_df.reset_index(inplace=True)
    return data_df


def CV_choice(split, shuffle=False):
    if split == 'LOO':
        return LeaveOneOut()
    return StratifiedKFold(n_splits=split, shuffle=shuffle)


def list_not_contain_empty(the_list):
    return any(map(list_not_contain_empty, the_list)) if isinstance(the_list, list) else False
