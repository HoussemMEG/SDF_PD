import json
import os
import sys
import time
from typing import Union, List

import mne
import mne.preprocessing
import numpy as np
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
import termcolor

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
