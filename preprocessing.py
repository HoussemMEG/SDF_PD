# Import dependencies
import json
import os
from typing import Union

import mne
import mne.preprocessing
import numpy as np

import utils
from utils import enable_print, disable_print, time_mask

# Parameters
preprocessing_verbose: bool = True
is_my_case: bool = False


class PreProcessing:
    def __init__(self, path,
                 save_path, save=False, save_precision='double', save_tmin=None, save_tmax=None, overwrite=False,
                 shift=False, t_shift=None,
                 sampling_freq=None,
                 filter=None, filter_picks='eeg', n_jobs=None,
                 rejection_type=None, reject_t_min=None, reject_t_max=None, ignore_chs=None,
                 ref_channel=None,
                 crop=False, crop_t_min=None, crop_t_max=None, include_t_max=True,
                 baseline=None):
        """
        Class that perform the main PreProcessing steps in this order:
            1/ Time shifting
            2/ Resampling (up or down sampling the data)
            3/ Filtering (low-pass, high-pass, band-pass, notch-filter)
            4/ Eye blink rejection (check the method _eye_blink_rejection)
            5/ Reference changing
            6/ Cropping the data
            7/ Saving the data

        :param path: Data location.

        :param save_path: Folder path to where save the processed data or / and the processing informations.
        :param save: (bool) To save or no the Processed data.
        :param save_precision: Saving precision.
        :param save_tmin: Save the data starting from save_tmin. If None, the data are saved starting from tmin.
        :param save_tmax: Save the data up to save_tmax. If None, the data are saved up to tmax.
        :param overwrite: If True overwrite the current files, else the new processing are not saved.

        :param shift: (bool) If True time shift the data.
        :param t_shift: Value of time shift (can be positive or negative) in [s]. Ignored if shift=False.

        :param sampling_freq: If None keep the same sampling frequency, if value is given in [Hz] down-sample or
                                up-sample the data.

        :param filter: (f_min, f_max). Filter the data from f_min to f_max using a notch filter. If f_min is None
                        apply a low pass filter, if f_max is None apply a high pass filter, if both are None no
                        filtering is applied to the data.
        :param filter_picks: Signal type to be filtered ['EEG', 'EOG'].
        :param n_jobs: Number of processing units used, "-1" to use the maximum available CPU cores.

        :param rejection_type: Rejection method can be 'pk-pk' (peak to peak thresholding) or None for no Eye Blink (EB)
                                rejection.
        :param reject_t_min: t_min to start the thresholding in peak to peak rejection.
        :param reject_t_max: t_max to end the thresholding in peak to peak rejection.
        :param ignore_chs: Channels to ignore in EB rejection (thresholding will not concern this channel). However,
                            this does not prevent the signals of this channel from being removed during an epoch removal
                            due to the presence of EB.

        :param ref_channel: Change the current channel reference to ref_channel. If None no channel re-referencing is
                             applied. 'average' for an average reference.

        :param crop: To whether crop or no the data.
                      /!\\ cropping must be applied if is_my_case=True and the cropping should at least be 450 ms from
                            the right extremity (crop_t_max). This is mandatory when manipulating the data without
                            passing by Step_2 since there is a delay of 450 ms for the arrival of Target and Novel
                            stimulus.
        :param crop_t_min: Lower limit of the cropping in [s]
        :param crop_t_max: Upper limit of the cropping in [s]
        :param include_t_max: If true keep the element corresponding to crop_t_max in the data, else remove it.

        :param baseline: (base_t_min [s], base_t_max [s]) Set the signal between base_t_min and base_t_max to zero mean.
                          If base_t_min is None the baseline would be [t_min, base_t_max] and viceversa.
                          If base_t_min and base_t_max are None, the signal average is removed.
                          If baseline is None no operation is made.
        """

        # Saving the preprocessing parameters
        self._saved_args = locals()

        # Important parameters
        self._raw: Union[mne.BaseEpochs, mne.io.BaseRaw, None] = None  # MNE object
        self._subject_info = None

        # Raw parameters
        self._sfreq = None
        self._t_min = None
        self._t_max = None
        self._times = None
        self._nb_point = None

        # Rejection parameters
        self._rejection_type = rejection_type  # None 'pk-pk' 'ICA eog correlation' 'pro_ICA'
        self._reject_criteria = dict(eeg=100e-6) if self._rejection_type == 'pk-pk' else None  # unit: V eog=250e-6
        self._reject_t_min = reject_t_min
        self._reject_t_max = reject_t_max
        self._ignore_chs = ignore_chs  # name of the channel to be ignored
        self._flat = None  # Just in case for latter
        self._num_exceeded = 30  # /!\ this should be defined ?

        # Baseline correction parameters
        # None or (None, None) for entire time or tuple (a, b) a: start / b: end in seconds
        self._baseline = tuple(baseline)

        # Filtering parameters
        self._filter = tuple(filter)
        self._filter_picks = filter_picks
        self._n_jobs = 1 if n_jobs is None else n_jobs

        # Down-sampling or Up-sampling parameters
        self._sampling_freq = sampling_freq  # [Hz]

        # Cropping parameters
        self._crop = crop
        self._crop_t_min = crop_t_min
        self._crop_t_max = crop_t_max
        self._include_t_max = include_t_max

        # Re-referencing parameters
        # ref channels can be: 'average', 'REST'?, ['ch_nameS'], [] for no re-referencing
        self._ref_channel = ref_channel if ref_channel is not None else []
        self._ref_channel_type = 'eeg'
        # other parameters exist but we use them if needed

        # Time shifting parameters
        self._shift = shift
        self._t_shift = t_shift

        # Saving parameters
        self._do_save = save
        self._init_save(path=path, save_path=save_path)
        self._save_precision = save_precision
        self._overwrite = overwrite
        self._save_tmin = 0 if save_tmin is None else save_tmin
        self._save_tmax = save_tmax
        utils.save_args(self._saved_args, verbose=True,
                        path=os.path.join(self.save_path, 'info'), save_name='preprocessing_parameters')

    def process(self):
        """
        The main method to perform all the pre-processing,
        The order of the pre-processing depend on type of data and if the data is loaded or no
        Pre-processing performed :
            /!\\ if is_my_case it will also crop the data and center the stimuli arrival
            1/ Time shifting
            2/ Resampling (up or down sampling the data)
            3/ Filtering (low-pass, high-pass, band-pass, notch-filter)
            4/ Eye blink rejection (check the method _eye_blink_rejection)
            5/ Reference changing
            6/ Cropping the data
            7/ Saving the data

        :return: instance: mne.BaseEpochs, mne.io.BaseRaw
        """
        enable_print() if preprocessing_verbose else disable_print()
        print('\tPreprocessing:')
        if self._raw is None:
            raise (AttributeError('MNE object not found'))

        if isinstance(self._raw, mne.BaseEpochs):  # Epoched data

            # Time shifting
            if self._shift:
                if self._t_shift is None:
                    raise ValueError('Time shifting parameter not given')
                else:
                    print('\t\tThe epochs have been shifted by {:+} [s]'.format(self._t_shift))
                    self._raw = self._raw.shift_time(tshift=self._t_shift, relative=True)
                    self._t_min = self._raw.tmin
                    self._t_max = self._raw.tmax
                    self._times = self._raw.times

            # Resampling
            if self._sampling_freq is not None:
                if self._sampling_freq != self._raw.info['sfreq']:
                    if self._sampling_freq < self._raw.info['sfreq']:
                        print('\t\tDown sampling data to {:.2f} [Hz]'.format(float(self._sampling_freq)))
                    elif self._sampling_freq > self._raw.info['sfreq']:
                        print('\t\tUp sampling data to {:.2f} [Hz]'.format(float(self._sampling_freq)))

                    self._raw = self._raw.resample(sfreq=self._sampling_freq, npad='auto')

                    self._sfreq = self._sampling_freq
                    self._nb_point = len(self._raw.times)
                    self._times = self._raw.times
                else:
                    print('\t\tNo resampling performed')
            else:
                print('\t\tNo resampling performed')

            # Filtering
            if self._filter is not None:
                if self._filter[0] is not None or self._filter[1] is not None:
                    if preprocessing_verbose:
                        mne.set_log_level(verbose='INFO')
                    self._raw.filter(l_freq=self._filter[0], h_freq=self._filter[1], picks=self._filter_picks,
                                     filter_length='auto',
                                     l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=self._n_jobs,
                                     phase='zero', method='fir', fir_window='hamming')
                    if preprocessing_verbose:
                        mne.set_log_level(verbose='WARNING')
                else:
                    print('\t\tNo filtering applied')
            else:
                print('\t\tNo filtering applied')

            # Eye blink rejection
            self._reject_eye_blink(picks=['eeg'])
            # self._reject_eye_blink(picks=['eeg'], this_chans_only=[0, 27, 28, 29, 57, 58])

            # Change of reference
            if len(self._ref_channel) > 0:
                print('\t\tRe-referencing the data to %s reference' % self._ref_channel)
                self._raw = self._raw.set_eeg_reference(ref_channels=self._ref_channel, ch_type=self._ref_channel_type,
                                                        verbose=False)
            else:
                print('\t\t%s data marked as already having the desired reference' % self._ref_channel_type.upper())

            # For my case only: shift target and standard stim by 450 ms
            # Cropping and shifting for the non-shifted data else cropping only
            if is_my_case:  # /!\ for final submit this should be deleted
                if not self._crop:
                    raise NotImplementedError('Cropping have to be done in our data')

                # Value consistency verification
                crop_t_min = self._t_min if self._crop_t_min is None else self._crop_t_min
                if crop_t_min < self._t_min:
                    raise ValueError('crop_t_min is less than tmin')
                crop_t_max = self._t_max if self._crop_t_max is None else self._crop_t_max
                if crop_t_max > self._t_max:
                    raise ValueError('crop_t_max is greater than tmax')
                if crop_t_max < crop_t_min:
                    raise ValueError('crop_t_max is less than crop_t_min')
                if (self._t_max - crop_t_max) < 0.45:
                    raise ValueError('Crop at least 450ms at the right extremity')

                # Index values
                tmask = time_mask(self._times, crop_t_min, crop_t_max, sfreq=self._sfreq,
                                  include_tmax=self._include_t_max)
                tmask_shift = time_mask(self._times, crop_t_min + 0.45, crop_t_max + 0.45, sfreq=self._sfreq,
                                        include_tmax=self._include_t_max)
                data = self._raw.get_data()
                res_data = np.zeros([data.shape[0], data.shape[1], np.sum(tmask)])
                for i in range(data.shape[0]):
                    if self._raw.events[i, -1] == 1 or self._raw.events[i, -1] == 2:
                        res_data[i] = data[i][:, tmask_shift]
                    elif self._raw.events[i, -1] == 3:
                        res_data[i] = data[i][:, tmask]
                print('\t\tCropping epochs between ({:}, {:}) [s]'.format(self._crop_t_min, self._crop_t_max))

                # Setting the new parameters
                self._raw._set_times(self._times[tmask])
                self._raw._raw_times = self._raw._raw_times[tmask]
                self._times = self._raw.times
                self._t_min = self._raw.tmin
                self._t_max = self._raw.tmax
                self._nb_point = len(self._times)
                self._raw._data = res_data

            else:
                # Cropping
                if self._crop and (self._crop_t_min is not None or self._crop_t_max is not None):
                    self._raw = self._raw.crop(tmin=self._crop_t_min, tmax=self._crop_t_max,
                                               include_tmax=self._include_t_max)
                    self._t_min = self._raw.tmin
                    self._t_max = self._raw.tmax
                    self._times = self._raw.times
                    # /!\ modify time in events ?
                    print('\t\tCropping epochs between ({:}, {:}) [s]'.format(self._crop_t_min, self._crop_t_max))
                else:
                    print('\t\tNo cropping performed')

            # Base line correction
            if self._baseline[0] is not None or self._baseline[1] is not None:
                t_min = self._t_min if self._baseline[0] is None else self._baseline[0]
                t_max = self._t_max if self._baseline[1] is None else self._baseline[1]
                self._raw.apply_baseline(baseline=self._baseline)
                print('\t\tBaseline correction applied on (%.1f, %.1f) [s]' % (t_min, t_max))
            else:
                print('\t\tNo baseline correction applied')

            # Saving data
            return self.save(save_format='-epo.fif', save=self._do_save)

        elif isinstance(self._raw, mne.io.BaseRaw):  # Continuous data
            # Resampling
            if self._sampling_freq is not None:
                if self._sampling_freq != self._raw.info['sfreq']:
                    if self._sampling_freq < self._raw.info['sfreq']:
                        print('\t\tDown sampling data to {:.2f} [Hz]'.format(float(self._sampling_freq)))
                    elif self._sampling_freq > self._raw.info['sfreq']:
                        print('\t\tUp sampling data to {:.2f} [Hz]'.format(float(self._sampling_freq)))

                    events = None  # maybe useful latter
                    if events is not None:
                        self._raw, events = self._raw.resample(sfreq=self._sampling_freq, npad='auto', n_jobs=1,
                                                               events=events)
                    else:
                        self._raw = self._raw.resample(sfreq=self._sampling_freq, npad='auto', n_jobs=1)

                    self._sfreq = self._sampling_freq
                    self._nb_point = len(self._raw.times)
                    self._times = self._raw.times
                else:
                    print('\t\tNo resampling performed')
            else:
                print('\t\tNo resampling performed')

            # Filtering
            if self._filter is not None:
                if self._filter[0] is not None or self._filter[1] is not None:
                    self._raw.filter(l_freq=self._filter[0], h_freq=self._filter[1], picks=self._filter_picks,
                                     filter_length='auto',
                                     l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=self._n_jobs,
                                     phase='zero', method='fir', fir_window='hamming', verbose=preprocessing_verbose)
                else:
                    print('\t\tNo filtering applied')
            else:
                print('\t\tNo filtering applied')

            # Change of reference
            if len(self._ref_channel) > 0:
                print('\t\tRe-referencing the data to %s reference' % self._ref_channel)
                self._raw = self._raw.set_eeg_reference(ref_channels=self._ref_channel, ch_type=self._ref_channel_type,
                                                        verbose=False)
            else:
                print('\t\t%s data marked as already having the desired reference' % self._ref_channel_type.upper())

            # Cropping
            if self._crop and (self._crop_t_min is not None or self._crop_t_max is not None):
                self._raw = self._raw.crop(tmin=self._crop_t_min, tmax=self._crop_t_max,
                                           include_tmax=self._include_t_max)
                self._times = self._raw.times
                print('\t\tCropping raw data between ({:}, {:}) [s]'.format(self._crop_t_min, self._crop_t_max))
            else:
                print('\t\tNo cropping performed')

            # Saving data
            return self.save(save_format='-raw.fif', save=self._do_save)

    def _reject_eye_blink(self, picks=None, this_chans_only=None):
        """
        Method to reject eye blink (EB) from data,
        Rejection method can be 'pk-pk' (peak to peak thresholding) or None for no EB rejection,
        other EB rejection methods can be added (check ICA decomposition).
        :param picks: channels type to check for eye blink rejection
        :param this_chans_only: list of the only channels to perform on the EB rejection
        :return: instance: mne.BaseEpochs, mne.io.BaseRaw
        """
        if self._rejection_type is None:
            print('\t\tNo eye blink rejection')

        elif self._rejection_type == 'pk-pk':
            # Get indexes to keep
            reject_i_min = None
            reject_i_max = None
            if (self._reject_t_min is not None) and (self._reject_t_max is not None):
                if self._reject_t_min >= self._reject_t_max:
                    raise ValueError('reject_tmin needs to be < reject_tmax')

            if self._reject_t_min is not None:
                if self._reject_t_min < self._t_min:
                    raise ValueError("reject_tmin needs to be None or >= tmin")
                else:
                    idx = np.nonzero(self._times >= self._reject_t_min)[0]
                    reject_i_min = idx[0]

            if self._reject_t_max is not None:
                if self._reject_t_max > self._t_max:
                    raise ValueError("reject_tmax needs to be None or <= tmax")
                else:
                    idx = np.nonzero(self._times <= self._reject_t_max)[0]
                    reject_i_max = idx[-1]

            reject_time = slice(reject_i_min, reject_i_max)
            data = self._raw.get_data()[:, :, reject_time]

            channel_type_idx = mne.io.pick.channel_indices_by_type(self._raw.info, picks=picks)
            bad_tuple = tuple()
            checkable = np.ones(len(self._raw.info['ch_names']), dtype=bool)
            if self._ignore_chs is not None:
                checkable[np.array([c in self._ignore_chs for c in self._raw.info['ch_names']], dtype=bool)] = False

            for reject_dic, fnct, reason in zip([self._reject_criteria, self._flat], [np.greater, np.less],
                                                ['pk-pk', 'flat']):
                if reject_dic is not None:
                    for key, thresh in reject_dic.items():
                        if this_chans_only is not None:
                            chs_idx = this_chans_only
                        else:
                            chs_idx = channel_type_idx[key]
                        name = key.upper()
                        if len(chs_idx) > 0:
                            data_idx = data[:, chs_idx, :]
                            deltas = np.max(data_idx, axis=2) - np.min(data_idx, axis=2)
                            checkable_idx = np.tile(checkable[chs_idx], [data.shape[0], 1])
                            idx_deltas = np.logical_and(fnct(deltas, thresh), checkable_idx)
                            ###########################################################################################
                            # how to deal with them ? should a number of channels
                            # has to be greater than the thresh or how ?
                            ###########################################################################################
                            if is_my_case:
                                to_reject = np.where(np.sum(idx_deltas, axis=1) > 0)[0]
                            else:
                                to_reject = np.where(np.sum(idx_deltas, axis=1) > self._num_exceeded)[0]
                            if len(to_reject) > 0:
                                if reason == 'pk-pk':
                                    print('\t\tRejecting %s epochs based on pk-pk rejection threshold exceeded. '
                                          'Type : %s' % (len(to_reject), name))
                                elif reason == 'flat':
                                    pass  # this should be changed if we reject based on flatness
                                bad_tuple += tuple(to_reject)
                                self._raw = self._raw.drop(indices=to_reject, reason='pk-pk rejection')
                            else:
                                print('\t\tNo bad epochs found')
                else:
                    if reason == 'pk-pk':
                        print('\t\tNo rejection dictionary found')
                    elif reason == 'flat':
                        pass  # this should be changed if we reject based on flatness
        else:
            pass  # To implement

    def set_data(self, raw, epoch_drop_idx=None, epoch_drop_reason='USER', channel_drop=None):
        """
        Setting the data in the class, extracting data information
        :param raw: mne instance, can be : mne.BaseEpochs, mne.io.BaseRaw
        :param epoch_drop_idx: list of indexes containing epochs to be dropper, None to not drop any epoch
        :param epoch_drop_reason: str explaining the dropping reason (usually "USER" choice)
        :param channel_drop: list of channels names to be dropped, None to not drop any channel
        :return: /
        """
        # Setting patient data
        self._raw = raw
        try:
            self._subject_info = json.loads(self._raw.info['description'])['subject_info']
        except KeyError:
            pass

        # Channels dropping
        if channel_drop is not None:
            if len(channel_drop) > 0:
                print('\t\tDropping %s channels' % channel_drop)
                self._raw = self._raw.drop_channels(ch_names=channel_drop)

        if isinstance(self._raw, mne.BaseEpochs):
            # Epoch dropping (manually)
            if epoch_drop_idx is not None:
                self._raw = self._raw.drop(indices=epoch_drop_idx, reason=epoch_drop_reason)
            # Init parameters
            self._sfreq = self._raw.info['sfreq']
            self._t_min = self._raw.tmin
            self._t_max = self._raw.tmax
            self._times = self._raw.times
            self._nb_point = len(self._times)

        elif isinstance(self._raw, mne.io.BaseRaw):
            # Init parameters
            self._sfreq = self._raw.info['sfreq']
            self._times = self._raw.times
            self._nb_point = len(self._times)

        # Load data
        self._raw = self._raw.load_data()

    def _init_save(self, path, save_path):
        """
        Verify if the saving path and folder are correct
        If the saving folder is not found, creates it
        :param path: from where the data are taken
        :param save_path: the name of the saving folder
        :return: (no return, modify self._save_path)
        """
        if path.split('/')[-2] == 'processed':
            if path.split('/')[-1] == save_path:
                self.save_path = path
            else:
                self.save_path = os.path.join(path, '..', save_path)
        else:
            self.save_path = os.path.join(path, 'processed', save_path)

        if not os.path.exists(self.save_path):  # create save_dir and save_dir/info if it doesn't exist
            os.makedirs(self.save_path)
            print('Saving folder created')
        if not os.path.exists(os.path.join(self.save_path, 'info')):
            os.makedirs(os.path.join(self.save_path, 'info'))
        if self._do_save:  # show saving folder
            print(' Saving folder:', self.save_path)

    def save(self, save_format, data=None, save=True) -> Union[mne.BaseEpochs, mne.io.BaseRaw]:
        """
        Saving method
        :param save_format: should be "-epo.fif" for epoched data or "-raw.fif" for continous data.
        :param data: the data to be saved.
        :param save: I don't remember why I did it.
        :return: /
        """
        if save and self._raw is not None:  # inner data saving (after pre-processing)
            self._save(raw=self._raw, save_format=save_format)
            print('\tData saved')
            return self._reset()
        if self._raw is None and data is not None:  # outer data saving (after channel extraction)
            self._save(raw=data, save_format=save_format)
            print('\tData saved')
        return self._reset()

    def _save(self, raw, save_format):
        description = raw.info['description']
        saving_name = raw.info['description']['file_name']
        save_to = os.path.join(self.save_path, saving_name + save_format)
        raw.info['description'] = json.dumps(description)

        if isinstance(raw, mne.BaseEpochs):
            raw.save(fname=save_to, fmt=self._save_precision, overwrite=self._overwrite)
        elif isinstance(raw, mne.io.RawArray):
            raw.save(fname=save_to, picks=None, tmin=self._save_tmin, tmax=self._save_tmax, proj=False,
                     fmt=self._save_precision, overwrite=self._overwrite)
        raw.info['description'] = description

    def _reset(self):
        """
        Resetting the class parameters to be used by the next data
        :return: instance: mne.BaseEpochs, mne.io.BaseRaw
        """
        enable_print()
        to_return = self._raw
        # Reset parameters
        self._raw = None
        self._subject_info = None

        self._sfreq = None
        self._t_min = None
        self._t_max = None
        self._times = None
        self._nb_point = None
        return to_return
