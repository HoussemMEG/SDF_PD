import json
import os
from typing import List, Optional
import mne
import numpy as np
from scipy.io import loadmat

import preprocessing
from utils import print_c

# Parameters
reader_verbose: bool = False


class Reader:
    def __init__(self, path, info_path, subj_info_keys=None, use_eog=False):
        """
        Main class for data reading
        :param path: Data location.
        :param info_path: subject_info.json file location.
        :param subj_info_keys: Key information to use and extract from subject_info.json
        :param use_eog: If True the 'EOG' data if present in the reading file are read, else they will be ignored.
        """
        # Paths initialization
        self._data_path = os.path.join(path)
        self._info_path = os.path.join(info_path)
        self.files: Optional[List[str]] = os.listdir(self._data_path)

        # Subject information
        self._subj_info = []
        self._subj_info_keys = [] if subj_info_keys is None else subj_info_keys

        # Parameters
        self.use_eog: bool = use_eog
        self._CAL = 1e-6  # calibration parameter (MNE data should be in Volts)

        self._init_and_check()

    def read_data(self, file=None):
        """
        Main reading method, that can read either only one file or serve as a generator to read multiple files.
        :param file: If file name is given, the class will only read this file, else read all the file present in the
                      folder and yield them (generator).
        :return: Read data.
        """
        if file:
            # Just in case where we specify only the number and session in single file reading
            if len(file.split('.')) == 1:  # if file have an extension, specify the right name
                file = [_file for _file in self.files if file.split('_')[0:2] == _file.split('_')[0:2]][0]
            data = self._read(file)
            return data
        else:
            def generator():
                for i, file in enumerate(self.files):
                    data = self._read(file, i)
                    yield data
            return generator()

    def _read(self, file, i=None):
        subj_info = None
        # Reading files
        text = '\nReading file: {:}'.format(file) if i is None else \
               '\nReading file: {:}  {:}/{:}'.format(file, i+1, len(self.files))
        print_c(text, bold=True)

        if self._data_extension == 'mat':
            raw_data = loadmat(os.path.join(self._data_path, file))
        elif self._data_extension == 'fif':
            if self._data_type == 'epo':
                raw_data = mne.read_epochs(os.path.join(self._data_path, file), preload=True, verbose=reader_verbose)
            elif self._data_type == 'raw':
                raw_data = mne.io.read_raw_fif(os.path.join(self._data_path, file), preload=True,
                                               verbose=reader_verbose)
        else:  # other reading procedures to implement
            raw_data = None
            pass

        # Importing the subject information
        if self._subj_info and self._subj_info_keys:
            subj_info = {'id': str(file.split('_')[0])}
            for key in self._subj_info_keys:
                subj_info[key] = self._subj_info[str(file.split('.')[0])][key]

        # Parsing
        raw_data = self._parser(raw=raw_data, file=file, subj_info=subj_info)
        return raw_data

    def _parser(self, raw, file, subj_info=None):
        """
        Method for parsing the data and put it in the right format for MNE.
        Parsing is skipped for ".fif" files as they have already been parsed.
        All the data information are kept as a dictionary inside the mne info at "info['description']".
         The information are in the following form: info['description'][subj_ID] = subj_info.
        :param raw: If a ".fif" file is given this step is skipped.
        :param file: File name just to be kept in the data information.
        :param subj_info: Subject information to add to the data.
        :return: Parsed data.
        """
        if self._data_extension == 'fif':  # no parsing for processed data
            return raw

        if self._data_extension == 'mat':
            eeg = raw['EEG'][0][0]

            # General information on epochs
            s_rate = float(np.squeeze(eeg['srate']))
            t_min = float(np.squeeze(eeg['xmin']))  # should be in seconds
            t_max = float(np.squeeze(eeg['xmax']))  # should be in seconds
            nb_ch = int(np.squeeze(eeg['nbchan']))
            trials = int(np.squeeze(eeg['trials']))
            times = eeg['times'][0]
            ref = np.squeeze(eeg['ref'])

            # Channels information
            # 'labels' / 'theta' 'radius' / 'X' 'Y' 'Z' / 'sph_theta' 'sph_phi' 'sph_radius' / 'urchan' 'ref'
            ch = eeg['chanlocs'][0]

            # ICA information
            ica_act = eeg['icaact']
            ica_w_inf = eeg['icawinv']
            ica_sphere = eeg['icasphere']
            ica_weights = eeg['icaweights']

            if trials == 1 and self._data_type == 'epo':
                raise ValueError("Reading type is {:} while the data contains only {:} of epochs, verify the validity"
                                 "of your reading type".format(self._data_type, trials))

            if trials > 1:  # epoched data case
                bad_epochs = raw['bad_epochs'][0][0].flatten()  # they removed the first column of bad epochs only
                data = eeg['data'].transpose((2, 0, 1)).astype('double') * self._CAL

                # n_samples = int((t_max - t_min) * s_rate + 1)
                n_samples = int(eeg['pnts'])

                if self.use_eog:
                    VEOG = eeg['VEOG'].T * self._CAL ** 2
                    if len(bad_epochs) == VEOG.shape[0]:
                        VEOG = np.delete(VEOG, bad_epochs == 1, axis=0)
                    ch_name = [*[ch[i][0][0] for i in range(len(ch))], 'VEOG']
                    ch_type = [*['eeg'] * nb_ch, 'eog']  # 'eog' should be last
                    data = np.concatenate((data, np.reshape(VEOG, [trials, 1, n_samples])), axis=1)
                else:
                    ch_name = [ch[i][0][0] for i in range(len(ch))]
                    ch_type = ['eeg'] * nb_ch

                # Events information
                epoch = eeg['epoch'][0]
                event_latency = epoch['eventlatency']
                event_type = epoch['eventtype']
                events = np.array([[i * n_samples, 0, self._event_name_map[event_type[i][event_latency[i] == 0][0][0]]]
                                   for i in range(epoch.shape[0])])

                """
                This is not returned but if needed it's here
                It gives an array containing at each index (epoch) the event latency and type
                of all stimuli during that epoch
                """
                # my_events = [{'event_latency': np.array(event_latency[i][0], dtype=np.int16),
                #               'event_type': np.array([event_map[x] for x in np.concatenate(event_type[i][0])],
                #               dtype=np.int16)}
                #               for i in range(len(epoch))]

                del eeg

                # Info object creation
                info = mne.create_info(ch_names=ch_name, sfreq=s_rate, ch_types=ch_type)
                # This have to be a str
                if subj_info is not None:
                    info['description'] = json.dumps(dict(file_name=file.split('.')[0], subject_info=subj_info,
                                                          decomposition=None))
                else:
                    info['description'] = json.dumps(dict(file_name=file.split('.')[0], decomposition=None))

                # Montage creation and electrodes position adjustments
                info.set_montage(montage='standard_1020', match_case=True)
                if self.use_eog:
                    info['chs'][-1]['cal'] = self._CAL
                for i in range(nb_ch):
                    info['chs'][i]['loc'][0] = -ch['Y'][i] * 1e-3
                    info['chs'][i]['loc'][1] = +ch['X'][i] * 1e-3
                    info['chs'][i]['loc'][2] = +ch['Z'][i] * 1e-3
                    info['chs'][i]['cal'] = self._CAL

                # epochs MNE object
                epochs = mne.EpochsArray(data=data, info=info, events=events, tmin=t_min, event_id=self._event_id_map,
                                         verbose=reader_verbose)
                return epochs

            elif trials == 1:  # continuous data case
                # Extracting the information
                data = eeg['data'] * self._CAL

                n_samples = int(eeg['pnts'])

                if self.use_eog:
                    VEOG = eeg['VEOG'] * self._CAL
                    ch_name = [*[ch[i][0][0] for i in range(len(ch))], 'VEOG']
                    ch_type = [*['eeg'] * nb_ch, 'eog']  # 'eog' should be last
                    data = np.concatenate((data, VEOG), axis=0)
                else:
                    ch_name = [ch[i][0][0] for i in range(len(ch))]
                    ch_type = ['eeg'] * nb_ch

                # Events information
                event_latency = eeg['event'][0]['latency']
                event_type = eeg['event'][0]['type']
                idx_to_del = np.where((event_type == 'boundary') | (event_type == 'S 85') | (event_type == 'S255'))
                event_latency = np.delete(event_latency, idx_to_del)
                event_type = np.delete(event_type, idx_to_del)
                events = np.array([[event_latency[i][0][0], 0, self._event_name_map[event_type[i][0]]]
                                   for i in range(len(event_latency))])

                del eeg

                # Info object creation
                info = mne.create_info(ch_names=ch_name, sfreq=s_rate, ch_types=ch_type)

                # This have to be a str
                if subj_info is not None:
                    info['description'] = dict(file_name=file.split('.')[0], subject_info=subj_info, decomposition=None)
                else:
                    info['description'] = dict(file_name=file.split('.')[0], decomposition=None)

                # Montage creation and electrodes position adjustments
                info.set_montage(montage='standard_1020', match_case=True)
                if self.use_eog:
                    info['chs'][-1]['cal'] = self._CAL
                for i in range(nb_ch):
                    info['chs'][i]['loc'][0] = -ch['Y'][i] * 1e-3
                    info['chs'][i]['loc'][1] = +ch['X'][i] * 1e-3
                    info['chs'][i]['loc'][2] = +ch['Z'][i] * 1e-3
                    info['chs'][i]['cal'] = self._CAL

                # epochs MNE object
                raw = mne.io.RawArray(data=data, info=info, first_samp=0, verbose=reader_verbose)

                # add events
                stim_data = np.zeros((1, len(raw.times)))
                stim_info = mne.create_info(ch_names=['STIM'], sfreq=raw.info['sfreq'], ch_types=['stim'])
                stim_raw = mne.io.RawArray(stim_data, stim_info, verbose=False)
                raw = raw.add_channels([stim_raw], force_update_info=True)
                raw.add_events(events, stim_channel=['STIM'], replace=True)
                return raw

    def _init_and_check(self):
        # Check if subject_info is found and remove if from files (case where we read from raw data)
        if 'info' in self.files:
            if os.path.exists(os.path.join(self._data_path, 'info', 'subject_info.json')):
                with open(os.path.join(self._data_path, 'info', 'subject_info.json'), 'r') as fp:
                    self._subj_info = json.load(fp)
            self.files.remove('info')
        else:  # case where we read from info directly
            if os.path.exists(os.path.join(self._info_path, 'subject_info.json')):
                with open(os.path.join(self._info_path, 'subject_info.json'), 'r') as fp:
                    self._subj_info = json.load(fp)

        # Remove processed in case we are in the primary main directory
        if 'processed' in self.files:
            self.files.remove('processed')

        # Read data extension and verify that all the files of the folders have the same extension
        self._data_extension: str = self.files[0].split('.')[-1]  # get the first file extension
        for file in self.files:
            if not file.endswith(self._data_extension):
                raise ValueError("All your files should have the same extension, recognised extension is {:} but the"
                                 "file {:} doesn't match that extension".format(self._data_extension, file))

        # Pick data type ('epochs' or 'raw')
        if self._data_extension == 'fif':
            self._data_type = file.split('.')[0].split('-')[-1]  # depending on the first file of the directory
        elif 'rest' in self._data_path.split('/'):
            self._data_type = 'raw'
        else:
            self._data_type = 'epo'

        # Set the event map for the subject information reading
        # event_map_should be in the form (in data name, class, class_id)
        if self._data_type == 'epo' and self._data_extension == 'mat':
            event_map = (('S200', 'target', 1), ('S201', 'standard', 2), ('S202', 'novel', 3))
        elif self._data_type == 'raw' and self._data_extension == 'mat':
            event_map = (('S  1', 'eyes_opened', 1), ('S  2', 'eyes_opened', 1),
                         ('S  3', 'eyes_closed', 2), ('S  4', 'eyes_closed', 2))
        else:
            event_map = None
        if event_map is not None:
            in_data_name, class_name, class_id = zip(*event_map)
            self._event_name_map = dict(zip(in_data_name, class_id))
            self._event_id_map = dict(zip(class_name, class_id))

        # If raw data (first time touching them) must shift the data and put them on the same time origin
        if self._data_extension == 'mat' and self._data_type == 'epo' and \
                'processed' not in self._data_path.split('/'):
            preprocessing.is_my_case = True

        print_c('Loading folder: {:}'.format(self._data_path), bold=True)
