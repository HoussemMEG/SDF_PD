import json
import os
import random
import warnings
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import utils
from plots import Plotter
from utils import print_c

np.set_printoptions(precision=2, suppress=True)
verbose = True
once = True


class Classifier:
    def __init__(self,
                 case=None,
                 clf_choice='LDA',
                 feature_extraction=True,
                 test_set=None,
                 n_repeat=None,
                 ch_pick=None,
                 stim_pick=None,
                 pars_pick=None,
                 select_freq: Union[List[Union[int, float]], None] = None,
                 switch=False,
                 use_x0=False,
                 score_thresh=100,
                 precision=0.01,
                 plot=False,
                 save_plot=False,
                 save_features=False,
                 **kwargs):
        """
        The classifier class is in charge of:
            1- Feature extraction and save them.
            2- Classification in a LOO procedure /!\\ not in a nested way.
            3- Classification in a nested CV procedure but not as described in the paper, for that see Step_5.
            4- Permutation testing.
        However due to some time limitation this class is only in charge of the feature extraction and saving.

        :param case: the running cases of the algorithm (generaly you would like to select 3 or other)
            case 0: to select only PD patients and generate a random label
            case 1: to select only CTL patients and generate a random label
            case 2: to select randomly 25 patients (12 PD and 13 CTL), without shuffling their labels
            case 3: to perform a permutation test (randomly shuffle the labels of all the patients)
            other: to perform a normal classification
        :param clf_choice: classifier choice {'LDA', 'QDA', 'SVM'}.
        :param feature_extraction: to weather use all the SDF or extract features F_1 and F_2.
        :param test_set: value to specify how the testing procedure is done.
            test_set == 'LOO': leave one out procedure
            test_set == 0.0: train and test on the whole data set
            0.0 < test_set < 1.0: split randomly the data set into train and test sets (depending on the
                                  specified value) redo the split x times.
            test_set (int): take test_set subjects as validation and test_set subjects as testing subjects,
                            the remaining are used as training subjects. Preferable don't use this but rather use the
                            Step_5 it contains the nested LOOCV.
        :param n_repeat: to specify the number of times we redo the training and testing set splitting in the case where
                          test_set is an (int), validation is not concerned. This parameter is ignored in other cases.
        :param ch_pick: list of channels to pick during the classification e.g. ['CP1', 'CPz', 'CP2'], other channels
                         are ignored.
        :param stim_pick: list of stimulus type to pick during the classification, other stims are skipped.
        :param pars_pick: list of parsimony to pick during the classification, other parsimony levels are skipped.
        :param select_freq:
        :param switch (bool): if False the order of the SDF is kept as in the paper, if True the SDF will be ordered
                               first by their frequency then by the time, it kinda does a transpose to the SDF.
        :param use_x0: whether use 'x0' in the feature extraction or use only 'u'.
        :param score_thresh: score threshold, if the score is above this threshold the color is green.
        :param precision: signal values under this precision are set to zero.
        :param plot: if True plot the SDF in two forms:
            1- if feature selection has been performed (feature_extraction=True):
                plot the extracted features in a correlogram manner.
            2- if no feature selection applied (feature_extraction=False):
                plot the features in fireflies manner showing white dots where (frequency and time) the features are
                 non-zero.
        :param save_plot: True in order to save the plotted figures, ignored when plot=False.
        :param save_features: True to save the extracted features in "./extracted features", this should be done to use
                                step_5.
        :param kwargs:
                * path: path folder that contain the generated SDF.
                * pca: to whether perform a PCA on the SDF or on the extracted features.
                * n_component: number of component to be selected by the PCA, ignored if pca=False.
                * merge: ignored if data_case!='epochs' (see DFG parameters), this parameter allow to have one
                          prediction per subject for the case of multiple epochs data for one subject using a voting
                          strategy.
        """

        self._case = case
        self._clf_choice = clf_choice.upper()

        # Reading param
        self.path = kwargs.get('path', os.path.join(os.getcwd(), 'generated SDF'))
        self._sessions = [os.path.join(self.path, folder) for folder in sorted(os.listdir(self.path), reverse=True)]
        self._param_files = ['DFG_parameters.json', 'preprocessing_parameters.json']
        self._features_files = ['generated_features.json']
        self._patient_case = ['on_medication', 'off_medication'][1]

        # Classification param
        self._clf = self._classifier_choice()
        self._param = None
        self._test_set = 0 if test_set is None else test_set
        self._n_repeat = n_repeat if n_repeat is not None else 500
        self._extract_feature = feature_extraction
        self._selection_freq = select_freq
        self._precision = precision
        self._use_x0 = use_x0
        self._pca = kwargs.get('pca', False)
        self._n_components = kwargs.get('n_components', 3)
        self._PCA = sklearn.decomposition.PCA(n_components=self._n_components, whiten=True)

        # Selection param (this part need to be modifier by the selection method)
        self._pick_channel = ch_pick
        self._pick_stim = stim_pick
        self._pick_pars = pars_pick
        self._remove_patient = ['894_1_PDDys_ODDBALL', '908_1_PDDys_ODDBALL']  # as James Cavanagh did, I don't know why
        self._switch = switch
        self._score_thresh = score_thresh

        # Running values
        self._curr_stim = None
        self._curr_ch = None
        self._curr_pars = None
        self._curr_sess = None

        # misc param
        self._merge = kwargs.get('merge', False)
        self._plot = plot
        self._save_plot = save_plot
        self._save = save_features
        self._to_save = {}
        self.max_score = 0
        self.max_test_score = 0
        self._print_parameters()

    def _read_param(self, session):
        """
        Method in charge of reading the JSON <parameters> files and extract the working parameters that are needed
        for the classifier to work.
            1- preprocessing_parameters.json
            2- DFG_parameters.json
        /!\\ This method is in charge of performing the channel selection and the parsimony selection

        :param session: the path of the current session
        """
        # Reading the JSON files
        self._param = {}
        for file in os.listdir(session):
            if file in self._param_files:
                param_path = os.path.join(session, file)
                with open(param_path) as f:
                    self._param.update(json.load(f))

        # Parameters reading
        self._model_freq = np.array(self._param['model_freq'])
        self._n_freq = self._param['n_freq']
        self._n_point = self._param['n_point']
        self._n_features = self._param['n_features']
        self._version = self._param.get('version', 0)
        if self._use_x0 and (self._version is None or self._version not in [1, 2]):
            print_c(r'/!\ Error no x0 found in data while use_x0 is {:}, use_x0 is set to False'
                    .format(self._use_x0), 'red')
            self._use_x0 = False
        self._pars = self._param['selection'] if self._param['selection'] is not None \
            else self._param.get('selection_alpha', None)
        self._data_case = self._param.get('data_case', 'evoked')
        self._alpha = self._param['alpha']

        # Channel reading
        self._channels = self._param['channel_picks']  # Used channels
        if self._channels == 'all':
            self._channels = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                              'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2',
                              'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                              'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4',
                              'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz']  # The order is important
        if 'VEOG' in self._channels:
            self._channels.remove('VEOG')

        # Channel and parsimony picks
        self._pick_channels()
        self._pick_parsimony()

        self._print_model_parameters()

    def _classifier_choice(self):
        """
        Set the classifier with its working parameters.
        Available classifiers {'QDA', 'LDA', 'SVM'}

        :return: clf (classifier).
        """
        if self._clf_choice == 'QDA':
            return QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0)
        elif self._clf_choice == 'LDA':
            return LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                              store_covariance=False, tol=0.0001, covariance_estimator=None)
        elif self._clf_choice == 'SVM':
            return SVC(C=1.0, kernel='linear')
        else:
            raise ValueError("Classifier supported {:}, but {:} were given".format(['LDA', 'QDA', 'SVM'],
                                                                                   self._clf_choice))

    def _select_data(self, x: np.ndarray, y: np.ndarray):
        """
        Method in charge of switching the run cases.
            case 0: to select only PD patients and generate a random label
            case 1: to select only CTL patients and generate a random label
            case 2: to select randomly 25 patients (12 PD and 13 CTL), without shuffling their labels
            case 3: to perform a permutation test (randomly shuffle the labels of all the patients)
            other: to perform a normal classification
        for case 3 the permutation is set in the reading method since we should keep the same permuted labels for all
         the channels and parsimony levels, however, at each global iteration the labels are permuted again.

        :param x: shape: (n_patient, n_features, n_path)
        :param y: shape: (n_patient,)
        :return: x, y
        """
        if self._case == 0 or self._case == 1:
            x = x[y == self._case]
            y = y[y == self._case]
            # label permutation
            permute_mask = random.sample(range(len(y)), k=int(len(y) / 2))
            y[permute_mask] = np.logical_not(y[permute_mask])
        elif self._case == 2:
            PD_mask = random.sample(list(np.where(y == 0)[0]), 12)
            CTL_mask = random.sample(list(np.where(y == 1)[0]), 13)
            mask = PD_mask + CTL_mask
            x, y = x[mask], y[mask]
        return x, y

    def classify(self):
        """
        Main classification method, performs a classification for each session present in the folder "./generated SDF".

        Just as a reminder
          X shape: List(n_stim)(List(n_patient)(List(n_channels)(n_features, n_path)))
          y shape: (n_patient, 1)

        :return: /
        """
        for sess in self._sessions:
            if sess.split('\\')[-1] == r"don't read":
                continue
            print_c('\nSessions: {:}'.format(sess.split('\\')[-1]), 'blue', bold=True)
            self._curr_sess = sess
            self._read_param(sess)

            X, y = self._read_data(sess)  # X: List(n_stim)(List(n_patient)(List(n_channels)(n_features, n_path)))

            for stim_idx, stim in enumerate(self._stim):
                self._curr_stim = stim
                self._print_stim(stim, stim_idx, len(self._stim))

                if self._data_case == 'evoked':
                    for i, ch_idx in enumerate(self._ch_pick_idx):
                        self._curr_ch = self._channels[ch_idx]
                        print_c('\n--- Channel: <<{:}>>   {:}/{:}'.format(self._curr_ch, i + 1,
                                                                          len(self._ch_pick_idx)), 'cyan', bold=True)
                        X_ = np.array([value[ch_idx] for value in X[stim_idx]])
                        self._classify(X_[:, :, self._pars_pick_idx], y)

                elif self._data_case == 'epochs':
                    print_c('\n--- Channel: <<{:}>>   {:}/{:}'.format(self._curr_ch, 1,
                                                                      len(self._ch_pick_idx)), 'cyan', bold=True)
                    X_ = np.vstack(X[stim_idx])
                    self._classify(X_[:, :, self._pars_pick_idx], y)

            # save the extracted features of each session
            self._save_(sess)

    def _pick_channels(self):
        """
        This method is in charge of picking the desired channels, see the argument ch_pick
        /!\\ Be careful while trying to change this method, the order of the channels is important

        :return: /
        """
        if not self._pick_channel:
            self._ch_pick_idx = range(len(self._channels))
        else:  # The order of enumerate(self._channels is important)
            self._ch_pick_idx = [i for i, value in enumerate(self._channels) if value in self._pick_channel]
            if len(self._ch_pick_idx) == 0:
                raise ValueError('No selected channel found, channel picks: {:}'.format(self._pick_channel))

    def _pick_parsimony(self):
        """
        This method is in charge of picking the desired parsimony level, see the argument pars_pick

        :return: /
        """
        if not self._pick_pars:
            self._pars_pick_idx = range(len(self._pars))

        else:
            self._pars_pick_idx = [i for i, value in enumerate(self._pars) if np.isclose(value, self._pick_pars).any()]
            # A verification if no desired parsimony level is found in the data
            if len(self._pars_pick_idx) == 0:
                raise ValueError('No parsimony level found in the data, parsimony picks {:}'.format(self._pick_pars))
            self._pars_pick_idx.sort()

    def _classify(self, x: np.ndarray, y: np.ndarray):
        """
        Main classification method that contain the classifier and all the train / test splitting.

        :param x: shape (n_patient, n_features, n_path)
        :param y: shape (n_patient,)
        :return: /
        """

        # Selecting the data depending on the run case
        x, y = self._select_data(x, y)

        path_scores = []  # Path score memory
        for pars_idx in range(len(self._pars_pick_idx)):
            self._curr_pars = self._pars[self._pars_pick_idx[pars_idx]]
            # frequency selection and switch rotation
            X = self._frequency_pick(x, pars_idx)

            # insignificant value removal
            X[np.where(np.abs(X) < self._precision)] = 0

            # feature extraction and non_zero masking
            if self._extract_feature:
                X = self._feature_extraction(X)
            elif not self._plot:  # if I want to plot I don't want to remove non_zero values from the array
                feature_mask = np.sum(X[:, :] != 0, axis=0) != 0
                X = X[:, feature_mask]

            # features saving
            self._save_features(X)

            if self._pca:
                X = self._PCA.fit_transform(X)
                print('Explained variance ratio', self._PCA.explained_variance_ratio_)

            # train and test procedure with warning catching
            with warnings.catch_warnings(record=True) as warning:
                score = self._train_procedure(X, y, self._curr_pars)
                path_scores.append(score)
            if any([issubclass(warn.category, UserWarning) for warn in warning]):
                print_c(r'/!\ Collinear variable warning', 'red')

            # plotting features
            self.plot_feature(X, y)

            if self._use_x0 and self._version == 2:
                return 0
        path_scores = np.array(path_scores)
        path_mean, path_std = path_scores.mean(), path_scores.std()
        print(' ' * 32, 'Score: {:3.1f} ± {:.1f} %'.format(path_mean, path_std))

    def _frequency_pick(self, x, pars_idx):
        """
        Method that perform the frequency selection, and the switch the arrangement of the features
            (from time then frequency -> frequency then time)

        :param x: shape (n_patient, n_features, n_path)
        :param pars_idx: current parsimony index
        :return: X: The array on which we will perform the classification on, shape: (n_patient, n_features)
        """
        # Generating the frequency selection mask
        if self._selection_freq:
            if all(isinstance(i, int) for i in self._selection_freq):
                self._freq_mask = sorted(list(set([i for i in self._selection_freq if i < self._n_freq])))
            elif all(isinstance(x, float) for x in self._selection_freq):
                min_, max_ = min(self._selection_freq), max(self._selection_freq)
                self._freq_mask = np.where((self._model_freq >= min_) &
                                           (self._model_freq <= max_))[0]
            else:
                raise ValueError('All frequency selection values must have the same type and must be int or float only,'
                                 ' given type {:}'.format([type(x) for x in self._selection_freq]))
            if len(self._freq_mask) == 0:
                raise ValueError('Unable to find any selected frequency {:} in the frequency model'
                                 .format(self._selection_freq))

        # Perform the frequency selection
        if self._selection_freq or self._switch:
            divide_by = self._n_features / self._n_freq
            X = np.swapaxes(np.array(np.split(x[:, :, pars_idx], divide_by, axis=1)), 0, 1)
            if self._selection_freq:
                X = X[:, :, self._freq_mask]
            if self._switch:
                X = np.array([np.ravel(X[i], order='F') for i in range(x.shape[0])])
            else:
                X = np.array([np.ravel(X[i], order='C') for i in range(X.shape[0])])
        else:
            X = x[:, :, pars_idx]
        return X

    def _train_procedure(self, x, y, pars_level):
        """
        Method that performs the test-train sets split following the specified training procedure
            test_set == 'LOO': leave one out procedure
            test_set == 0.0: train and test on the whole data set
            0.0 < test_set < 1.0: split randomly the data set into train and test sets (depending on the
                                  specified value) redo the split x times.
            test_set (int): take test_set subjects as validation and test_set subjects as testing subjects,
                            the remaining are used as training subjects. Preferable don't use this but rather use the
                            Step_5 it contains the nested LOOCV.

        :param x: features, shape (n_patient, n_features)
        :param y: label, shape (n_patient,)
        :param pars_level: parsimony level used only for print
        :return: /
        """
        self._h = []
        if self._test_set == 'LOO':  # Leave One Out procedure
            score_memory = []
            for exclude_idx, y_i in enumerate(y):
                mask = np.ones(y.shape, bool)
                mask[exclude_idx] = False
                X_train, X_test, y_train, y_test = x[mask, :], x[[exclude_idx], :], y[mask], y[[exclude_idx]]

                # Train
                self._clf.fit(X=X_train, y=y_train)

                # Test
                # self._h.append(self._clf.predict_proba(X_test)[0][-1])
                self._h.append(self._clf.predict(X_test)[0])
                score = 100 * self._clf.score(X=X_test, y=y_test)
                score_memory.append(score)
            # print(score_memory)
            score_memory = np.array(score_memory)
            mean, std = score_memory.mean(), score_memory.std()
            self._print_score(pars_level, mean)

            # store the best values
            if mean > self.max_score:
                self.max_score = mean
                self.max_score_pars = pars_level
            return mean

        elif self._test_set == 0:  # Train on all the data then evaluate performance on all the data
            # Train
            self._clf.fit(X=x, y=y)

            # Test
            score = 100 * self._clf.score(X=x, y=y)
            if score > self.max_score:
                self.max_score = score
                self.max_score_pars = pars_level

            if self._merge and self._data_case == 'epochs':
                y_real = y[5::20]
                y_pred = self._clf.predict(X=x)
                y_pred = np.array([int(np.sum(y_pred[i * 20:(i + 1) * 20]) > 10) for i in range(50)])
                score = np.logical_not(np.logical_xor(y_real, y_pred)).sum() / 50 * 100
            self._print_score(pars_level, score)
            return score

        elif 0.0 < self._test_set < 1.0:  # Training and testing sets, repeat the splitting x times
            score_memory = []  # trial score memory
            for _ in range(10000):  # x times repetition
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self._test_set,
                                                                    shuffle=True, stratify=y)
                # Train
                self._clf.fit(X=X_train, y=y_train)

                # Test
                score = 100 * self._clf.score(X=X_test, y=y_test)
                score_memory.append(score)

            # store the best values
            score_memory = np.array(score_memory)
            mean, std = score_memory.mean(), score_memory.std()
            self._print_score(pars_level, mean, std)
            self.max_score = mean if mean > self.max_score else self.max_score
            return mean

        # Training, validation and test sets, repeat the splitting self._n_repeat times
        elif isinstance(self._test_set, int):
            merged_idx = self._train_idx + self._val_idx
            train_size = 1 - self._test_set / len(merged_idx)
            score_memory_train = []
            score_memory_val = []
            score_memory_test = []
            for _ in range(self._n_repeat):
                train_idx, val_idx, _, _ = train_test_split(merged_idx, y[merged_idx], train_size=train_size,
                                                            stratify=y[merged_idx])
                # Train
                self._clf.fit(X=x[train_idx], y=y[train_idx])
                score_train = 100 * self._clf.score(X=x[train_idx], y=y[train_idx])
                score_memory_train.append(score_train)

                # Validate
                score_val = 100 * self._clf.score(X=x[val_idx], y=y[val_idx])
                score_memory_val.append(score_val)

                # Test
                score_test = 100 * self._clf.score(X=x[self._test_idx], y=y[self._test_idx])
                score_memory_test.append(score_test)

            train_mean, train_std = np.array(score_memory_train).mean(), np.array(score_memory_train).std()
            val_mean, val_std = np.array(score_memory_val).mean(), np.array(score_memory_val).std()
            test_mean, test_std = np.array(score_memory_test).mean(), np.array(score_memory_test).std()

            if val_mean > self.max_score:
                self.max_score = val_mean
                self.max_score_pars = pars_level
                self.max_test_score = test_mean

            if self._param['selection'] is not None:
                if val_mean < self._score_thresh:
                    print('\t\tParsimony level: {:3.0f} %        Train score: {:3.1f} ± {:3.1f} %'
                          '     Val score: {:3.1f} ± {:3.1f} %     Test score: {:3.1f} ± {:3.1f} %'.format(
                        100 * pars_level, train_mean, train_std, val_mean, val_std, test_mean, test_std))
                else:
                    print_c('\t\tParsimony level: {:3.0f} %        Train score: {:3.1f} ± {:3.1f} %'
                            '     Val score: {:3.1f} ± {:3.1f} %     Test score: {:3.1f} ± {:3.1f} %\t Good'.format(
                        100 * pars_level, train_mean, train_std, val_mean, val_std, test_mean, test_std), 'green',
                        bold=True)
            else:
                if val_mean < self._score_thresh:
                    print('\t\tParsimony alpha: {: <7}    Train score: {:3.1f} ± {:3.1f} %'
                          '     Val score: {:3.1f} ± {:3.1f} %     Test score: {:3.1f} ± {:3.1f} %'.format(
                        100 * pars_level, train_mean, train_std, val_mean, val_std, test_mean, test_std))
                else:
                    print_c('\t\tParsimony alpha: {: <7}    Train score: {:3.1f} ± {:.1f} %'
                            '     Val score: {:3.1f} ± {:3.1f} %     Test score: {:3.1f} ± {:3.1f} %\t Good'
                            .format(100 * pars_level, train_mean, train_std, val_mean, val_std,
                                    test_mean, test_std), 'green', bold=True)
            return val_mean

    @utils.execution_time
    def _read_data(self, session):
        """
        Method in charge of reading the data from <generated_features.json>, decompress them and parse them.

        :param session: the path of the current session
        :return: X SDF, shape: List(n_stim)(List(n_patient)(List(n_channels)(n_features, n_path)))
        :return: y class label (PD=0, CTL=1), shape: (n_patient, 1)
        """
        for file in os.listdir(session):
            if file not in self._features_files:
                continue

            file_path = os.path.join(session, file)
            with open(file_path) as f:
                data: dict = json.load(f)

            self._stim = list(data.keys())
            if self._pick_stim:
                self._stim = [stim for stim in self._pick_stim if stim.lower() in self._stim]
            if not self._stim:
                raise ValueError('No stim type {:} found in data'.format(self._pick_stim))

            X = []
            for stim in self._stim:
                X_, y, ID = [], [], []
                for subj, subj_data in data[stim].items():
                    if subj in self._remove_patient:
                        continue
                    subj_info = subj_data['subject_info']['subject_info']
                    ID.append(subj)

                    if self._use_x0:
                        subj_feat = subj_data['x0']  # shape: List(n_channels)(n_features, n_path)
                    else:
                        # subj_feat shape: List(n_channels)(n_features, n_path)
                        subj_feat = utils.decompress(subj_data['features'], n_features=self._n_features)

                    # Consistence verification (that the n_path is equal between all the subjects)
                    # If not equal then repeat the last solution <repeat> times at the end
                    if self._version == 2 or self._version == 1:
                        for ch in range(len(subj_feat)):
                            temp = np.array(subj_feat[ch])
                            if temp.shape[1] < len(self._pars):
                                repeat = len(self._pars) - temp.shape[1]
                                subj_feat[ch] = np.hstack((temp, np.tile(temp[:, [-1]], repeat)))

                    X_.append(subj_feat)
                    if self._data_case == 'evoked':
                        y.append(1 if subj_info['category'] == 'CTL' else 0)
                    elif self._data_case == 'epochs':
                        y.extend([1] * len(subj_feat) if subj_info['category'] == 'CTL' else [0] * len(subj_feat))

                X.append(X_)

            if self._save:
                self._to_save['y_true'] = y
            y = np.array(y)

            self._IDs = ID
            if isinstance(self._test_set, int) and self._test_set != 0:
                if self._test_set % 2 == 1:
                    raise ValueError('test_set should be multiple of 2, got instead {:}'.format(self._test_set))
                train_size = 1 - 2 * self._test_set / len(self._IDs)
                self._train_idx, rem_idx, _, y_rem = train_test_split(range(len(self._IDs)),
                                                                      y, train_size=train_size, stratify=y)
                self._val_idx, self._test_idx, _, _ = train_test_split(rem_idx, y_rem, test_size=0.5, stratify=y_rem)

        if self._case == 3:  # permutation test only
            np.random.shuffle(y)
        return X, y

    def _print_parameters(self):
        global once
        if once:
            if self._case == 0:
                print_c(r'/!\ Careful, PD data only are taken', 'red', bold=True)
            elif self._case == 1:
                print_c(r'/!\ Careful, CTL data only are taken', 'red', bold=True)
            elif self._case == 2:
                print_c(r'/!\ Careful, 25 data samples are taken', 'red', bold=True)
            elif self._case == 3:
                print_c(r'/!\ Careful, permutation test', 'red', bold=True)
            else:
                print_c(r'Normal training', 'green', bold=True)
            print_c(' Classifier choice: ', 'green', self._clf_choice)
            print_c(' Merge: ', 'green', str(self._merge))
            print_c(' Test set: ', 'green', str(self._test_set))
            print_c(' Patient case: {:}'.format(self._patient_case), 'green')
            print_c(' Channel pick: ', 'green', self._pick_channel)
            print_c(' Stim pick: ', 'green', self._pick_stim)
            print_c(' Parsimony pick: ', 'green', self._pick_pars)
            print_c(' Score threshold: {:.1f}%'.format(self._score_thresh), 'green')
            print_c(' Precision = <{:}>'.format(self._precision), 'green')
            print_c(' Feature extraction: ', 'green', self._extract_feature)
            print_c(' Frequency selection: {:}'.format(self._selection_freq), 'green')
            print_c(' Plot : {:}'.format(self._plot), 'green')
            if self._use_x0:
                print_c(r'/!\ Careful use X0 enabled', 'red')
            if self._pca:
                print_c(r'/!\ Careful PCA enabled, n_components = {:}'.format(self._n_components), 'red')
        once = False

    def _print_model_parameters(self):
        global verbose
        if verbose:
            print_c(' Data case: ', highlight=self._data_case)
            print_c(' Version: ', highlight=str(self._version))
            print_c(' Alpha: ', highlight=self._alpha)
            print(' Channels: {:}'.format(self._param['channel_picks']))
            print(' Model frequencies: {:}'.format(self._model_freq))
            print(' N_freq = {:}'.format(self._n_freq))
            print_c(' N_point = ', highlight=self._n_point)
            print(' Parsimony: {:}'.format(np.array(self._pars)))

    @staticmethod
    def _print_stim(stim, idx, l):
        print_c('\n{:} -------------------------------------'.format(' ' * 47), 'yellow', bold=True)
        print_c('{:} Stimulus: <{:}>    {:}/{:}'.format(' ' * 51, stim.capitalize(), idx + 1, l), 'yellow', bold=True)
        print_c('{:} -------------------------------------'.format(' ' * 47), 'yellow', bold=True)

    def _print_score(self, pars_level, score, std=None):
        if std is None:
            if self._param['selection'] is not None:
                if score < self._score_thresh:
                    print("\t\tParsimony level: {:3.0f} %   Score: {:3.1f} %".format(100 * pars_level, score))
                else:
                    print_c("\t\tParsimony level: {:3.0f} %   Score: {:3.1f} %\t Good".format(100 * pars_level, score),
                            'green', bold=True)
            else:
                if score < self._score_thresh:
                    print("\t\tParsimony alpha: {: <7}  Score: {:3.1f} %".format(pars_level, score))
                else:
                    print_c("\t\tParsimony alpha: {: <7}  Score: {:3.1f} %\t Good".format(pars_level, score),
                            'green', bold=True)
        else:
            if self._param['selection'] is not None:
                if score < self._score_thresh:
                    print('\t\tParsimony level: {:3.0f} %   Score: {:3.1f} ± {:.1f} %'.format(100 * pars_level,
                                                                                              score, std))
                else:
                    print_c('\t\tParsimony level: {:3.0f} %   Score: {:3.1f} ± {:.1f} %\t Good'
                            .format(100 * pars_level, score, std), 'green', bold=True)
            else:
                if score < self._score_thresh:
                    print('\t\tParsimony alpha: {: <7}  Score: {:3.1f} ± {:.1f} %'.format(pars_level, score, std))
                else:
                    print_c('\t\tParsimony alpha: {: <7}  Score: {:3.1f} ± {:.1f} %\t Good'
                            .format(pars_level, score, std), 'green', bold=True)

    def plot_feature(self, x, y, vmax=0.1):
        """
        Plot the SDF in two forms:
            1- if feature selection has been performed (feature_extraction=True):
                plot the extracted features in a correlogram manner.
            2- if no feature selection applied (feature_extraction=False):
                * if cmap="black_white": plot the features in fireflies manner showing white dots where
                    (frequency and time) the features are non-zero (to observe general patterns) for the PD only / CTL
                    only and both of them on the same figure.
                * if cmap="blue_red": plot the features in a red blue gradient, (blue for negative value and red for
                    positive value of the SDF), showing where the features are non-zero for the PD only / CTL only
                    and both of them on the same figure.

        :param x: SDF or extracted features. shape: (n_patient, n_features)
        :param y: patient labels
        :param: vmax: if the SDF value is above vmax or below -vmax this value is considered as max value so will be
                        assigned  either blue or red color (edge color of the cmap), it only works for
                        feature_extraction=False and cmap='blue_red'.
        :return: /
        """
        if not self._plot:
            return

        if self._extract_feature:
            if x.shape[-1] == 2:
                Plotter.decision_boundary(x, y, self._clf)
            elif x.shape[-1] == 1:
                Plotter.scatter(x, y, self._clf)
            else:
                Plotter.correlogram(x, y)

        else:  # if not feature selection show the big colored diagram grouping all the patients
            PD = x[np.where(y == 0)]
            CTL = x[np.where(y == 1)]
            # np.random.shuffle(PD), np.random.shuffle(CTL)  # don't forget to shuffle the labels also
            PD_id = ['_'.join(id.split('_')[0:2]) for i, id in enumerate(self._IDs) if i in np.where(y == 0)[0]]
            CTL_id = ['_'.join(id.split('_')[0:2]) for i, id in enumerate(self._IDs) if i in np.where(y == 1)[0]]
            ID = PD_id + CTL_id

            # normalizing the vectors (so they have the same max amplitude)
            PD, CTL = PD / np.std(PD, axis=1)[:, np.newaxis], CTL / np.std(CTL, axis=1)[:, np.newaxis]

            session = self._curr_sess.split('\\')[-1]
            fig_name = '_'.join([session, self._curr_stim, self._curr_ch, str(self._curr_pars)])
            Plotter.plot_control(x=PD, freq=self._model_freq, cmap='blue_red',  # 'blue_red', 'black_white'
                                 group_by='all', vmax=vmax, id_label=PD_id,
                                 fig_name='PD_' + fig_name, save=self._save_plot, show=True)
            Plotter.plot_control(x=CTL, freq=self._model_freq, cmap='blue_red',  # 'blue_red', 'black_white'
                                 group_by='all', vmax=vmax, id_label=CTL_id,
                                 fig_name='CTL_' + fig_name, save=self._save_plot, show=True)
            Plotter.plot_control(x=np.vstack((PD, CTL)), freq=self._model_freq,
                                 cmap='blue_red',  # 'blue_red', 'black_white'
                                 group_by='all', vmax=vmax, id_label=ID,
                                 fig_name='ALL_' + fig_name, save=self._save_plot, show=True)
        plt.show(block=True)

    def _feature_extraction(self, x: np.ndarray) -> np.ndarray:
        """
        Feature extraction method, extract F_1 and F_2 from the SDF.
            F_1 = the instant of the lowest activation amplitude of all oscillators defined by: argmin(U[0, 500] ms)
            F_2 = the average excitation forces occurring between 180 ms and 500 ms defined by: mean(U[180, 500] ms)

        :param x: shape: (n_patient, n_features)
        :return: x: shape: (n_patient, m_features) with m < n
        """
        # Each feature should have the dimension of (n_patient,)
        f = [np.argmin(x, axis=1) // self._n_freq * 2,                       # F_1
             np.sum(x[:, self._n_freq * 90:self._n_freq * 250], axis=1)]     # F_2 (/!\\ 90 and 250 are the indexes)

        # Consistency verification
        if not f:
            raise ValueError("No feature picked, please select some features")
        for i, feature in enumerate(f):
            if feature.shape != (x.shape[0],):
                raise ValueError("The feature <{:}> doesn't have the correct dimension,"
                                 " expected {:} but got instead {:}".format(i, (x.shape[0],), feature.shape))
        return np.array(f)[range(i + 1)].T

    def _save_features(self, X):
        """
        Method to save the extracted features if any with their parsimony levels.

        :param X: current (at one level of parsimony) of the extracted features, to be stacked in the dictionary
                   self._to_save.
        :return: /
        """
        if not self._save:
            if not self._extract_feature:
                raise ValueError("Unable to save features as features extraction is set to <{:}>".
                                 format(self._extract_feature))
            return

        if self._to_save.get(self._curr_ch) is None:
            self._to_save[str(self._curr_ch)] = {'parsimony': list(np.array(self._pars)[self._pars_pick_idx]),
                                                 'features': [X.tolist()]}
        else:
            features = self._to_save[self._curr_ch]['features']
            features.append(X.tolist())
            self._to_save[str(self._curr_ch)] = {'parsimony': list(np.array(self._pars)[self._pars_pick_idx]),
                                                 'features': features}

    def _save_(self, sess):
        """
        :param sess: the name of the current session
        :return: /
        """
        if not self._save:
            return

        save_path = os.path.join(os.getcwd(), 'extracted features')
        utils.save_args(self._to_save, path=save_path, save_name=sess.split('\\')[-1], verbose=True)


max_score, max_score_pars, max_score_test = [], [], []
for k in range(1):  # in the case of permutation test, set to the number of repetition desired
    # case 0: PD_only / 1: CTL_only / 2: 25_random_PD_CTL / 3: permutation_test / else: normal_case
    classifier = Classifier(case=None,
                            use_x0=False,
                            clf_choice='LDA',  # {'LDA', 'QDA', 'SVM'}
                            feature_extraction=True,
                            test_set='LOO',  # float: 0.0 <= val < 1.0 or 'LOO': leave one out
                            n_repeat=200,
                            stim_pick=[],  # 'standard', 'novel', 'target', 'standard_novel_target'
                            ch_pick=[],  # ['CP1', 'CPz', 'CP2']
                            pars_pick=[],
                            select_freq=[],
                            score_thresh=74,
                            merge=True,
                            switch=False,
                            pca=False, n_components=3,
                            save_features=True,
                            plot=False, save_plot=False)
    classifier.classify()

    if k > 0:
        print_c(r'Max score: {:.2f} %'.format(classifier.max_score), 'red', bold=True)
        max_score.append(classifier.max_score)
        max_score_test.append(classifier.max_test_score)
        print('\nk =', k)
        print('Train\n {:} \n\t µ = {:.2f} ± {:.2f}\n'.format(max_score, np.array(max_score).mean(), np.array(max_score).std()))
        print('Test \n {:} \n\t µ = {:.2f} ± {:.2f}\n'.format(max_score_test, np.array(max_score_test).mean(),
              np.array(max_score_test).std()))
