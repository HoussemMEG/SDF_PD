import json
import os
from math import comb

import numpy as np
import sklearn.base
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import print_c

path = os.path.join(os.getcwd(), 'extracted features')


class Classifier:
    def __init__(self, test_size, val_size, repeat_test, repeat_val, vote_prob, merge_on_test, y_true, clf='LDA'):
        """
        Main classifier that uses the nested CV and nested LOOCV procedure presented in the paper. This class is also
        the one in charge of the voting strategy.
        This class is badly writen with hard coding and a lot of bad habits due to lack of time (but it is working well)

        :param: clf: Classifier choice, available options are {'LDA', 'QDA', 'SVM'}, default 'LDA'.
        :param test_size: handle the learning set / testing set splitting procedure (so it only concerns the outer-loop)
                           The splitting is always performed in a random stratified fashion, available options:
                            {0, 'BLOO', int, 'LOO'}
                            1- 0: for no testing set, all the available data is used as a learning set.
                            2- BLOO: stands for bad LOO procedure (with data leakage to compare with James Cavanagh
                                        paper, where we permit ourself to select the best working parameters on the
                                        testing set).
                            3- int < n_patient: take test_size patients in the test set and repeat repeat_test times
                                                this splitting.
                                                Special case where test_size is a divider of n_patients, then a Cross
                                                Validation procedure is used in this case and repeat_test is ignored.
                                                For our case n_patients = 50, so test_size=10 is the 5-fold CV and
                                                test_size=25 is the 2-fold CV.
                            4- 'LOO': Leave One Out procedure as described in the paper.
        :param val_size: handle the training set / validation set splitting procedure (so it only concerns the
                          inner-loop) The splitting is always performed in a random stratified fashion,
                          available options: {'LOO', int}:
                            1- 'LOO': perform a LOO procedure for the learning set to form the training and validation
                                      set. There is no need to implement the k-fold CV since we can have a better
                                      estimate.
                            2- int: select val_size subjects as a validation set from the learning set and repeat this
                                    selection repeat_val times.
        :param repeat_test: In the case where no CV is possible for the outer-loop, we should repeat the test splits
                            repeat_test times. ignored if test_size={'BLOO', 'LOO', 0, n_patient // test_size == 0}.
        :param repeat_val: Repeat the validation splitting repeat_val times (inner-loop). ignored if val_size='LOO'.
        :param vote_prob: If True use the subject confidence in the vote else use the subject class,
                            see clf.predict_prob(...).
        :param merge_on_test: If True prior to test evaluation, the classifier is trained on the learning set, if False
                                the classifier is trained only on the training set (the validation set is not used in
                                the outer-loop)
        :param y_true: Subjects labels.
        """

        # Parameters
        self._clf_choice_ = clf
        self._test_size = test_size
        self._val_size = val_size
        self._repeat_test = repeat_test
        self._repeat_val = repeat_val
        self._vote_prob = vote_prob
        self._weight = [0.33]  # voting weight for each channel (0.33 will give them the same voting power)
        self._n_subj = 50
        self._merge_on_test = merge_on_test
        self._all_idx = np.arange(self._n_subj)
        self._first = True

        if isinstance(self._test_size, int) and self._test_size != 0:
            if self._n_subj % self._test_size == 0:
                n_split = self._n_subj // self._test_size
                n_repeat = self._repeat_test // n_split
                rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat)
                self._split = list(rskf.split(self._all_idx, y_true))

        self._model = self._clf_choice(self._clf_choice_)
        self._model_CP1 = self._clf_choice(self._clf_choice_)
        self._model_CPz = self._clf_choice(self._clf_choice_)
        self._model_CP2 = self._clf_choice(self._clf_choice_)

        # Test-Validation
        self._validation = True
        self._test = True
        self._test_idx = None
        self._test_y = None
        self._train_idx = None
        self._train_y = None
        self._check()

        # Memory
        self._m_train_idx = []
        self._m_train_y = []
        self._m_val_idx = []
        self._m_val_y = []
        self.memory = {'CP1': {'train': [], 'val': [], 'test': [], 'param': []},
                       'CPz': {'train': [], 'val': [], 'test': [], 'param': []},
                       'CP2': {'train': [], 'val': [], 'test': [], 'param': []},
                       'Merged': {'train': [], 'val': [], 'test': [], 'param': []},
                       'Voted': {'train': [], 'val': [], 'test': [], 'param': []}}

    @staticmethod
    def _clf_choice(choice):
        """
        Set the classifier with its working parameters.
        Available classifiers {'QDA', 'LDA', 'SVM'}

        :return: clf (classifier).
        """
        if choice == 'QDA':
            return QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0)
        elif choice == 'LDA':
            return LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                              store_covariance=False, tol=0.0001, covariance_estimator=None)
        elif choice == 'SVM':
            return SVC(C=1.0, kernel='linear')

    def _check(self):
        """
        Some verification
        :return: /
        """
        if isinstance(self._test_size, int) and self._test_size != 0:
            print_c('Test size <{:}> corresponds to {:}/50 subjects'.format(self._test_size, self._test_size),
                    'green', bold=True)
            print_c('The number of possible combinations is: {:,}'.format(
                comb(len(y_true), self._test_size), ',').replace(',', ' ').replace('.', ','),
                    'green', bold=True)
        else:
            print_c('Test size <{:}>'.format(self._test_size), 'green', bold=True)

        if self._test_size == 0:
            self._validation, self._repeat_val = False, 1
            self._test, self._repeat_test = False, 1
        elif self._test_size == 'BLOO':
            self._test, self._repeat_test = False, 1
            self._repeat_val = self._n_subj
        elif self._test_size == 'LOO':
            self._repeat_test = self._n_subj
            if self._val_size == 'LOO':
                self._repeat_val = self._n_subj - 1
        elif self._val_size == 'LOO' and isinstance(self._test_size, int):
            self._repeat_val = self._n_subj - self._test_size

    def classify(self, y_true, CP1, CP1_alpha, CPz, CPz_alpha, CP2, CP2_alpha):
        """
        Main classification class

        :return: /
        """
        for i in range(self._repeat_test):
            # test / learning split
            if self._test_size == 0 or self._test_size == 'BLOO':
                self._train_idx, self._train_y = self._all_idx, y_true

            elif isinstance(self._test_size, int):
                if self._n_subj % self._test_size == 0:
                    self._train_idx, self._test_idx = self._split[i]
                    self._train_y, self._test_y = y_true[self._train_idx], y_true[self._test_idx]
                else:
                    self._train_idx, self._test_idx, self._train_y, self._test_y = train_test_split(self._all_idx,
                                                                                                    y_true,
                                                                                                    test_size=self._test_size,
                                                                                                    shuffle=True,
                                                                                                    stratify=y_true)
            if self._test_size == 'LOO':
                mask = np.ones(y_true.shape, bool)
                mask[i] = False
                self._train_idx, self._test_idx, self._train_y, self._test_y = self._all_idx[mask], self._all_idx[[i]], \
                                                                               y_true[mask], y_true[[i]]

            # train / validation split
            if self._validation:
                for j in range(self._repeat_val):
                    if self._test_size == 'BLOO':
                        mask = np.ones(y_true.shape, bool)
                        mask[j] = False
                        train_idx, val_idx, train_y, val_y = self._all_idx[mask], self._all_idx[[j]], \
                                                             y_true[mask], y_true[[j]]
                    else:
                        if self._val_size == 'LOO':
                            mask = np.ones(len(self._train_idx), bool)
                            mask[j] = False
                            train_idx, val_idx, train_y, val_y = self._train_idx[mask], self._train_idx[[j]], \
                                                                 self._train_y[mask], self._train_y[[j]]
                        else:  # int split
                            train_idx, val_idx, train_y, val_y = train_test_split(self._train_idx, self._train_y,
                                                                                  test_size=self._val_size,
                                                                                  shuffle=True, stratify=self._train_y)
                    # memory to keep track of the splittings in order to have the same splittings for all the channels
                    self._m_train_idx.append(train_idx)
                    self._m_train_y.append(train_y)
                    self._m_val_idx.append(val_idx)
                    self._m_val_y.append(val_y)
            else:
                self._m_train_idx.append(self._train_idx)
                self._m_train_y.append(self._train_y)

            # Train Val Test
            print_c('\nIteration = {:}/{:}'.format(i + 1, self._repeat_test), 'red', bold=True)
            pars_CP1 = self._train(CP1, CP1_alpha, 'CP1')
            pars_CPz = self._train(CPz, CPz_alpha, 'CPz')
            pars_CP2 = self._train(CP2, CP2_alpha, 'CP2')

            # Global system
            merged = np.hstack((CP1[:, pars_CP1, :], CPz[:, pars_CPz, :], CP2[:, pars_CP2, :]))[:, np.newaxis, :]
            self._train(merged, [0], 'Merged')

            # Voting system
            self._classify_vote(CP1[:, pars_CP1, :], CPz[:, pars_CPz, :], CP2[:, pars_CP2, :], 'Voted')

            # Exit
            self.clean_memory()
            self._first = False
        self._print_final_result()

    __call__ = classify

    def _train(self, channel: np.ndarray, alpha, key):
        train_score_m = np.zeros((self._repeat_val, len(alpha)))
        val_score_m = np.zeros((self._repeat_val, len(alpha)))
        test_score_m = np.zeros((self._repeat_val, len(alpha)))
        if self._merge_on_test:
            test_score_m = np.zeros((1, len(alpha)))

        for i in range(len(alpha)):
            for j in range(self._repeat_val):
                # Train
                train_idx = self._m_train_idx[j]
                train_y = self._m_train_y[j]
                self._model.fit(X=channel[train_idx, i], y=train_y)
                train_score_m[j, i] = self._model.score(X=channel[train_idx, i], y=train_y)

                # Validation
                if self._validation:
                    val_idx = self._m_val_idx[j]
                    val_y = self._m_val_y[j]
                    val_score_m[j, i] = self._model.score(X=channel[val_idx, i], y=val_y)

                # Test
                if self._test and not self._merge_on_test:
                    test_score_m[j, i] = self._model.score(X=channel[self._test_idx, i], y=self._test_y)
            if self._test and self._merge_on_test:
                self._model.fit(X=channel[self._train_idx, i], y=self._train_y)
                test_score_m[0, i] = self._model.score(X=channel[self._test_idx, i], y=self._test_y)

        best_pars = self._print(alpha, train_score_m, val_score_m, test_score_m, key)
        self.memory[key]['train'].append(np.mean(train_score_m, axis=0)[best_pars])
        self.memory[key]['val'].append(np.mean(val_score_m, axis=0)[best_pars])
        self.memory[key]['test'].append(np.mean(test_score_m, axis=0)[best_pars])
        self.memory[key]['param'].append(alpha[best_pars])
        return best_pars

    def _classify_vote(self, CP1, CPz, CP2, key):
        train_score_m = np.zeros((self._repeat_val, len(self._weight)))
        val_score_m = np.zeros((self._repeat_val, len(self._weight)))
        test_score_m = np.zeros((self._repeat_val, len(self._weight)))

        for i, w in enumerate(self._weight):
            for j in range(self._repeat_val):
                # Train
                model_CP1 = self._model_CP1.fit(X=CP1[self._train_idx], y=self._train_y)
                model_CPz = self._model_CPz.fit(X=CPz[self._train_idx], y=self._train_y)
                model_CP2 = self._model_CP2.fit(X=CP2[self._train_idx], y=self._train_y)

                if self._vote_prob:
                    train_score_m[j, i] = self._vote(model_CP1.predict_proba(X=CP1[self._train_idx])[:, -1],
                                                     model_CPz.predict_proba(X=CPz[self._train_idx])[:, -1],
                                                     model_CP2.predict_proba(X=CP2[self._train_idx])[:, -1],
                                                     self._train_y, w)
                else:
                    train_score_m[j, i] = self._vote(model_CP1.predict(X=CP1[self._train_idx]),
                                                     model_CPz.predict(X=CPz[self._train_idx]),
                                                     model_CP2.predict(X=CP2[self._train_idx]), self._train_y, w)

                if self._test_size == 'BLOO':
                    val_idx = self._m_val_idx[j]
                    val_y = self._m_val_y[j]
                    if self._vote_prob:
                        val_score_m[j, i] = self._vote(model_CP1.predict_proba(X=CP1[val_idx])[:, -1],
                                                       model_CPz.predict_proba(X=CPz[val_idx])[:, -1],
                                                       model_CP2.predict_proba(X=CP2[val_idx])[:, -1], val_y, w)
                    else:
                        val_score_m[j, i] = self._vote(model_CP1.predict(X=CP1[val_idx]),
                                                       model_CPz.predict(X=CPz[val_idx]),
                                                       model_CP2.predict(X=CP2[val_idx]), val_y, w)

                # Test
                if self._test:
                    if self._vote_prob:
                        test_score_m[j, i] = self._vote(model_CP1.predict_proba(X=CP1[self._test_idx])[:, -1],
                                                        model_CPz.predict_proba(X=CPz[self._test_idx])[:, -1],
                                                        model_CP2.predict_proba(X=CP2[self._test_idx])[:, -1],
                                                        self._test_y, w)
                    else:
                        test_score_m[j, i] = self._vote(model_CP1.predict(X=CP1[self._test_idx]),
                                                        model_CPz.predict(X=CPz[self._test_idx]),
                                                        model_CP2.predict(X=CP2[self._test_idx]), self._test_y, w)

        best_param = self._print(self._weight, train_score_m, val_score_m, test_score_m, key, weight=True)
        self.memory[key]['train'].append(np.mean(train_score_m, axis=0)[best_param])
        self.memory[key]['val'].append(np.mean(val_score_m, axis=0)[best_param])
        self.memory[key]['test'].append(np.mean(test_score_m, axis=0)[best_param])
        self.memory[key]['param'].append(self._weight[best_param])

    @staticmethod
    def _vote(pred_CP1, pred_CPz, pred_CP2, true, w):
        res = (1 - w) * 0.5 * pred_CP1 + w * pred_CPz + (1 - w) * 0.5 * pred_CP2
        res = np.array([1 if x >= 0.5 else 0 for x in res])
        score = 1 - (np.bitwise_xor(res, true).sum() / len(true))
        return score

    def _print_final_result(self):
        print_c('Final result', 'red', bold=True)
        for channel in self.memory.keys():
            print_c('\t {:}'.format(channel), 'blue', bold=True)
            for key, val in self.memory[channel].items():
                if key != 'param':
                    val = 100 * np.array(val)
                    val_1, val_2 = np.mean(val), np.std(val)
                    print_c('\t\t {:<5}:  {:} {:4.1f} %    {:} {:4.1f} %'.format(key.capitalize(), 'mean', val_1,
                                                                                 'std', val_2), bold=True)
            print('')

    def _print(self, param, train_score_m, val_score_m, test_score_m, key, weight=None):
        # Title
        string = '  {:<7}'.format(key)
        if not self._first:
            train = np.array(self.memory[key]['train'])
            string += '                  Train: {:3.1f} ± {:4.1f} %'.format(100 * train.mean(), 100 * train.std())

            if self._validation:
                val = np.array(self.memory[key]['val'])
                string += '    Val: {:4.1f} ± {:4.1f} %'.format(100 * val.mean(), 100 * val.std())
            if self._test:
                test = np.array(self.memory[key]['test'])
                string += '    Test: {:3.1f} ± {:4.1f} %'.format(100 * test.mean(), 100 * test.std())
        print_c(string, 'blue', bold=True)

        # Best parameter choice
        if self._test_size == 0:
            best_param = np.argmax(np.mean(train_score_m, axis=0))
        else:
            best_param = np.argmax(np.mean(val_score_m, axis=0))

        # Print parameter varying
        for i, param in enumerate(param):
            string = '\t Pars level: {:.2f}'.format(param) if weight is None else '\t Weight = {:.2f}   '.format(param)
            string += '      Train: {:3.1f} ± {:4.1f} %'.format(100 * train_score_m[:, i].mean(),
                                                                100 * train_score_m[:, i].std())
            if self._validation:
                string += '    Val: {:4.1f} ± {:4.1f} %'.format(100 * val_score_m[:, i].mean(),
                                                                100 * val_score_m[:, i].std())
            if self._test:
                string += '    Test: {:3.1f} ± {:4.1f} %'.format(100 * test_score_m[:, i].mean(),
                                                                 100 * test_score_m[:, i].std())

            if best_param == i:
                print_c(string, 'green', bold=True)
            else:
                print(string)
        print('')
        return best_param

    def clean_memory(self):
        self._m_train_idx = []
        self._m_train_y = []
        self._m_val_idx = []
        self._m_val_y = []


for session in [os.path.join(path, folder) for folder in sorted(os.listdir(path), reverse=True)]:
    with open(session) as f:
        data: dict = json.load(f)
    # parsing data from json
    y_true = np.array(data['y_true'])

    # read parsimony levels
    CP1_alpha = data['CP1']['parsimony']
    CPz_alpha = data['CPz']['parsimony']
    CP2_alpha = data['CP2']['parsimony']
    # read features
    CP1 = np.swapaxes(np.array(data['CP1']['features']), 0, 1)
    CPz = np.swapaxes(np.array(data['CPz']['features']), 0, 1)
    CP2 = np.swapaxes(np.array(data['CP2']['features']), 0, 1)

    print_c('\nSessions: {:}'.format(session.split('\\')[-1]), 'blue', bold=True)
    classifier = Classifier(test_size='LOO',  # {0, 'BLOO', int, 'LOO'}
                            val_size='LOO',  # {int / 'LOO'}
                            repeat_test=1000,
                            repeat_val=400,
                            vote_prob=False,
                            merge_on_test=True,
                            y_true=y_true)  # bad coding but no time
    classifier(y_true, CP1, CP1_alpha, CPz, CPz_alpha, CP2, CP2_alpha)

