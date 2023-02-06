import os
import time
import itertools

import numpy as np

from utils import print_c, mean, select_data, reset_outer_eval, read_parameters, read_data, smoothing_scores, \
    update_table, prints, update_experience_values, CV_choice, list_not_contain_empty
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC


# Parameters
read_only = True
do_permutation_test = False
shuffle = True
smoothing = True
testing_split = ['LOO', 5, 2][0]
validation_split = ['LOO', 7, 5][0]
k_feat = 1
k_chan = 1


# Selection
select_stim = [None, ['target'], ['novel'], ['standard']][0]
select_channel = [None, [['CP1']], [['CPz']], [['CP2']], [['CPz'], ['CP1'], ['CP2']]][0]
select_feature = [None, [['argmin']], [['my_mean', 'my_argmin']]][0]


never_use = []
print_c(f'{never_use = }', 'yellow')
print_c(f'{testing_split = }', 'yellow')
print_c(f'{validation_split = }', 'yellow')


# Classifier
clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, covariance_estimator=None)
# clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], covariance_estimator=None)
# clf = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.0)
# clf = LinearSVC(tol=1e-8, class_weight='balanced', max_iter=1e5)


# read and set parameters
path = r"./generated SDF"
param_files = ['DFG_parameters.json', 'preprocessing_parameters.json']
session = os.listdir(path)[0]
# session = ['2022-03-02 18;00'][0]  # if you redo many sessions you should probably use this instead of the above
path_session = os.path.join(path, session)
param = read_parameters(path_session, param_files)
data_df = read_data(session, path_session, read_only, param, never_use)
category = data_df.groupby(by='subject')['category'].apply('first').to_numpy()

# init parameters
channels = param['channel_picks']
if 'VEOG' in channels:
    channels.remove('VEOG')
n_subj = len(data_df['subject'].unique())
n_repeat = 33333 if do_permutation_test or (shuffle and testing_split != 'LOO') else 1
parsimony = np.array(param['selection'] if param['selection'] is not None else param.get('selection_alpha', None))
parsimony = np.arange(0.04, 1.02, 0.02)

k_feat = k_feat if select_feature is None else len(select_feature[0])
k_chan = k_chan if select_channel is None else len(select_channel[0])
select_stim = data_df['stim'].unique() if select_stim is None else select_stim
select_channel = list(itertools.combinations(channels, k_chan)) if select_channel is None else select_channel
select_feature = list(itertools.combinations(data_df['feature'].unique(), k_feat)) if select_feature is None else select_feature

fold_test = CV_choice(testing_split, shuffle=shuffle)
fold_validation = CV_choice(validation_split)
test_p_value, test_val, learning_val, validation_val, vote_test_val, vote_learning_val = [], [], [], [], [], []


for _ in range(n_repeat):
    y = category
    if do_permutation_test:
        y = np.random.permutation(category)
        print_c(r'/!\ Permutation test label permutation enabled.', 'red', bold=True)
        print('\tShuffle quality: {:.1f} %'.format(np.logical_xor(y, category).mean() * 100))

    timed = 0
    experience_values = {'validation_acc': np.zeros((3,), dtype=np.float64),
                         'channel': [],
                         'stim': [],
                         'predicted_category': np.zeros((3, n_subj)),
                         'learning_category': [[], [], []],
                         'test': 0}

    grouped = data_df.groupby(by=['stim', 'feature', 'parsimony'])

    # store folds splitting
    folds = []
    for idx_learning, subj_test in fold_test.split(np.zeros(n_subj), y):
        folds.append((idx_learning, subj_test))

    for i, stim in enumerate(select_stim):
        for j, channel in enumerate(select_channel):
            channel = list(channel)

            for k, feature in enumerate(select_feature):
                feature = list(feature)
                tic = time.time()
                outer_m = reset_outer_eval(n_subj=n_subj, parsimony=parsimony)
                table = [['Parsimony (%)'], ['Train (%)'], ['Validation (%)'], ['Remark']]
                for idx_learning, subj_test in folds:
                    val_splits = list(fold_validation.split(idx_learning, y[idx_learning]))
                    for pars_idx, pars in enumerate(parsimony):
                        X = select_data(grouped, stim, feature, pars, channel)
                        # this is added to not bias the average of the resulting scores even if the number of
                        # subjects in the folds are not the same
                        counter = -1
                        # For each test subject out and level of parsimony perform the validation and save it in inner-mem
                        inner_m = {'train': [], 'validation': []}  # inner scores memory
                        for temp_idx_train, temp_subj_validation in val_splits:
                            counter += 1
                            idx_train = idx_learning[temp_idx_train]
                            subj_validation = idx_learning[temp_subj_validation]
                            clf.fit(X[idx_train, :], y[idx_train])
                            inner_m['train'].append(clf.score(X[idx_train, :], y[idx_train]) * 100 * len(idx_train))
                            inner_m['validation'].append(clf.score(X[subj_validation, :], y[subj_validation]) * 100 * len(subj_validation))
                        outer_m['train'][subj_test, pars_idx] = sum(inner_m['train']) / (len(idx_learning) * counter)
                        outer_m['validation'][subj_test, pars_idx] = sum(inner_m['validation']) / len(idx_learning)

                    # For the test subject out: select the best performing parsimony level obtained on validation set
                    smoothed_validation_score = smoothing_scores(outer_m['validation'][subj_test[0], :], smoothing)
                    best_pars = parsimony[np.argmax(smoothed_validation_score)]

                    X = select_data(grouped, stim, feature, best_pars, channel)
                    clf.fit(X[idx_learning, :], y[idx_learning])  # use learning set to learn instead of training only
                    outer_m['learning'].append(clf.score(X[idx_learning, :], y[idx_learning]) * 100)  # learning score
                    outer_m['learning_category'].append(clf.predict(X[idx_learning, :]))  # learning category for vote learning evaluation
                    outer_m['selected validation'].append(np.max(outer_m['validation'][subj_test, :]))
                    outer_m['test'].append(clf.score(X[subj_test, :], y[subj_test]) * 100)  # test score
                    outer_m['test_category'][subj_test] = clf.predict(X[subj_test, :])  # predicted label

                update_table(table, parsimony, outer_m, highlight_above=75)
                update_experience_values(experience_values, outer_m, channel, stim)
                # print(experience_values['learning_category'])  # here
                timed = time.time() - tic
                prints(i, j, k, stim, select_stim, channel, select_channel, feature, select_feature, table, outer_m,
                       timed, highlight_above=75)

    print('\nBest validation accuracy obtained for the session {:} is: {:.1f} % for the stim <{:}> and channel {:}'
          ' using <{:}> features and <{:}> channels'.format(session, max(experience_values['validation_acc']),
                                                            experience_values['stim'], experience_values['channel'], k_feat, k_chan))
    print('\t3 best validation accuracies: {:}'.format([float('{:.1f}'.format(val)) for val in experience_values['validation_acc']]))

    # voted part
    if all(map(lambda x: len(x) != 0, experience_values['learning_category'])):
        learning_vote_acc = 0
        for i, (idx_learning, subj_test) in enumerate(folds):
            voted = sum(map(lambda x: x[i], experience_values['learning_category']))
            voted = np.array([1 if x > 1.5 else 0 for x in voted])
            score = 100 - np.logical_xor(voted, category[idx_learning]).mean() * 100
            learning_vote_acc += score * len(idx_learning)
        voted = np.sum(experience_values['predicted_category'], axis=0)
        voted = np.array([1 if x > 1.5 else 0 for x in voted])
        vote_test_val.append(float('{:.1f}'.format(100 - np.logical_xor(voted, category).mean() * 100)))
        vote_learning_val.append(float('{:.1f}'.format(learning_vote_acc / sum(map(len, experience_values['learning_category'][0])))))
        print('\t\tVoted accuracy learning = {:.1f} % \t test = {:.1f}'.format(vote_learning_val[-1], vote_test_val[-1]))
        print_c('\nLearning vote {:}'.format(vote_learning_val), 'cyan', bold=True)
        print_c('Testing  vote {:}\n'.format(vote_test_val), 'cyan', bold=True)

    if not do_permutation_test:
        learning_val.append(float('{:.1f}'.format(mean(outer_m['learning']))))
        validation_val.append(float('{:.1f}'.format(mean(outer_m['selected validation']))))
        test_val.append(float('{:.1f}'.format(experience_values['test'])))
        print_c('Learning      {:}'.format(learning_val), 'cyan', bold=True)
        print_c('Validation    {:}'.format(validation_val), 'cyan', bold=True)
        print_c('Testing       {:}\n'.format(test_val), 'cyan', bold=True)

    else:
        test_p_value.append(float('{:.1f}'.format(experience_values['test'])))
        print_c('Compute p val {:}\n\n'.format(test_p_value), 'cyan', bold=True)

