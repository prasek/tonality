#
# classify.py
#
# Auto feature selection and SVC training
#

from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV, RFE
from audio import LABEL_MUSIC, LABEL_SPEECH ,debug
import numpy as np

def auto_select_features(df, labels):
    """
    Uses recursive feature elimination with cross validation to select the best features

    :arg df: pandas DataFrame with feature vectors
    :arg labels: labels for data frame rows
    :return: best_features, fsel, features, feature_labels
     - best_features: best feature ids
     - features: all feature ids for the selected groups
     - feature_ranking: rank values for features, keyed to the features list
     - grid_scores: grid scores for recursive feature elimination
    """
    if df is None:
        return None, None, None, None

    X = df.values
    y = labels

    # RFECV for best feature selection using cross validation, not just simple ranking
    svc = SVC(kernel="linear", C=1)
    fsel = RFECV(estimator=svc, step=1, cv=StratifiedKFold(3), scoring='accuracy')
    fsel.fit(X, y)

    best_features = df.columns[fsel.support_]

    lb = LabelBinarizer()
    yB = lb.fit_transform(y)

    #RFE for ranking the features for display text (so we can see the rank of the auto selected features)
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(svc, n_features_to_select=1)
    rfe.fit(X, yB.ravel())

    #do as Python lists so more friendly for the GUI, since they have an index function vs. numpy arrays which do not
    best_features = [f for f in best_features]
    grid_scores = fsel.grid_scores_
    features = [f for f in df.columns.values]
    feature_ranking = rfe.ranking_

    return best_features, features, feature_ranking, grid_scores


def train_svc(df, labels, **kwargs):
    """
    :arg df: pandas DataFrame with feature vectors
    :arg labels: labels for data frame rows
    :kwarg status: optional status callback
    :return: trained svc for classifying audio
    """
    # optional callback function to report status back to the caller since this is a long running operation
    status = kwargs.get('status', lambda msg: debug(msg))

    features = df.columns.values

    status('--------------------------------------')
    status('SELECTED FEATURES: %d' % len(features))
    status('--------------------------------------')

    for feature in features:
        status('  - %s' % feature)

    if df is None:
        status('No features found. Click Extract Features.')
        return

    svc = SVC()

    ids = df.index.values
    X = df.values
    y = labels

    # initial results with all 52 features
    skf = StratifiedKFold(3)
    skf.get_n_splits((X, y))

    status('')
    status('--------------------------------------')
    status('GROUND TRUTH: IS MUSIC?')
    status('--------------------------------------')
    for i in np.arange(len(ids)):
        status('# %s: %s' % (ids[i].rjust(10), y[i]))

    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ids_train, ids_test = ids[train_index], ids[test_index]
        svc.fit(X_train, y_train)

        status('')
        status('--------------------------------------')
        status('TEST RESULTS - FOLD: %d' % fold)
        status('--------------------------------------')
        fold += 1
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in np.arange(len(ids_test)):
            id = ids_test[i]
            expected = y_test[i]
            result = svc.predict([df.ix[id]])[0]
            if result == expected:
                status('# %s: got %s, CORRECT: expected %s' % (id.rjust(10), result.rjust(3), expected))
            else:
                status('# %s: got %s,   WRONG: expected %s' % (id.rjust(10), result.rjust(3), expected))
            if expected == result == LABEL_MUSIC:
                tp += 1
            elif expected == result == LABEL_SPEECH:
                tn += 1
            elif result == LABEL_MUSIC:
                fp += 1
            elif result == LABEL_SPEECH:
                fn += 1

        mp = tp / (tp + fp)
        mr = tp / (tp + fn)

        status('')
        status('Accuracy:  %0.2f' % svc.score(X_test, y_test))
        status('Precision: %0.2f = %d / (%d + %d)' % (mp, tp, tp, fp))
        status('Recall:    %0.2f = %d / (%d + %d)' % (mr, tp, tp, fn))

    status('')
    status('--------------------------------------')
    status('Results Summary')
    status('--------------------------------------')
    lb = LabelBinarizer()
    yB = lb.fit_transform(y)

    scores = cross_val_score(svc, X, y, cv=StratifiedKFold(3), scoring='accuracy')
    status("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    for score in scores:
        status(" - Accuracy: %0.2f" % score)

    scores = cross_val_score(svc, X, yB.ravel(), cv=StratifiedKFold(3), scoring='precision')
    status("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    for score in scores:
        status(" - Precision: %0.2f" % score)

    scores = cross_val_score(svc, X, yB.ravel(), cv=StratifiedKFold(3), scoring='recall')
    status("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    for score in scores:
        status(" - Recall: %0.2f" % score)

    return svc

