import pytest
from oligogym.search import RandomSearch
from oligogym.data import DatasetDownloader
from oligogym.models import XGBoostModel
from oligogym.features import KMersCounts
from oligogym.search import RandomSearch
from oligogym import metrics


downloader = DatasetDownloader()
data = downloader.download('siRNAmod',verbose=1)
x_train, x_test, y_train, y_test = data.split(split_strategy='nucleobase')


def kmer():
    featurizer = KMersCounts(k=[1,2,3],modification_abundance=True)
    return featurizer


@pytest.mark.parametrize("featurizer", [kmer()])
def test_random_search(featurizer):
    feat_x_train = featurizer.fit_transform(x_train)
    feat_x_test = featurizer.transform(x_test)
    model = XGBoostModel()
    clf = RandomSearch(model)
    param_distributions = {
        "n_estimators": range(50, 500),
        "max_depth": range(1, 5),
        "learning_rate": [0.01, 0.05, 0.1],
    }
    clf.random_search(param_distributions=param_distributions)
    clf.fit(feat_x_train, y_train)
    y_pred = clf.predict(feat_x_test) 
    assert abs(y_pred[0]) > 0
    metrics_dict = metrics.regression_metrics(y_test,y_pred)
    assert abs(metrics_dict['r2_score']) > 0 
    assert abs(metrics_dict['mean_absolute_error']) > 0