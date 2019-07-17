
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from brainDecode.towardMoabbIntegration.brainDecodeSKLearnWrapper.ShallowFBCSPNet_GeneralTrainer import ShallowFBCSPNet_GeneralTrainer


def bayes_optimize(classifier=None, X_train=None, Y_train=None):
    classifier = ShallowFBCSPNet_GeneralTrainer(nb_epoch=25)

    searches = {            # This should containt Dimension instances (Reals, Integer, or Categorical)
        "n_filters_time":Integer(5, 15),
        "filter_time_length":Integer(50,75),
        "n_filters_spat":Integer(3,10),
        "pool_time_length":Integer(50,80),
        "pool_time_stride":Integer(20,40)
    }

    bayes_optmizer = BayesSearchCV(classifier, searches)

    # now fit and profit
    bayes_optmizer.fit(X_train, Y_train)