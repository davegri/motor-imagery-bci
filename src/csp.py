from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.preprocessing import laplacian
import mne
import numpy as np
from src.pipeline import show_pipeline_steps
from skopt.space import Real, Integer
import json

name = "csp"


class Preprocessor:
    def __init__(self):
        self.epoch_tmin = 1
        self.l_freq = 7
        self.h_freq = 30
        self.do_laplacian = True

    def set_params(self, epoch_tmin, l_freq, h_freq, do_laplacian):
        self.epoch_tmin = epoch_tmin
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.do_laplacian = do_laplacian

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        epochs = mne.filter.filter_data(epochs, 125, self.l_freq, self.h_freq, verbose=False)
        epochs = epochs[:, :, int(125 * self.epoch_tmin):]
        if self.do_laplacian:
            epochs = laplacian(epochs)
        return epochs


class CSP_features:
    def __init__(self):
        self.CSP = mne.decoding.CSP(transform_into="csp_space")

    def set_params(self, n_components, **kwargs):
        self.CSP = mne.decoding.CSP(n_components=n_components, transform_into="csp_space")
        self.params = kwargs
        print()

    def fit(self, data, labels):
        self.CSP.fit(data, labels)
        return self

    def transform(self, epochs):
        components = self.CSP.transform(epochs)
        features = []
        power = components ** 2
        if not self.params.get('total_power') and not self.params.get('log_mean_power') and not self.params.get(
                'entropy') and not self.params.get('var'):
            return power.sum(axis=2)
        if self.params.get('total_power'):
            total_power = power.sum(axis=2)
            features.append(total_power)
        if self.params.get('log_mean_power'):
            log_mean_power = np.log10(power.mean(axis=2))
            features.append(log_mean_power)
        if self.params.get('entropy'):
            entropy = (power * np.log(power)).sum(axis=2)
            features.append(entropy)
        if self.params.get('var'):
            var_sum = np.sum(np.stack([np.var(components[:, i, :], axis=1) for i in range(components.shape[1])]),
                             axis=0)
            var = np.stack([np.var(components[:, i, :], axis=1) / var_sum for i in range(components.shape[1])], axis=1)
            features.append(np.log(var))
        features = np.concatenate(features, axis=1)
        return features


bayesian_search_space = {
    "preprocessing__epoch_tmin": Real(0, 3),
    "preprocessing__l_freq": Real(1, 14),
    "preprocessing__h_freq": Real(15, 50),
    "preprocessing__do_laplacian": [True, False],
    "csp__n_components": Integer(8, 13),
    "csp__log_mean_power": [True, False],
    "csp__total_power": [True, False],
    "csp__entropy": [True, False],
    "csp__var": [True, False],
}

default_hyperparams = {
    "csp__log_mean_power": True,
    "csp__total_power": False,
    "csp__entropy": False,
    "csp__var": False,
    "csp__n_components": 12,
    "preprocessing__do_laplacian": False,
    "preprocessing__epoch_tmin": 0.0,
    "preprocessing__h_freq": 29.077596766188705,
    "preprocessing__l_freq": 9.348780724669755
}

def create_pipeline(hyperparams={}, model=LinearDiscriminantAnalysis):
    if hyperparams:
        hyperparams = {**default_hyperparams, **hyperparams}
    else:
        hyperparams = default_hyperparams
    pipeline = Pipeline(
        [('preprocessing', Preprocessor()), ('csp', CSP_features()), ('model', model())])
    pipeline.set_params(**hyperparams)
    print(f'Creating CSP pipeline: {show_pipeline_steps(pipeline)}')
    print(f'With hyperparams: {json.dumps(hyperparams, indent=4)}')

    return pipeline
