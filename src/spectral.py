from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.preprocessing import laplacian
import mne
from mne_features.univariate import compute_pow_freq_bands
import mne_features.univariate as mnf
import numpy as np
from src.pipeline import show_pipeline_steps, filter_hyperparams_for_pipeline
from skopt.space import Categorical, Integer, Real
import json
import scipy

name = "spectral"

all_freq_bands = {
    "1": [6, 7],
    "2": [7, 11],
    "3": [15, 19],
    "4": [17, 21],
    "5": [24, 26],
}


class Preprocessor:
    def __init__(self):
        self.l_freq = 7
        self.h_freq = 30
        self.do_laplacian = True

    def set_params(self, l_freq=None, h_freq=None, do_laplacian=None):
        if l_freq is not None:
            self.l_freq = l_freq
        if h_freq is not None:
            self.h_freq = h_freq
        if do_laplacian is not None:
            self.do_laplacian = do_laplacian

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        epochs = mne.filter.filter_data(epochs, 125, self.l_freq, self.h_freq, verbose=False)
        if self.do_laplacian:
            epochs = laplacian(epochs)
        # epochs = epochs[:, :3, :]
        return epochs


class FeatureExtractor:
    def __init__(self):
        self.epoch_tmin = 0
        self.freq_bands = {
            "1": [8, 12],
            "2": [12, 30]
        }
        self.params = {
            "freq_1": True,
            "freq_2": True,
            "freq_3": True,
            "freq_4": True,
            "freq_5": True,
        }
        self.n_per_seg = 100
        self.n_overlap = 0.5

    def set_params(self, epoch_tmin, n_per_seg=None, freq_bands=None, n_overlap=None, **kwargs):
        if n_per_seg is not None:
            self.n_per_seg = n_per_seg
        if epoch_tmin is not None:
            self.epoch_tmin = epoch_tmin
        if freq_bands is not None:
            self.freq_bands = freq_bands
        if n_overlap is not None:
            self.n_overlap = n_overlap
        self.params = {**self.params, **kwargs}

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        features = np.zeros((epochs.shape[0], 0))
        freq_bands = {}
        if self.params["freq_1"]:
            freq_bands["1"] = all_freq_bands["1"]
        if self.params["freq_2"]:
            freq_bands["2"] = all_freq_bands["2"]
        if self.params["freq_3"]:
            freq_bands["3"] = all_freq_bands["3"]
        if self.params["freq_4"]:
            freq_bands["4"] = all_freq_bands["4"]
        if self.params["freq_5"]:
            freq_bands["5"] = all_freq_bands["5"]
        if not freq_bands:
            freq_bands = {
                "1": [0, 2]
            }
        sfreq = 125
        tmin = self.epoch_tmin
        imagination_start = int(tmin * sfreq)
        n = len(epochs[0, 0, imagination_start:])
        if self.n_per_seg > n:
            self.n_per_seg = n

        psd_params = {"welch_n_fft": 512, "welch_n_per_seg": self.n_per_seg,
                      "welch_n_overlap": int(self.n_per_seg * self.n_overlap)}
        # band_power = [epoch_band_powers(epoch[:, imagination_start:], sfreq, freq_bands, welch_params) for epoch in
        #               epochs]

        band_power = np.array(
            [compute_pow_freq_bands(sfreq, epoch[:, imagination_start:], self.freq_bands, normalize=True,
                                    psd_params=psd_params) for
             epoch in
             epochs]
        )
        n = len(epochs[0, 0, :imagination_start])
        if self.n_per_seg > n:
            self.n_per_seg = n
        psd_params = {"welch_n_fft": 512, "welch_n_per_seg": self.n_per_seg,
                      "welch_n_overlap": int(self.n_per_seg * self.n_overlap)}
        band_power_calib = np.array(
            [compute_pow_freq_bands(sfreq, epoch[:, :imagination_start], self.freq_bands, normalize=True,
                                    psd_params=psd_params) for
             epoch in
             epochs]
        )
        # feature_funcs = [mnf.compute_mean, mnf.compute_std, mnf.compute_rms, mnf.compute_wavelet_coef_energy,
        #                  mnf.compute_samp_entropy, mnf.compute_app_entropy,
        #                  lambda data: mnf.compute_spect_entropy(125, data), mnf.compute_hjorth_mobility]
        # for func in feature_funcs:
        #     features = np.concatenate(
        #         (features, np.array([func(epoch[:, imagination_start:]) for epoch in epochs])), axis=1)

        sre = band_power_calib / band_power
        features = np.concatenate((band_power, sre), axis=1)
        return features


def epoch_band_powers(epoch, fs, bands, welch_params):
    powers = []
    for band in bands.values():
        powers.append(bandpower(epoch, fs, band[0], band[1], welch_params))
    return np.concatenate(powers)


def bandpower(x, fs, fmin, fmax, welch_params):
    f, Pxx = scipy.signal.welch(x, fs=fs, **welch_params)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return np.trapz(Pxx[:, ind_min: ind_max], f[ind_min: ind_max])


class CategoricalList(Categorical):
    def __init__(self, categories, **categorical_kwargs):
        super().__init__(self._convert_hashable(categories), **categorical_kwargs)

    def _convert_hashable(self, list_of_lists):
        return [self._HashableListAsDict(list_)
                for list_ in list_of_lists]

    class _HashableListAsDict(dict):
        def __init__(self, arr):
            self.update({i: val for i, val in enumerate(arr)})

        def __hash__(self):
            return hash(tuple(sorted(self.items())))

        def __repr__(self):
            return str(list(self.values()))

        def __getitem__(self, key):
            return list(self.values())[key]


bayesian_search_space = {
    "feature_extraction__epoch_tmin": Real(0.5, 3),
    "feature_extraction__n_per_seg": Integer(50, 300),
    "feature_extraction__n_overlap": Real(0, 0.9),
    "preprocessing__do_laplacian": [True, False],
    "feature_extraction__freq_1": [True, False],
    "feature_extraction__freq_2": [True, False],
    "feature_extraction__freq_3": [True, False],
    "feature_extraction__freq_4": [True, False],
    "feature_extraction__freq_5": [True, False],
}

default_hyperparams = {
    "preprocessing__l_freq": 6,
    "preprocessing__h_freq": 30,
    "preprocessing__do_laplacian": False,
    "feature_extraction__epoch_tmin": 1.9,
}

def create_pipeline(hyperparams=None, model=LinearDiscriminantAnalysis):
    if hyperparams:
        hyperparams = {**default_hyperparams, **hyperparams}
    else:
        print("no hyperparams passed, using default")
        hyperparams = default_hyperparams

    pipeline = Pipeline(
        [('preprocessing', Preprocessor()), ('feature_extraction', FeatureExtractor()), ('model', model())])
    hyperparams = filter_hyperparams_for_pipeline(hyperparams, pipeline)
    print("Creating pipeline with hyperparams")
    pipeline.set_params(**hyperparams)
    print(f'Creating spectral pipeline: {show_pipeline_steps(pipeline)}')
    print(json.dumps(hyperparams, indent=4))
    return pipeline
