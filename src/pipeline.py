import mne
from src.Marker import Marker
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict
from src.data_utils import load_recordings
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mne.set_log_level('warning')
import matplotlib.pyplot as plt

def evaluate_pipeline(pipeline: Pipeline, epochs, labels, n_splits=10, n_repeats=1):
    n_splits = get_n_splits(n_splits, labels)
    print(f'Evaluating pipeline performance ({n_splits} splits, {n_repeats} repeats, {len(labels)} epochs)...')
    results = cross_validate(pipeline, epochs, labels, cv=cross_validation(n_splits, n_repeats),
                             return_train_score=True, n_jobs=-1)
    return results
    # print(format_results(results))
    # pred = cross_val_predict(pipeline, epochs, labels, cv=4, n_jobs=-1)
    # con_matrix = confusion_matrix(labels, pred, normalize="true")
    # disp = ConfusionMatrixDisplay(con_matrix, display_labels=["Left/Right", "Tounge", "Idle"])
    # disp.plot()
    # plt.show()

def format_results(results):
    line1 = f'\nTraining Accuracy: \n mean: {np.round(np.mean(results["train_score"]), 2)} \n std: {np.round(np.std(results["train_score"]), 3)}'
    line2 = f'\nTesting Accuracy: \n mean: {np.round(np.mean(results["test_score"]), 2)} \n std: {np.round(np.std(results["test_score"]), 3)}'
    return f'{line1} \n {line2}'

def get_n_splits(n_splits, labels):
    _, label_counts = np.unique(labels, return_counts=True)
    n_splits = 2 if n_splits >= min(label_counts) else n_splits
    return n_splits


def filter_hyperparams_for_pipeline(hyperparams, pipeline: Pipeline):
    return {key: hyperparams[key] for key in hyperparams if
            key.split("__")[0] in pipeline.get_params().keys()}


def show_pipeline_steps(pipeline: Pipeline):
    return " => ".join(list(pipeline.named_steps.keys()))


def cross_validation(n_splits=10, n_repeats=1):
    return RepeatedStratifiedKFold(n_splits=n_splits)


def bayesian_opt(epochs, labels, pipeline: Pipeline):
    pipe = pipeline.create_pipeline()
    n_splits = get_n_splits(10, labels)
    opt = BayesSearchCV(
        pipe,
        pipeline.bayesian_search_space,
        verbose=20,
        n_iter=5,
        cv=cross_validation(n_splits=n_splits),
        n_jobs=-1,
    )
    opt.fit(epochs, labels)

    print("Best parameter (CV score=%0.3f):" % opt.best_score_)
    print(opt.best_params_)
    return opt.best_params_, opt.best_score_, opt.cv_results_["std_test_score"][opt.best_index_]


def grid_search_pipeline_hyperparams(epochs, labels, pipeline: Pipeline):
    gs = GridSearchCV(pipeline.create_pipeline(), pipeline.grid_search_space, cv=cross_validation(), n_jobs=-1,
                      verbose=10,
                      error_score="raise")
    gs.fit(epochs, labels)
    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
    print(gs.best_params_)
    return gs.best_params_


def get_epochs(raws, trial_duration, calibration_duration, markers=[Marker.LEFT, Marker.RIGHT, Marker.IDLE],
               reject_bad=False,
               on_missing='warn'):
    reject_criteria = dict(eeg=100e-6)  # 100 ??V
    flat_criteria = dict(eeg=1e-6)  # 1 ??V

    epochs_list = []
    for raw in raws:
        events = mne.find_events(raw)

        epochs = mne.Epochs(raw, events, markers, tmin=-calibration_duration, tmax=trial_duration, picks="data",
                            on_missing=on_missing, baseline=None)
        epochs_list.append(epochs)
    epochs = mne.concatenate_epochs(epochs_list)

    # running get data triggers dropping of epochs, we want to make sure this happens now so that the labels are
    # consistent with the epochs
    epochs.get_data()
    labels = epochs.events[:, -1]
    print(f'Found {len(labels)} epochs')

    return epochs, labels


if __name__ == "__main__":
    raw, params = load_recordings("Ori")
