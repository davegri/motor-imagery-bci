from pathlib import Path

import mne.preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from src.recording import run_session
from src.pipeline import evaluate_pipeline, get_epochs, bayesian_opt
from src.data_utils import load_recordings, load_hyperparams, save_hyperparams, load_rec_params
import src.spectral as spectral
import src.csp as csp
import sklearn
import pandas as pd

models = [
    {"model": sklearn.discriminant_analysis.LinearDiscriminantAnalysis, "search_space": {
    }},
    {"model": sklearn.svm.SVC, "search_space": {
        'model__C': Real(1e-6, 1e+1, 'log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, 'log-uniform'),
        'model__degree': Integer(1, 12),
        'model__kernel': ['linear', 'poly', 'rbf'],
    }},
    {"model": sklearn.ensemble.RandomForestClassifier, "search_space": {
        "model__max_features": Integer(1, 8),
        "model__n_estimators": Integer(1, 200),
        "model__max_depth": Integer(1, 10),
    }},
    {"model": sklearn.ensemble.GradientBoostingClassifier, "search_space": {
        "model__max_features": Integer(1, 8),
        "model__n_estimators": Integer(1, 200),
        "model__max_depth": Integer(1, 13),
        'model__learning_rate': Real(0.00005, 0.9, prior="log-uniform"),
    }},
    {"model": sklearn.ensemble.AdaBoostClassifier, "search_space": {
        "model__n_estimators": Integer(1, 200),
        'model__learning_rate': Real(0.00005, 0.9, prior="log-uniform"),
    }},
    {"model": sklearn.neighbors.KNeighborsClassifier, "search_space": {
        'model__leaf_size': Integer(10, 100),
        'model__n_neighbors': Integer(5, 25),
    }},
]


def record_and_create_pipeline(pipeline=spectral):
    rec_params = load_rec_params()
    run_session(rec_params)
    create_pipeline_for_subject(rec_params["subject"], pipeline)
    return pipeline


def create_pipeline_for_subject(subject, pipeline=spectral, choose=False):
    """
    Create pipline of type in: ["spectral", "csp"]
    """
    print(f'Creating pipeline for subject {subject}...')
    epochs, labels = load_epochs_for_subject(subject, choose)
    hyperparams = load_hyperparams(subject, pipeline.name)

    pipe = pipeline.create_pipeline(hyperparams)
    pipe.fit(epochs, labels)

    return pipe


def find_best_hyperparams_for_subject(subject=None, pipeline=spectral, choose=False):
    epochs, labels = load_epochs_for_subject(subject, choose)
    best_hyperparams = bayesian_opt(epochs, labels, pipeline)
    save_hyperparams(best_hyperparams, pipeline.name, subject)


def find_best_pipeline_for_subject(subject=None, pipeline=csp):
    Path(f'../{subject}_pipeline_results').mkdir(exist_ok=True)
    epochs, labels = load_epochs_for_subject(subject)
    results = []
    for model in models:
        pipe = pipeline.create_pipeline(model=model["model"])
        opt = BayesSearchCV(
            pipe,
            {
                **pipeline.bayesian_search_space,
                **model["search_space"],
            }
            ,
            verbose=0,
            n_iter=100,
            cv=RepeatedStratifiedKFold(n_repeats=1, n_splits=10),
            n_jobs=-1,
        )
        opt.fit(epochs, labels)
        print("Best parameter (CV score=%0.3f):" % opt.best_score_)
        print(opt.best_params_)
        result = {
            **opt.best_params_,
            "pipeline": pipeline.name,
            "model": model["model"].__name__,
            "accuracy": opt.best_score_,
            "std": opt.cv_results_["std_test_score"][opt.best_index_],
        }
        results.append(result)
    pd.DataFrame(results).to_csv(f'../{subject}_pipeline_results/{pipeline.name}_results.csv')


def record_with_live_retraining(subject, pipeline=spectral, choose=False):
    epochs, labels = load_epochs_for_subject(subject, choose=choose)
    rec_params = load_rec_params()
    run_session(rec_params, retrain_pipeline=pipeline, epochs=epochs, labels=labels)
    create_pipeline_for_subject(rec_params["subject"], pipeline=pipeline)


def load_epochs_for_subject(subject, choose=False):
    raws, rec_params = load_recordings(subject, choose)
    epochs, labels = get_epochs(raws, rec_params["trial_duration"], rec_params["calibration_duration"],
                                reject_bad=not rec_params['use_synthetic_board'])
    return epochs.get_data(), labels


if __name__ == "__main__":
    # find_best_hyperparams_for_subject("David7", pipeline=spectral)
    record_with_live_retraining("David7", csp)
