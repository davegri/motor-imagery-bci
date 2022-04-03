import os
import json
import datetime
from src.constants import *
from pathlib import Path
from tkfilebrowser import askopendirnames
import mne
import pickle

SYNTHETIC_SUBJECT_NAME = "Synthetic"


def now_datestring():
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


def save_raw(raw, rec_params):
    folder_path = create_session_folder(rec_params['subject'])
    raw.save(os.path.join(folder_path, "raw.fif"))
    json_dump(rec_params, os.path.join(folder_path, "params.json"))
    return os.path.basename(folder_path)


def create_session_folder(subj):
    folder_name = f'{now_datestring()}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)
    return folder_path


def load_rec_params():
    rec_params = json_load(RECORDING_PARAMS_PATH)

    if rec_params["use_synthetic_board"]:
        rec_params["subject"] = SYNTHETIC_SUBJECT_NAME

    return rec_params


def load_recording(rec_folder):
    return mne.io.read_raw_fif(os.path.join(RECORDINGS_DIR, rec_folder, 'raw.fif'))


def get_file_date(filename):
    return filename.split("--")[0]


def get_subject_rec_folders(subj):
    all_recs = os.listdir(RECORDINGS_DIR)
    subj_recs = [rec for rec in all_recs if rec.split("_")[-1] == subj]
    return subj_recs


def load_recordings(subj="", choose=False):
    """
    Load all the recordings, all from the most recent day.
    """
    if choose:
        subj_recs_recent = askopendirnames(initialdir=RECORDINGS_DIR)
    else:
        print(f'Loading recordings for subject {subj}...')
        subj_recs = get_subject_rec_folders(subj)

        if len(subj_recs) == 0:
            raise ValueError(f'No recordings found for subject: {subj}')

        most_recent_rec_date = get_file_date(sorted(subj_recs)[-1])
        subj_recs_recent = [rec for rec in subj_recs if get_file_date(rec) == most_recent_rec_date]

        if len(subj_recs_recent) != len(subj_recs):
            print(f'There are recordings taken from multiple days, using the most recent {most_recent_rec_date}')

    raws = [load_recording(rec) for rec in subj_recs_recent]

    # When multiple recordings are loaded, the recording_params.json is taken from the first recording
    with open(os.path.join(RECORDINGS_DIR, subj_recs_recent[0], 'params.json')) as file:
        rec_params = json.load(file)

    return raws, rec_params


def save_pipeline(pipeline, subject):
    save_path = os.path.join(PIPELINES_DIR, f'{now_datestring()}_{subject}_pipeline.pickle')
    pickle_dump(pipeline, save_path)


def load_pipeline(subj):
    all_pipelines = os.listdir(PIPELINES_DIR)
    subj_pipelines = sorted([p for p in all_pipelines if p.split("_")[1] == f"{subj}.pickle"])
    latest_pipeline = subj_pipelines[-1]
    load_path = os.path.join(PIPELINES_DIR, latest_pipeline)
    return pickle_load(load_path)


def save_hyperparams(hyperparams, subject, pipeline):
    save_path = os.path.join(HYPERPARAMS_DIR, f'{now_datestring()}_{pipeline}_{subject}_hyperparams.json')
    json_dump(hyperparams, save_path)


def load_hyperparams(subject, pipeline):
    print(f'Loading hyperparams for subject {subject} and pipeline {pipeline}...')
    all_hyperparams = os.listdir(HYPERPARAMS_DIR)
    subj_hyperparams = sorted(
        [p for p in all_hyperparams if (p.split("_")[1] == subject and p.split("_")[2] == pipeline)]
    )

    if len(subj_hyperparams) == 0:
        print(f'No hyperparams found for subject {subject}')
        return None

    if len(subj_hyperparams) > 1:
        print("Multiple hyperparam files found, taking most recent")

    latest_hyperparams = subj_hyperparams[-1]
    hyperparams = json_load(os.path.join(HYPERPARAMS_DIR, latest_hyperparams))
    print(json.dumps(hyperparams, indent=4))
    return hyperparams


def json_load(load_path):
    with open(load_path) as file:
        return json.load(file)


def json_dump(obj, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def pickle_load(load_path):
    with open(load_path, "rb") as file:
        return pickle.load(file)


def pickle_dump(obj, save_path):
    with open(save_path, "wb") as file:
        pickle.dump(obj, file)
