import src.constants
from src.recording import run_session
from src.Workflows import load_epochs_for_subject, create_pipeline_for_subject, record_with_live_retraining
from src.Marker import Marker
import src.csp as csp
from src.data_utils import json_dump
import os

params = {
    "language": "french",
    "use_synthetic_board": True,
    "full_screen": False,
    "wait_on_start": False,
    "subject": "Synthetic",
    "trial_duration": 0.5,
    "trials_per_stim": 2,
    "get_ready_duration": 0.5,
    "calibration_duration": 0.5,
    "display_online_result_duration": 1,
    "retrain_num": 2
}

def test_full(monkeypatch, tmp_path):
    monkeypatch.setattr(src.constants, 'RECORDINGS_DIR', tmp_path)
    params_path = os.path.join(tmp_path, "params.json")
    monkeypatch.setattr(src.constants, 'RECORDING_PARAMS_PATH', params_path)
    json_dump(params, params_path)

    # test recording
    run_session(params)

    # test creating pipeline
    pipeline = create_pipeline_for_subject(params["subject"], pipeline=csp)

    # test recording with live retraining
    record_with_live_retraining(params["subject"], pipeline=csp)

    # finally, test that all epochs exist (from both regular and live retraining sessions)
    epochs, labels = load_epochs_for_subject(params["subject"])
    assert len(epochs) == params["trials_per_stim"] * len(Marker.all()) * 2


