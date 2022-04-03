import src.constants
from src.recording import run_session
from src.Workflows import load_epochs_for_subject, create_pipeline_for_subject
from src.Marker import Marker
import src.csp as csp

params = {
    "use_synthetic_board": True,
    "full_screen": False,
    "wait_on_start": False,
    "subject": "Synthetic",
    "trial_duration": 0.5,
    "trials_per_stim": 2,
    "get_ready_duration": 0.5,
    "calibration_duration": 0.5,
    "display_online_result_duration": 1,
    "retrain_num": 10
}


def test_full(monkeypatch, tmp_path):
    monkeypatch.setattr(src.constants, 'RECORDINGS_DIR', tmp_path)
    run_session(params)
    epochs, labels = load_epochs_for_subject(params["subject"])
    assert len(epochs) == params["trials_per_stim"] * len(Marker.all())
    pipeline = create_pipeline_for_subject(params["subject"], pipeline=csp)
