import json
import os.path

import edifice as ed
from edifice.components.forms import Form
from src.data_utils import load_rec_params, save_rec_params, load_recording, load_hyperparams
from edifice import View, Button, alert, Label, TabView, Dropdown
from src.recording import run_session
from src.Workflows import create_pipeline_for_subject, load_epochs_for_subject
from src.health_check import health_check
import mne
from PyQt5.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView
from src.pipeline import get_epochs, show_pipeline_steps, evaluate_pipeline, format_results
from src.constants import RECORDINGS_DIR
from src import csp, spectral
from src.Marker import Marker
import numpy as np

button_style = {"margin": 10, "padding": 10, "font-size": 20}
pipeline_modules = [csp, spectral]
pipelines_modules_dict = {module.name: module for module in pipeline_modules}


class App(ed.Component):
    def __init__(self):
        super().__init__()
        rec_params = load_rec_params()
        self.rec_params_state = ed.StateManager(rec_params)
        self.pipeline_name = "csp"
        self.hyperparams_state = {module.name: ed.StateManager(module.default_hyperparams) for module in
                                  pipeline_modules}
        self.epochs, self.labels = load_epochs_for_subject(self.rec_params["subject"])
        self.pipeline = None
        self.eval_results = None

    def start_recording(self, e):
        run_session(self.rec_params_state)

    def create_pipeline(self, e):
        pipeline = self.pipeline_module.create_pipeline(self.hyperparams_state[self.pipeline_name].as_dict())
        pipeline.fit(self.epochs, self.labels)
        self.set_state(pipeline=pipeline)

    def eval_pipeline(self, e):
        results = evaluate_pipeline(self.pipeline, self.epochs, self.labels)
        self.set_state(eval_results=results)

    @property
    def rec_params(self):
        return self.rec_params_state.as_dict()

    def save_recording_params(self, e):
        save_rec_params(self.rec_params)
        alert("params saved")

    def health_check(self, e):
        health_check()

    def load_epochs_from_subject(self, e):
        epochs, labels = load_epochs_for_subject(self.rec_params["subject"])
        self.set_state(epochs=epochs, labels=labels)

    @property
    def pipeline_module(self):
        return pipelines_modules_dict[self.pipeline_name]

    def folder_dialog(self, start_path):
        file_dialog = QFileDialog(None, None, start_path)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            paths = file_dialog.selectedFiles()
            return paths

    def load_epochs_from_recording(self, e):
        rec_folders = self.folder_dialog(RECORDINGS_DIR)
        rec_folders = [folder for folder in rec_folders if os.path.isfile(os.path.join(folder, 'raw.fif'))]
        raws = [mne.io.read_raw_fif(os.path.join(rec_folder, 'raw.fif')) for rec_folder in rec_folders]

        # When multiple recordings are loaded, the recording_params.json is taken from the first recording
        with open(os.path.join(rec_folders[0], 'params.json')) as file:
            rec_params = json.load(file)

        epochs, labels = get_epochs(raws, rec_params["trial_duration"], rec_params["calibration_duration"],
                                    reject_bad=not rec_params['use_synthetic_board'])
        self.set_state(epochs=epochs, labels=labels)

    def set_pipeline_module(self, pipeline_name):
        default_hyperparams = pipelines_modules_dict[pipeline_name].default_hyperparams
        self.set_state(pipeline_name=pipeline_name)

    def render(self):
        epochs_msg = f'Loaded {len(self.epochs)} epochs' if len(self.epochs) else "no epochs loaded"
        pipeline_msg = f'loaded pipeline: \n {show_pipeline_steps(self.pipeline)}' if self.pipeline else ""
        results_msg = f'{format_results(self.eval_results)}' if self.eval_results else ""
        epochs_labels = [Marker(label).name for label in self.labels]
        unique, counts = np.unique(epochs_labels, return_counts=True)
        epoch_class_count_labels = []
        hyperparam_forms = {name: Form(self.hyperparams_state[name]) for name in self.hyperparams_state.keys()}
        for name, count in zip(unique, counts):
            epoch_class_count_labels.append(Label(f'{name} : {count}', style=button_style))
        return ed.Window()(
            TabView(["Recording", "Pipeline Creation"])(
                View(layout="row", style={"margin": 20})(
                    View(layout="column", style={"margin": 20})(
                        Form(self.rec_params_state),
                        Button("Save Recording Params", on_click=self.save_recording_params,
                               style=button_style),
                    ),
                    View(layout="column", style={"margin": 20, "align": "top"})(
                        Button("Health Check", on_click=self.health_check, style=button_style),
                        Button("Start Recording", on_click=self.start_recording, style=button_style),
                    ),
                ),
                View(layout="column", style={"margin": 20, "align": "top"})(
                    Label(f'Subject: {self.rec_params["subject"]}', style={**button_style, "font-size": 20}),
                    View(layout="row")(
                        View(layout="column", style={"margin": 20, "align": "top"})(
                            Button("Load epochs for subject", on_click=self.load_epochs_from_subject,
                                   style=button_style),
                            Button("Load epochs from recording folders",
                                   on_click=self.load_epochs_from_recording,
                                   style=button_style),
                            Label(epochs_msg, style=button_style),
                            *epoch_class_count_labels
                        ),
                        View(layout="column", style={"margin": 20, "align": "top"})(
                            Button("Load hyperparams for subject", on_click=self.load_epochs_from_subject,
                                   style=button_style),
                            View(layout="column")(
                                Label("csp hyperparams:", style=button_style),
                                hyperparam_forms["csp"],
                            ),
                            View(layout="column")(
                                Label("spectral hyperparams:", style=button_style),
                                hyperparam_forms["spectral"],
                            )
                        ),
                        View(layout="column", style={"margin": 20, "align": "top"})(
                            Dropdown(
                                selection=self.pipeline_module.name,
                                options=[module.name for module in pipeline_modules],
                                on_select=self.set_pipeline_module),
                            Button("Create Pipeline", on_click=self.create_pipeline, style=button_style),
                            Label(pipeline_msg, style=button_style),
                            Button("Evaluate Pipeline", on_click=self.eval_pipeline, style=button_style),
                            Label(results_msg, style=button_style),
                        ),
                    )
                )
            )
        )


if __name__ == "__main__":
    ed.App(App()).start()
