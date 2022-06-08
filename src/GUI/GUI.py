import edifice as ed
from edifice.components.forms import Form
from src.data_utils import load_rec_params, save_rec_params
from edifice import View, Button, alert
from src.recording import run_session
from src.Workflows import create_pipeline_for_subject
from src.health_check import health_check

button_style = {"margin": 10, "padding": 10, "font-size": 20}

class App(ed.Component):
    def __init__(self):
        super().__init__()
        rec_params = load_rec_params()
        self.rec_params_state = ed.StateManager(rec_params)

    def start_recording(self, e):
        run_session(self.rec_params_state.as_dict())

    def create_pipeline(self, e):
        subject = self.rec_params["subject"]
        create_pipeline_for_subject(subject)
        alert("pipeline created")

    @property
    def rec_params(self):
        return self.rec_params_state.as_dict()

    def save_recording_params(self, e):
        save_rec_params(self.rec_params)
        alert("params saved")

    def health_check(self, e):
        health_check()

    def render(self):
        return ed.Window()(
            View(layout="row", style={"margin": 20})(
                View(layout="column", style={"margin": 20})(
                    Form(self.rec_params_state),
                    Button("Save Recording Params", on_click=self.save_recording_params, style=button_style),
                ),
                View(layout="column", style={"margin": 20, "align": "top"})(
                    Button("Health Check", on_click=self.health_check, style=button_style),
                    Button("Start Recording", on_click=self.start_recording, style=button_style),
                    Button("Create Pipeline", on_click=self.create_pipeline, style=button_style),
                )
            )
        )

if __name__ == "__main__":
    ed.App(App()).start()