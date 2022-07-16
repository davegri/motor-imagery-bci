import numpy as np
from src.Marker import Marker
from src.board import Board

from src.pipeline import get_epochs, evaluate_pipeline
from src.data_utils import load_rec_params, save_raw, load_hyperparams, json_load
import src.spectral as spectral
import os
from src.constants import TEXT_DIR, AUDIO_DIR

BG_COLOR = "black"
STIM_COLOR = "white"

visual = None
core = None
event = None
sound = None


def run_session(params, retrain_pipeline=None, predict_pipeline=None, epochs=None, labels=None):
    """
    Run a recording session, if pipeline is passed display prediction after every epoch
    params = the json file contains all recording parameters
    retrain_pipeline = used within co-adaptive sessions. determine the pipeline by which the model will be trained.
    predict_pipeline = used within co-adaptive sessions. predicting without retraining.
    epochs, labels = used within co-adaptive session. the pre-loaded epochs and labels for the re-classifying
    """

    # import psychopy only here to prevent pygame loading.
    from psychopy import visual as vis, core as cor, event as eve, sound as snd
    global visual, core, event, sound
    visual, core, event, sound = vis, cor, eve, snd

    # create list of random trials
    trial_markers = Marker.all() * params["trials_per_stim"]
    np.random.shuffle(trial_markers)

    # open psychopy window and display starting message
    win = visual.Window(units="norm", color=BG_COLOR, fullscr=params["full_screen"])

    # import text based on recording language
    language_texts = json_load(os.path.join(TEXT_DIR, f'{params["language"]}.json'))

    if retrain_pipeline:
        visual.TextStim(win=win, text=language_texts["please_wait"], color=STIM_COLOR).draw()
        win.flip()
        hyperparams = load_hyperparams(params["subject"], retrain_pipeline.name)
        predict_pipeline = retrain_pipeline.create_pipeline(hyperparams)
        predict_pipeline.fit(epochs, labels)
        best_score = np.mean(evaluate_pipeline(predict_pipeline, epochs, labels)["test_score"])

    if params["wait_on_start"]:
        msg1 = f'{language_texts["hello_msg"].format(subject_name=params["subject"])}\n{language_texts["record_instructions"]}'
        loop_through_messages(win, [msg1])

    # Start recording
    with Board(use_synthetic=params["use_synthetic_board"]) as board:
        for i, marker in enumerate(trial_markers):

            # "get ready" period
            progress_text_stim = progress_text(win, i + 1, len(trial_markers), marker, language_texts)

            show_stim_for_duration(win, params["get_ready_duration"], progress_text_stim,
                                   aud_stim_start=sound.Sound(marker.sound_path(params["language"])))

            # calibration period
            core.wait(params["calibration_duration"])

            # motor imagery period
            START_BEEP = sound.Sound("a", secs=0.1)
            END_BEEP = sound.Sound("c", secs=0.1)
            show_stim_for_duration(win, params["trial_duration"], marker_stim(win, marker), aud_stim_start=START_BEEP,
                                   aud_stim_end=END_BEEP, run_before_flip=lambda: board.insert_marker(marker))

            if predict_pipeline:
                # We need to wait a short time between the end of the trial and trying to get it's data to make sure
                # that we have recorded (trial_duration * sfreq) samples after the latest marker (otherwise the epoch
                # will be too short)
                core.wait(0.5)

                # get latest epoch and make prediction
                raw = board.get_data()
                new_epochs, new_labels = get_epochs([raw], params["trial_duration"], params["calibration_duration"],
                                                    on_missing='ignore')
                new_epochs = new_epochs.get_data()
                prediction = predict_pipeline.predict(new_epochs)[-1]

                # display prediction result
                show_stim_for_duration(win, params["display_online_result_duration"], classification_result_txt(win, marker, prediction),
                                       aud_stim_start=classification_result_sound(marker, prediction, params["language"]),
                                       )

            if retrain_pipeline and i % params["retrain_num"] == 0 and i != 0:
                text_stim(win, language_texts["retraining_model"]).draw()
                win.flip()

                # train new pipeline
                total_epochs = np.concatenate((epochs, new_epochs), axis=0)
                total_labels = np.concatenate((labels, new_labels), axis=0)
                new_pipeline = retrain_pipeline.create_pipeline(hyperparams)
                new_pipeline.fit(total_epochs, total_labels)

                # evaluate new pipeline
                results = evaluate_pipeline(new_pipeline, total_epochs, total_labels)
                score = np.mean(results['test_score'])
                msg = f'Finished retraining \nold model score: {best_score} \nnew model score: {score}'
                win.flip()
                predict_pipeline = new_pipeline

                if score > best_score:
                    best_score = score
                    msg += f'\n{language_texts["good_job"]}'

                msg += f'\n{language_texts["continue_instructions"]}'

                text_stim(win, msg).draw()
                win.flip()
                if params["wait_on_start"]:
                    keys_pressed = event.waitKeys()
                    if 'escape' in keys_pressed:
                        win.close()
                        return
        core.wait(0.5)
        win.close()
        raw = board.get_data()
    save_raw(raw, params)


def loop_through_messages(win, messages):
    for msg in messages:
        visual.TextStim(win=win, text=msg, color=STIM_COLOR).draw()
        win.flip()
        keys_pressed = event.waitKeys()
        if 'escape' in keys_pressed:
            win.close()
        if 'backspace' in keys_pressed:
            break

def marker_stim(win, marker):
    if Marker(marker).what_to_show == "shape":
        return visual.ShapeStim(win, vertices=Marker(marker).shape, fillColor=STIM_COLOR, size=.5)
    return visual.ImageStim(win, image=Marker(marker).image_path, size=(0.6, 0.6))


def show_stim_for_duration(win, duration, vis_stim, aud_stim_start=None, aud_stim_end=None, run_before_flip=None):
    # Adding this code here is an easy way to make sure we check for an escape event before showing every stimulus
    if 'escape' in event.getKeys():
        win.close()

    vis_stim.draw()  # draw stim on back buffer
    if aud_stim_start:
        aud_stim_start.play()
    if run_before_flip:
        run_before_flip()
    win.flip()  # flip the front and back buffers and then clear the back buffer
    core.wait(duration)
    if aud_stim_end:
        aud_stim_end.play()
    win.flip()  # flip back to the (now empty) back buffer


def text_stim(win, text, color=STIM_COLOR):
    return visual.TextStim(win=win, text=text, color=color, bold=True, alignHoriz='center', alignVert='center')


def progress_text(win, done, total, stim, language_texts):
    return text_stim(win, f'trial {done}/{total}\n {Marker(stim).get_ready_text(language_texts)}')

def classification_result_sound(marker, prediction, language):
    if marker == prediction:
        return sound.Sound(os.path.join(AUDIO_DIR, language, "good job!.ogg"))
    return sound.Sound(os.path.join(AUDIO_DIR, language, "try again.ogg"))

def classification_result_txt(win, marker, prediction):
    if marker == prediction:
        msg = 'correct prediction'
        col = (0, 1, 0)
    else:
        msg = 'incorrect prediction'
        col = (1, 0, 0)
    return text_stim(win, f'label: {Marker(marker).name}\nprediction: {Marker(prediction).name}\n{msg}', col)


def marker_image(win, marker):
    return visual.ImageStim(win=win, image=Marker(marker).image_path, units="norm", size=2, color=(1, 1, 1))

if __name__ == "__main__":
    rec_params = load_rec_params()
    run_session(rec_params)
