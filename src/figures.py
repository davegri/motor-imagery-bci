import os
from constants import *
import mne
from pathlib import Path
from Marker import Marker
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import json
from preprocessing import preprocess
from data_utils import get_subject_rec_folders, load_recordings
from pipeline import get_epochs

FREQ_RANGE = (6, 30)


def create_and_save_plots(rec_folder_name, bad_electrodes=[]):
    plt.ioff()
    raw, rec_params = load_raw(rec_folder_name)
    raw = preprocess(raw)
    raw.info['bads'] = bad_electrodes

    fig_path = create_figures_folder(rec_folder_name)

    fig_raw = create_raw_fig(raw)
    fig_raw.savefig(os.path.join(fig_path, "raw.png"))

    fig_psd = create_psd_fig(raw)
    fig_psd.savefig(os.path.join(fig_path, "psd.png"))

    epochs, labels = get_epochs([raw], rec_params["trial_duration"], rec_params["calibration_duration"])
    epochs = mne.preprocessing.compute_current_source_density(epochs)

    electrodes = ["C3", "C4", "Cz"]
    class_spectrogram_fig = create_class_spectrogram_fig(epochs, electrodes, rec_params["calibration_duration"])
    class_spectrogram_fig.savefig(os.path.join(fig_path, f'class_spectrogram_{"_".join(electrodes)}.png'))

    class_psd_fig = create_class_psd_fig(epochs, electrodes, rec_params["trial_duration"],
                                         rec_params["calibration_duration"])
    class_psd_fig.savefig(os.path.join(fig_path, f'class_psd_{"_".join(electrodes)}.png'))


def create_plots_for_subject(subject):
    raws, rec_params = load_recordings(subject)
    epochs, labels = get_epochs(raws, rec_params["trial_duration"], rec_params["calibration_duration"],
                                reject_bad=not rec_params['use_synthetic_board'])
    save_folder = f'../figures/{subject}'
    Path(save_folder).mkdir(exist_ok=True)

    electrodes = ["C3", "C4", "Cz"]
    class_spectrogram_fig = create_class_spectrogram_fig(epochs, electrodes, rec_params["calibration_duration"])
    class_spectrogram_fig.savefig(
        os.path.join(save_folder, f'class_spectrogram_{"_".join(electrodes)}.png'))
    class_psd_fig = create_class_psd_fig(epochs, electrodes, rec_params["trial_duration"],
                                         rec_params["calibration_duration"])
    class_psd_fig.savefig(os.path.join(save_folder, f'class_psd_{"_".join(electrodes)}.png'))


def create_psd_fig(raw):
    fig = mne.viz.plot_raw_psd(raw, fmin=FREQ_RANGE[0], fmax=FREQ_RANGE[1], show=False)
    return fig


def create_raw_fig(raw):
    events = mne.find_events(raw)
    event_dict = {marker.name: marker.value for marker in Marker}
    fig = mne.viz.plot_raw(raw, events=events, clipping=None, show=False, event_id=event_dict, show_scrollbars=False,
                           start=10)
    return fig


def create_figures_folder(rec_folder_name):
    rec_folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name)
    fig_path = os.path.join(rec_folder_path, "figures")
    Path(fig_path).mkdir(exist_ok=True)
    return fig_path


def load_raw(rec_folder_name):
    rec_folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name)
    raw_path = os.path.join(rec_folder_path, 'raw.fif')
    raw = mne.io.read_raw_fif(raw_path)
    with open(os.path.join(rec_folder_path, 'params.json')) as file:
        rec_params = json.load(file)
    return raw, rec_params


def calc_average_spectrogram(epochs, sfreq):
    segments_per_second = 3
    fft_params = {
        "nperseg": int(sfreq / segments_per_second),
        "noverlap": int(sfreq / segments_per_second) * 0.7,
        "nfft": 256,
    }

    # we calculate the power for the first epoch separately so that we have a variable of the right dimensions to sum
    # onto
    freq, time, avg_power = signal.spectrogram(epochs[0], sfreq, **fft_params)
    for epoch in epochs[1:]:
        _, _, power = signal.spectrogram(epoch, sfreq, **fft_params)
        avg_power += power / len(epochs)

    freq_idxs = (freq >= FREQ_RANGE[0]) & (freq <= FREQ_RANGE[1])
    freq = freq[freq_idxs]
    avg_power = avg_power[freq_idxs]
    return time, freq, avg_power


def calc_average_psd(epochs, sfreq):
    fft_params = {
        'nfft': 512,
        'nperseg': 200,
        'noverlap': int(0.5 * 200),
    }

    # calculate the first fft
    freq, avg_power = signal.welch(epochs[0], sfreq, **fft_params)

    for epoch in epochs[1:]:
        _, power = signal.welch(epoch, sfreq, **fft_params)
        avg_power += power / len(epochs)

    freq_idxs = (freq >= FREQ_RANGE[0]) & (freq <= FREQ_RANGE[1])
    freq = freq[freq_idxs]
    avg_pxx = avg_power[freq_idxs]
    return avg_pxx, freq


def create_class_psd_fig(epochs, electrodes, trial_duration, calibration_period):
    channels = [epochs.info.ch_names.index(elec) for elec in electrodes]
    fig, axs = plt.subplots(len(channels), len(Marker.all()), figsize=(22, 11))
    epochs.crop(tmin=1.1)  # remove calibration period
    for i, electrode in enumerate(electrodes):
        for j, cls in enumerate(Marker):
            cls_chan_epochs = epochs[str(cls.value)].pick_channels([electrode]).get_data().squeeze(axis=1)
            power, freq = calc_average_psd(cls_chan_epochs, epochs.info["sfreq"])
            ax = axs[i, j]
            ax.plot(freq, power)
            ax.set_ylabel('Power Density')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_title(f'{cls.name} {electrode}')
            ax.set_xticks(np.arange(FREQ_RANGE[0], FREQ_RANGE[1] + 1, 2))

    fig.tight_layout()
    return fig


def create_class_spectrogram_fig(epochs, electrodes, calibration_period):
    fig, axs = plt.subplots(len(electrodes), len(Marker.all()), figsize=(22, 11))
    for i, electrode in enumerate(electrodes):
        for j, cls in enumerate(Marker):
            cls_chan_epochs = epochs[str(cls.value)].pick_channels([electrode]).get_data().squeeze(axis=1)
            time, freq, power = calc_average_spectrogram(cls_chan_epochs, epochs.info["sfreq"])
            ax = axs[i, j]
            mesh = ax.pcolormesh(time, freq, power, shading='gouraud', cmap="jet")
            plt.colorbar(mesh, ax=ax)
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Frequency [Hz]')
            ax.axvline(calibration_period, color='r', linestyle="dashed", label="stimulus")
            ax.legend()
            ax.set_title(f'{cls.name} {electrode}')
    fig.tight_layout()
    return fig


def save_plots_for_subject(subject_name):
    rec_folders = get_subject_rec_folders(subject_name)
    [create_and_save_plots(folder) for folder in rec_folders]


if __name__ == "__main__":
    plt.ioff()  # don't display plots while creating them
    create_plots_for_subject("Robert")
