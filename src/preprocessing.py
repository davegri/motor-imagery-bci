import mne
import numpy as np

from src.board import EEG_CHAN_NAMES
import scipy.stats

LAPLACIAN = {
    "C3": ["FC5", "FC1", "CP5", "CP1"],
    "Cz": ["FC1", "FC2", "CP1", "CP2"],
    "C4": ["FC2", "FC6", "CP2", "CP6"]
}

LAPLACIAN = {EEG_CHAN_NAMES.index(key): [EEG_CHAN_NAMES.index(chan) for chan in value] for key, value in
             LAPLACIAN.items()}


def laplacian(epochs):
    filtered_epochs = np.copy(epochs)
    for chan, adjacent_chans in LAPLACIAN.items():
        filtered_epochs[:, chan, :] -= np.mean(epochs[:, adjacent_chans, :], axis=1)
    return filtered_epochs


def preprocess(raw):
    raw.load_data()
    raw.filter(7, 30)
    return raw


def reject_epochs(epochs, labels):
    bad_epochs = dict()
    for epoch_idx, epoch in enumerate(epochs):
        reasons = {
            'bad_chans': [],
            'reason': []
        }
        chan_index = list(range(len(epoch[:, 1])))
        for chan in range(len(chan_index)):
            curr_chan = epoch[chan, :]
            if abs(curr_chan.min()) < 1 * 1e-7:
                reasons['bad_chans'].append(chan)
                reasons['reason'].append('amp low')
            elif abs(curr_chan.max()) > 200 * 1e-6:
                reasons['bad_chans'].append(chan)
                reasons['reason'].append('amp high')
            all_chan_except = chan_index.remove(chan)
            # average_corr_chan = get_average_corr(curr_chan, epoch[all_chan_except, :])
            # if abs(average_corr_chan) < 0.05:
            #     reasons['bad_chans'].append(chan)
            #     reasons['reason'].append('corr low')
            # elif abs(average_corr_chan) > 0.9:
            #     reasons['bad_chans'].append(chan)
            #     reasons['reason'].append('corr high')

        if len(reasons['bad_chans']) > 3:
            bad_epochs[epoch_idx] = reasons
    n_epochs_removed = len(bad_epochs.keys())
    print(f"{n_epochs_removed} epochs rejected")
    if n_epochs_removed:
        print(bad_epochs)
    return np.delete(epochs, list(bad_epochs.keys()), axis=0), np.delete(labels, list(bad_epochs.keys()), axis=0)


def find_average_voltage(epochs):
    vol_per_chan = {}
    for chan_inx in range(len(epochs[0, :, 0])):
        vol_per_chan[chan_inx + 1] = np.mean(epochs[:, chan_inx, :])
    return vol_per_chan


def get_average_corr(chan, all_chans):
    total_corr = 0
    for other_chan in all_chans:
        if len(chan) != len(other_chan) or len(chan) < 2 or len(other_chan) < 2:
            continue
        corr, _ = scipy.stats.pearsonr(chan, other_chan)
        if not np.isnan(corr):
            total_corr += corr
    avg_corr = total_corr / len(all_chans)
    return avg_corr
