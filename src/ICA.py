from data_utils import load_recordings
import mne
from pipeline import get_epochs
from mne.preprocessing import ICA
from Marker import Marker
import matplotlib.pyplot as plt

raw, rec_params = load_recordings("David7")
raw.load_data()
raw.filter(1, 50)

events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, Marker.all(), tmax=rec_params["trial_duration"], picks="data", baseline=None)

epochs.load_data()
ica = ICA(n_components=13, max_iter='auto', random_state=97)
ica.fit(epochs)
ica.plot_sources(epochs, show_scrollbars=False)
ica.plot_components()
ica.plot_properties(epochs, [0, 5, 7])
plt.show()
print()
