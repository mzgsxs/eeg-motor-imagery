import numpy as np

# model
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import mne
# for fetching the training data
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf, read_raw_fif

# for connecting to eeg stream
from mne_lsl.stream import StreamLSL

# for establishing control signal stream
from mne_lsl.lsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_streams,
)
import uuid


def make_control_stream(sfreq):
  sinfo = StreamInfo(
      name="control-stream",
      stype="command",
      n_channels=2,
      sfreq=32,
      dtype="float32",
      source_id=uuid.uuid4().hex,
  )
  sinfo.set_channel_names(["left", "right"])
  sinfo.set_channel_types("command")
  sinfo.set_channel_units("None")
  outlet = StreamOutlet(sinfo)
  return outlet


tmin = -1.0
tmax = 5.0
train_tmin = 1.5
train_tmax = 2.5
win_duration = train_tmax - train_tmin


highpass = 7.
lowpass = 30.

csp_num = 4


def fit_model():
  # standard training data and model fitting
  subject = 1
  runs = [6, 10, 14]  # motor imagery: hands vs feet
  raw_fnames = eegbci.load_data(subject, runs)
  raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
  train_win_size = int(raw.info['sfreq']*win_duration) 
  eegbci.standardize(raw)

  montage = make_standard_montage("standard_1005")
  raw.set_montage(montage)
  raw.annotations.rename(dict(T1="hands", T2="feet"))
  raw.set_eeg_reference(projection=True)

  # Apply band-pass filter
  raw.filter(highpass, lowpass, fir_design="firwin", skip_by_annotation="edge")
  picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
  electrodes_to_keep = ['Cz', 'Fz', 'C3', 'C4']
  picks = mne.pick_channels(raw.ch_names, include=electrodes_to_keep, ordered=True)

  epochs = Epochs(
      raw,
      event_id=["hands", "feet"],
      tmin=tmin,
      tmax=tmax,
      proj=True,
      picks=picks,
      baseline=None,
      preload=True,
  )
  epochs_train = epochs.copy().crop(tmin=train_tmin, tmax=train_tmax)
  labels = epochs.events[:, -1] - 1
  epochs_data = epochs.get_data(copy=False) # num_event x num_chs x num_time_points
  epochs_data_train = epochs_train.get_data(copy=False) # num_event x num_chs x num_time_points

  # Assemble a classifier
  csp = CSP(n_components=csp_num, reg=None, log=True, norm_trace=False)
  lda = LinearDiscriminantAnalysis()
  print(epochs_data_train.shape, labels.shape)
  X_train = csp.fit_transform(epochs_data_train, labels)
  print(X_train.shape, labels.shape)
  lda.fit(X_train, labels)
  return csp, lda, train_win_size


csp, lda, train_win_size = fit_model() 
print(csp, lda, train_win_size)
# Establishing lsl control signal stream 
control_stream = make_control_stream(20)


import scipy
import time

print("\nStarting online decoder ... ")
try:
  # connect to eeg signal stream
  # make sure is prefiltered bandpass 7.0 - 30.0 hz
  eeg_stream = StreamLSL(name="obci_eeg1", bufsize=3600)
  eeg_stream.connect(acquisition_delay=0.01, processing_flags="all")
  win_size = eeg_stream.info['sfreq']*win_duration
  # wait for eeg stream have enough data
  time.sleep(1)
  while True:
    data, ts = eeg_stream.get_data(win_duration)
    # resample data to fit to the training data shape
    if win_size != train_win_size: 
      resampled_data = scipy.signal.resample(data, train_win_size, axis=1)
    else:
      resampled_data = data 
    X = np.expand_dims(data, axis=0)
    X_test = csp.transform(X)
    y_predict = lda.predict_proba(X_test)
    print(y_predict)
    # push this command to stream 
    control_stream.push_sample(y_predict[0])
except KeyboardInterrupt:
  print("\nProgram terminated by user")
finally:
  print("\nProgram exited")


