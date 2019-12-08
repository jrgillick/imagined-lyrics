import librosa, numpy as np, pandas as pd, audioread, itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
from keras.preprocessing.sequence import pad_sequences as keras_pad_seqs
from collections import defaultdict
import text_utils
from sklearn.utils import shuffle


"""
# Useful functions for loading audio files
"""

# librosa.load() but return only the signal, not (y, sr)
def librosa_load_without_sr(f, sr=None,offset=None,duration=None):
    if offset is not None and duration is not None:
        return librosa.load(f, sr=sr,offset=offset,duration=duration)[0]
    else:
        return librosa.load(f, sr=sr)[0]

# Runs librosa.load() on a list of files in parallel, returns [y1, y2, ...]
def parallel_load_audio_batch(files,n_processes,sr=None,offsets=None,
    durations=None):
    if offsets is not None and durations is not None:
        return Parallel(n_jobs=n_processes)(
            delayed(librosa_load_without_sr)(files[i],sr=sr,offset=offsets[i],
                duration=durations[i]) for i in tqdm(range(len(files))))
    else:
        return Parallel(n_jobs=n_processes)(
            delayed(librosa_load_without_sr)(f,sr=sr) for f in tqdm(files))

def get_audio_length(path):
    with audioread.audio_open(path) as f:
        return f.duration

"""
# Sequence utils
"""
def pad_sequences(sequences, pad_value=None, max_len=None):
    # If a list of features are supposed to be sequences of the same length
    # But are not, then zero pad the end
    # Expects the sequence length dimension to be the first dim (axis=0)
    # Optionally specify a specific value `max_len` for the sequence length.
    # If none is given, will use the maximum length sequence.
    if max_len is None:
        #lengths = [len(ft) for ft in sequences]
        max_len = max([len(ft) for ft in sequences])
    # Pass along the pad value if provided
    kwargs = {'constant_values': pad_value} if pad_value is not None else {}

    sequences = [librosa.util.fix_length(
        np.array(ft), max_len, axis=0, **kwargs) for ft in sequences]
    return sequences

# This function is for concatenating subfeatures that all have
# the same sequence length
# e.g.  feature_list = [mfcc(40, 12), deltas(40, 12), rms(40, 1)]
# output would be (40, 25)
# The function calls pad_sequences first in case any of the
# sequences of features are off-by-one in length
def concatenate_and_pad_features(feature_list):
    feature_list = pad_sequences(feature_list)
    return np.concatenate(feature_list, axis=1)

"""
# Feature Utils
"""
def featurize_mfcc(f=None, offset=0, duration=None, y=None, sr=None,
        augment_fn=None, **kwargs):
    """ Accepts either a filepath with optional offset/duration
    or a 1d signal y and sample rate sr. But not both.
    """
    if f is not None and y is not None:
            raise Exception("You should only pass one of `f` and `y`")

    if (y is not None) ^ bool(sr):
            raise Exception("Can't use only one of `y` and `sr`")

    if y is None:
        try:
            y, sr = librosa.load(f, sr=sr, offset=offset, duration=duration)
        except:
            import pdb; pdb.set_trace()

    # Get concatenated and padded MFCC/delta/RMS features
    S, phase = librosa.magphase(librosa.stft(y, hop_length=int(sr/100)))
    rms = librosa.feature.spectral.rms(S=S).T
    mfcc_feat = librosa.feature.mfcc(y,sr,n_mfcc=13, n_mels=13,
        hop_length=int(sr/100), n_fft=int(sr/40)).T[:,1:]
    deltas = librosa.feature.delta(mfcc_feat.T).T
    delta_deltas = librosa.feature.delta(mfcc_feat.T, order=2).T
    feature_list = [rms, mfcc_feat, deltas, delta_deltas]
    feats = concatenate_and_pad_features(feature_list)
    return feats

def featurize_melspec(f=None, offset=0, duration=None, y=None, sr=None,
        augment_fn=None, **kwargs):
    """ Accepts either a filepath with optional offset/duration
    or a 1d signal y and sample rate sr. But not both.
    """
    if f is not None and y is not None:
        raise Exception("You should only pass one of `f` and `y`")

    if (y is not None) ^ bool(sr):
        raise Exception("Can't use only one of `y` and `sr`")

    if y is None:
        try:
            y, sr = librosa.load(f, sr=sr, offset=offset, duration=duration)
        except:
            import pdb; pdb.set_trace()

    if augment_fn is not None:
        y = augment_fn(y)
    S = librosa.feature.melspectrogram(y, sr, hop_length=int(sr/100)).T
    S = librosa.amplitude_to_db(S, ref=np.max)
    return S

#def load_audio_file_segments(f, sr, segments):
#    """ Method to load multiple segments of audio from one file. For example,
#    if there are multiple annotations corresponding to different points in the
#    file.
#    Returns: The clipped audio file
#
#    """

def featurize_audio_segments(segments, feature_fn, f=None, y=None, sr=None):
    """ Method to load features for multiple segments of audio from one file.
    For example, if annotations correspond to different points in the file.
    Accepts either a path to an audio file (`f`), or a preloaded signal (`y`)
    and sample rate (`sr`).

    Args:
    segments: List of times in seconds, of the form (offset, duration)
    feature_fn: A function to compute features for each segment
        feature_fn must accept params (f, offset, duration, y, sr)
    f: Filename of audio file for which to get feature
    y: Preloaded 1D audio signal.
    sr: Sample rate
    Returns: A list of audio features computed by feature_fn, for each
        segment in `segments`.
    """

    if f is not None and y is not None:
            raise Exception("You should only pass one of `f` and `y`")

    if (y is not None) ^ bool(sr):
            raise Exception("Can't use only one of `y` and `sr`")

    feature_list = []
    for segment in segments:
        feature_list.append(feature_fn(f=f, offset=segment[0],
            duration=segment[1], y=y, sr=sr))
    return feature_list


"""
# Collate Functions
# For use in Pytorch DataLoaders
# These functions are applied to the list of items returned by the __get_item__
# method in a Pytorch Dataset object.  We need to follow this pattern in order
# to get the benefit of the multi-processing implemented
# in torch.utils.data.DataLoader
"""
def pad_sequences_with_labels(seq_label_tuples, sequence_pad_value=0,
    label_pad_value=None, input_vocab=None, output_vocab=None,
    max_seq_len=None,  max_label_len=None, one_hot_labels=False,
    one_hot_inputs=False, expand_channel_dim=False):

    """ Args:
            seq_label_tuples: a list of length batch_size. If the entries in this
            list are already tuples (i.e. type(seq_label_tuples[0]) is tuple),
            we're dealing with the "Basic" setup, where __get_item__ returns 1
            example per file. In that case, we don't need to do anything extra.
            But if seq_label_tuples[0] is a list, then that means we have a
            list of examples for each file, so we need to combine those lists
            and store the results.

            Pads at the beginning for input sequences and at the end for
            label sequences.

    """
    # First remove any None entries from the list
    # These may have been caused by too short of a sequence in the dataset or some
    # other data problem.
    seq_label_tuples = [s for s in seq_label_tuples if s is not None]

    if len(seq_label_tuples) == 0:
        return None

    try:
      if type(seq_label_tuples[0]) is list:
          # Each file has multiple examples need to combine into one larger
          # list of batch_size*n_samples tuples, instead of a list of lists of tuples
          combined_seq_label_tuples = []
          for i in range(len(seq_label_tuples)):
              combined_seq_label_tuples += seq_label_tuples[i]
          seq_label_tuples = combined_seq_label_tuples
    except:
      import pdb; pdb.set_trace()


    if (output_vocab is None and one_hot_labels) or (input_vocab is None and one_hot_labels):
        raise Exception("Need to provide vocab to convert labels to one_hot.")

    sequences, labels = unpack_list_of_tuples(seq_label_tuples)

    sequences = keras_pad_seqs(sequences, maxlen=max_seq_len, dtype='float32',
        padding='pre', truncating='post', value=sequence_pad_value)

    #if one_hot_inputs:
    #    sequences = text_utils.np_onehot(sequences.astype(np.int32), depth=len(input_vocab))
    #sequences = pad_sequences(sequences, sequence_pad_value, max_len=max_seq_len)

    # If there are no labels, then expect the batch of labels as [None, None...]
    if labels[0] is not None:
        if label_pad_value is not None:
            # label_pad_value should be the string value, not the integer in the voc
            labels = pad_sequences(labels, label_pad_value, max_len=max_label_len)

        # Convert vocab to integers after padding
        if output_vocab is not None:
            labels = [text_utils.sequence_to_indices(l, output_vocab) for l in labels]

        if one_hot_labels:
            labels = [text_utils.np_onehot(l, depth=len(output_vocab)) for l in labels]

    if expand_channel_dim:
        sequences = np.expand_dims(sequences, 1)
    return sequences, labels

"""
# Data Augmentation Functions
"""
# Speed up or slow down audio by `factor`. If factor is >1,
# we've sped up, so we need to pad. If factor <1, take the beginning of y
# set length of new_y to match length of y
def set_length(new_y, y):
    if len(new_y) < len(y):
        new_y = librosa.util.fix_length(new_y, len(y))
    elif len(new_y) > len(y):
        new_y = new_y[0:len(y)]
    return new_y

def random_speed(y, sr, prob=0.5, min_speed=0.8, max_speed=1.2):
    if np.random.uniform(0,1) < prob:
        factor = np.random.uniform(min_speed, max_speed)
        new_sr = sr*factor
        new_y = librosa.core.resample(y,sr,new_sr)
        return set_length(new_y, y)
    else:
        return y

def random_stretch(y, sr, prob=0.5, min_stretch=0.8, max_stretch=1.2):
    if np.random.uniform(0,1) < prob:
        factor = np.random.uniform(min_stretch, max_stretch)
        new_y = librosa.effects.time_stretch(y, factor)
        return set_length(new_y, y)
    else:
        return y

def random_pitch(y, sr, prob=0.5, min_shift=-16, max_shift=25, bins_per_octave=48):
    if np.random.uniform(0,1) < prob:
        steps = np.random.randint(min_shift,max_shift)
        new_y = librosa.effects.pitch_shift(y, sr, n_steps=steps, bins_per_octave=bins_per_octave)
        return set_length(new_y, y)
    else:
        return y

def random_noise(y, sr, noise_signals, min_noise_factor=0.0, max_noise_factor=0.5, prob=0.5):
    if np.random.uniform(0,1) < prob:
        noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)
        noise_signal = np.random.choice(noise_signals)
        if len(noise_signal) < len(y):
            raise Exception("length of the background noise signal is too short")
        noise_start = int(np.random.uniform(0, len(noise_signal)-len(y)))
        noise = noise_signal[noise_start:noise_start+len(y)]
        # adjust the intended noise factor based on the energy of the signal
        noise_rms = np.maximum(0.001, librosa.feature.rms(noise).mean())
        sig_rms = librosa.feature.rms(y).mean()
        energy_ratio = np.minimum(20, (noise_rms / sig_rms)) #librosa.feature.rms(y).mean()/librosa.feature.rms(noise).mean()
        noise_factor *= energy_ratio

        max_noise = np.max(noise)
        max_y = np.max(y)
        return noise_factor * noise + (1.0 - noise_factor) * y
    else:
        return y

def random_augment(y, sr, noise_signals):
    functions = shuffle([random_speed, random_stretch, random_pitch,
        partial(random_noise, noise_signals=noise_signals)])
    for fn in functions:
        y = fn(y, sr=sr)
    return y

"""
# Misc Functions
"""
def unpack_list_of_tuples(list_of_tuples):
    return [list(tup) for tup in list(zip(*list_of_tuples))]

def combine_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def reverse_sequence(feature_seq):
    return np.flip(feature_seq, axis=0)

def reverse_sequence_batch(batch_feats):
    for i in range(len(batch_feats)):
        batch_feats[i] = reverse_sequence(batch_feats[i])
    return batch_feats

def dedup_list(l):
    l = copy.deepcopy(l)
    i = 1
    while i < len(l):
        if l[i] == l[i-1]:
            del l[i]
        else:
            i += 1
    return l

def times_overlap(start1, end1, start2, end2):
    if end1 < start2 or end2 < start1:
        return False
    else:
        return True

def start_end_to_offset_duration(start, end):
    return start, end-start

def subsample_time(offset, duration, audio_file_length, subsample_length=1., padding_length=0.5):
    start_time = np.maximum(0, offset-padding_length)
    end_time = np.minimum(offset+duration+padding_length, audio_file_length)
    start = np.maximum(start_time,np.random.uniform(start_time, end_time-subsample_length))
    return (start, subsample_length)
