import pandas as pd, numpy as np, math, os
from praatio import tgio
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
import text_utils
import audio_utils
import itertools
from tqdm import tqdm

"""
# Dataset or Format Specific Utils
"""

######################## TIMIT ###############################

def read_phoneme_file_to_dataframe(f, samples_to_seconds=True, sr=16000.):
    # 3 rows in a ' ' separated text file, corresponding to (start, end, label)
    # For TIMIT, start and end are times in SAMPLES, at 16K sample rate
    df = pd.read_csv(f, sep=' ', names=['start','end','label'],
        dtype={'start': int, 'end':int, 'label':str})
    if samples_to_seconds:
        df.start /= sr
        df.end /= sr
    return df

def get_timit_phoneme_labels(f, **kwargs):
    # Returns a list of the phoneme labels in a TIMIT phoneme file
    # Discards the start and end times
    # The TIMIT labels start and end with '#h', so let's update that
    # Replace the start '#h' with SOS_token and end '#h' with EOS_token
    # The make_vocab function recognizes '###_START_###' and '###_END_###'
    labels = list(read_phoneme_file_to_dataframe(f).label)
    labels[0] = text_utils.START_SYMBOL
    labels[-1] = text_utils.END_SYMBOL
    return labels

def sample_timit_features_and_labels(f, label_file, feature_fn, **kwargs):
    """ feature_fn must accept `segments`, a list of (offset, duration) tuples """
    label_fn_args = kwargs['label_fn_args'] if 'label_fn_args' in kwargs else {}
    feature_fn_args = kwargs['feature_fn_args'] if 'feature_fn_args' in kwargs else {}

    time_segments, label_subseqs = sample_timit_labels_and_times(label_file, **label_fn_args)
    #time_segments, label_subseqs = sample_timit_labels_and_times(f, l, min_len,
    #    max_len, n_samples)

    feature_subseqs = []
    for segment in time_segments:
        feature_subseqs.append(feature_fn(f=f, offset=segment[0],
            duration=segment[1]-segment[0]))

    return list(zip(feature_subseqs, label_subseqs))

def sample_timit_labels_and_times(f, **kwargs):
    """ Function to randomly extract time segments of a label file
        along with the corresponding label sequence. This function
        is dataset specific, but should be copied with minimal
        changes for any other dataset.

        For TIMIT, label files are ' ' separated text files with
        3 columns: (start, end, label) - start and end are
        times in SAMPLES at 16K, labels are phonemes (strings). This will
        be read into a pandas dataframe with cols ('start', 'end', 'label')
        Start and end times are converted into seconds based on
        the given sample rate `sr`.

        In TIMIT, label files also start and end with '#h', which correspond to
        silence at the beginning and end. We convert these to START_SYMBOL
        and END_SYMBOL. When sampling a subsequence of labels, ignore the input
        START and END symbols, and then add those back to each subsequence.

        Args:
            f: path to a TIMIT label file (extension .phn)
            min_len: Minimum length of the sampled subsequences of labels
            max_len: Maximum length of the sampled subsequences of labels
            n_samples: Number of subsequenecs of labels and times to sample

        Returns:
            segments: a list of length `n_samples` containing "time segment"
                tuples in SECONDS of the form (offset, duration).

            label_seqs: a list of "label sequences" corresponding to the labels
                at each segment. Length of the individual sequences can
                be different. Each sequence length is a random number between
                `min_len` and `max_len`. Time length of the segment depends on
                start time of the first label and end time of the last label.
    """

    min_len = kwargs['min_len'] if 'min_len' in kwargs else 1
    max_len = kwargs['max_len'] if 'max_len' in kwargs else 10
    n_samples = kwargs['n_samples'] if 'n_samples' in kwargs else 1

    # Load and convert SAMPLES to SECONDS
    df = read_phoneme_file_to_dataframe(f, samples_to_seconds=True, sr=16000.)
    # Drop the silence and start and end
    df=df[1:-1].reset_index(drop=True)

    segments = []
    label_seqs = []

    if max_len > len(df): max_len = len(df)
    #if min_len < len(df): min_len = len(df)

    # Choose a random sequence length for each of the n_samples
    # e.g. Sample of list of 10 phonemes, or 3 phonemes, etc.
    # Must be done before choosing start point.
    seq_lengths = np.random.randint(min_len, max_len, n_samples)

    for i in range(n_samples):
        # Choose a start point in the label sequence
        start_index = np.random.randint(0, len(df) - seq_lengths[i])
        end_index = start_index + seq_lengths[i]
        segment = (df.start[start_index], df.end[end_index])
        label_seq = [text_utils.START_SYMBOL] + list(
            df.label[start_index:end_index]) + [text_utils.END_SYMBOL]
        segments.append(segment)
        label_seqs.append(label_seq)

    return segments, label_seqs


##################### TEXTGRID - LibriSpeech #############################


def get_words_from_textgrid_file(text_grid_path):
    return [e.label for e in tgio.openTextgrid(text_grid_path).tierDict['words'].entryList]

def convert_word_list_to_phonemes(word_list, phoneme_dict):
    """ Assume that phoneme_dict has all UPPERCASE words
        Removes OOV words from the word_list in the process
    """
    phoneme_list = []
    new_word_list = []
    for w in word_list:
        if w.upper() in phoneme_dict:
            phoneme_list += (phoneme_dict[w.upper()])
            new_word_list.append(w)
    return new_word_list, phoneme_list

def sample_textgrid_features_and_labels(f, label_file=None, feature_fn=None, **kwargs):
    """ Function to randomly extract time segments w/ features
        along with the corresponding label sequence. This function
        is specific to textgrid, but should be copied with minimal
        changes for any other dataset.

        For this function, features are phoneme lists, labels are words.
        For modeling phoneme_sequence --> word_sequence.

        For textgrid files, words are loaded using praatio.

        In this case the input vocab is converted to indices, but the
        output vocab is not. This is left to the padding function that gets
        called later.

        Args:
            f: path to a TIMIT label file (extension .phn)
            min_len: Minimum length of the sampled subsequences of labels
            max_len: Maximum length of the sampled subsequences of labels
            n_samples: Number of subsequenecs of labels and times to sample
            phoneme_dict: a preloaded python dict to convert words->phonemes
            phoneme_vocab: a preloaded python dict to convert phonemes->ints

        Returns:
            features: a list of length `n_samples`. Each entry in the list
                is a list of words.

            label_seqs: a list of "label sequences" corresponding to the labels
                at each segment. Length of the individual sequences can
                be different. Each sequence length is a random number between
                `min_len` and `max_len`.
    """
    min_len = kwargs['min_len'] if 'min_len' in kwargs else 1
    max_len = kwargs['max_len'] if 'max_len' in kwargs else 10
    n_samples = kwargs['n_samples'] if 'n_samples' in kwargs else 1
    phoneme_dict = kwargs['phoneme_dict']
    phoneme_vocab = kwargs['phoneme_vocab'] if 'phoneme_vocab' in kwargs else None

    # Load and convert to words and phonemes
    word_list = get_words_from_textgrid_file(f)
    word_seqs = []
    phoneme_seqs = []

    if max_len > len(word_list): max_len = len(word_list)

    if min_len>=max_len:
         return None

    # Choose a random sequence length for each of the n_samples
    # e.g. Sample of list of 10 phonemes, or 3 phonemes, etc.
    # Must be done before choosing start point.
    seq_lengths = np.random.randint(min_len, max_len, n_samples)


    for i in range(n_samples):
        # Choose a start point in the word sequence
        start_index = np.random.randint(0, len(word_list) - seq_lengths[i])
        end_index = start_index + seq_lengths[i]
        segment = word_list[start_index:end_index]
        word_seq, phoneme_seq = convert_word_list_to_phonemes(segment, phoneme_dict)
        word_seq = [text_utils.START_SYMBOL] + segment + [text_utils.END_SYMBOL]
        word_seqs.append(word_seq)
        if phoneme_vocab is not None:
            phoneme_seq = text_utils.sequence_to_indices(phoneme_seq, phoneme_vocab, one_hot=True)
            #phoneme_seq = [phoneme_vocab[p] for p in phoneme_seq]
        phoneme_seqs.append(phoneme_seq)

    return list(zip(phoneme_seqs, word_seqs))



##################### METROLYRICS  #############################

def load_metrolyrics_from_csv(filepath):
    df = pd.read_csv(filepath)
    lyrics = list(df.lyrics)
    lyrics = [l for l in lyrics if str(l) != 'nan']
    lyrics = [l.split('\n') for l in lyrics]
    lyrics = [l for l in lyrics if len(l) >= 8 and len(l) < 200]
    return lyrics

def sample_metrolyrics_features_and_labels(lyrics_list, **kwargs):
    """ Function to randomly extract time segments w/ features
        along with the corresponding label sequence. This function
        is specific to metrolyrics, but should be copied with minimal
        changes for any other dataset.

        For this function, features are phoneme lists, labels are words.
        For modeling phoneme_sequence --> word_sequence.

        Each actual sample is one pair of lines tokenized and then
        joined with a newline, e.g.:
        ['Morning', 'has', 'broken', '\n' 'Blackbird', 'has', 'spoken']

        In this case the input vocab is converted to indices, but the
        output vocab is not. This is left to a padding function that gets
        called later.


        Args:
            lyrics_list: a list of untokenized lyric lines e.g.
                ['Morning has broken', 'Blackbird has spoken', ...]
            index: an index into the list
            n_samples: Number of subsequences of labels and times to sample
            n_lines: The number of lines to take (1, 2, 3, etc.) default to 2.
            phoneme_dict: a preloaded python dict to convert words->phonemes
            phoneme_vocab: a preloaded python dict to convert phonemes->ints

        Returns:
            features: a list of length `n_samples`. Each entry in the list
                is a list of phonemes.

            label_seqs: a list of "label sequences" corresponding to the labels
                at each segment. Length of the individual sequences can
                be different. Each sequence length is a random number between
                `min_len` and `max_len`.
    """
    n_samples = kwargs['n_samples'] if 'n_samples' in kwargs else 1
    n_lines = kwargs['n_lines'] if 'n_lines' in kwargs else 2
    phoneme_drop_prob = kwargs['phoneme_drop_prob'] if 'phoneme_drop_prob' in kwargs else 0
    phoneme_swap_prob = kwargs['phoneme_swap_prob'] if 'phoneme_swap_prob' in kwargs else 0
    phoneme_dict = kwargs['phoneme_dict']
    phoneme_vocab = kwargs['phoneme_vocab'] if 'phoneme_vocab' in kwargs else None

    # Load and convert to words and phonemes
    word_seqs = []
    phoneme_seqs = []


    # If each of the n samples is a pair of lines, so we want to start at
    # even numbered lines only.
    start_points = np.random.randint(0, len(lyrics_list)/n_lines, n_samples) * n_lines

    for start_point in start_points:
        # Choose a start point in the word sequence
        lines = [word_tokenize(lyrics_list[start_point+i].lower()) for i in range(n_lines)]
        combined_lines = [w for w in itertools.chain.from_iterable(lines)]
        #line1 = word_tokenize(lyrics_list[start_point].lower())
        #line2 = word_tokenize(lyrics_list[start_point+1].lower())

        #word_list_for_phoneme_dict = [w.upper() for w in line1+line2]
        #phoneme_seq = convert_word_list_to_phonemes(word_list_for_phoneme_dict,
        #    phoneme_dict)
        word_seq, phoneme_seq = convert_word_list_to_phonemes(combined_lines,
            phoneme_dict)


        # Randomly swap phonemes into the sequence to add noise
        to_swap = list(np.random.binomial(1,phoneme_swap_prob,len(phoneme_seq)).nonzero()[0])
        for ind in to_swap:
        	random_phoneme = list(phoneme_vocab.keys())[np.random.randint(len(phoneme_vocab))]
        	phoneme_seq[ind] = random_phoneme

        # Randomly remove some phonemes from the sequence to add noise
        to_drop = list(np.random.binomial(1,phoneme_drop_prob,len(phoneme_seq)).nonzero()[0])
        for ind in to_drop:
        	phoneme_seq[ind] = None
        phoneme_seq = [e for e in phoneme_seq if e is not None]


        #word_seq = [START_SYMBOL] + line1 + line2 + [END_SYMBOL]
        word_seq = [text_utils.START_SYMBOL] + word_seq + [text_utils.END_SYMBOL]

        word_seqs.append(word_seq)
        if phoneme_vocab is not None:
            phoneme_seq = text_utils.sequence_to_indices(phoneme_seq, phoneme_vocab, one_hot=True)

        phoneme_seqs.append(phoneme_seq)

    return list(zip(phoneme_seqs, word_seqs))

def load_all_metrolyrics_words(filepath, num_workers=8,
	banned_words = None):
    print("Reading lyrics csv...")
    df = pd.read_csv(filepath)
    lyrics = list(df.lyrics)
    print("Filtering step 1 of 5...")
    lyrics = [l for l in tqdm(lyrics) if str(l) != 'nan']
    print("Filtering step 2 of 5...")
    lyrics = [l.split('\n') for l in tqdm(lyrics)]
    print("Filtering step 3 of 5...")
    lyrics = [l for l in tqdm(lyrics) if len(l) >= 8 and len(l) < 200]

    all_lines = []
    for l in lyrics: all_lines += l

    if banned_words is not None:
      print("Filtering step 4 of 5...")
      #all_lines = [l for l in tqdm(all_lines) if not str_contains_any(l, banned_words)]
      all_lines = Parallel(n_jobs=num_workers)(
            delayed(lambda x:x if not str_contains_any(x, banned_words) else None)(l) for l in tqdm(all_lines))
      all_lines = [l for l in all_lines if l is not None]

    print("Filtering step 5 of 5...")
    token_lists = Parallel(n_jobs=num_workers)(
            delayed(word_tokenize)(l) for l in tqdm(all_lines))

    all_tokens = []
    for tl in token_lists: all_tokens += tl

    return all_tokens

def str_contains_any(s, word_list):
	for w in word_list:
		if w in s or w in s.lower():
			return True
	return False


##################### CMU PRONUNCIATION  #############################


def read_cmu_pronunciation_dict(dict_path):
    """ Method to load the CMU pronunciation dictionary into a python dict
        This dictionary contains UPPERCASE words only.
    """

    lines = open(dict_path).read().split('\n')
    pronunciation_dict = {}
    for line in lines:
        split_line = [l for l in line.split(' ') if l!='']
        if len(split_line) > 1:
            k = split_line[0]
            v = split_line[1:]
            pronunciation_dict[k] = v
    return pronunciation_dict


##################### SWITCHBOARD (Laughter Detection)  #############################

# methods for getting files from Switchboard Corpus
def get_train_val_test_folders(t_root):
    t_folders = [t_root + f for f in os.listdir(t_root) if os.path.isdir(t_root + f)]
    t_folders.sort()
    train_folders = t_folders[0:23]
    val_folders = t_folders[23:26]
    test_folders = t_folders[26:30]
    train_folders.sort(); val_folders.sort(); test_folders.sort()
    return (train_folders, val_folders, test_folders)

def get_transcription_files(folder):
    return [f for f in librosa.util.find_files(folder,ext='text') if f.endswith('word.text')]

def get_laughter_rows_from_file(f):
    #return [l for l in get_text_from_file(f) if 'laughter' in l]
    return [l for l in get_text_from_file(f) if '[laughter]' in l] # doesn't allow laughter with words together

def get_audio_file_from_id(d, all_audio_files):
    files = [f for f in all_audio_files if d in f]
    if len(files) == 1:
        return files[0]
    elif len(files) > 1:
        print("Warning: More than 1 audio file matched id %d" % (int(d)))
        return None
    else:
        print("Warning: No audio file matched id %d" % (int(d)))
        return None

def get_id_from_row(row):
    return row[2:6]

def get_id_from_file(f):
    return get_id_from_row(get_text_from_file(f)[0])

def get_audio_file_from_row(row, all_audio_files):
    return get_audio_file_from_id(get_id_from_row(row))

def get_audio_file_from_transcription_text(t, all_audio_files):
    return get_audio_file_from_id(get_id_from_row(t[0]))

def get_audio_file_from_transcription_file(f, all_audio_files):
    t = open(f).read().split('\n')
    return get_audio_file_from_id(get_id_from_row(t[0]), all_audio_files)

def extract_times_from_row(row):
    return (float(row.split()[1]), float(row.split()[2]))

def get_length_from_transcription_file(t_file):
    try:
        return float(open(t_file).read().split('\n')[-2].split()[-2])
    except:
        print(t_file)

# a_or_b should be either 'A' or 'B' - referring to label of which speaker
def get_transcriptions_files(folder, a_or_b):
    files = []
    subfolders = [folder + "/" + f for f in os.listdir(folder)]
    for f in subfolders:
        fs = [f + "/" + fname for fname in os.listdir(f) if 'a-word.text' in fname and a_or_b in fname]
        files += fs
    files.sort()
    return files

def get_all_transcriptions_files(folder_list, a_or_b):
    files = []
    for folder in folder_list:
        files += get_transcriptions_files(folder, a_or_b)
    files.sort()
    return files

def get_audio_files_from_transcription_files(transcription_files, all_audio_files):
    files = []
    transcription_files_to_remove = []
    for f in transcription_files:
        audio_file = get_audio_file_from_transcription_file(f, all_audio_files)
        if audio_file is None:
            transcription_files_to_remove.append(f)
        else:
            files.append(audio_file)
    #files = list(set(files))
    transcription_files = [t for t in transcription_files if t not in transcription_files_to_remove]
    return transcription_files, files

# Check if laughter is present in a region of an audio file by looking at the transcription file
def no_laughter_present(t_files,start,end):
    for t_file in t_files:
        all_rows = get_text_from_file(t_file)
        for row in all_rows:
            region_start, region_end = extract_times_from_row(row)
            if audio_utils.times_overlap(float(region_start), float(region_end), float(start), float(end)):
                if 'laughter' in row.split()[-1]:
                    return False
    return True

def get_random_speech_region_from_files(t_files, audio_length, region_length):
    contains_laughter = True
    tries = 0
    while(contains_laughter):
        tries += 1
        if tries > 10:
            print("audio length %f" % (audio_length))
            print("region length %f" % (region_length))
            return None
        start = np.random.uniform(1.0, audio_length - region_length - 1.0)
        end = start + region_length
        if no_laughter_present(t_files,start,end):
            contains_laughter = False
    return (start, end)

def get_laughter_regions_from_file(t_file):
    rows = get_laughter_rows_from_file(t_file)
    times = []
    for row in rows:
        try:
            start, end = extract_times_from_row(row)
            if end - start > 0.05:
                times.append((start,end))
        except:
            continue
    return times

def get_text_from_file(f):
    return (open(f).read().split("\n"))[0:-1]

def combine_overlapping_regions(regions_A, regions_B):
    all_regions = regions_A + regions_B
    overlap_found = True
    while(overlap_found):
        i = 0; j = 0
        overlap_found = False
        while i < len(all_regions):
            while j < len(all_regions):
                if i < j:
                    start1 = all_regions[i][0]; end1 = all_regions[i][1]
                    start2 = all_regions[j][0]; end2 = all_regions[j][1]
                    if audio_utils.times_overlap(start1, end1, start2, end2):
                        overlap_found = True
                        all_regions.pop(i); all_regions.pop(j-1)
                        all_regions.append((min(start1, start2), max(end1, end2)))
                j += 1
            i += 1
    return sorted(all_regions, key=lambda r: r[0])

def get_laughter_regions_and_speech_regions(text_A, text_B, audio_file):
    laughter_regions_A = get_laughter_regions_from_file(text_A)
    laughter_regions_B = get_laughter_regions_from_file(text_B)
    laughter_regions = combine_overlapping_regions(laughter_regions_A, laughter_regions_B)
    speech_regions = []
    audio_length = get_length_from_transcription_file(text_A)
    for laughter_region in laughter_regions:
        region_length = laughter_region[1] - laughter_region[0]
        speech_regions.append(get_random_speech_region_from_files([text_A, text_B], audio_length, region_length))
    laughter_regions = [l for l in laughter_regions if l is not None]
    speech_regions = [s for s in speech_regions if s is not None]
    laughter_regions = [audio_utils.start_end_to_offset_duration(s,e) for s,e in laughter_regions]
    speech_regions = [audio_utils.start_end_to_offset_duration(s,e) for s,e in speech_regions]
    # Add padding on each side for windowing later
    laughter_regions = [audio_utils.subsample_time(
        l[0], l[1], audio_length, subsample_length=0.75, padding_length=0.375) for l in laughter_regions]
    speech_regions = [audio_utils.subsample_time(
        s[0], s[1], audio_length, subsample_length=0.75, padding_length=0.375) for s in speech_regions]
    return laughter_regions, speech_regions

def get_laughter_speech_text_lines(t_file_a, t_file_b, a_file):
    lines = []
    laugh_regions, speech_regions = get_laughter_regions_and_speech_regions(
        t_file_a,t_file_b,a_file)
    for r in laugh_regions:
        line = list(r) + [a_file] + [1]
        lines.append('\t'.join([str(l) for l in line]))
    for r in speech_regions:
        line = list(r) + [a_file] + [0]
        lines.append('\t'.join([str(l) for l in line]))
    return lines

def sample_switchboard_features_and_labels(audio_file, t_file_a, t_file_b, feature_fn, **kwargs):
    laughter_regions, speech_regions = get_laughter_regions_and_speech_regions(
        t_file_a, t_file_b, audio_file)
    laughter_feats = audio_utils.featurize_audio_segments(laughter_regions, feature_fn=feature_fn,f=audio_file, sr=8000)
    speech_feats = audio_utils.featurize_audio_segments(speech_regions, feature_fn=feature_fn,f=audio_file, sr=8000)
    X = laughter_feats+speech_feats
    y = list(np.ones(len(laughter_regions))) + list(np.zeros(len(speech_regions)))
    return list(zip(X,y))
