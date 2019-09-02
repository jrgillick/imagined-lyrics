#python make_libris_peech_phoneme_files_from_dictionary.py /data/jrgillick/librispeech/LibriSpeech/english.dict /data/jrgillick/librispeech/LibriSpeech/train-clean-100

import os, sys, librosa
from tqdm import tqdm

def read_pronunciation_dict(dict_path):
    lines = open(dict_path).read().split('\n')
    pronunciation_dict = {}
    for line in lines:
        split_line = [l for l in line.split(' ') if l!='']
        if len(split_line) > 1:
            k = split_line[0]
            v = split_line[1:]
            pronunciation_dict[k] = v
    return pronunciation_dict

def make_phoneme_files(librispeech_text_file, pronunciation_dict):
    directory = os.path.dirname(librispeech_text_file)
    lines = open(librispeech_text_file).read().split('\n')
    for line in lines:
        # Just skip the whole sentence if there's an OOV word
        try:
            toks = line.split(' ')
            path = toks[0]
            words = toks[1:]
            phonemes = []
            phonemes += ['h#']
            for word in words:
                phonemes += pronunciation_dict[word.upper()]
            phonemes += ['h#']
            output_file = os.path.join(directory, path) + '.phn'
            with open(output_file, 'w') as f:
                out_str = '\n'.join(phonemes)
                f.write(out_str)
        except:
            print(line)

if __name__ == '__main__':
    pronunciation_dict = sys.argv[1]
    libri_speech_dir = sys.argv[2]

    pronunciation_dict = read_pronunciation_dict(pronunciation_dict)
    train_texts = librosa.util.find_files(libri_speech_dir, ext='txt')
    for librispeech_text_file in tqdm(train_texts):
        make_phoneme_files(librispeech_text_file, pronunciation_dict)

