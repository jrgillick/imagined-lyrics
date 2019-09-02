#python make_pronunciation_vocab.py /data/jrgillick/librispeech/LibriSpeech/english.dict /home/jrgillick/projects/phoneme/libri_speech_phonemes.vocab

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

if __name__ == '__main__':
    dict_path = sys.argv[1]
    output_vocab_file = sys.argv[2]
    d = read_pronunciation_dict(dict_path)

    phoneme_lists = list(d.values())

    vocab = []
    for l in phoneme_lists:
        vocab += l
        vocab = list(set(vocab))
    vocab = sorted(vocab)

    with open(output_vocab_file, 'w') as f:
        out_str = '\n'.join(vocab)
        f.write(out_str)
