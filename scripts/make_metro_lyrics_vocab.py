import audio_utils

ml_tokens = audio_utils.load_all_metrolyrics_words('/data/jrgillick/metrolyrics/lyrics.csv')
big_str = ' '.join(ml_tokens)
with open('/data/jrgillick/metrolyrics/texts/words.txt', 'w') as f:
    f.write(big_str.lower())

#python get_vocab_counts.py --glove_dir=/data/jrgillick/projects/long_text/GloVe/ --input_files_dir=/data/jrgillick/metrolyrics/texts/ --output_files_dir=/data/jrgillick/metrolyrics/vocabs/ --min_count=24