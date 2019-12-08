import sys
sys.path.append('../utils/')
sys.path.append('../')
import dataset_utils, json

profane_words_path = '../data/profane-words/words.json'
with open(profane_words_path) as f:
    profane_words = json.load(f)

ml_tokens = dataset_utils.load_all_metrolyrics_words('../data/metrolyrics/lyrics.csv', banned_words = profane_words)
big_str = ' '.join(ml_tokens)
with open('../data/metrolyrics/texts/words.txt', 'w') as f:
    f.write(big_str.lower())

