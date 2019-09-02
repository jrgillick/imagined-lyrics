#python make_librispeech_phoneme_files.py /data/jrgillick/librispeech/LibriSpeech

import pandas as pd, numpy as np, librosa, sys
from praatio import tgio
from tqdm import tqdm

# assumes 16K sample rate to match TIMIT
def secs_to_samps(secs):
    return int(secs * 16000)

def process_text_grid_file(text_grid_path):
    out_path = text_grid_path.replace('.TextGrid', '.phn')
    tg = tgio.openTextgrid(text_grid_path)
    phoneTier = tg.tierDict['phones']
    entryList = phoneTier.entryList
    matrix = [[secs_to_samps(start), secs_to_samps(end), label] for start, end, label in entryList]
    df = pd.DataFrame(np.array(matrix))
    df.to_csv(out_path, header=False, index=False, sep=' ')
    
if __name__ == '__main__':
    libri_speech_dir = sys.argv[1]
    textgrid_files = librosa.util.find_files(libri_speech_dir, ext='TextGrid', case_sensitive=True)
    print(len(textgrid_files))
    for f in tqdm(textgrid_files):
        process_text_grid_file(f)