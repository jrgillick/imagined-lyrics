import sys, os, random, time, math, copy, librosa, torch, numpy as np, argparse
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
sys.path.append('../')
from models import *
import audio_utils, dataset_utils, data_loaders, file_utils, torch_utils, text_utils
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter

#############  METROLYRICS TRAINING Phonemes --> Words  #############
# Phoneme Labels --> Words

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--teacher_forcing_ratio', type=str, default='0.3')
parser.add_argument('--lstm_size', type=str, default='256')
parser.add_argument('--batch_size', type=str, default='64')
parser.add_argument('--dropout', type=str, default='0.7')
parser.add_argument('--loss_type', type=str, default='x_ent') # 'ctc' or 'x_ent'
parser.add_argument('--min_word_len', type=str, default='5')
parser.add_argument('--max_word_len', type=str, default='25')
parser.add_argument('--max_label_len', type=str, default='28') 
parser.add_argument('--max_seq_len', type=str, default='60')
parser.add_argument('--n_lines', type=str, default='2')
parser.add_argument('--phoneme_drop_prob', type=str, default='0')
parser.add_argument('--phoneme_swap_prob', type=str, default='0')

args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_path= os.path.join(checkpoint_dir, 'last.pth.tar')

teacher_forcing_ratio = float(args.teacher_forcing_ratio)
lstm_size = int(args.lstm_size)
batch_size = int(args.batch_size)
dropout = float(args.dropout)
loss_type = args.loss_type

min_word_len = int(args.min_word_len)
max_word_len = int(args.max_word_len)

# the number of lyric lines to take per sample (default 2)
n_lines = int(args.n_lines) 
phoneme_drop_prob = float(args.phoneme_drop_prob)
phoneme_swap_prob = float(args.phoneme_swap_prob)

HID_DIM = lstm_size 
ENC_DROPOUT = dropout
DEC_DROPOUT = dropout

# LIBRI SPEECH #
# Set desired range of lengths for subsampling phoneme sequences
max_label_len = int(args.max_label_len)
max_seq_len = int(args.max_seq_len)

# And set up some arguments to be used later
label_fn = dataset_utils.get_timit_phoneme_labels

#subsampling_label_fn_args = {'min_len':min_phoneme_len, 'max_len':max_phoneme_len, 'n_samples':1}
#subsampling_kwargs = {'label_fn_args': subsampling_label_fn_args}


ROOT_DIR = '/data/jrgillick/librispeech/LibriSpeech'

train_wavs = librosa.util.find_files(ROOT_DIR+'/train-clean-100/', ext='flac')
test_wavs = librosa.util.find_files(ROOT_DIR+'/dev-clean/', ext='flac')

def get_libri_speech_phoneme_file_from_audio_file(phoneme_file):
    return phoneme_file.replace(".flac", ".phn")

def get_libri_speech_audio_file_from_phoneme_file(audio_file):
    return audio_file.replace(".phn", ".flac")

train_phoneme_files = [get_libri_speech_phoneme_file_from_audio_file(f) for f in train_wavs]
test_phoneme_files = [get_libri_speech_phoneme_file_from_audio_file(f) for f in test_wavs]

glove = text_utils.glove2dict("/data/jrgillick/word-vectors/glove.6B.100d.txt")

d = dataset_utils.read_cmu_pronunciation_dict('/data/jrgillick/librispeech/LibriSpeech/english.dict')
input_vocab = text_utils.make_vocab(
    filepaths=['/home/jrgillick/projects/phoneme/data/libri_speech_phonemes.vocab'],
    token_fn=lambda x: open(x).read().split('\n'),
    include_pad_symbol=True, include_start_symbol=True,
    include_end_symbol=True, include_oov_symbol=True,
    standard_special_symbols=True
)

word_output_vocab = text_utils.make_vocab(filepaths=['/data/jrgillick/metrolyrics/vocabs/words.txt'],
                token_fn=lambda f: list(text_utils.glove2dict(f).keys()),
                include_pad_symbol=True, include_start_symbol=True,
                include_end_symbol=True, include_oov_symbol=True,
                standard_special_symbols=True)

reverse_word_output_vocab = text_utils.make_reverse_vocab(word_output_vocab)

output_embedding_matrix = text_utils.make_embedding_matrix(glove, word_output_vocab, 100)

word_to_phoneme_subsampling_args = {'n_samples':1,
                                    'n_lines':n_lines,
                                    'phoneme_drop_prob': phoneme_drop_prob,
                                    'phoneme_swap_prob': phoneme_swap_prob,
                                    'phoneme_dict': d,
                                    'phoneme_vocab': input_vocab}


# FOR METROLYRICS
metrolyrics_db_file = '/data/jrgillick/metrolyrics/lyrics.csv'

train_dataset = data_loaders.OneFileDataset(metrolyrics_db_file, 
                    load_fn=dataset_utils.load_metrolyrics_from_csv,
                     feature_and_label_fn=dataset_utils.sample_metrolyrics_features_and_labels,
                     start_index=0, end_index=200000,
                     **word_to_phoneme_subsampling_args)

test_dataset = data_loaders.OneFileDataset(metrolyrics_db_file, 
                    load_fn=dataset_utils.load_metrolyrics_from_csv,
                     feature_and_label_fn=dataset_utils.sample_metrolyrics_features_and_labels,
                     start_index=200000, end_index=253962,
                     **word_to_phoneme_subsampling_args)

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                    sequence_pad_value=0, label_pad_value=text_utils.PAD_SYMBOL, 
                    max_seq_len=max_seq_len,max_label_len=max_label_len,
                    input_vocab=input_vocab, output_vocab=word_output_vocab,
                    one_hot_inputs=True)

training_generator = torch.utils.data.DataLoader(
    train_dataset, num_workers=2, batch_size=64, shuffle=True,
    collate_fn=collate_fn)

test_generator = torch.utils.data.DataLoader(
    test_dataset, num_workers=2, batch_size=64, shuffle=True,
    collate_fn=collate_fn)

INPUT_DIM = len(input_vocab) 
OUTPUT_DIM = len(word_output_vocab)
DEC_EMB_DIM = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, HID_DIM, ENC_DROPOUT)
dec = EmbeddingDecoder(OUTPUT_DIM, HID_DIM, DEC_EMB_DIM, output_embedding_matrix, word_output_vocab, DEC_DROPOUT, device)

model = Seq2Seq(enc, dec, device, max_label_len).to(device)

torch_utils.count_parameters(model)
model.apply(torch_utils.init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = word_output_vocab[text_utils.PAD_SYMBOL])


if os.path.exists(checkpoint_path):
    torch_utils.load_checkpoint(checkpoint_path, model, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)
    print("Beginning training...")

print("Batch Size: ", batch_size)


def text_log(checkpoint_dir):
    text_path = os.path.join(checkpoint_dir, 'sample_preds.txt')
    i = np.random.randint(len(test_phoneme_files))
    device = model.device
    model.set_device(torch.device('cpu'))
    p = torch_utils.OneFileDatasetPredictor(test_dataset, index=i, model=model,
              reverse_vocab=reverse_word_output_vocab,
              collate_fn=collate_fn)
    preds, true_labs = p.predict()[0]
    pred_comp = list(zip(preds[0], true_labs[0]))
    with open(text_path, 'a') as f:
        f.write("############ Prediction at step: " + str(model.global_step) + " ############\n")
        for line in pred_comp:
            f.write(str(line)+"\n")
    model.set_device(device)

epochs = 10

for i in range(epochs):
    torch_utils.run_training_loop(n_epochs=1, model=model, device=device, loss_type=loss_type, checkpoint_dir=checkpoint_dir, 
        optimizer=optimizer, iterator=training_generator, teacher_forcing_ratio=teacher_forcing_ratio, val_iterator=test_generator,
        gradient_clip=1., verbose=True)

    text_log(checkpoint_dir)



