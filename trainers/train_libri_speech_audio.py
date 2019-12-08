import sys, os, random, time, math, copy, librosa, torch, numpy as np, argparse
sys.path.append('../data/utils/')
sys.path.append('../')
from models import *
import audio_utils, dataset_utils, data_loaders, file_utils, torch_utils, text_utils
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter

#############  LIBRISPEECH TRAINING  #############
# 5-10 words w/ subsampling
# Audio -->> Phoneme Labels

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--teacher_forcing_ratio', type=str, default='0.3')
parser.add_argument('--lstm_size', type=str, default='128')
parser.add_argument('--batch_size', type=str, default='16')
parser.add_argument('--dropout', type=str, default='0.7')
parser.add_argument('--loss_type', type=str, default='ctc') # 'ctc' or 'x_ent'
parser.add_argument('--min_phoneme_len', type=str, default='5')
parser.add_argument('--max_phoneme_len', type=str, default='10')
parser.add_argument('--max_label_len', type=str, default='13') # should be ~max_phoneme_len + 3
parser.add_argument('--max_seq_len', type=str, default='180')

args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_path= os.path.join(checkpoint_dir, 'last.pth.tar')

teacher_forcing_ratio = float(args.teacher_forcing_ratio)
lstm_size = int(args.lstm_size)
batch_size = int(args.batch_size)
dropout = float(args.dropout)
loss_type = args.loss_type

HID_DIM = lstm_size 
ENC_DROPOUT = dropout
DEC_DROPOUT = dropout

# LIBRI SPEECH #
# Set desired range of lengths for subsampling phoneme sequences
min_phoneme_len = int(args.min_phoneme_len)
max_phoneme_len = int(args.max_phoneme_len)
max_label_len = int(args.max_label_len)
max_seq_len = int(args.max_seq_len)

# And set up some arguments to be used later
label_fn = dataset_utils.get_timit_phoneme_labels

subsampling_label_fn_args = {'min_len':min_phoneme_len, 'max_len':max_phoneme_len, 'n_samples':1}
subsampling_kwargs = {'label_fn_args': subsampling_label_fn_args}


ROOT_DIR = '/data/jrgillick/librispeech/LibriSpeech'

train_wavs = librosa.util.find_files(ROOT_DIR+'/train-clean-100/', ext='flac')
test_wavs = librosa.util.find_files(ROOT_DIR+'/dev-clean/', ext='flac')

def get_libri_speech_phoneme_file_from_audio_file(phoneme_file):
    return phoneme_file.replace(".flac", ".phn")

def get_libri_speech_audio_file_from_phoneme_file(audio_file):
    return audio_file.replace(".phn", ".flac")

train_phoneme_files = [get_libri_speech_phoneme_file_from_audio_file(f) for f in train_wavs]
test_phoneme_files = [get_libri_speech_phoneme_file_from_audio_file(f) for f in test_wavs]

# Then we'll need to remove any files from the data whose total length is less than the minimum subsampling length
# And also filter any missing or incorrect file paths

bad_train_indices = []
bad_test_indices = []

print("Loading data...")

for i in tqdm(range(len(train_wavs))):
    if not (os.path.exists(train_wavs[i]) and os.path.exists(train_phoneme_files[i]) ):
        bad_train_indices.append(i)
    elif (len(label_fn(train_phoneme_files[i])) - 2 <= min_phoneme_len): # don't count START and END in length measurements
        bad_train_indices.append(i)

for i in tqdm(range(len(test_wavs))):
    if not (os.path.exists(test_wavs[i]) and os.path.exists(test_phoneme_files[i]) ):
        bad_test_indices.append(i)
    elif (len(label_fn(test_phoneme_files[i])) - 2 <= min_phoneme_len): # don't count START and END in length measurements
        bad_test_indices.append(i)

train_wavs = [f for i, f in enumerate(train_wavs) if i not in bad_train_indices]
train_phoneme_files = [f for i, f in enumerate(train_phoneme_files) if i not in bad_train_indices]

test_wavs = [f for i, f in enumerate(test_wavs) if i not in bad_test_indices]
test_phoneme_files = [f for i, f in enumerate(test_phoneme_files) if i not in bad_test_indices]
        
print(len(train_wavs), len(test_wavs))

output_vocab = text_utils.make_vocab(
    filepaths=['/home/jrgillick/projects/phoneme/data/libri_speech_phonemes.vocab'],
    token_fn=lambda x: open(x).read().split('\n'),
    include_pad_symbol=True, include_start_symbol=True,
    include_end_symbol=True, include_oov_symbol=True,
    standard_special_symbols=True
)

reverse_output_vocab = text_utils.make_reverse_vocab(output_vocab)


# WITH SUBSAMPLING


train_dataset = data_loaders.AudioDataset(train_wavs, 
                 feature_and_label_fn=dataset_utils.sample_timit_features_and_labels,
                 label_paths=train_phoneme_files, feature_fn=audio_utils.featurize_mfcc,
                 does_subsample=True, **subsampling_kwargs)

test_dataset = data_loaders.AudioDataset(test_wavs,
                 feature_and_label_fn=dataset_utils.sample_timit_features_and_labels,
                 label_paths=test_phoneme_files, feature_fn=audio_utils.featurize_mfcc,
                 does_subsample=True, **subsampling_kwargs)

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                    sequence_pad_value=0, label_pad_value=text_utils.PAD_SYMBOL, 
                    max_seq_len=max_seq_len,max_label_len=max_label_len,
                    output_vocab=output_vocab, one_hot_labels=False)

training_generator = torch.utils.data.DataLoader(
    train_dataset, num_workers=6, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn)

test_generator = torch.utils.data.DataLoader(
    test_dataset, num_workers=6, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn)




INPUT_DIM = 37
OUTPUT_DIM = len(output_vocab)


print("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_DROPOUT, device)

model = Seq2Seq(enc, dec, device, max_label_len).to(device)

torch_utils.count_parameters(model)
model.apply(torch_utils.init_weights)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index = output_vocab[text_utils.PAD_SYMBOL])
criterion_ctc = nn.CTCLoss(blank=output_vocab[text_utils.PAD_SYMBOL])




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
    p = torch_utils.Predictor(train_dataset,test_wavs[i:i+1],
        model=model.cpu(),label_paths=test_phoneme_files[i:i+1],collate_fn=collate_fn,
        reverse_vocab=reverse_output_vocab,label_fn=label_fn)
    preds, true_labs = p.predict()[0]
    pred_comp = list(zip(preds[0], true_labs[0]))
    with open(text_path, 'a') as f:
        f.write("############ Prediction at step: " + str(model.global_step) + " ############\n")
        for line in pred_comp:
            f.write(str(line)+"\n")
    model.set_device(device)


while model.global_step < 200000:
    torch_utils.run_training_loop(n_epochs=1, model=model, device=device, loss_type=loss_type, checkpoint_dir=checkpoint_dir, 
        optimizer=optimizer, iterator=training_generator, teacher_forcing_ratio=teacher_forcing_ratio, val_iterator=test_generator,
        gradient_clip=1., verbose=True)

    text_log(checkpoint_dir)



