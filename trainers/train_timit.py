import sys, os, random, time, math, copy, librosa, torch, numpy as np, argparse
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
sys.path.append('../')
from models import *
import audio_utils, dataset_utils, data_loaders, file_utils, torch_utils, text_utils
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter

#############  TIMIT TRAINING  #############
# Full sentences w/ no subsampling
# Audio -->> Phoneme Labels

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--teacher_forcing_ratio', type=str, default='0.3')
parser.add_argument('--lstm_size', type=str, default='128')
parser.add_argument('--batch_size', type=str, default='32')
parser.add_argument('--dropout', type=str, default='0.7')
parser.add_argument('--loss_type', type=str, default='x_ent') # 'ctc' or 'x_ent'

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


def get_phoneme_path_from_audio_path(f):
    return f.replace('.WAV','.PHN')

ROOT_DIR = '/data/jrgillick/timit/'

print("Loading data...")
train_wavs = librosa.util.find_files(ROOT_DIR+'/TRAIN/')
train_wavs = [t for t in train_wavs if not t.endswith('.WAV.wav')][0:4608]

test_wavs = librosa.util.find_files(ROOT_DIR+'/TEST/')
test_wavs = [t for t in test_wavs if not t.endswith('.WAV.wav')][0:1664]

train_phoneme_files = [get_phoneme_path_from_audio_path(f) for f in train_wavs]
test_phoneme_files = [get_phoneme_path_from_audio_path(f) for f in test_wavs]

print("Training on ", len(train_wavs), " audio files.")
print("Evaluating on ", len(test_wavs), " audio files.")

print("Loading vocab...")
output_vocab = text_utils.make_vocab(filepaths=train_phoneme_files,
                token_fn=dataset_utils.get_timit_phoneme_labels,
                include_pad_symbol=True, include_start_symbol=True,
                include_end_symbol=True, include_oov_symbol=True,
                standard_special_symbols=True)

reverse_output_vocab = text_utils.make_reverse_vocab(output_vocab)

label_fn = dataset_utils.get_timit_phoneme_labels

label_seq_lens = [len(dataset_utils.get_timit_phoneme_labels(f)) for f in tqdm(train_phoneme_files)]
max_label_len = max(label_seq_lens)

#NO SUBSAMPLING
basic_label_fn_args = {'vocab': output_vocab}
basic_feature_fn_args = {'vocab': output_vocab}
basic_kwargs = {'label_fn_args': basic_label_fn_args, 'feature_fn_args': basic_feature_fn_args}

train_dataset = data_loaders.AudioDataset(train_wavs, feature_fn=audio_utils.featurize_mfcc,
                 label_paths=train_phoneme_files, 
                 label_fn=label_fn, **basic_kwargs)

test_dataset = data_loaders.AudioDataset(test_wavs, feature_fn=audio_utils.featurize_mfcc,
                 label_paths=train_phoneme_files, 
                 label_fn=label_fn, **basic_kwargs)

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                    sequence_pad_value=0, label_pad_value=text_utils.PAD_SYMBOL, 
                    max_seq_len=450, max_label_len=max_label_len,
                    output_vocab=output_vocab, one_hot_labels=False)

training_generator = torch.utils.data.DataLoader(
    train_dataset, num_workers=8, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn)

test_generator = torch.utils.data.DataLoader(
    test_dataset, num_workers=8, batch_size=batch_size, shuffle=True,
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

epochs = 10

for i in range(epochs):
    torch_utils.run_training_loop(n_epochs=1, model=model, device=device, loss_type=loss_type, checkpoint_dir=checkpoint_dir, 
        optimizer=optimizer, iterator=training_generator, teacher_forcing_ratio=teacher_forcing_ratio, val_iterator=test_generator,
        gradient_clip=1., verbose=True)

    text_log(checkpoint_dir)

