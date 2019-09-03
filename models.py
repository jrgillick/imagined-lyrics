import sys, random, copy, numpy as np, librosa
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
import audio_utils, dataset_utils, data_loaders, file_utils, torch_utils, text_utils
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Encoder(nn.Module):
	def __init__(self, input_dim, hid_dim, dropout):
		super().__init__()

		self.input_dim = input_dim
		self.hid_dim = hid_dim

		#self.dropout = nn.Dropout(dropout)
		self.rnn = nn.GRU(input_dim, hid_dim, num_layers=2, dropout=dropout)

	def forward(self, src):
		#src = [src sent len, batch size, input_dim]
		#src = self.dropout(src)

		outputs, hidden = self.rnn(src) #no cell state

		# outputs dim: [src sent len, batch size, hid dim * n directions]
		# hidden dim: [n layers * n directions, batch size, hid dim]

		#outputs are always from the top hidden layer
		return hidden


class Decoder(nn.Module):
	def __init__(self, output_dim, hid_dim, dropout, device):
		super().__init__()

		self.hid_dim = hid_dim
		self.output_dim = output_dim
		self.dropout = dropout
		self.device = device


		self.rnn = nn.GRU(output_dim + hid_dim, hid_dim, num_layers=2)

		#self.projection_dim = 80
		#self.projection = nn.Linear(output_dim + hid_dim * 2, self.projection_dim)

		self.out = nn.Linear(output_dim + hid_dim * 2, output_dim)
		#self.out = nn.Linear(self.projection_dim, self.output_dim)

		self.dropout = nn.Dropout(dropout)

		#self._y_onehot = torch.FloatTensor(batch_size, output_dim).to(device)

	# project to self.projection_dim, then to output_dim, with dropout
	def output_forward(self, rnn_output):
		projected = self.dropout(self.projection(rnn_output))
		prediction = self.out(projected)
		return prediction

	def forward(self, input, hidden, context):

		# input dim: [batch size, output_dim]
		# hidden dim: = [n layers * n directions, batch size, hid dim]
		# context dim: = [n layers * n directions, batch size, hid dim]
		input = input.unsqueeze(1) #[batch size, 1]

		# Make the inputs (labels at previous timesteps) one_hots here
		input = torch_utils.torch_one_hot(input, self.device, n_dims = self.output_dim)
		input = input.permute((1,0,2))
		input = input.repeat((2,1,1)) # 2 Layer GRU

		#dim = [1, batch size, emb dim + hid dim]
		input_and_context = torch.cat((input, context), dim = 2)

		input_and_context = self.dropout(input_and_context)
		output, hidden = self.rnn(input_and_context, hidden)

		# output dim: [sent len, batch size, hid dim * n directions]
		# hidden dim: [n layers * n directions, batch size, hid dim]

		output = torch.cat((input[:-1].squeeze(0), hidden[:-1].squeeze(0),
			context[:-1].squeeze(0)), dim = 1)

		# output dim: [batch size, output_dim + hid dim * 2]

		prediction = self.out(output) # dim: [batch size, output dim]
		#prediction = self.output_forward(output)

		return prediction, hidden

class EmbeddingDecoder(nn.Module):
	def __init__(self, output_dim, hid_dim, embedding_dim, embedding_matrix, output_vocab, dropout, device):
		super().__init__()

		self.hid_dim = hid_dim
		self.output_dim = output_dim
		self.embedding_dim = embedding_dim
		self.dropout = dropout
		self.device = device

		self.embeds = torch_utils.create_embedding_layer(embedding_matrix)
		self.output_vocab = output_vocab

		self.rnn = nn.GRU(embedding_dim + hid_dim, hid_dim, num_layers=2)

		#self.projection_dim = 80
		#self.projection = nn.Linear(output_dim + hid_dim * 2, self.projection_dim)

		self.out = nn.Linear(embedding_dim + hid_dim * 2, output_dim)
		#self.out = nn.Linear(self.projection_dim, self.output_dim)

		self.dropout = nn.Dropout(dropout)

	# project to self.projection_dim, then to output_dim, with dropout
	def output_forward(self, rnn_output):
		projected = self.dropout(self.projection(rnn_output))
		prediction = self.out(projected)
		return prediction

	def forward(self, input, hidden, context):
		# input dim: [batch size, output_dim]
		# hidden dim: [n layers * n directions, batch size, hid dim]
		# context dim: [n layers * n directions, batch size, hid dim]

		input = input.unsqueeze(1) # dim: [batch_size, 1]

		# Make the inputs (labels at previous timesteps) type Long
		input=input.type(torch.LongTensor).to(self.device)

		# Get the input embeddings
		input = self.embeds(input)

		input = input.permute((1,0,2))
		input = input.repeat((2,1,1))

		input_and_context = torch.cat((input, context), dim = 2)

		input_and_context = self.dropout(input_and_context)
		output, hidden = self.rnn(input_and_context, hidden)

		# output dim: [sent len, batch size, hid dim * n directions]
		# hidden dim: [n layers * n directions, batch size, hid dim]

		output = torch.cat((input[:-1].squeeze(0), hidden[:-1].squeeze(0),
			context[:-1].squeeze(0)), dim = 1) #[batch size, output_dim + hid dim * 2]

		prediction = self.out(output)
		#prediction = self.output_forward(output)

		return prediction, hidden


class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, device, max_label_len, start_symbol_value=1):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.device = device
		self.max_label_len = max_label_len
		self.start_symbol_value = start_symbol_value

		self.global_step = 0
		self.epoch = 0
		self.best_val_loss = np.inf

		assert encoder.hid_dim == decoder.hid_dim, \
			"Hidden dimensions of encoder and decoder must be equal!"

	def increment_global_step():
		self.global_step += 1

	def set_device(self, device):
		self.decoder.device = device
		self.device = device
		self.to(device)

	def forward(self, src, trg=None, teacher_forcing_ratio = 0.7):
		# src dim: [src sent len, batch size, input_dim]
		# trg dim: [trg sent len, batch size]

		#src = src.permute((1,0,2))
		batch_size = src.shape[1]

		if trg is not None:
			max_len = trg.shape[0]
		else:
			max_len = self.max_label_len
		trg_vocab_size = self.decoder.output_dim

		#tensor to store decoder outputs
		outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

		#last hidden state of the encoder is the context for the decoder
		context = self.encoder(src)

		#context also used as the initial hidden state of the decoder
		hidden = context

		#first input to the decoder is the <sos> tokens
		if trg is not None:
			input = trg[0,:]
		else:
			#input = torch.Tensor(batch_size).fill_(output_vocab[text_utils.START_SYMBOL]).float().to(self.device)
			input = torch.Tensor(batch_size).fill_(self.start_symbol_value).float().to(self.device)

		for t in range(1, max_len):
			output, hidden = self.decoder(input, hidden, context)
			outputs[t] = output

			teacher_force = random.random() < teacher_forcing_ratio

			m = Categorical(logits=output) # Sampling
			top1 = m.sample()

			#top1 = output.max(1)[1] #Argmax

			if trg is not None:
				input = (trg[t] if teacher_force else top1)
			else:
				input = top1

		# fix the final outputs to all start with start symbol instead of PAD
		# Just for ease of printout...
		outputs[0, :] = 0
		#outputs[0, :, output_vocab[text_utils.START_SYMBOL]] = 1
		outputs[0, :, 1] = 1

		return outputs








############ PREDICTION METHODS ############

def predict(model, input_file, label_file=None, reverse_output_vocab=None):
	if label_file is None:
		tmp_dataset = data_loaders.AudioDataset([input_file],
			feature_fn=audio_utils.featurize_mfcc, **subsampling_kwargs)

	else:
		tmp_dataset = data_loaders.AudioDataset([input_file],
			feature_fn=audio_utils.featurize_mfcc, label_paths=[label_file],
			label_fn=dataset_utils.get_timit_phoneme_labels,
			**subsampling_kwargs)

	tmp_generator = torch.utils.data.DataLoader(
		tmp_dataset, num_workers=0, batch_size=1, shuffle=False,
		collate_fn=collate_fn)

	batch = [batch for i_batch, batch in enumerate(tmp_generator)][0]

	seqs, labs = batch

	# Run model forward
	with torch.no_grad():
		src = torch.from_numpy(np.array(seqs)).float().permute((1,0,2))#.to(device)
		output = model(src) # No labels given

	# Remove batch dimension, get arxmax from one hot, and convert to numpy
	output_seq = np.argmax(output.cpu().numpy().squeeze(), axis=1)

	# Convert to readable labels with reverse vocab
	return output_seq, text_utils.readable_outputs(output_seq, reverse_output_vocab)

def predict_subsampling(model, input_file, label_file, reverse_output_vocab=None):
	#input_feats = np.array([featurize_chars(input_file, output_vocab)])

	tmp_dataset = data_loaders.AudioDataset([input_file],
				 feature_and_label_fn=dataset_utils.sample_timit_features_and_labels,
				 label_paths=[label_file], feature_fn=audio_utils.featurize_mfcc,
				 does_subsample=True, **subsampling_kwargs)

	tmp_generator = torch.utils.data.DataLoader(
		tmp_dataset, num_workers=0, batch_size=1, shuffle=False,
		collate_fn=collate_fn)

	batch = [batch for i_batch, batch in enumerate(tmp_generator)][0]

	seqs, labs = batch

	seqs = seqs[0:1]
	labs = labs[0:1]

	# Run model forward
	with torch.no_grad():
		src = torch.from_numpy(np.array(seqs)).float().permute((1,0,2))#.to(device)
		output = model(src, teacher_forcing_ratio=0) # No labels given

	# Remove batch dimension, get arxmax from one hot, and convert to numpy
	output_seq = np.argmax(output.cpu().numpy().squeeze(), axis=1)

	# Convert to readable labels with reverse vocab
	return output_seq, text_utils.readable_outputs(
		output_seq, reverse_output_vocab), text_utils.readable_outputs(
		np.array(labs[0]), reverse_output_vocab)

def dedup_list(l):
	l = copy.deepcopy(l)
	i = 1
	while i < len(l):
		if l[i] == l[i-1]:
			del l[i]
		else:
			i += 1
	return l

def predict_windows(model, input_file, hop_length=1., window_length=1.,
	sr=16000, offset=0, duration=None, dedup=True, reverse_output_vocab=None):

	# windows of the form (start, duration)
	def _get_time_windows(total_length, hop_length,
		window_length, offset, duration):
		if duration is None:
			latest_window_start = total_length - (window_length + offset)
		else:
			latest_window_start = offset + duration - (window_length + offset)
		windows = []
		start = offset
		while start < latest_window_start:
			windows.append((start, window_length))
			start += hop_length
		return windows

	# First get a list of times to do the windowing over the file
	file_length = librosa.core.get_duration(filename=input_file)

	windows = _get_time_windows(file_length, hop_length, window_length,
		offset, duration)

	seqs = audio_utils.featurize_audio_segments(windows,
		audio_utils.featurize_mfcc, f=input_file)

	padded_seqs = audio_utils.keras_pad_seqs(seqs, maxlen=100, dtype='float32',
		padding='pre', truncating='post', value=0)

	# Run model forward
	with torch.no_grad():
		src = torch.from_numpy(np.array(seqs)).float().permute((1,0,2))#.to(device)
		encodings = model.encoder(src)
		output = model(src) # No labels given

	# get arxmax from one hot, and convert to numpy
	output_seqs = np.argmax(output.cpu().detach().numpy().squeeze(), axis=2)

	# Convert to readable labels with reverse vocab
	readable_seqs = []
	for output_seq in output_seqs.T:
		readable_seq = text_utils.readable_outputs(output_seq, reverse_output_vocab)
		readable_seq = list(filter(lambda e: e not in [text_utils.START_SYMBOL, text_utils.END_SYMBOL], readable_seq) )
		readable_seq = ['pause' if x is text_utils.OOV_SYMBOL else x for x in readable_seq]
		if dedup:
			readable_seqs.append(dedup_list(readable_seq))
		else:
			readable_seqs.append(readable_seq)
	return output_seqs, readable_seqs
