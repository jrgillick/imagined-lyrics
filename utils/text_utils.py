import numpy as np, csv
from collections import defaultdict


"""
# Vocab Utils
"""

PAD_SYMBOL = '###_PAD_###'  #  -> 0
START_SYMBOL = '###_START_###'  #  -> 1
END_SYMBOL = '###_END_###'  #  -> 2
OOV_SYMBOL = '###_OOV_###'  #  -> 3


def make_vocab(filepaths=None, token_fn=None, token_lists=None,
    include_start_symbol=False, include_end_symbol=False,
    include_oov_symbol=False, include_pad_symbol=False,
    standard_special_symbols=False, verbose=False):
    """ Create a vocabulary dict for a dataset.
    Accepts either a list of filepaths together with a `token_fn` to read and
    tokenize the files, or a list of token_lists that have already been
    processed. Optionally includes special symbols.

    In order to make it easy to adding padding, start/end, or OOV tokens
    at other times, it's helpful to give special entries standard values, which
    can be set by setting standard_special_symbols=True.

    '###_PAD_###' --> 0
    '###_START_###' --> 1
    '###_END_###' --> 2
    '###_OOV_###' --> 3
    """

    # Validate args
    if bool(filepaths) and bool(token_lists):
            raise Exception("You should only pass one of `filepaths` and `token_lists`")

    if bool(filepaths) ^ bool(token_fn):
            raise Exception("Can't use only one of `filepaths` and `token_fn`")

    if standard_special_symbols and not (include_start_symbol and \
        include_end_symbol and include_oov_symbol and include_pad_symbol):
        raise Exception("standard_special_symbols needs to include all 4 symbol.")

    # Initialize special symbols
    special_symbols = []
    if include_pad_symbol:
        special_symbols.append(PAD_SYMBOL)
    if include_start_symbol:
        special_symbols.append(START_SYMBOL)
    if include_end_symbol:
        special_symbols.append(END_SYMBOL)
    if include_oov_symbol:
        special_symbols.append(OOV_SYMBOL)

    counter = 0

    # Make vocab dict and initialize with special symbols
    vocab = {}
    for sym in special_symbols:
        vocab[sym] = counter
        counter += 1

    if token_lists is None: # Get tokens from filepaths and put in token_lists
        if verbose:
            token_lists = [token_fn(f) for f in tqdm(filepaths)]
        else:
            token_lists = [token_fn(f) for f in filepaths]

    # Loop through tokens and add to vocab
    if verbose: token_lists = tqdm(token_lists)

    for sequence in token_lists:
        for token in sequence:
            if token not in vocab:
                vocab[token] = counter
                counter += 1

    return vocab

def make_reverse_vocab(vocab, default_type=str, merge_fn=None):
    # Flip the keys and values in a dict.
    """ Straightforward function unless the values of the vocab are 'unhashable'
        i.e. a list. For example, a phoneme dictionary maps 'SAY' to
        ['S', 'EY1']. In this case, pass in a function merge_fn, which specifies
        how to combine the list items into a hashable key. This could be a
        lambda fn, e.g merge_fn = lambda x: '_'.join(x).

        It's also possible that there could be collisions - e.g. with
        homophones. If default_type is list, collisions will be combined into
        a list. If not, they'll be overwritten.

        Args:
            merge_fn: a function to combine lists into hashable keys

    """
    rv = defaultdict(default_type)
    for k in vocab.keys():
        if merge_fn is not None:
            if default_type is list:
                rv[merge_fn(vocab[k])].append(k)
            else:
                rv[merge_fn(vocab[k])] = k
        else:
            if default_type is list:
                rv[vocab[k]].append(k)
            else:
                rv[vocab[k]] = k
    return rv

def filter_vocab(vocab, word_list):
    # Filters a vocab dict to only words in the given word_list
    v = {}
    for key, value in tqdm(vocab.items()):
        if key in word_list:
            v[key] = value
    return v

def readable_outputs(seq, reverse_vocab):
    """ Convert a sequence of output indices to readable string outputs
    from a given (reverse) vocab """
    return [reverse_vocab[s] for s in seq]

def sequence_to_indices(sequence, vocab, one_hot=False):
    """ Converts a sequence to a list of indices using a given vocab dict.
    Checks for OOV tokens along the way, with the assumption that the standard
    OOV token is OOV_SYMBOL.

    This can be used for labels in multiclass classification or for
    inputs to models with discrete inputs.

    Example: Suppose `vocab` = {OOV_SYMBOL:0, 'hi': 1, 'there': 2}, and
    `sequence` =  ['hi', 'there' 'Bob'].
    Returns [1, 2, 0]
    """

    include_oov_symbol = OOV_SYMBOL in vocab

    # Look up tokens in vocab
    indices = []
    for token in sequence:
        if token not in vocab:
            if include_oov_symbol:
                indices.append(vocab[OOV_SYMBOL])
            else:
                raise Exception("Out of Vocab token but no OOV Symbol in vocab")
        else:
            indices.append(vocab[token])

    if one_hot:
        return np_onehot(indices, depth=len(vocab))
    else:
        return indices

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

"""
# Modeling Utils
"""

def np_onehot(indices, depth, dtype=np.int32):
    """Converts 1D array of indices to a one-hot 2D array with given depth."""
    onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
    onehot_seq[np.arange(len(indices)), indices] = 1.0
    return onehot_seq

def make_embedding_matrix(embedding_dict, vocab, embedding_dim, scale=0.01):
	matrix_len = len(vocab)
	weight_matrix = np.zeros((matrix_len, embedding_dim))
	counter = 0
	for i, word in enumerate(vocab):
	    try:
	        weight_matrix[i] = embedding_dict[word]
	        counter += 1
	    except KeyError:
	        weight_matrix[i] = np.random.normal(scale=scale,
	        	size=(embedding_dim,))
	return weight_matrix