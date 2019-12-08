"""
Uses the script from https://github.com/stanfordnlp/GloVe to efficiently make a vocabulary file from a text file

example use:
python get_vocab_counts.py --glove_dir=/data/jrgillick/projects/long_text/GloVe/ \
--input_files_dir=../data/metrolyrics/texts/ \
--output_files_dir=../data/metrolyrics/vocabs --min_count=15
"""


# call the script from GloVe to quickly create one vocab file each for a directory of text files
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--glove_dir', type=str)
parser.add_argument('--input_files_dir', type=str)
parser.add_argument('--output_files_dir', type=str)
parser.add_argument('--min_count', type=str)

args = parser.parse_args()

glove_dir = args.glove_dir
input_files_dir = args.input_files_dir
output_files_dir= args.output_files_dir
min_count = args.min_count

input_files = [f for f in os.listdir(input_files_dir)]

for f in input_files:
	input_text_file = os.path.join(input_files_dir, f)
	output_vocab_file = os.path.join(output_files_dir, f)
	cmd = os.path.join(glove_dir,'build/vocab_count -min-count %s < %s > %s' % (min_count, input_text_file, output_vocab_file) ) 
	os.system(cmd)

#build/vocab_count -min-count $VOCAB_MIN_COUNT < $CORPUS > $VOCAB_FILE
