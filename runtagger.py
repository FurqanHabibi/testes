# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch

from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, PackedSequence
from torch.utils.data import DataLoader

from buildtagger import Model, read_file, indexize, integerize, MyDataset, collate, tags

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
	# use torch library to load model_file
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device : {}".format(dev))

    checkpoint = torch.load(model_file)
    word_to_idx = checkpoint["word_to_idx"]
    model_state_dict = checkpoint["model_state_dict"]

    test_feats = read_file(test_file, with_tag=False)
    test_feats_idx = indexize(test_feats, word_to_idx)
    test_feats_int = integerize(test_feats)
    test_dataset = MyDataset(test_feats_int, test_feats_idx, test_feats_idx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate)

    if os.path.exists(out_file):
        os.remove(out_file)

    model = Model(word_to_idx)
    model.load_state_dict(model_state_dict)
    model.to(dev)
    model.eval()

    cur_line = 0
    test_feats_raw = read_file(test_file, with_tag=False, upper=False)
    for X_int, X_idx, _, batch_len in test_loader:
        X_int = pack_sequence(X_int)
        X_idx = pack_sequence(X_idx)
        X_int = X_int.to(dev)
        X_idx = X_idx.to(dev)
        y_tilde = model(X_int, X_idx)
        y_tilde_label = torch.argmax(y_tilde, dim=1)
        y_tilde_label = PackedSequence(y_tilde_label, X_idx.batch_sizes)
        y_tilde_label_pad, lengths = pad_packed_sequence(y_tilde_label, batch_first=True, padding_value=0)

        with open(out_file, 'a') as writer:
            for line_idx in range(batch_len):
                for word_idx in range(lengths[line_idx]):
                    writer.write("{}/{}".format(test_feats_raw[cur_line][word_idx], tags[y_tilde_label_pad[line_idx][word_idx]]))
                    if word_idx < lengths[line_idx]-1:
                        writer.write(" ")
                    else:
                        writer.write("\n")
                cur_line += 1

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
