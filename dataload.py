import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class Fra2Eng(Dataset):
    def __init__(self, fra_sent, eng_sent, fra_word2idx, eng_word2idx, fra_wordcnt, eng_wordcnt, max_seq_len):
        self.fra_sent = fra_sent
        self.eng_sent = eng_sent
        self.fra_word2idx = fra_word2idx
        self.eng_word2idx = eng_word2idx
        self.fra_wordcnt = fra_wordcnt
        self.eng_wordcnt = eng_wordcnt
        self.max_seq_len = max_seq_len
        self.eng_unk = set()
        self.fra_unk = set()

    def __len__(self):
        return len(fra_sentences)

    def __getitem__(self, idx):
        fra_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        eng_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        fra_sentence = self.fra_sent[idx].split()
        eng_sentence = self.eng_sent[idx].split()
        # Append EOS tokens at the end of each sequence
        fra_sentence.append('<\s>')
        eng_sentence.append('<\s>')

        # Load word indices
        for i, word in enumerate(fra_sentence):

