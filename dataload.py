import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class Fra2Eng(Dataset):
    def __init__(self, fra_sent, eng_sent, fra_word2idx, eng_word2idx, max_seq_len):
        self.fra_sent = fra_sent
        self.eng_sent = eng_sent
        self.fra_word2idx = fra_word2idx
        self.eng_word2idx = eng_word2idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(fra_sentences)

    def __getitem(self, idx):
        