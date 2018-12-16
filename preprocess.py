import torch
import torch.nn as nn
import os.path
import glob as g

PAR_DIR='.\data'

class DataProcess():
    def __init__(self, path=PAR_DIR):
        self.path = path
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def read_data(self):
        # Read data from the specified path
        data = {}
        file_paths = g.glob(os.path.join(self.path, '*'))
        try:
            for i in range(len(file_paths)):
                with open(file_paths[i]) as fp:
                    data['lang' + str(i+1)] = fp.read()
        except IOError as e:
            print('File open failed: ', e.strerror)
        return data

    def build_dict(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.word2cnt[word] = 1
                self.idx2word[self.n_words] = word
            else:
                self.word2cnt[word] += 1

    def preprocess(self):
        reader = self.read_data()
        data = {}
        data['eng'] = reader['lang1'].split('\n')
        data['fra'] = reader['lang2'].split('\n')
        print('Number of English sentences:', len(data['eng']))
        print('Number of French sentences:', len(data['fra']))
        print('[Example pair]:\n', data['eng'][3], '\n', data['fra'][3])
        
        max_seq_len = 0
        # Calculate the longest sequence length in both English and French sentences
        for key, val in data.items():
            for sentence in val:
                self.build_dict(sentence) # Make dictionary from vocabs in dataset
                length = len(sentence.split())
                max_seq_len = max(max_seq_len, length)

        data['max_len'] = max_seq_len + 1

        return data

if __name__ == '__main__':
    d = DataProcess()
    d.preprocess()