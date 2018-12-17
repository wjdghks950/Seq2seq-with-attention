import torch
import torch.nn as nn
import os.path
import glob as g
import unicodedata

PAR_DIR='.\data'

class DataProcess():
    def __init__(self, path=PAR_DIR):
        self.path = path
        self.eng_word2cnt = {}
        self.fra_word2cnt = {}
        self.idx2word = {0: 'SOS', 1: 'EOS'}

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

    def getCount(self, dict_data):
        return dict_data[1]

    def build_dict(self, data):
        for s in data['eng']:
            for word in s.split():
                if word in self.eng_word2cnt:
                    self.eng_word2cnt[word] += 1
                else:
                    self.eng_word2cnt[word] = 1
        for s in data['fra']:
            for word in s.split():
                if word in self.fra_word2cnt:
                    self.fra_word2cnt[word] += 1
                else:
                    self.fra_word2cnt[word] = 1
        # Add the end of sentence token to each word count
        self.eng_word2cnt['</s>'] = len(data['eng'])
        self.fra_word2cnt['</s>'] = len(data['fra'])
        # Sort according to word occurrence
        sorted_eng = sorted(self.eng_word2cnt.items(), key=self.getCount, reverse=True)
        sorted_fra = sorted(self.fra_word2cnt.items(), key=self.getCount, reverse=True)

        print(sorted_eng[:10])
        print('\n', sorted_fra[:10])

    def to_ascii(self, sentence):
        return ''.join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')

    def preprocess(self):
        reader = self.read_data()
        data = {}
        data['eng'] = reader['lang1'].split('\n')
        data['fra'] = reader['lang2'].split('\n')
        print('Number of English sentences:', len(data['eng']))
        print('Number of French sentences:', len(data['fra']))
        print('[Example pair]:\n', data['eng'][3], '\n', data['fra'][3])
        
        self.build_dict(data) # Make dictionary from vocabs in dataset

        max_seq_len = 0
        # Calculate the longest sequence length in both English and French sentences
        for key, val in data.items():
            for sentence in val:
                length = len(sentence.split())
                max_seq_len = max(max_seq_len, length)

        data['max_len'] = max_seq_len + 1

        return data

if __name__ == '__main__':
    d = DataProcess()
    d.preprocess()