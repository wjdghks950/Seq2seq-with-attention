import os.path
import glob as g
import unicodedata
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAR_DIR='.\data'
SOS_token = '<\s>'
EOS_token = '<\e>'

class Language():
    def __init__(self, lang='UNK'):
        self.lang = lang
        self.sentences = []
        self.word2cnt = {}
        self.word2idx = {SOS_token: 0, EOS_token: 1}
        self.idx2word = {0: SOS_token, 1: EOS_token}
        self.num_words = 2
    
    def parseSentence(self, sentence):
        sentence_indices = []
        self.sentences = sentence.split('\n')
        for s in self.sentences:
            for word in s.split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.num_words
                    self.word2cnt[word] = 1
                    self.idx2word[self.num_words] = word
                    self.num_words += 1
                else:
                    self.word2cnt[word] += 1

    def maxLen(self):
        max_seq_len = 0
        for s in self.sentences:
            length = len(s.split())
            max_seq_len = max(max_seq_len, length)
        return max_seq_len

    def sentence2tensor(self, max_seq_len):
        # Turns each sequence from words to indices in a tensor form
        sent_tensor = []
        cnt = 0
        for s in self.sentences:
            indices = []
            s += ' ' + EOS_token # Append EOS token at the end of each sentence
            for word in s.split():
                indices.append(self.word2idx[word])
            sent_tensor.append(indices)
        print(len(sent_tensor))
        return torch.tensor(sent_tensor, dtype=torch.long, device=device).view(len(sent_tensor), max_seq_len)

    def langName(self):
        return self.lang

class DataProcess():
    def __init__(self, path=PAR_DIR):
        self.path = path

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

    def to_ascii(self, sentence):
        return ''.join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')

    def makePair(self, lang1, lang2):
        return (lang1, lang2)

    def preprocess(self):
        reader = self.read_data()
        eng = Language('eng')
        fra = Language('fra')
        eng.parseSentence(reader['lang1'])
        fra.parseSentence(reader['lang2'])
        print('Number of English sentences:', len(reader['lang1'].split('\n')))
        print('Number of French sentences:', len(reader['lang2'].split('\n')))
        max_seq_len = 0
        # Calculate the longest sequence length from both English and French sentences
        max_seq_len = max(eng.maxLen(), fra.maxLen())
        print('Longest sequence in both langs: ', max_seq_len)
        data = {}
        data['max_len'] = max_seq_len + 1
        data['eng'] = eng.sentence2tensor(data['max_len'])
        data['fra'] = fra.sentence2tensor(data['max_len'])

        seq_pair = self.makePair(data['eng'], data['fra'])

        return seq_pair

if __name__ == '__main__':
    d = DataProcess()
    d.preprocess()