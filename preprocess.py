import os.path
import glob as g
import unicodedata
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAR_DIR='.\data'
SOS_token = '<\s>'
EOS_token = '<\e>'
PAD_token = '<\p>'

class Language():
    def __init__(self, lang='UNK'):
        self.lang = lang
        self.sentences = []
        self.word2cnt = {}
        self.word2idx = {SOS_token: 0, EOS_token: 1, PAD_token: -1}
        self.idx2word = {-1: PAD_token, 0: SOS_token, 1: EOS_token}
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
            s_char = s.split()
            if len(s_char) < max_seq_len:
                # Pad sequences shorter than the longest sequence
                s_char += [PAD_token] * (max_seq_len - len(s_char))
            for word in s_char:
                indices.append([self.word2idx[word]])
            sent_tensor.append(indices)
            
        return torch.tensor(sent_tensor, dtype=torch.long, device=device)

    def getnwords(self):
        return self.num_words

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
        lang_pair = []
        len_lang1 = list(lang1.size())[0]
        len_lang2 = list(lang2.size())[0]
        len_data = max(len_lang1, len_lang2)
        for i in range(len_data):
            pair = list((lang1[i], lang2[i]))
            lang_pair.append(pair)
        
        return lang_pair

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
        data['eng'] = eng
        data['fra'] = fra

        eng_tensor = eng.sentence2tensor(data['max_len'])
        fra_tensor = fra.sentence2tensor(data['max_len'])
        print('English tensor size:', eng_tensor.size(), 'French tensor size:', fra_tensor.size())
        seq_pair = self.makePair(eng_tensor, fra_tensor)

        return data, seq_pair

if __name__ == '__main__':
    d = DataProcess()
    d.preprocess()