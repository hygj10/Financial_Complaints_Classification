#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=3GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=itp@nyu.edu
#SBATCH --output=slurm_%j.out



import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import reload
import numpy as np
import os
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.model_selection import KFold
import random
from torchtext import data
from torchtext.data import Dataset
import argparse
import sys
from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torchtext import data
from torchtext.data import TabularDataset

class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes, kernel_num, kernel_sizes, dropout):
        super(TextCNN, self).__init__()

        
        V = vocab_size
        D = embed_dim
        C = num_classes
        Cin = 1
        Cout = kernel_num
        Ks = kernel_sizes


        self.embeding = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Cin, Cout, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(Ks) * Cout, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embeding(x)

        x = x.unsqueeze(1)
        x= [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        '''
                x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
                x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
                x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
                x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)
        out = self.fc(x)
        return out
    
    
class TextLSTM(nn.Module):
    """
    lstm model of text classify.
    """
    def __init__(self, vocab_size, hidden_size, linear_hidden_size, 
                 embed_dim, num_classes, embedding_path):
        self.name = "TextLstm"
        super(TextLSTM, self).__init__()
        

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(input_size = embed_dim,
                            hidden_size = hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = False)

        self.linears = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(linear_hidden_size, num_classes),
         
        )

        if embedding_path:
            self.embedding.weight.data.copy_(torch.from_numpy
                                             (np.load(embedding_path)))
    

    def forward(self, x):
        x = self.embedding(x)
        print (x)
        lstm_out, _ = self.lstm(x)

      
        out = self.linears(lstm_out[:, -1, :])
       
        return out
    
    
    
def fit(train, val, model, cuda, lr, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    h = []
    v = []

    t = []
 
    
    if cuda:
        model.cuda()
    steps = 0
    model.train()
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        for batch in train:
            print(batch)
            feature, target = batch.consumer_complaint_narrative, batch.category_id
      
            
            feature.data.t_()
            target.data.sub_(1)

            if cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()

            out = model(feature)
            
      
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 2 == 0:
                # print torch.max(out, 1)[1].view(target.size()).data
                corrects = (torch.max(out, 1)[1].view(target.size()).data == target.data).double().sum()
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
                
            if steps % 10 == 0:
                current_accuracy = eval(val, model, cuda)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy

                    print("save model at step %s, accuracy is %d".format(str(steps), current_accuracy))
   
                    torch.save(model, 'save_pathff')
                
                h.append(steps)
                v.append(current_accuracy)
            
    plt.xlabel('Steps')
    plt.ylabel('Development Accuracy')    
    plt.plot(h, v)
        


def eval(data_iter, model, cuda):
    model.eval()
    
    corrects, avg_loss = 0, 0


    for batch in data_iter:
        feature, target = batch.consumer_complaint_narrative, batch.category_id
        target.data.sub_(1)  # batch first, index align
        feature.data.t_()
      

        if cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).double().sum()

    size = len(data_iter.dataset)
    avg_loss = loss.item() / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                        size))
    return accuracy

class MyDataset(data.TabularDataset):
    """
    """

    def splits(self, fields, dev_ratio=.15, shuffle=True, **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            fields: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
        """
        examples = self.examples
        if shuffle: random.shuffle(examples)

        dev_index = -1 * int(dev_ratio * len(examples))
        return (Dataset(fields=fields, examples=examples[:dev_index]),
                Dataset(fields=fields, examples=examples[dev_index:]))

    def kfold(self, k):
        """
        kfold using sklearn
        :param k:
        :return: index of kfolded
        """
        kf = KFold(k)
        examples = self.examples
        return kf.split(examples)

    def get_fold(self, fields, train_indexs, test_indexs, shuffle=True):
        """
        get new batch
        :return:
        """
        examples = np.asarray(self.examples)

        if shuffle: random.shuffle(examples)
        print (list(train_indexs))
        return (Dataset(fields=fields, examples=examples[list(train_indexs)]),
                Dataset(fields=fields, examples=examples[list(test_indexs)]))
    
TEXT = data.Field()
LABELS = data.Field(sequential=False)
dataset = MyDataset(path='./complaints_cleaned_label.csv', format='csv', 
                    fields=[("product", None), ('consumer_complaint_narrative', TEXT), 
                            ('category_id', LABELS)])
train, dev = dataset.splits(fields=[("product", None), ('consumer_complaint_narrative', TEXT), 
                                    ('category_id', LABELS)])

# print(train.fields)
# print(len(train))
# print(vars(train[0]))

LABELS.build_vocab(dataset.category_id)
TEXT.build_vocab(dataset.consumer_complaint_narrative)

print(TEXT.vocab.freqs.most_common(10))
print(LABELS.vocab.itos)

train_iter, dev_iter = data.Iterator.splits((train, dev), sort=False,
                                            batch_sizes=(1000, len(dev)))

EMBEDDING_FILE = 'd2v_model_dbow.doc2vec'
MAX_NB_WORDS = 50000
embedding_matrix_path = "./embedding_matrix.npy"
########################################
# prepare embeddings
########################################

print('Preparing embedding matrix')
word2vec = Doc2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(TEXT.vocab))

embedding_matrix = np.zeros((nb_words, 300))
c = 0
for i, word in enumerate(TEXT.vocab.itos):
    if word in word2vec.wv.vocab and i < 50000:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
    else:
        c += 1
        # print word
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
np.save(embedding_matrix_path, embedding_matrix)



vocab_size = len(TEXT.vocab)
num_classes = 12
embedding_path = embedding_matrix_path
kernel_sizes = [1, 1, 1]
embed_dim = 3
kernel_num = 30
cuda = False
dropout = 0.3
lr = 0.001
epochs = 256
cnn = TextCNN(vocab_size, embed_dim, num_classes, kernel_num, kernel_sizes, dropout)

#fit(train_iter, dev_iter, cnn, cuda, lr, epochs)



hidden_size = 7
linear_hidden_size = 10
embed_dim = 3

EMBEDDING_FILE = 'd2v_model_dbow.doc2vec'
MAX_NB_WORDS = 66808
embedding_matrix_path = "./embedding_matrix.npy"
########################################
# prepare embeddings
########################################

print('Preparing embedding matrix')
word2vec = Doc2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(TEXT.vocab))

embedding_matrix = np.zeros((nb_words, 3))
c = 0
for i, word in enumerate(TEXT.vocab.itos):
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
    else:
        c += 1
        # print word
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
np.save(embedding_matrix_path, embedding_matrix)

lstm = TextLSTM(vocab_size, hidden_size, linear_hidden_size, embed_dim, num_classes, embedding_path)
fit(train_iter, dev_iter, lstm, cuda, lr, epochs)