import os

import torch
import flair

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def main():

    columns = {0: 'text', 1: 'tag'}

    data_folder = os.path.join('extractor', 'simdata')

    corpus: Corpus = ColumnCorpus(
        data_folder, columns,
        train_file='train.txt',
        test_file='test.txt',
        dev_file='valid.txt'
    )

    tag_dictionary = corpus.make_tag_dictionary('tag')

    embeddings = FlairEmbeddings(os.path.join(
        'embeddings', 'full', 'spec_newline', 'best-lm.pt'))

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type='tag',
        use_crf=True
    )

    trainer : ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        os.path.join('extractor', 'simmodel'),
        learning_rate=0.1,
        mini_batch_size=64,
        max_epochs=150
    )

if __name__=='__main__':

    if not torch.cuda.is_available():
        print("Warning: GPU not available. Defaulting to CPU.")
    else:
        print("Using device: ", flair.device)
    print(torch.cuda.is_available())
    main()