"""
This module trains a custom language model, to be later used by the sequence labeler.
Only needs to be run once for each model.
"""
import os

import torch
import flair
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

def main():
    # forward vs backward
    is_forward_lm = True

    # most legal characters are available in the default char dictionary
    dictionary: Dictionary = Dictionary.load('chars')

    # with the exception of a special newline character we will use to allow 
    # the tagger to use newlines in ColumnCorpus
    dictionary.add_item('[newline]')

    # our dataset is mostly made up of thousands of arbitrary abbreviations,
    # so we specify a character-level model
    corpus = TextCorpus(os.path.join('embeddings', 'full', 'corpus'),
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # a hidden size of 512 offers a good balance of complexity to training time
    language_model = LanguageModel(dictionary,
                                is_forward_lm,
                                hidden_size=1024,
                                nlayers=1)

    # train model
    trainer = LanguageModelTrainer(language_model, corpus)
    trainer.train('full',
                sequence_length=125,
                mini_batch_size=50,
                max_epochs=10)

if __name__=='__main__':

    if not torch.cuda.is_available():
        print('Warning: GPU not available. Flair will default to CPU.')
    else:
        print('Using device:', flair.device)

    main()