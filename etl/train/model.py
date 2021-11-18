"""
This module trains a custom language model, to be later used by the sequence labeler.
Only needs to be run once.
"""
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

def main():
    # forward vs backward
    is_forward_lm = True

    # all legal characters for this project are contained in the default dictionary
    dictionary: Dictionary = Dictionary.load('chars')

    # our dataset is mostly made up of thousands of arbitrary abbreviations,
    # so we specify a character-level model
    corpus = TextCorpus('corpus',
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # a hidden size of 512 offers a good balance of complexity to training time
    language_model = LanguageModel(dictionary,
                                is_forward_lm,
                                hidden_size=512,
                                nlayers=1)

    # train model
    trainer = LanguageModelTrainer(language_model, corpus)
    trainer.train('models',
                sequence_length=50,
                mini_batch_size=20,
                max_epochs=10)

if __name__=='__main__':
    main()