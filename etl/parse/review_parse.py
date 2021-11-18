"""
Methods for parsing structured information from raw OCR data from the
Book Review Index.

A typical review string looks something like this:

Choice  - v35 - D '97 - p690 - [51-250] - H - Net - F '98 - ONL [501+]

This string contains information on two reviews, segmented into the following general format.

journal - volume - month/season - day - year - page number - length

In the best case, each field is separated by a dash. However, this is not a guarantee. Some
journal abbreviations contain dashes ("H-Net"). Sometimes dashes were missed by the OCR
process or replaced with a different character. Additionally, fields are not guaranteed 
to be present. Some reviews contain only journal, year, and page.
Sometimes OCR errors blur the boundaries between fields.

As such, accurately parsing a review string is a challenge. The simplest method is a regular
expression designed to accomodate the most common OCR errors. An alternative method is
a custom LSTM sequence tagger that makes use of custom Flair character embeddings.
"""
import os
from flair.models import SequenceTagger
from .definitions import ROOT_DIR # broken

def get_flair_parser():

    model_path = os.path.join(ROOT_DIR, 'etl/train/labeler/models/best-model.pt')
    return SequenceTagger.join(model_path)

