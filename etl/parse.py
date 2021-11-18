"""
Parser class with methods for parsing structured information from raw OCR data from
scans of the Book Review Index.

A typical review string looks something like this:

Choice  - v35 - D '97 - p690 - [51-250] - H - Net - F '98 - ONL [501+]

This string contains information on two reviews, segmented into the following general format.

journal - volume - month/season - day - year - page number - length

In the best case, each field is separated by a dash. However, this often is not true. Some
journal abbreviations contain dashes ("H-Net"). Sometimes dashes were missed by the OCR
process or replaced with a different character. Additionally, fields are not guaranteed
to be present. Some reviews contain only journal, year, and page.
Sometimes OCR errors blur the boundaries between fields.

As such, accurately parsing a review string is a challenge. The simplest method is a regular
expression designed to accomodate the most common OCR errors. An alternative, preferred method
is a custom LSTM sequence tagger that makes use of custom Flair character embeddings.
"""
import re

from flair.models import SequenceTagger
from flair.data import Sentence

class ReviewParser:
    """
    A parser that provides two separate methods for extracting fields from Index entries.
    If a path to a flair model is provided, it uses a flair SequenceTagger to extract fields.
    Otherwise, it defaults to a regular expression method.
    In either case, it returns a list of review dicts, where keys are fields.

    Guide to fields.

    J - journal
    V - volume
    M - month/season
    D - day
    Y - year
    P - page
    L - length

    The SequenceTagger method extracts all fields, whereas the regex method extracts only journals.

    """

    def __init__(self, model_path: str = ''):

        self.model_path = model_path

        if model_path:
            self.tagger = SequenceTagger.load(model_path)
        else:
            self.tagger = None

    def parse(self, review_sentence_obj: Sentence):
        """
        Given a flair Sentence object, parses BRI fields and a returns a list of dicts,
        where each dict is field-value pairs for one review.
        """

        if self.tagger:
            return self._tagger_parse(review_sentence_obj)
        else:
            return self._regex_parse(review_sentence_obj)

    def _tagger_parse(self, review_sentence_obj: Sentence):

        reviews = []
        current_review = {}
        self.tagger.predict(review_sentence_obj)
        for span in review_sentence_obj.get_spans('tag'):

            for label in span.labels:

                # each review can only have exactly one of each field
                if label.value in current_review:

                    reviews.append(current_review)
                    current_review = {label.value: span.text}
                else:
                    current_review[label.value] = span.text

        # final review
        reviews.append(current_review)
        return reviews

    def _regex_parse(self, review_sentence_obj: Sentence):

        review_string = review_sentence_obj.to_original_text()

        # First we remove bracketed clauses, which trip up the parser.
        # Find:
        # 1) parentheses or bracket open
        # 2) one or more characters that are digits, dash (-), plus (+), or common OCR errors JIl
        # 3) parentheses or bracket close
        pattern1 = r'[\[\(]*[ 0-9\-\+IlJ]+[\]\)]'

        # The end of an entry is marked by a page number.
        # Find:
        # 1) dash, then optionally a space
        # 2) p, then optionally i or a capital letter
        # 3) some number of digits, spaces, and common OCR error digits
        # 4) finally, it may optionally end with a lowercase letter
        pattern2 = r'\- ?(?:p[A-Zi ]*[ \+0-9lO!Ili]+ [a-z]?|ONL)'
        if re.search(pattern1, review_string):
            review_string = re.sub(pattern1, '', review_string)
        raw_reviews = re.split(pattern2, review_string)
        edited_reviews = []
        for review in raw_reviews:
            new_review = review
            if new_review:
                # sometimes the review will be preceded by a space or certain lowercase letters,
                # which we remove
                while new_review and new_review[0] in ' tcyxr-':
                    new_review = new_review.replace(new_review[0], '', 1)
                if new_review: # in very rare cases the previous process eliminates the whole string
                    edited_reviews.append(new_review)
        journals = [r.split('- ')[0].strip() for r in edited_reviews]
        return [{'J': journal} for journal in journals]
