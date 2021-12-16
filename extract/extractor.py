
from dataclasses import dataclass

from flair.models import SequenceTagger
from flair.data import Sentence

from tokenize import ExtractTokenizer

@dataclass
class Entry:
    """
    Class for representing an entry extracted from raw OCR of the Book Review Index.
    """
    full_string: str = ''
    parsed_author: list = []
    parsed_title: list = []
    reviews: list = []
    start_pos: int = None
    end_pos: int = None

    # To do: infer header from full_string and review position
    # def bad_header(self):

    #     ans = self.header_string.replace(' ', '') == (self.parsed_author + self.parsed_title).replace(' ', '')
    #     return not ans

    def add_by_tag(self, tag: str, value: str):
        if tag == 'A':
            self.parsed_author.append(value)
        elif tag == 'T':
            self.parsed_title.append(value)
        elif tag == 'R':
            self.reviews.append(value)
        return

class Extractor:
    """
    Main class used for extracting individual fields from the raw OCR text of
    the Book Review Index. 

    Note that the model was trained on the 1965-1984 data specifically, which had
    the largest gaps from the Regex-based extraction method.
    """

    def __init__(self,
        model: SequenceTagger, 
        tokenizer: ExtractTokenizer,
        chunk_size: int = 10000):

        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def extract(self, text: str, verbose: bool = False):
        """
        :param str text: input text from which to extract fields
        :param bool verbose: whether to give progress updates
        """

        chunk_start = 0
        total = len(text)
        final = False
        entries = []

        while not final:

            chunk_end = chunk_start + self.chunk_size
            if chunk_end > total:
                chunk = text[chunk_start:]
                final = True
            else:
                chunk = text[chunk_start:chunk_end]
            sentence = Sentence(
                chunk,
                use_tokenizer=self.tokenizer,
            )
            self.model.predict(sentence)
            current_entry = Entry(start_pos=0)
            for span in sentence.get_spans():

                for label in span.labels:

                    if current_entry.reviews and (label.value == 'T' or label.value == 'A'):
                        
                        current_entry.full_string = text[current_entry.start_pos:current_entry.end_pos]
                        entries.append(current_entry)
                        current_entry = Entry(start_pos=current_entry.end_pos)
                        current_entry.add_by_tag(label.value, span.text)
                        current_entry.end_pos = span.end_pos

                    else:
                        current_entry.add_by_tag(label.value, span.text)
                        current_entry.end_pos = span.end_pos
            
            if not final:
                chunk_start = chunk_start + entries[-2].end_pos # end pos is indexed within chunk, not within full text
                entries = entries[:-1]              # remove the last book, which becomes the beginning of the next chunk
                                                    # we do this because we can't be sure it is a complete entry
            else:
                entries.append(current_entry)
            if verbose >= 1:
                percent_complete = round(chunk_start / total, 5) * 100
                print(f'Extracted {len(entries)} books. {percent_complete}% of text parsed.')

        return entries


