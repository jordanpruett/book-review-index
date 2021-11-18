"""
Specialized tokenizer classes for parsing the Book Review Index.
"""
from typing import List
from flair.data import Tokenizer, Token

# we have to declare a custom tokenizer that splits on spaces and apostrophes,
# which it removes, or on dashes, which it keeps
class ReviewTokenizer(Tokenizer):
    """
    Custom tokenizer used for review tagger. Splits on spaces and apostrophes, which it removes,
    as well as on dashes, which it keeps.
    """

    # Flair library includes super delegations of this type, but they appear to be redundant (?)
    #def __init__(self):
        #super(ReviewTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[Token]:
        return ReviewTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        """
        Primary tokenization method. Splits on spaces and apostrophes, which it removes, 
        as well as on dashes, which it keeps. Returns a list of Flair Token objects.
        """

        tokens: List[Token] = []

        word = ""
        index = -1
        for index, char in enumerate(text):

            if char in " '":
                if len(word) > 0:
                    whitespace_after = char == " "
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word,
                            start_position=start_position,
                            whitespace_after=whitespace_after
                        )
                    )
                word = ""
            elif char == "-":

                if len(word) > 0:
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word,
                            start_position=start_position,
                            whitespace_after=False
                        )
                    )
                whitespace_after = text[index+1] == " "
                tokens.append(
                    Token(
                        text="-",
                        start_position=index,
                        whitespace_after=whitespace_after
                    )
                )
                word = ""
            else:
                word += char

        # increment for last token in sentence if not followed by whitespace
        index += 1
        if len(word) > 0:
            start_position = index - len(word)
            tokens.append(
                Token(text=word,
                start_position=start_position,
                whitespace_after=False)
            )

        return tokens
