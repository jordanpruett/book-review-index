"""
Specialized tokenizer classes for parsing the Book Review Index.
"""
from typing import List
from flair.data import Tokenizer, Token

class ReviewTokenizer(Tokenizer):
    """
    Custom tokenizer used for review tagger. Splits on spaces and apostrophes, which it removes,
    as well as on dashes, which it keeps.
    """

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
                try:
                    whitespace_after = text[index+1] == " "
                except IndexError:
                    whitespace_after = False
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

class ExtractTokenizer(Tokenizer):
    """
    Custom tokenizer used for extracting tagged fields from the raw OCR data.
    Splits on spaces, which it removes, as well as on dashes, which it keeps.
    Replaces \n char with [newline], which is treated as a single token.
    """

    def tokenize(self, text: str) -> List[Token]:
        return ExtractTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        """
        Primary tokenization method. Splits on spaces, which it removes,
        as well as on dashes, which it keeps. Returns a list of Flair Token objects.
        Replaces \n char with [newline], which is treated as a single token.
        Returns a list of Token objects.
        """

        tokens: List[Token] = []

        word = ""
        index = -1
        for index, char in enumerate(text):

            if char == ' ':
                if len(word) > 0:
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word,
                            start_position=start_position,
                            whitespace_after=True
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
                try:
                    whitespace_after = text[index+1] == " "
                except IndexError:
                    whitespace_after = False
                tokens.append(
                    Token(
                        text="-",
                        start_position=index,
                        whitespace_after=whitespace_after
                    )
                )
                word = ""
            elif char == "\n":

                if len(word) > 0:
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word,
                            start_position=start_position,
                            whitespace_after=False
                        )
                    )
                try:
                    whitespace_after = text[index+1] == " "
                except IndexError:
                    whitespace_after = False
                tokens.append(
                    Token(
                        text="[newline]",
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
