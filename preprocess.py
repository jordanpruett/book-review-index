import os

import pandas as pd
from flair.data import Sentence

from extract.reviewparser import ReviewParser
from extract.reviewtokenizer import ReviewTokenizer


def main():

    # fnames for raw data
    filenames = [
        '1965-1984.csv',
        '1985-1992.csv',
        '1993-1997.csv',
        '1998-2000.csv'
    ]

    # instantiate parser object
    parser = ReviewParser(
        os.path.join('extract', 'train', 'labeler', 'models', 'best-model.pt')
    )
    tokenizer = ReviewTokenizer()

    for i, fn in enumerate(filenames):

        path = os.path.join('data', 'raw', fn)
        raw = pd.read_csv(path, encoding='latin-1')

        print(f'Parsing spreadsheet #{i}')
        # list of dicts, where each dict is one parsed review
        review_df_rows = []
        count = 0
        for row in raw.itertuples(index=False):

            author = row.Author
            title = row.Title
            review_string = row.Review
            sentence = Sentence(
                review_string,
                use_tokenizer = tokenizer
            )
            reviews = parser.parse(sentence) # returns list of dicts
            for review in reviews:
                review['author'] = author
                review['title'] = title
            review_df_rows.extend(reviews)
            count += 1
            if count % 1000 == 0:
                print(f'Parsed {count} rows.')

        print(f'Saving spreadsheet #{i}')
        df = pd.DataFrame(review_df_rows)
        df = df[['author', 'title', 'J', 'V', 'M', 'D', 'Y', 'P', 'L']]
        df.to_csv(
            os.path.join('data', 'processed', f'data{i}.tsv'),
             index=False,
             sep='\t'
        )

if __name__=='__main__':

    main()

