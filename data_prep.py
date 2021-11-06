import re, os
from collections import defaultdict

import pandas as pd

def load_tags(path: list, ocr_fixes_path: str):
    """Returns a dictionary that maps BRI abbreviations to journal titles"""

    # this spreadsheet contains mappings between BRI abbreviations and journal names
    tag_df = pd.read_csv(path, sep='\t')

    # this one contains quick fixes for the most common OCR erros
    fixes_df = pd.read_csv(ocr_fixes_path, sep='\t')

    # generate a dict to map between abbreviations and names
    tag_map = {tag: title for (tag, title) in zip(tag_df['tag'], tag_df['title'])}
    for tag, title in zip(fixes_df['tag'], fixes_df['title']):
        if tag not in tag_map:
            tag_map[tag] = title

    # the most common OCR error is a missing space
    # luckily, we can safely strip spaces and check against that
    for tag in list(tag_map):
        if ' ' in tag:
            tag_map[tag.replace(' ', '')] = tag_map[tag]

    return tag_map

def count_reviews(raw_df, tag_map: dict):
    """Counts reviews and returns a new dataframe with titles
    as rows and journal review counts as columns."""

    books = {}
    bad_tags = defaultdict(int) # used for identifying common OCR errors
    for author, title, review_string in zip(raw_df.Author, raw_df.Title, raw_df.Review):

        book_id = title + ' || ' + author
        if book_id not in books:
            books[book_id] = defaultdict(int)

        # all reviews end with a page number of format p[number]
        # this regex checks for that, as well as common OCR errors identified in testing
        reviews = re.split(r'pi?[A-Z]?[0-9lO!]+ [a-z]?', review_string.strip())
        reviews = [r.split('-')[0].strip() for r in reviews]
        for review in reviews:
            if review in tag_map:
                books[book_id][tag_map[review]] += 1
            else:
                bad_tags[review] += 1

    books_df = pd.DataFrame.from_dict(books, orient='index')
    books_df = books_df.fillna(0)
    books_df = books_df.astype(int)

    return books_df

def author_compile(books_df):
    """Compiles book review counts into author-review counts."""

    books_df['author_name'] = books_df.index.to_series().str.split('\\|\\|').str[1].str.strip()
    return books_df.groupby('author_name').sum()

def main():
    """Loads, preprocesses, and saves review data for replication notebooks."""

    tag_paths = [
        'tags/1965-1985.tsv',
        'tags/2000.tsv'
    ]
    ocr_fixes_path = 'tags/OCR_corrections_1965.tsv'
    raw_data_fnames = [
        '1965-1984.csv',
        '1985-1992.csv',
        '1993-1997.csv',
        '1998-2000.csv',
    ]
    books_dest_path = 'data/processed/book_reviews_full.tsv'
    authors_dest_path = 'data/processed/author_reviews_full.tsv'

    # load tags
    print('Loading tags.')
    tag_map = {}
    for path in tag_paths:
        tag_map = tag_map | load_tags(path, ocr_fixes_path)

    # raw data comes as title-level rows with a single cell for ALL reviews for that title
    dfs = []
  
    for fname in raw_data_fnames:

        print(f'Loading raw review data: {fname}')
        path = os.path.join('data/raw', fname)
        df = pd.read_csv(path, encoding='latin-1')
        df = df[['Author', 'Title', 'Review']]
        df['Period'] = fname.split('.csv')[0]
        dfs.append(df)
    raw_df = pd.concat(dfs, axis=0)

    print('Counting book-level reviews.')
    books_df = count_reviews(raw_df=raw_df, tag_map=tag_map)
    books_df.to_csv(books_dest_path, sep='\t')

    print('Compiling author-level reviews.')
    # TO DO: fix this so that it correctly joins dates
    author_compile(books_df).to_csv(authors_dest_path, sep='\t')

if __name__ == '__main__':

    main()
