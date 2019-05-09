# External libraries
import pandas as pd

# Local libraries
import string_processing

# Constants
MAX_LEN = 30
MIN_LEN = 2

# Get reviews from csv file.
df = pd.read_csv("rotten_tomatoes_reviews.csv", encoding='iso-8859-1')  # s/o nicolas-gervais from r/datasets

# Generate tokens from the text reviews.
df['Review_Clean'] = df['Review'].apply(lambda x: string_processing.normalize(x))
df['Tokens'] = df['Review_Clean'].apply(lambda x: string_processing.tokenize(x))
df.drop(['Review', 'Review_Clean'], axis=1, inplace=True)

# Get rid of reviews with word count below the minimum length.
df = df[df['Tokens'].apply(lambda x: len(x) > MIN_LEN)]
df.reset_index(drop=True, inplace=True)

# Generate a vocabulary.
vocab = string_processing.build_vocab(df['Tokens'].tolist(), 'vocabulary.txt')

# Replace tokens with their respective numbers in the vocabulary.
df['Tokens'] = df['Tokens'].apply(lambda x: string_processing.get_numbers(x, vocab))

# Add zero-padding.
df['Tokens'] = df['Tokens'].apply(lambda x: string_processing.padding(x, MAX_LEN))

# Save into a csv file.
df['Tokens'] = df['Tokens'].apply(lambda x: [str(i) for i in x])
df['Tokens'] = df['Tokens'].apply(lambda x: ' '.join(x))
df.to_csv("reviews.csv", header=False, index=False, encoding='iso-8859-1')
