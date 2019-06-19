# Rotten Tomatoes Freshness Analysis
A LSTM sentiment analysis on Rotten Tomatoes reviews. It identifies reviews between **Fresh**, meaning a positive review, and **Rotten**, meaning a negative one.\
Tested with Python 3.6.1 (Anaconda version).
## Data
The data is obtained from the subreddit `r/datasets`, thanks to `u/nicolas-gervais` by scraping the Rotten Tomatoes website. It consists of 480 thousand reviews, 240 thousand Fresh and 240 thousand Rotten ones. There are two columns, the first consisting on `0` meaning Rotten or `1` meaning Fresh and the second consisting on the text of the review itself. The dataset can be found [here](https://drive.google.com/file/d/1N8WCMci_jpDHwCVgSED-B9yts-q9_Bb5/view?usp=sharing).\
Sample of the data:

Freshness | Review
--- | ---
1 | Manakamana doesn't answer any questions, yet makes its point: Nepal, like the rest of our planet, is a picturesque but far from peaceable kingdom.
1 | Wilfully offensive and powered by a chest-thumping machismo, but it's good clean fun.
0 | It would be difficult to imagine material more wrong for Spade than Lost & Found.
0 | Despite the gusto its star brings to the role, it's hard to ride shotgun on Hector's voyage of discovery.
0 | If there was a good idea at the core of this film, it's been buried in an unsightly pile of flatulence jokes, dog-related bad puns and a ridiculous serial arson plot.

## Files
`rotten_tomatoes_reviews.csv` Original dataset. Too large to be included in the repo.\
`reviews.csv` Intermediate file created by `prepare_data.py` containing the tokens that represent the review without headers. Too large to be included in the repo. Sample of the data:

Freshness | Review
--- | ---
1 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 29509 1841 39337 55254 29358 37531 33200 28125 40779 37260 36962 16626 36243 26373
1 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 54407 34405 38009 8187 49423 29107 19955 8796 18565
0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 54895 12644 23431 30066 54959 45881 28661 18056
0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12304 20829 46557 6139 41466 21307 41092 44189 21717 53526 12930
0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 19955 23264 10094 17201 6592 52243 37030 17511 25379 13326 40439 3393 39033 41098 43575 2503 37401

`vocabulary.txt` Intermediate file created by `prepare_data.py` containing all the words in the vocabulary and the number by which they are represented.\
`string_processing.py` Python file containing functions that help in translating a text review into number tokens.\
`prepare_data.py` Python file that converts all the text reviews in the original dataset to number tokens.\
`lstm_sentiment.py` Python file containing the class declaration of the model used.\
`data_utils.py` Python file containing functions that help in processing the data to serve as input for the model to be trained.\
`train.py` Python file that creates an instance of the model and trains it.\
`predict.py` Python file that loads `model` file (not included in the repo, output of `train.py`) and uses it to predict a sentence given by the user.
## Acknowledgements
Dataset obtained thanks to `u/nicolas-gervais` from the `r/datasets` subreddit.\
Done roughly following Samarth Agrawal's tutorial on LSTM Sentiment Analysis in `towardsdatascience.com` that can be found [here](https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948).
