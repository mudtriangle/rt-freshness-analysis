# Rotten Tomatoes Freshness Analysis
A LSTM sentiment analysis on Rotten Tomatoes reviews. It identifies reviews between **Fresh**, meaning a positive review, and **Rotten**, meaning a negative one.
## Data
The data is obtained from the subreddit `r/datasets`, thanks to `u/nicolas-gervais` by scraping the Rotten Tomatoes website. It consists of 480000 reviews, 240000 Fresh and 240000 Rotten ones. There are two columns, the first consisting on `0` meaning Rotten or `1` meaning Fresh and the second consisting on the text of the review itself. The dataset can be found [here](https://drive.google.com/file/d/1N8WCMci_jpDHwCVgSED-B9yts-q9_Bb5/view?usp=sharing).\
Sample of the data:

Freshness | Review
--- | ---
1 | Manakamana doesn't answer any questions, yet makes its point: Nepal, like the rest of our planet, is a picturesque but far from peaceable kingdom.
1 | Wilfully offensive and powered by a chest-thumping machismo, but it's good clean fun.
0 | It would be difficult to imagine material more wrong for Spade than Lost & Found.
0 | Despite the gusto its star brings to the role, it's hard to ride shotgun on Hector's voyage of discovery.
0 | If there was a good idea at the core of this film, it's been buried in an unsightly pile of flatulence jokes, dog-related bad puns and a ridiculous serial arson plot.
