"""
this file gets emotion/sentiment prepared for each year   

Author: Sheng Yang 
Date: 2021/05/16
"""

# TODO: add Text2Emotion version of this, maybe?

import pandas as pd 
import multiprocessing as mp

# word processors
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer

# constants 
news_path = 'data/archive'

def nltk_score_by_year(year):
    """
    a helper function to read in a year, sum up the nltk sentiment score  
    """
    print(f'Processing year {year}')
    # read in
    words_series = pd.read_csv(
        news_path + f'/df_{year}.csv', usecols=['sentence'], squeeze=True)
    # make prediction
    sia = SentimentIntensityAnalyzer()
    polarity_scores = sia.polarity_scores
    score_by_news = words_series.apply(polarity_scores).apply(pd.Series)
    # sum up
    score_sum = score_by_news.sum()
    print(f'Finish processing year {year}')
    return score_sum / score_sum.sum()


def nltk_score_for_all_years():
    """
    read in all years, compute the scores for four nltk categories, and write to disk in a csv format 
    the csv contains five columns, corresponding to year, 'neg', 'neu', 'pos', and 'comp' respectively
    """
    years = list(range(1920, 2020 + 1))
    # process with multiprocessing
    with mp.Pool() as pool:
        stacked_scores = pool.map(nltk_score_by_year, years)  # use map to ensure sequence
    # write to disk 
    pd.DataFrame(stacked_scores, index=years).to_csv('preprocess/nltk_scores_by_year.csv')


if __name__ == '__main__':
    # run the preprocessing 
    nltk_score_for_all_years()
