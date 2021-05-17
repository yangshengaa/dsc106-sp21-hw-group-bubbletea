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
from text2emotion import get_emotion

# constants 
news_path = 'data/archive'

# ---------- functions to process news data --------------

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


# def t2e_score_by_year(year):
#     """
#     a helper function to read in a year, sum up the t2e sentiment score  
#     """
#     print(f'Processing year {year}')
#     # read in
#     words_series = pd.read_csv(
#         news_path + f'/df_{year}.csv', usecols=['sentence'], squeeze=True)
#     # make prediction
#     score_by_news = words_series.apply(get_emotion).apply(pd.Series)
#     # sum up
#     score_sum = score_by_news.sum()
#     print(f'Finish processing year {year}')
#     return score_sum / score_sum.sum()


# ------------- function to process timbre data -------------- 

def read_timbre_feature():
    """ read in timbre feature """ 
    file_path = 'data/year_prediction.csv'
    return pd.read_csv(file_path).rename(columns={'label': 'year'})


def get_timbre_avg_by_year():  
    """ read in timbre feature and write timbre average feature by year """
    timbre_cols = [f"TimbreAvg{i}" for i in range(1, 13)]
    timbre_avgs = timbre[["year"] + timbre_cols]
    timbre_df = read_timbre_feature()[timbre_avgs]
    timbre_df.groupby('year').mean().to_csv('preprocess/timbre_avg_by_year.csv')


#  def get_timbre_avg_by_decade(): 
#     """ read in timbre feature and write timbre average feature by decade """ 
#     # compute decades
#     def transform_year_to_decade(year):
#         return 'd' + str(year)[2] + '0'
#     # transform and obtain long form data for plotting
#     timbre_avgs['decade'] = timbre_avgs.year.transform(transform_year_to_decade)
#     timbre_avgs_by_decade = timbre_avgs.drop(
#         columns=['year']).groupby('decade').mean()


if __name__ == '__main__':
    # run the preprocessing 
    # nltk score
    # nltk_score_for_all_years()
    # t2e score
    print(t2e_score_by_year(1924))
