"""
this file gets emotion/sentiment prepared for each year   

Author: Sheng Yang 
Date: 2021/05/16
"""

import pandas as pd 
import multiprocessing as mp

# word processors
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer

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

# TODO: modify the function below to incorporate a 'year' in front 

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


# ------------- function to process timbre data -------------- 

def transform_year_to_decade(year):
    """ convert year into decade """
    return 'd' + str(year)[2] + '0'


def read_timbre_feature():
    """ read in timbre feature """ 
    file_path = 'data/year_prediction.csv'
    return pd.read_csv(file_path).rename(columns={'label': 'year'})


def prepare_timbre_avg_by_year_and_decade():  
    """ read in timbre feature and write timbre average feature by year """
    # read in data
    timbre_cols = [f"TimbreAvg{i}" for i in range(1, 13)]
    timbre_df = read_timbre_feature()[["year"] + timbre_cols]
    # compute annual average
    timbre_avg_by_year = timbre_df.groupby('year').mean().reset_index()
    # append decade information 
    timbre_avg_by_year['decade'] = timbre_avg_by_year.year.transform(transform_year_to_decade)
    # write avg file
    timbre_avg_by_year.to_csv('preprocess/timbre_avg_by_year.csv', index=False)
    # write decade file 
    timbre_avg_by_decade = timbre_avg_by_year.drop(columns=['year']).groupby('decade').mean()
    timbre_avg_by_decade.to_csv('preprocess/timbre_avg_by_decade.csv')


if __name__ == '__main__':
    # run the preprocessing 
    # nltk score
    nltk_score_for_all_years()
    # timbre groupby mean
    prepare_timbre_avg_by_year_and_decade()
