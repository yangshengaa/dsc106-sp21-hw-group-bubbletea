"""
This file contains the necessary auxiliary functions to load, process, and plot data

Author: Jasmine Guan and Sheng Yang
Date: 2021/05/14
"""

# load packages 
import os 
import re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import multiprocessing as mp

# word processors 
from wordcloud import WordCloud
# from nltk.sentiment import SentimentIntensityAnalyzer


# constants 
news_path = 'data/archive'
report_path = 'report'
to_drop = ['a', 'an', 'the', 'and', 'so', 'therefore', 'for',
           'is', 'are', 'was', 'were', 'am', 'be', 'as', 'has', 'had',
           'of', 'to', 'by', 'then', 'on', 'at', 'with', 'from', 'in',
           'it', 'this', 'that', 'those', 'who', 'whose', 'whom', 'which',
           'his', 'he', 'her', 'she', 'they', 'them', 'their', 's'
           ]  # words to drop, meaningless to consider by themselves


# ------ functions for news analysis only --------

def read_csv_as_txt(year):
    """ 
    read in the csv of a given year into a strings, and discard all "new york times" 
    
    :param year: an integer, the year we are interested in loading in
    """
    return open(news_path + f'/df_{year}.csv').read().lower().replace('new york times', '')


def count_words_by_year(year):
    """ 
    load in csv of a specific year and perform value count

    :param year: the year we are interested in 
    :return a pandas Series value_counts by word 
    """
    # read in as a text file
    words_of_a_year = read_csv_as_txt(year)
    # count words by value counts 
    words_count = pd.Series(re.findall(r'[a-z]+', words_of_a_year.lower())).value_counts()
    # drop meaningless words 
    return words_count.drop(index=to_drop)


def document_frequency(year, word):
    """
    compute tf_idf of a word in a given year 
    # (not used, probably irrelevant)

    :param year: the year we are interested in exploring 
    :param word: a specific word that we are interested in computing the tf_idf 
    :return tf_idf
    """
    # read in csv 
    df = pd.read_csv(news_path + f'/df_{year}.csv', index_col=0)
    return df.sentence.str.contains(word).mean()


def plot_word_cloud_of_year(year):
    """ 
    make a wordcloud plot of a give year 
    
    :param year: an integer, the year we are interested in loading in
    """ 
    words_of_a_year = read_csv_as_txt(year)
    wd_plot = WordCloud(max_words=100, 
                        width=2560, height=1600,  # for better resolution
                        background_color='white', 
                        min_word_length=5  # to eliminate trivial words
                        ).generate(words_of_a_year)
    # save to a local file 
    wd_plot.to_file(os.path.join('report', f'word_cloud_plot_{year}.png'))
    # plot in notebook 
    plt.imshow(wd_plot, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# TODO: more helper methods to add, especially those for plotting

# ------ functions for both dataset --------

def transform_year_to_decade(year):
    """ 
    convert year into decade string, e.g. 'd10' means 1910s, and 'd00' means 2000s 

    :param year: a year integer to be converted 
    """
    return ("d" + str(year)[2] + '0')


def transform_year_to_decade_start(year):
    """ 
    convert year into decade start, e.g. 1925 -> 1920. 
    This function is catered to Ridgeline plot implementation. 

    :param year: the year to be converted 
    """
    return int(year / 10) * 10


def corr_timbre_nltk():
    """ compute the correlation between each timbre column and each nltk column """
    # read in df 
    timbre_avg_by_year = pd.read_csv('preprocess/timbre_avg_by_year.csv')
    nltk_scores_by_year = pd.read_csv('preprocess/nltk_scores_by_year.csv')
    # merge data 
    merged_df = timbre_avg_by_year.merge(nltk_scores_by_year, on='year')
    # compute correlation 
    nltk_cols = ['neg', 'neu', 'pos', 'compound']
    timbre_cols = [f'TimbreAvg{i}' for i in range(1, 13)]
    corr_mat = []
    for nltk_col in nltk_cols:
        corr_row = [] 
        for timbre_col in timbre_cols:
            corr_row.append(
                np.corrcoef(merged_df[timbre_col], 
                            merged_df[nltk_col]
                            )[0, 1]
                            )
        corr_mat.append(corr_row)
    return pd.DataFrame(corr_mat, index=nltk_cols, columns=timbre_cols)
            

# TODO: add ridgeline plot function here
# TODO: modify function comment 
# TODO: probably accept some positional argument to adjust colors? (manual adding is fine)
# TODO: plots needed include TimbreAvg1
# def ...


if __name__ == '__main__':
    print(corr_timbre_nltk())
