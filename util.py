"""
This file contains the necessary auxiliary functions to load, process, and plot data

Author: Sheng Yang
Date: 2021/05/14
"""

# load packages 
import os 
import re 
import pandas as pd
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS


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

# ------ functions for combining two dataset --------
