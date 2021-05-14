"""
This file contains the necessary auxiliary functions to perform ploting 

Author: Sheng Yang
Date: 2021/05/14
"""

# load packages 
import os 
import re 
import pandas as pd

# constants 
news_path = 'data/archive'
to_drop = ['a', 'an', 'the', 'and', 'so', 'therefore', 'for',
           'is', 'are', 'was', 'were', 'am', 'be', 'as', 'has', 'had',
           'of', 'to', 'by', 'then', 'on', 'at', 'with', 'from', 'in',
           'it', 'this', 'that', 'those', 'who', 'whose', 'whom', 'which',
           'his', 'he', 'her', 'she', 'they', 'them', 'their', 's'
           ]  # words to drop, meaningless to consider by themselves


# ------ functions --------

def count_words_by_year(year):
    """ 
    load in csv of a specific year and perform value count

    :param year: the year we are interested in 
    :return a pandas Series value_counts by word 
    """
    # read in as a text file
    words_of_a_year = open(news_path + f'/df_{year}.csv').read()
    # count words by value counts 
    words_count = pd.Series(re.findall(r'[a-z]+', words_of_a_year.lower())).value_counts()
    # drop meaningless words 
    return words_count.drop(index=to_drop)


def document_frequency(year, word):
    """
    compute tf_idf of a word in a given year 

    :param year: the year we are interested in finding 
    :param word: a specific word that we are interested in computing the tf_idf 
    :return tf_idf
    """
    # read in csv 
    df = pd.read_csv(news_path + f'/df_{year}.csv', index_col=0)
    return df.sentence.str.contains(word).mean()

# TODO: more helper methods to add, especially those for plotting 
