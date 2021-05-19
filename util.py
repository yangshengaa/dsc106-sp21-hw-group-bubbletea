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
import altair as alt
import datetime
from vega_datasets import data

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
sentiment_cat = ['neg', 'neu', 'pos', 'compund']
timbre_cols = [f'TimbreAvg{i}' for i in range(1, 13)]


# ------ functions for news analysis only --------

def read_csv_as_txt(path):
    """ 
    read in the csv of a given year into a strings, and discard all "new york times" 
    
    :param year: an integer, the year we are interested in loading in
    """
    return open(path).read().lower().replace('new york times', '')


def count_words_by_year(path):
    """ 
    load in csv of a specific year and perform value count

    :param year: the year we are interested in 
    :return a pandas Series value_counts by word 
    """
    # read in as a text file
    words_of_a_year = read_csv_as_txt(path)
    # count words by value counts 
    words_count = pd.Series(re.findall(r'[a-z]+', words_of_a_year.lower())).value_counts()
    # drop meaningless words 
    return words_count.drop(index=to_drop)


def plot_word_cloud_of_year(path):
    """ 
    make a wordcloud plot of a give year (for demo only)
    
    :param year: an integer, the year we are interested in loading in
    """ 
    words_of_a_year = read_csv_as_txt(path)
    wd_plot = WordCloud(max_words=100, 
                        width=2560, height=1600,  # for better resolution
                        background_color='white', 
                        min_word_length=5  # to eliminate trivial words
                        ).generate(words_of_a_year)
    # save to a local file 
    wd_plot.to_file(os.path.join('report', 'word_cloud_plot_1920.png'))
    # plot in notebook 
    plt.imshow(wd_plot, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def generate_demo_sentiment_word_cloud():
    """ make a word cloud on four words (negative, neutral, positive, compound) only """
    sample_sentiments = 'negative negative, neutral, neutral' + \
        ' neutral positive positive positive positive positive compound compound'
    wd_plot = WordCloud(width=1200, height=800,  # for better resolution
                        background_color='white').generate(sample_sentiments)
    plt.imshow(wd_plot, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# ------ functions for both dataset --------

def read_avg_by_year():
    """ read in average features by year """
    timbre_avg_by_year = pd.read_csv('preprocess/timbre_avg_by_year.csv')
    nltk_scores_by_year = pd.read_csv('preprocess/nltk_scores_by_year.csv')
    return timbre_avg_by_year, nltk_scores_by_year


def transform_year_to_decade(year):
    """ 
    convert year into decade string, e.g. 'd10' means 1910s, and 'd00' means 2000s 

    :param year: a year integer to be converted 
    """
    return ("d" + str(year)[2] + '0')


def corr_timbre_nltk(timbre_avg_by_year, nltk_scores_by_year):
    """ compute the correlation between each timbre column and each nltk column """
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
            

def year_to_decade_datetime(year):
    """ convert year into decade in for the format that can be recognized as datetime object"""
    return str(year)[:3] + "0"


def transform_date(year): 
    """ transform year to it's datetime object, auto-filling in January 1st"""
    return datetime.datetime.strptime(str(year), "%Y")


def plot_ridgeline(input_df, time_unit_field, value_col):
    """ graph ridgeline plot of given dataframe, time unit field and value column"""
    df = input_df.copy()
    df[time_unit_field] = df[time_unit_field].apply(transform_date)
    step = 30   # adjust height of each kde
    overlap = 1
    to_transform = 'mean(' + value_col + ')'
    ridgeline = alt.Chart(df, height=step).transform_timeunit(
        as_ = "Decade", timeUnit="year", field=time_unit_field
    ).transform_joinaggregate(
        mean_val=to_transform, groupby=["Decade"]
    ).transform_bin(
        ['bin_max', 'bin_min'], value_col, bin=alt.Bin(maxbins=10)
    ).transform_aggregate(
        value='count()', groupby=["Decade", 'mean_val', 'bin_min', 'bin_max']
    ).transform_impute(
        impute='value', groupby=['Decade', 'mean_val'], key='bin_min', value=0
    ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='lightgray',
        strokeWidth=0.5
    ).encode(
        alt.X('bin_min:Q', bin='binned', title='Timbre 2 Average By Decades'),
        alt.Y(
            'value:Q',
            scale=alt.Scale(range=[step, -step * overlap]),   
            axis=None
        ),
        alt.Fill(
            'mean_val:Q',
            legend=None,
            scale=alt.Scale(domain=[50, -100], scheme='redyellowblue')  # adjust color 
        )
    ).facet(
        row=alt.Row(
            'Decade:T',    # only accepts T type: convert things to T type first 
            title=None,
            header=alt.Header(labelAngle=0, labelAlign='right', format='%Y'+"s")
        )
    ).properties(
        title='Timbre Avg by Decade (Ridgeline)',
        bounds='flush'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ).configure_title(
        anchor='end'
    )
    return ridgeline


# TODO: plots needed include TimbreAvg1


def visualize_corr_timbre_nltk(timbre_avg_by_year, nltk_scores_by_year): 
    """ visualize the correlation above """
    corr_mat = corr_timbre_nltk(timbre_avg_by_year, nltk_scores_by_year)
    corr_mat_long = corr_mat.reset_index().melt('index', var_name="Timbre Avg",
                                                value_name="Correlation").rename(columns={'index': 'sentiment'})
    corr_mat_long['Correlation'] = corr_mat_long.Correlation.round(2) # round to 2 decimal places for good display
    return alt.Chart(corr_mat_long).mark_rect().encode(
        y=alt.Y('sentiment:O', sort=sentiment_cat),
        x=alt.X('Timbre Avg:O', sort=timbre_cols),
        tooltip=['sentiment', 'Timbre Avg', 'Correlation'],
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme="redblue"))
    ).properties(
        title='Correlation Between Timbre Avgs and Sentiment',
        width=800, height=240
    )


def visualize_timbre_nltk_regression(timbre_avg_by_year, nltk_scores_by_year):
    """ for demo purpose only """ 
    # merge dataframes 
    merged_df = timbre_avg_by_year.merge(nltk_scores_by_year, on='year')
    merged_df['decade'] = (
        (merged_df['year'] / 10).astype(int) * 10).astype(str) + 's'
    
    # make a scatter plot 
    selection = alt.selection_multi(fields=['decade'])
    color = alt.condition(selection,
                        alt.Color('decade:N', legend=None,
                                    scale=alt.Scale(scheme='magma')),
                        alt.value('lightgray'))

    scatter = alt.Chart(merged_df).mark_point().encode(
        x=alt.X('neg:Q', scale=alt.Scale(zero=False), axis=alt.Axis(grid=False)),
        y=alt.Y('TimbreAvg3:Q', axis=alt.Axis(grid=False)),
        color=color,
        tooltip=['year', 'neg', 'TimbreAvg3']
    ).properties(
        title='Annual Average Timbre Feature and Negative Sentiment'
    )
    legend = alt.Chart(merged_df).mark_rect().encode(
        y=alt.Y('decade:N', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(selection)
    return (scatter + scatter.transform_regression('neg', 'TimbreAvg3').mark_line()
    .encode(strokeDash=alt.value([5, 5]), color=alt.value('darkblue'))) | legend


def visualize_agg_plot(timbre_avg_by_year, nltk_scores_by_year):
    """ 
    this function makes the aggregate plot that helps select each year for individual checking

    :param timbre_avg_by_year: the df read in directly from preprocess 
    :param nltk_scores_by_year: the df read in directly from preprocess
    """
    corr_mat = corr_timbre_nltk(timbre_avg_by_year, nltk_scores_by_year)
    corr_mat_long = corr_mat.reset_index().melt('index', var_name="Timbre Avg",
                                                value_name="Correlation").rename(columns={'index': 'sentiment'})
    corr_mat_long['Correlation'] = corr_mat_long.Correlation.round(
        2)  # round to 2 decimal places for good display
    # process data
    timbre_avg_by_year['decade'] = (
        (timbre_avg_by_year.year / 10).astype(int) * 10).astype(str) + 's'
    timbre_long = timbre_avg_by_year.melt(
        ['decade', 'year'], var_name='Timbre Avg', value_name='Timbre Avg Value')

    nltk_scores_by_year['decade'] = (
        (nltk_scores_by_year.year / 10).astype(int) * 10).astype(str) + 's'
    nltk_long = nltk_scores_by_year.melt(
        ['decade', 'year'], var_name='sentiment', value_name='sentiment value')

    timbre_nltk_long = timbre_long.merge(nltk_long, on=['decade', 'year'])

    # set up selector
    var_sel_cor = alt.selection_single(fields=['sentiment', 'Timbre Avg'], clear=True,
                                    init={'sentiment': 'neg', 'Timbre Avg': 'TimbreAvg3'})

    # set up correlation heat map as the base
    base = alt.Chart(corr_mat_long).mark_rect().encode(
        y=alt.Y('sentiment:O', sort=sentiment_cat),
        x=alt.X('Timbre Avg:O', sort=timbre_cols),
        tooltip=['sentiment', 'Timbre Avg', 'Correlation'],
        color=alt.condition(var_sel_cor, alt.value('gray'), alt.Color(
            'Correlation:Q', scale=alt.Scale(scheme='redblue')))
    ).properties(
        title='Correlation Between Timbre Avgs and Sentiment',
        width=800, height=240
    ).add_selection(var_sel_cor)

    # add text
    text = alt.Chart(corr_mat_long).encode(
        y=alt.Y('sentiment:O', sort=sentiment_cat),
        x=alt.X('Timbre Avg:O', sort=timbre_cols)
    ).mark_text().encode(
        text='Correlation',
        color=alt.condition(
            abs(alt.datum['Correlation']) > 0.5,
            alt.value('white'),
            alt.value('black')
        )
    )

    # set up selected scatter plot
    selection = alt.selection_multi(fields=['decade'])
    color = alt.condition(selection,
                        alt.Color('decade:N', legend=None, scale=alt.Scale(scheme='magma')
                                    ),
                        alt.value('lightgray'))

    scat_plot = alt.Chart(timbre_nltk_long).transform_filter(
        var_sel_cor
    ).mark_point().encode(
        x=alt.X('sentiment value:Q', scale=alt.Scale(
            zero=False), axis=alt.Axis(grid=False)),
        y=alt.Y('Timbre Avg Value:Q', axis=alt.Axis(grid=False)),
        color=color
    ).properties(
        title='Scatter Plot between Sentiment and Timbre Avgs',
        width=790, height=400
    )

    # set up selection
    legend = alt.Chart(timbre_nltk_long).transform_filter(
        var_sel_cor
    ).mark_rect().encode(
        y=alt.Y('decade:N', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(selection)

    return alt.vconcat((base + text),
                (scat_plot + scat_plot.transform_regression('sentiment value', 'Timbre Avg Value').mark_line().encode(
                    strokeDash=alt.value([5, 5]), color=alt.value('darkblue')
                )) | legend)
