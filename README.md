# dsc106-sp21-hw-group-bubbletea

UCSD DSC 106 SP21 Project by Jasmine Guan and Sheng Yang

## Overview

This is a Data Vis project that attempts to analyse the correlation between timbre of US pop music and news sentiment of a period. The leisure or ennui of a decade could be reflected by the soft and comforting tone color of songs, and the occurrence of social unrest may likely be coupled with a sharp and aggressive timbre in music.

## Dependencies

please pip install the following python packages:
`Altair`
`NLTK`
`WordCloud`

## Data Source

- **Timbre Features**: [Audio features of songs ranging from 1922 to 2011](https://www.kaggle.com/uciml/msd-audio-features?select=year_prediction.csv); [Analyzer Documentation](http://modelai.gettysburg.edu/2012/music/docs/EchoNestAnalyzeDocumentation.pdf)

| Timbre Basis    | Explanation |
| ----------- | ----------- |
| TimbreAvg1  | average loudness of the segment  |
| TimbreAvg2  | brightness |
| TimbreAvg3  | more closely correlated to the flatness of a sound |
| TimbreAvg4  | stronger attack |

- **News Sentiment**: [New York Times Articles 1920-2020](https://www.kaggle.com/tumanovalexander/nyt-articles-data).

## Citation

- **WordCloud Tutorial**: [Generating WordClouds in Python](https://www.datacamp.com/community/tutorials/wordcloud-python)
- **NLTK Tutorial**: [Sentiment Analysis: First Steps With Python's NLTK Library](https://realpython.com/python-nltk-sentiment-analysis/)

Example: Word Cloud in year 2000 ![Word Cloud in 2000](report/word_cloud_plot_2000.png)


### For developer references

- **Color Schemes**: [Available Color Schemes](https://vega.github.io/vega/docs/schemes/#reference)


### Task assignment
