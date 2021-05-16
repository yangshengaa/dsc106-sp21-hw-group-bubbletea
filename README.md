# dsc106-sp21-hw-group-bubbletea

UCSD DSC 106 SP21 Project by Jasmine Guan and Sheng Yang

## Overview

This is a Data Vis project that attempts to analyse the correlation between timbre of US pop music and news sentiment of a period. The leisure or ennui of a decade could be reflected by the soft and comforting tone color of songs, and the occurrence of social unrest may likely be coupled with a sharp and aggressive timbre in music.

## Dependencies

please pip install the following python packages:
`Altair`
`NLTK`
`WordCloud`
`Text2Emotion`

## Data Source

- **Timbre Features**: [Audio features of songs ranging from 1922 to 2011](https://www.kaggle.com/uciml/msd-audio-features?select=year_prediction.csv); [Analyzer Documentation](http://modelai.gettysburg.edu/2012/music/docs/EchoNestAnalyzeDocumentation.pdf)
- **News Sentiment**: [New York Times Articles 1920-2020](https://www.kaggle.com/tumanovalexander/nyt-articles-data).

## Citation

- **WordCloud Tutorial**: [Generating WordClouds in Python](https://www.datacamp.com/community/tutorials/wordcloud-python)
- **NLTK Tutorial**: [Sentiment Analysis: First Steps With Python's NLTK Library](https://realpython.com/python-nltk-sentiment-analysis/)
- **Text2Emotion Tutorial**: [Demo](https://colab.research.google.com/drive/1sCAcIGk2q9dL8dpFYddnsUin2MlhjaRw?usp=sharing#scrollTo=nxET8yW3HIvL)

Example: Word Cloud in year 2000 ![Word Cloud in 2000](report/word_cloud_plot_2000.png)

## For developers (plan)

Meeting 05/16 issues:

TODO: discuss and determine which sentiment analyzer to use (also check demo in the notebook when discussing).
There are:

- **NLTK**: mainstreamed Natural Language Toolkit, could employ a large pre-trained network for emotion classification (implies accuracy)
- **Text2Emotion**: A light sentence emotion classifier (may not be as accurate as NLTK)
- **AFINN**: this is a joke. It could only be classified into positive, negative, and neutral (useless)

| Package      | Emotion Category Supported | Accuracy |
| ----------- | ----------- | ----------- |
| NLTK      | negative, neutral, positive, compound       | high |
| Text2Emotion   | Happy, Angry, Surprise, Sad, Fear | Not sure|

TODO: discuss how to use these emotion
