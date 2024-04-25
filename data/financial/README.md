---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- text-classification
pretty_name: Financial Tweets with Sentiment class
dataset_info:
  features:
  - name: tweet
    dtype: string
  - name: sentiment
    dtype:
      class_label:
        names:
          '0': neutral
          '1': bullish
          '2': bearish
  - name: url
    dtype: string
  splits:
  - name: train
    num_bytes: 6848991
    num_examples: 38091
  download_size: 2648082
  dataset_size: 6848991
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- sentiment
- twitter
- finance
- crypto
- stocks
- tweet
- collection
---

# Financial Sentiment Analysis Dataset

## Overview
This dataset is a comprehensive collection of tweets focused on financial topics, meticulously curated to assist in sentiment analysis in the domain of finance and stock markets. It serves as a valuable resource for training machine learning models to understand and predict sentiment trends based on social media discourse, particularly within the financial sector.

## Data Description
The dataset comprises tweets related to financial markets, stocks, and economic discussions. Each tweet is labeled with a sentiment value, where '1' denotes a positive sentiment, '2' signifies a negative sentiment, and '0' indicates a neutral sentiment. The dataset has undergone thorough preprocessing, including sentiment mapping and the removal of duplicate entries, to ensure data quality and consistency.

### Dataset Structure
- **Tweet**: The text of the tweet, providing insights into financial discussions.
- **Sentiment**: A numerical label indicating the sentiment of the tweet (1 for bullish, 2 for bearish, and 0 for neutral).

## Dataset Size
- **Bullish Sentiments**: 17,368
- **Bearish Sentiments**: 8,542
- **Neutral Sentiments**: 12,181

## Sources
This dataset is an amalgamation of data from various reputable sources, each contributing a unique perspective on financial sentiment:

- [FIQA Sentiment Classification](https://huggingface.co/datasets/ChanceFocus/fiqa-sentiment-classification): A sentiment analysis dataset with 721 positive, 379 negative, and 11 neutral sentiments.
- [Stock Market Tweets Data](https://ieee-dataport.org/open-access/stock-market-tweets-data): A collection of tweets with 523 positive, 420 neutral, and 341 negative sentiments.
- [Stock Related Tweet Sentiment](https://www.kaggle.com/datasets/mattgilgo/stock-related-tweet-sentiment): A dataset featuring 5005 positive, 741 neutral, and 736 negative sentiments.
- [Master Thesis Data](https://github.com/moritzwilksch/MasterThesis/tree/main): Includes 3711 positive, 2784 neutral, and 2167 negative sentiments.
- [Twitter Stock Sentiment](https://github.com/poojathakoor/twitter-stock-sentiment): Comprises 702 positive, 595 negative, and 481 neutral sentiments.
- [Crypto Sentiment](https://github.com/surge-ai/crypto-sentiment/tree/main): Sentiment data for cryptocurrency-related tweets with 296 positive and 256 negative sentiments.
- [Stock Sentiment](https://github.com/surge-ai/stock-sentiment/tree/main): Sentiment analysis on stock-related tweets, including 327 positive and 173 negative sentiments.
- [Stockmarket Sentiment Dataset](https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset): Features 3685 positive and 2106 negative sentiments.
- [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment): Contains 2398 positive, 1789 negative, and 7744 neutral sentiments.

## Usage
This dataset is ideal for training and evaluating machine learning models for sentiment analysis, especially those focused on understanding market trends and investor sentiment. It can be used for academic research, financial market analysis, and developing AI tools for financial institutions.

## Acknowledgments
We extend our heartfelt gratitude to all the authors and contributors of the original datasets. Their efforts in data collection and curation have been pivotal in creating this comprehensive resource.

## License
This dataset is made available under the MIT license, adhering to the licensing terms of the original datasets.