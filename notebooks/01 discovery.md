---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: 'Python 3.9.13 (''.venv'': venv)'
    language: python
    name: python3
---

# Create Keywords from Text using TF-IDF
This is a redo with some programmatic enhancements; all credits to [Kavita Ganesan]([https://kavita-ganesan.medium.com/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667]) on Medium

## Settings
### Libraries and Modules

```python
# General libraries
from pathlib import Path
import pandas as pd
import numpy as np

# Local libraries
from config import SETTINGS
from preprocess import pre_process
```

```python
SOURCE = SETTINGS['SOURCE']
```

## Example Dataset
There are two files that we are going to use:

* a large dataset ("corpus") containing a collection (20,000) of items. That collection represents the general population of the individual member that we're going to look at. 
    * `stackoverflow-data-idf.json`
* a single member of the general population, which will be the focus of the analysis:
    * `stackoverflow-test.json`

```python
def read_json_file(path: Path) -> pd.DataFrame:
    df = (
        pd.read_json(
            SOURCE / path,
            lines = True
        )
        .assign(
            text = lambda x: x.title + x.body
        )
        .assign(
            text = lambda x: x.text.apply(lambda x: pre_process(x))
        )
    )

    return df
```

```python
df_idf = read_json_file(SOURCE / 'stackoverflow-data-idf.json')
df_test = read_json_file(SOURCE / 'stackoverflow-test.json')
```

Notice that this Stack Overflow dataset contains 19 fields including post title, body, tags, dates, and other metadata which we don’t need for this tutorial. For this tutorial, we are mostly interested in the body and title. These will become our source of text for keyword extraction.

We will now create a field that combines both body and title so we have the two in one field. We will also print the second text entry in our new field just to see what the text looks like.

```python
df_idf.text.loc[0]
```

# Create Vocabulary and Word Counts for IDF
In this section we need to create the vocabulary and start the counting process. Let us use a standard Class from `sklearn`, `CountVectorizer`. We are going to apply that to the entire collection of `text` in the TF-IDF dataframe, followed by word counts in the vocabulary:

```python
from sklearn.feature_extraction.text import CountVectorizer
import re
```

When working with text mining applications, we often hear of the term “stop words” or “stop word list” or even “stop list”. Stop words are basically a set of commonly used words in any language, not just English.

The reason why stop words are critical to many applications is that, if we remove the words that are very commonly used in a given language, we can focus on the important words instead. For example, in the context of a search engine, if your search query is “how to develop information retrieval applications”, If the search engine tries to find web pages that contained the terms “how”, “to” “develop”, “information”, ”retrieval”, “applications” the search engine is going to find a lot more pages that contain the terms “how”, “to” than pages that contain information about developing information retrieval applications because the terms “how” and “to” are so commonly used in the English language. If we disregard these two terms, the search engine can actually focus on retrieving pages that contain the keywords: “develop” “information” “retrieval” “applications” – which would bring up pages that are actually of interest. This is just the basic intuition for using stop words.

Stop words can be used in a whole range of tasks and here are a few:

* Supervised machine learning – removing stop words from the feature space
* Clustering – removing stop words prior to generating clusters
* Information retrieval – preventing stop words from being indexed
* Text summarization- excluding stop words from contributing to summarization scores & removing stop words when computing ROUGE scores

```python
def get_stop_words(stop_file_path: Path) -> frozenset:
    """Generates a list of stop words as a frozenset using some input file

    Parameters
    ----------
    stop_file_path : Path
        Path to the file (text) to ready

    Returns
    -------
    frozenset
        Frozen set (immutable set) of unique stop words
    """
    with open(stop_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stop_set = set(
            m.strip() for m in stopwords
        )
        return frozenset(stop_set)
```

```python
# Try this on a generic list of stop words
stopwords = get_stop_words(SOURCE / "stopwords.txt")
```

```python
# Process the text column from the dataframe we loaded
cv = CountVectorizer(
    max_df=0.85,
    stop_words=stopwords,
    max_features=10000,
    ngram_range=(1,2)
)
word_count_vector = cv.fit_transform(df_idf.text)
```

```python
# Figure out which words are most common
word_count_dict = dict(
    zip(
        cv.get_feature_names_out(),
        np.asarray(word_count_vector.sum(axis=0))[0]
        )
)

# Then sort and display the top-10 word list from that corpus count
top_n = {k : v for k, v in sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True)[:20]}
top_n

```

# TF-IDF Transform
We are going to use the `CountVectorizer` sparse matrix produced earlier, and we're going to capture the IDF quantity.

```python
from sklearn.feature_extraction.text import TfidfTransformer
```

```python
# The TF-IDF transformer is based on the large corpus
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
```

Extremely important point: the IDF should always be based on a large corpora, and **should be representative of texts you would be using to extract keywords**.

I’ve seen several articles on the Web that compute the IDF using a handful of documents. You will defeat the whole purpose of IDF weighting if its not based on a large corpora as:

* your vocabulary becomes too small, and
* you have limited ability to observe the behavior of words that you do know about.

```python
# Pick our test document
doc = df_test.text.loc[5]

# Generate the TF-IDF for the selected document based on the corpus vocabulary
# TO-DO: use pipeline on this one
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
```

```python
# Use the same logic as prior but now use the new quantity TF-IDF to estimate most relevant words
word_count_dict = dict(
    zip(
        cv.get_feature_names_out(),
        # this only works as this is a one-dimensional vector
        np.asarray(tf_idf_vector.sum(axis=0))[0]
        )
)
```

```python
# Then sort and display the top-10 word list from that corpus count
top_n = {k : v for k, v in sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True)[:10]}
top_n
```

Here we can map back the TF-IDF from each word in this test document, based on the vocabulary established in the corpus, and using the TF-IDF statistic computed from that test document on the basis of the corpus. These are the words that really "pop" based on the corpus. We note that it's important to analyze that particular item against the rest of the corpus. Otherwise, it's not as meaningful (e.g. a medical corpus, and an item about football would probably pop).

```python
top_n
```

```python
print(
    {
        "I analyzed" : doc,
        "I got these keywords" : top_n
    }
)
```

```python

```
