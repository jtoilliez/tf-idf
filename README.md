# tf-idf
 Implementation of TF-IDF for keyword extraction in documents of various formats

# What is TF IDF
In information retrieval, `TF-IDF `(also TF*IDF, TFIDF, TF-IDF, or TF-IDF), short for **term frequency–inverse document frequency**, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The `TF-IDF `value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.

`TF-IDF` is one of the most popular term-weighting schemes today. A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries use `TF-IDF`.

# Which Libraries to Use
The one problem that I noticed with these libraries is that they are meant as a pre-step for other tasks like clustering, topic modeling, and text classification (e.g., `sklearn`).

TF-IDF can actually be used to extract important keywords from a document to get a sense of what characterizes a document. For example, if you are dealing with Wikipedia articles, you can use `TF-IDF` to extract words that are unique to a given article. These keywords can be used as a very simple summary of a document, and for text-analytics when we look at these keywords in aggregate.