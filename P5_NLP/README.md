## Project Instruction
update later...

## Financial Sentiment Analysis
In this section, I will construct a simple model to analysis the sentiment consistency based on company year and half year financial reports. `NLP_on_Finacial_Statement
.ipynb` is the source code, check it feel free.

  1. We download data by `requests`, and clean data by `re` and `BeautifulSoup4` 
  2. Then we can transform data to Bag of Words to calculate TF-IDF vectors. We use a sentiment word dict wchich is called Loughran McDonald Sentiment Word Lists(Chinese version). Youcan find in folder named "dict". 
  3. We use each vector of document bettwen time and neighbor to calcualte their similarity. We can choose `jaccard_score` or `cosine_similarity` method.
  4. As the conclusion, I found that the larger cosistency sentiment value, the higher the company's market cap.

## Concepts
NLP Data process step below: 

Note that, Chinese don't need to nomalized and Lemmatize. Chinese tokenize toolkit in python I apply is `pkuseg`, the stop words also apply `nltk`.

<img src="images/1.jpg" width="500px">

TF-IDF formula below:

The TF means a word frequency of one document. The IDF means is the word is nomal. TF-IDF express the word importance in all documents. We can look as a weight of key-word

<img src="images/2.jpg" width="500px">
