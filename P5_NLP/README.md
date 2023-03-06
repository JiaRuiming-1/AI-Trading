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
 
