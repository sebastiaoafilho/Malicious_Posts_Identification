# Detection of Malicious Posts in the Dark Web using Supervised Machine Learning

## General Description

This repository contains the artifacts developed for the identification of malicious posts on the Dark Web using supervised machine learning.

## Datasets

- **dataset_I:**
  - Labeled based on the simultaneous occurrence of IoCs and contextual keywords.
  - Contains 17,675 posts from the Dark Web (1,665 Relevant, 16,010 Not Relevant).

- **dataset_II:**
  - Labeled considering IoCs, keywords, and manual analysis.
  - Contains 26,575 posts from the Dark Web (3,341 Relevant, 23,234 Not Relevant).

- **dataset_III:**
  - Contains 7,498 previously unlabeled posts from the Dark Web.
  - The labels for these posts were assigned through the developed post identification model, assigning a probability between 0 and 1 for the relevance of the post.

...

## Tools and Notebooks

- **ioc_explorer:**
  - ioc_explorer.ipynb: Jupyter Notebook for IoC identification and extraction.

- **lda_topics:**
  - keyword_search.ipynb: Marks posts with contextual keywords.
  - lda_topics.ipynb: Generates LDA topics and analyzes frequent words.

- **ml_models:**
  - ml_models.ipynb: Machine learning model training.
  - ml_best_model.ipynb: Application of the trained model to new posts.


## Pre-processing

- **pre_process:**
  - pre_process_I.ipynb: Prepares data for IoC extraction.
  - pre_process_II.ipynb: Prepares data for supervised machine learning model training.

