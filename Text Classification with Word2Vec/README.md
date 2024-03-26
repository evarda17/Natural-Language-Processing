# Text Classification with Word2Vec

In this assignment, we explore text classification using Word2Vec (w2v) embeddings within the context of logistic regression. Leveraging the Triage dataset from previous weeks, our objectives are twofold: to utilize pre-trained w2v embeddings and to experiment with w2v embeddings trained on the Triage dataset itself.

## Overview

### Part I: Pre-trained Word2Vec Embeddings

We start by employing pre-trained w2v embeddings from the `gensim` library to represent our text data. Notable embeddings such as `glove-wiki-gigaword-300` and `word2vec-google-news-300` are explored for their potential in capturing semantic information from the dataset.

### Task Highlights

- **Sentence Embedding**: Implement a function to average word vectors within a sentence, handling out-of-vocabulary words with zero vectors.
- **Logistic Regression Classification**: Utilize `sklearn`'s logistic regression to classify text, based on vector representations derived from pre-trained embeddings.

### Part II: Training Custom Word2Vec Embeddings

Moving beyond pre-trained models, we train our w2v embeddings directly on the Triage training dataset. This allows for potentially more tailored semantic representations specific to the disaster-related context of our data.

### Implementation Details

- **Vectorization**: Sentence embeddings are obtained by averaging the w2v vectors of words in a sentence, with preprocessing including tokenization.
- **Model Training**: We explore the training of custom w2v models with varying hyperparameters, followed by logistic regression classification.
- **Evaluation**: The effectiveness of the models is assessed based on accuracy metrics for both the training and development (dev) datasets.

## Results and Analysis

Our exploration yields insights into the comparative performance of pre-trained versus custom-trained w2v embeddings:

- **Pre-trained Embeddings**: Offer a quick and effective way to leverage existing semantic knowledge, showing promising results with close train and test accuracies.
- **Custom-trained Embeddings**: While requiring more computational resources and time, custom-trained embeddings provide tailored representations that can potentially enhance model performance.

### Comparative Analysis

| Model                   | Train Accuracy | Dev Accuracy |
|-------------------------|----------------|--------------|
| NB                      | 0.845          | 0.731        |
| LR (Unigram)            | 0.800          | 0.741        |
| LR (Unigram + Bigram)   | 0.979          | 0.763        |
| LR (TF-IDF)             | 0.935          | 0.758        |
| LR (Pre-trained w2v)    | 0.762          | 0.759        |
| LR (Custom w2v)         | 0.771          | 0.773        |

- **General Observations**: Models utilizing word embeddings (pre-trained and custom) tend to exhibit better generalization, as evidenced by closer train and test accuracies.
- **Best Performance**: The custom-trained w2v model with logistic regression showcases the highest dev accuracy, indicating superior generalization capabilities among the explored methods.

## Conclusion

This exploration underscores the value of word embeddings in text classification tasks, particularly within domains requiring nuanced semantic understanding. Both pre-trained and custom-trained embeddings offer distinct advantages, with custom models providing a slight edge in accuracy and generalization for this specific dataset and task.
