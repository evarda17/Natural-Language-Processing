# Text Classification with Logistic Regression

This assignment builds upon our previous work with the Triage dataset, advancing from Naive Bayes to exploring text classification using Logistic Regression. Unlike the previous homework, where models were provided or utilized from libraries, this week we will implement a logistic regression classifier from scratch, with no reliance on high-level APIs like `sklearn` for the implementation of the logistic regression algorithm itself.

## Overview

- **Objective**: To explore different methods of text to vector conversion and implement a logistic regression classifier for text classification.
- **Dataset**: The Triage dataset, used in previous assignments, consisting of text messages, social media posts, and news article snippets related to various disasters.

## Part I: Text Input Representation

### Conversion Methods

1. **Count Vectors**: Utilizing `sklearn.feature_extraction.text.CountVectorizer` to convert text documents into a matrix of token counts.
   
2. **TF-IDF Vectors**: Employing `sklearn.feature_extraction.text.TfidfVectorizer` to reflect how important a word is to a document in a collection or corpus.

### Implementation Highlights

- The CountVectorizer and TfidfVectorizer are explored to prepare the data for logistic regression classification.
- Various forms of vectorization include using unigrams and bigrams to capture single words and word pairs, respectively.

## Part II: Implementing Logistic Regression

A logistic regression classifier is implemented from scratch, following the standard components of a linear classifier:

1. **Features**: Represented by the vectors obtained from Part I.
2. **Forward Pass**: Utilizing the sigmoid function to predict the probability of a class.
3. **Loss Function**: The cross-entropy loss function is used to quantify the difference between the predicted and actual class labels.
4. **Optimizer**: Gradient descent is applied to minimize the loss function, adjusting the model's weights.

The implementation details include initializing parameters, making predictions, calculating the loss function, computing the gradient, and updating the parameters iteratively.

## Part III: Model Comparison and Analysis

The logistic regression model is trained using different text vector representations:

- Unigram count vectors
- Bigram count vectors
- TF-IDF vectors

Normalization is applied to vectors before feeding them into the logistic regression model to ensure proper convergence.

## Observations and Conclusion

- The logistic regression model's performance is evaluated based on accuracy, comparing results across different vector representations.
- The comparison indicates how vectorization methods impact the classifier's ability to generalize from training to unseen data.
- Observations on overfitting are made based on the discrepancy between training and validation accuracies, particularly with complex models like those using bigram counts or TF-IDF vectors.

## Wrapping Up

The assignment concludes with reflections on the models' performance compared to the Naive Bayes classifier from the previous homework. The exploration reveals insights into the trade-offs between model complexity, overfitting, and generalization capabilities.
