
# Text Classification with Naive Bayes

This series of homework assignments involve the use of various models and input representations for performing text classification on consistent datasets across three weeks. The focal point of this homework is the application of the Naive Bayes (NB) model for text classification.

## Dataset Overview

The dataset comprises text messages, social media posts (primarily from Twitter), and snippets from news articles related to various disasters. All entries have been translated and annotated by humans via the crowdsourcing platform CrowdFlower (now known as Appen), leading to a mix of translations with occasional non-English words. This real-world scenario represents the challenges often faced in NLP tasks.

The classification task is binary: to determine whether a document relates to aid (class AID) or does not (class NOT). Aid-related documents include requests for assistance like food, water, shelter, reports on dire situations, and disaster relief efforts.

## Project Structure

- `util.py`: Contains data loading functions and classes essential for handling the dataset.

### Homework Breakdown

#### HW4.1 Naive Bayes
- **Part I**: Implementation of the Naive Bayes classifier from scratch, based on algorithms outlined in the textbook "Speech and Language Processing, 3rd Edition" by Jurafsky and Martin, specifically Chapter 4.2.

### Dataset Exploration

To explore the dataset, `util.py` provides functions and classes for loading and handling the data. The dataset is divided into training, validation (development), and test sets. The primary focus is on training the model with the training set and evaluating it on both the training and development sets.

#### Loading and Exploring Data
```python
from util import load_data
# Load dataset
dataset = load_data("./data/triage")
# Explore dataset structure
train_data = dataset.train
```

### Implementing Naive Bayes

- **Training**: Implement the `trainNB` function following the algorithm provided in the textbook.
- **Inference**: Implement the `NB_inference` function to classify new documents based on the trained model.

#### Example Usage
```python
# Training the Naive Bayes classifier
log_prior, log_likelihood, V = trainNB(dataset)
# Evaluating the model
testNB(dataset.dev, log_prior, log_likelihood, V, [0,1])
testNB(dataset.train, log_prior, log_likelihood, V, [0,1])
```

## Tips and Recommendations

- Training might take a significant amount of time. Consider using a progress bar (e.g., tqdm) for better tracking.
- Expected accuracy: ~73% on development data and ~83% on training data. Achieving similar metrics indicates a well-performing model.
- Explore `util.py` for additional functionalities that might enhance your project.

## Additional Resources

- For a detailed understanding of the crowdsourcing translation effort and its challenges, refer to [this paper](https://nlp.stanford.edu/pubs/munro2010translation.pdf) by Munro.
- Naive Bayes implementation guidelines can be found in Chapter 4.2 of [Speech and Language Processing, 3rd Edition](https://web.stanford.edu/~jurafsky/slp3/4.pdf) by Jurafsky and Martin.

