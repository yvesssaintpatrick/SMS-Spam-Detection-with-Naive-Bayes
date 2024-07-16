import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# preprocess text data
def preprocess(text):
    
    text = text.lower()
    # Tokenize text into words
    words = text.split()
    return words

# train Naive Bayes classifier
def train(X_train, y_train):
    num_messages = len(X_train)
    num_spam = np.sum(y_train)
    num_ham = num_messages - num_spam

    # Calculate prior probabilities
    p_spam = num_spam / num_messages
    p_ham = num_ham / num_messages

    # dictionaries to store conditional probabilities
    word_given_spam = {}
    word_given_ham = {}

    # Calculate conditional probabilities
    for message, label in zip(X_train, y_train):
        words = preprocess_text(message)
        for word in words:
            if label == 1:
                word_given_spam[word] = word_given_spam.get(word, 0) + 1
            else:
                word_given_ham[word] = word_given_ham.get(word, 0) + 1

    # counts to probabilities
    total_words_spam = sum(word_given_spam.values())
    total_words_ham = sum(word_given_ham.values())
    for word in word_given_spam:
        word_given_spam[word] /= total_words_spam
    for word in word_given_ham:
        word_given_ham[word] /= total_words_ham

    return p_spam, p_ham, word_given_spam, word_given_ham

# predict the labels for new samples
def predict(X_test, p_spam, p_ham, word_given_spam, word_given_ham):
    y_pred = []
    for message in X_test:
        spam_prob = np.log(p_spam)
        ham_prob = np.log(p_ham)
        words = preprocess_text(message)
        for word in words:
            spam_prob += np.log(word_given_spam.get(word, 1e-10))
            ham_prob += np.log(word_given_ham.get(word, 1e-10))
        if spam_prob > ham_prob:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return np.array(y_pred)

# load and preprocess the dataset
def load(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["Label", "SMS"])
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    X = df['SMS'].values
    y = df['Label'].values
    return X, y

# perform train-test split
def perform(X, y, test_size=0.2, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# calculate true negatives, false positives, false negatives, and true positives
def calculate_matrix(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp

# calculate precision
def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)

# calculate recall
def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

# calculate F1 score
def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

# calculate accuracy
def calculate_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

# load and preprocess data
X, y = load("SMSSpamCollection")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = perform(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
p_spam, p_ham, word_given_spam, word_given_ham = train(X_train, y_train)

# Predict on test set
y_pred = predict(X_test, p_spam, p_ham, word_given_spam, word_given_ham)

# Calculate  matrix
tn, fp, fn, tp = calculate_matrix(y_test, y_pred)

# Calculate precision
precision = calculate_precision(tp, fp)

# Calculate recall
recall = calculate_recall(tp, fn)

# Calculate F1 score
f1 = calculate_f1_score(precision, recall)

# Calculate accuracy
accuracy = calculate_accuracy(tp, tn, fp, fn)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)
