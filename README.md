Overview

The goal of this project is to classify SMS messages as either spam or ham (non-spam) using a Naive Bayes classifier. The dataset used is the SMS Spam Collection Dataset, which contains labeled messages that are either spam or ham.

Data Preprocessing

The dataset is loaded and preprocessed using pandas. The text data is converted to lowercase and tokenized into words.

Training the Naive Bayes Classifier

The training function calculates the prior probabilities of spam and ham messages, as well as the conditional probabilities of each word given spam or ham.

Prediction

The prediction function calculates the log probabilities of a message being spam or ham and classifies the message based on these probabilities.

Model Evaluation

The model's performance is evaluated using various metrics such as precision, recall, F1-score, and accuracy. These metrics are calculated based on the true positives, true negatives, false positives, and false negatives.

Results
After training and testing the model, the following metrics are calculated:

Precision: The ratio of correctly predicted spam messages to the total predicted spam messages.
Recall (Sensitivity): The ratio of correctly predicted spam messages to all the actual spam messages.
F1 Score: The weighted average of Precision and Recall.
Accuracy: The ratio of correctly predicted messages (both spam and ham) to the total messages.

Conclusion

This project demonstrates the implementation of a Naive Bayes classifier to detect spam messages in a dataset of SMS messages. The code includes data preprocessing, training, predicting, and evaluating the model using various classification metrics.
