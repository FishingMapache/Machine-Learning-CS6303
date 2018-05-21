import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

if __name__ == "__main__":

    # Import data and split
    data = pd.read_csv('~/Desktop/sentiment labelled sentences/imdb_labelled.txt', delimiter="\t", header = None)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3)

    # Counting occurance
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    print(X_train_counts.shape)
    print(count_vect.vocabulary_.get(u'sad'))

    # Counting frequency
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    # Learning from features
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # Model Testing
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = clf.predict(X_test_tfidf)

    # Model Evaluation
    accuracy = np.mean(predicted == y_test)
    print('Accuracy: ', accuracy, '\n')
    print('Classification Report: ', metrics.classification_report(y_test, predicted))
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, predicted))
