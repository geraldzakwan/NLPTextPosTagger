import sys
import convert
import features
import pickle
import numpy as np
import nltk
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    if (sys.argv[1] == 'save'):
        # Save or load dataset to list of list of tuple
        if(sys.argv[2] == '1'):
            print "Using data ../UD_Indonesian_Conll2017/"
            path_name = '../UD_Indonesian_Conll2017/'
        elif(sys.argv[2] == '2'):
            print "Using data ../UD_Indonesian_v2.0/"
            path_name = '../UD_Indonesian_v2.0/'
        else:
            sys.exit("Option error")

        fload_train = path_name + 'list_of_tuple_train.pickle'
        fload_test = path_name + 'list_of_tuple_dev.pickle'
        train_tagged_sentences = convert.load_list(fload_train)
        test_tagged_sentences = convert.load_list(fload_test)

        training_sentences = train_tagged_sentences
        test_sentences = test_tagged_sentences

        chunk = int(sys.argv[3])

        # Number of train and test
        print 'Training data : ' + str(len(training_sentences))
        print 'Testing data : ' + str(len(test_sentences))

        X, y = features.transform_to_dataset(training_sentences)
        X_test, y_test = features.transform_to_dataset(test_sentences)

        # Number of X and y
        print 'X : ' + str(len(X))
        print 'y : ' + str(len(y))

        if(sys.argv[4] == '1'):
            # Using Perceptron
            print "\n\n============Using Perceptron============\n"
            clf_2 = Perceptron(alpha=.0001, penalty='l2', n_jobs=-1,
            #                       #shuffle=True, n_iter=10,
                                  verbose=1)
        elif(sys.argv[4] == '2'):
            # Using PassiveAggressiveClassifier
            print "\n\n============Using PassiveAggressiveClassifier============\n"
            clf_2 = PassiveAggressiveClassifier(n_jobs=-1,
            #                       #shuffle=True, n_iter=10,
                                  verbose=1)
        elif(sys.argv[4] == '3'):
            # Using SGD Classifier
            print "\n\n============Using SGDClassifier============\n"
            clf_2 = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1,
            #                       #shuffle=True, n_iter=10,
                                  verbose=1)
        elif(sys.argv[4] == '4'):
            # Using Multinomial Naive Bayes
            print "\n\n============Using MultinomialNB============\n"
            clf_2 = MultinomialNB(alpha=.0001)
        elif(sys.argv[4] == '5'):
            # Using Bernoulli Naive Bayes
            print "\n\n============Using BernoulliNB============\n"
            clf_2 = BernoulliNB(alpha=.0001)
        elif(sys.argv[4] == '6'):
            # Using Multi-layer Perceptron classifier.
            print "\n\n============Using MLPClassifier============\n"
            clf_2 = MLPClassifier(alpha=.0001)

        vect = DictVectorizer(sparse=False)

        total_word = len(X)
        total_processed = 0
        first_word = 0
        last_word = chunk
        while(total_processed <= total_word):
            print "Running Partial Fit from : " + str(first_word) + " to " + str(last_word)

            if(first_word == 0):
                vX = vect.fit_transform(X[first_word:last_word])
            else:
                vX = vect.transform(X[first_word:last_word])
            vy = y[first_word:last_word]
            clf_2.partial_fit(vX, vy, classes=np.unique(vy))

            total_processed = total_processed + chunk
            if(last_word + chunk < total_word - 1):
                first_word = first_word + chunk
                last_word = last_word + chunk
            else:
                total_processed = total_processed + chunk
                first_word = first_word + chunk
                last_word = total_word

        print 'Training completed'

        save_filename = "dataset_" + sys.argv[2] + "_classifier_" + sys.argv[4] + "_chunk_" + sys.argv[3] + ".pickle"
        pickle.dump(clf_2, open(save_filename, "wb"))

        vX = vect.transform(X_test)
        vy = y_test
        print "Accuracy:", clf_2.score(vX, vy)

        input_sentence = sys.argv[5]
        print features.pos_tag(nltk.word_tokenize(input_sentence), clf_2)
    elif (sys.argv[1] == 'load'):
        clf_2 = pickle.load(open(sys.argv[2], "rb" ))

        # Save or load dataset to list of list of tuple
        if("dataset_1" in sys.argv[2]):
            print "Using data ../UD_Indonesian_Conll2017/"
            path_name = '../UD_Indonesian_Conll2017/'
        elif("dataset_2" in sys.argv[2]):
            print "Using data ../UD_Indonesian_v2.0/"
            path_name = '../UD_Indonesian_v2.0/'
        else:
            sys.exit("Option error")

        fload_train = path_name + 'list_of_tuple_train.pickle'
        fload_test = path_name + 'list_of_tuple_dev.pickle'
        train_tagged_sentences = convert.load_list(fload_train)
        test_tagged_sentences = convert.load_list(fload_test)

        training_sentences = train_tagged_sentences
        test_sentences = test_tagged_sentences

        X, y = features.transform_to_dataset(training_sentences)
        X_test, y_test = features.transform_to_dataset(test_sentences)

        vect = DictVectorizer(sparse=False)
        input_sentence = sys.argv[3]

        # new_test_sentences = []
        # for i in range(0, len(test_sentences[0])):
        #     new_test_sentences.append(test_sentences[0][i][0])
        # print(new_test_sentences)
        # print features.pos_tag(X_test[0:2], clf_2)

        # print(X_test)
        # print clf_2.predict(X_test)
        print features.pos_tag(nltk.word_tokenize(sys.argv[3]), clf_2)
