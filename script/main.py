import sys
import pprint
import convert
import features
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features.extract(features.untag(tagged), index))
            y.append(tagged[index][1])

    return X, y

if __name__ == '__main__':
    # Save or load dataset to list of list of tuple
    if (sys.argv[1] == 'save'):
        fname = '../UD_Indonesian_Conll2017/id-ud-train.conllu'
        fsave = '../UD_Indonesian_Conll2017/list_of_tuple.pickle'
        tagged_sentences = convert.save_list(fname, fsave)
    elif (sys.argv[1] == 'load'):
        fload = '../UD_Indonesian_Conll2017/list_of_tuple.pickle'
        tagged_sentences = convert.load_list(fload)

    # Check if list loaded well
    # print tagged_sentences[0]
    # print '-----------------'
    # print tagged_sentences[1]

    # Number of sentences
    # print "Tagged sentences: ", len(tagged_sentences)

    pprint.pprint(features.extract(['This', 'is', 'a', 'sentence'], 2))

    # Split the dataset for training and testing
    cutoff = int(.75 * len(tagged_sentences))
    training_sentences = tagged_sentences[:cutoff]
    test_sentences = tagged_sentences[cutoff:]

    print len(training_sentences)   # 2935
    print len(test_sentences)         # 979

    X, y = transform_to_dataset(training_sentences)

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])

    clf.fit(X[:10000], y[:10000])   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

    print 'Training completed'

    X_test, y_test = transform_to_dataset(test_sentences)

    print "Accuracy:", clf.score(X_test, y_test)

    # Accuracy: 0.904186083882
    # not bad at all :)
