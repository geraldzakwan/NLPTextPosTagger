import sys
import convert
import features
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    # Save or load dataset to list of list of tuple
    if(sys.argv[2] == '1'):
        path_name = '../UD_Indonesian_Conll2017/'
    elif(sys.argv[2] == '2'):
        path_name = '../UD_Indonesian_v2.0/'
    else:
        sys.exit("Option error")

    if (sys.argv[1] == 'save'):
        fname_train = path_name + 'id-ud-train.conllu'
        fsave_train = path_name + 'list_of_tuple_train.pickle'
        fname_test = path_name + 'id-ud-dev.conllu'
        fsave_test = path_name + 'list_of_tuple_dev.pickle'
        train_tagged_sentences = convert.save_list(fname_train, fsave_train)
        test_tagged_sentences = convert.save_list(fname_test, fsave_test)
    elif (sys.argv[1] == 'load'):
        fload_train = path_name + 'list_of_tuple_train.pickle'
        fload_test = path_name + 'list_of_tuple_dev.pickle'
        train_tagged_sentences = convert.load_list(fload_train)
        test_tagged_sentences = convert.load_list(fload_test)

    # Split the dataset for training and testing
    # cutoff = int(.75 * len(tagged_sentences))
    # training_sentences = tagged_sentences[:cutoff]
    # test_sentences = tagged_sentences[cutoff:]

    training_sentences = train_tagged_sentences
    test_sentences = test_tagged_sentences

    # Number of train and test
    print 'Training data : ' + str(len(training_sentences))
    print 'Testing data : ' + str(len(test_sentences))

    X, y = features.transform_to_dataset(training_sentences)

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])

    # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
    clf.fit(X[:10000], y[:10000])

    print 'Training completed'

    X_test, y_test = features.transform_to_dataset(test_sentences)

    print "Accuracy:", clf.score(X_test, y_test)
