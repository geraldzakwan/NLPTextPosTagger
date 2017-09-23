import sys
import convert
import features
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

chunk = 20000

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

    # Number of X and y
    print 'X : ' + str(len(X))
    print 'y : ' + str(len(y))

    # clf = Pipeline([
    #     ('vectorizer', DictVectorizer(sparse=False)),
    #     ('classifier', DecisionTreeClassifier(criterion='entropy'))
    # ])

    # # Iterate per chunk sample, so it fits in memory
    # # Salah, ni ngk bisa soalnya mesti support partial fit
    # # Kalo ngk support sama aja training ulang
    # total_word = len(X)
    # total_processed = 0
    # first_word = 0
    # last_word = chunk
    # while(total_processed < total_word):
    #     clf.fit(X[first_word:last_word], y[first_word:last_word])
    #     total_processed = total_processed + chunk
    #     if(last_word + chunk < total_word - 1):
    #         first_word = first_word + chunk
    #         last_word = last_word + chunk
    #     else:
    #         first_word = first_word + chunk
    #         last_word = total_word

    # # Bukti : yg atas bakal sama aja kek gini
    # clf.fit(X[80000:len(X)], y[80000:len(y)])

    # Use partial fit instead
    # partial_fit with partial_fit(X, y[, classes, sample_weight])
    # clf_2 = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1,
    #                       #shuffle=True, n_iter=10,
    #                       verbose=1)

    clf_2 = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, verbose=1))
    ])

    total_word = len(X)
    total_processed = 0
    first_word = 0
    last_word = chunk
    while(total_processed < total_word):
        print "Running Partial Fit from : " + str(first_word) + " to " + str(last_word)
        clf_2.partial_fit(X[first_word:last_word], y[first_word:last_word])
        # clf_2.partial_fit(X[:99, :], y[:99, 0], classes=[0, 1])
        total_processed = total_processed + chunk
        if(last_word + chunk < total_word - 1):
            first_word = first_word + chunk
            last_word = last_word + chunk
        else:
            first_word = first_word + chunk
            last_word = total_word

    print 'Training completed'

    X_test, y_test = features.transform_to_dataset(test_sentences)

    print "Accuracy:", clf.score(X_test, y_test)
