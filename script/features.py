import sys
import nltk
# nltk.download()

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def pos_tag(sentence, clf):
    tagged_sentence = []
    # Untuk tree only
    tags = clf.predict([extract(sentence, index) for index in range(len(sentence))])

    # list_to_predict = []
    # list_to_predict.append([extract(sentence, index) for index in range(len(sentence))])
    # print(list_to_predict)
    # tags = clf.predict(list_to_predict)
    return zip(sentence, tags)

def extract(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'prefix-4': sentence[index][:4],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'suffix-4': sentence[index][-4:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(extract(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y
