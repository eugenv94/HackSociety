import distance as dist
import Levenshtein as lev
import numpy as np
import pylab as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn import svm, linear_model

def jaccard(s1, s2):
    """
    Function that returns the Jaccard distance between two strings, s1 and s2

    :param s1: the 1st string
    :param s2: the 2nd string
    :return: the jaccard distance between the two strings
    """
    n = len(s1.intersection(s2))
    m = float(len(s1) + len(s2) - n)
    return n/m

def get_n_grams(a, grams_len):
    return [a[i:i+grams_len] for i in range(len(a) - 1)]

def dice_coeff(s1, s2, grams_len = 2):

    """
    Function that returns the dice coefficient for 2 given strings s1 and s2

    :param s1:      the 1st string
    :param s2:      the 2nd string
    :param grams_len:   the length of the n-grams
    :return:    the dice coefficient of the two strings
    """

    if not len(s1) or not len(s2):
        return 0.0

    # A simple case, where the two strings are equal
    if s1 == s2 :
        return 1.0



    s1_ngram = get_n_grams(s1)
    s2_ngram = get_n_grams(s2)

    s1_ngram.sort()
    s2_ngram.sort()

    lng_s1 = len(s1_ngram, grams_len)
    lng_s2 = len(s2_ngram, grams_len)

    matches = x = y = 0

    while (x<lng_s1 and y<lng_s2) :
        if s1_ngram[x] == s2_ngram[y] :
            #YAY! We have a match
            matches += grams_len
            x += 1
            y += 1
        elif s1_ngram[x] < s2_ngram[y] :
            x += 1
        else :
            y += 1

    coeff = float(matches) / float(lng_s1 + lng_s2)
    return coeff

def get_TDIDF(source):
    """
        Function that returns a couple of statistics based on the strings we have to work on
    :param source:  The source file where to get the data from
    :return:
    """

    def cnt_underscores(s):
        return s.count('_')

    train_set = list()

    with open(source) as input_file:
        for i, line in enumerate(input_file):
            data = line.split(";")
            if (len(data) == 2):
                if (cnt_underscores(data[1].rstrip()) < 2):
                    train_set.append(data[1].lower().rstrip())

    train_set = sorted(list(set(train_set)))
    dict_train = dict()

    for i, product in enumerate(train_set):
        dict_train[product] = i

    # Getting weights matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

    # Getting tri-grams
    train_setTrigrams = []
    for product in train_set:
        tmp = [e for e in get_n_grams(product, 3) if len(e) == 3]
        train_setTrigrams.append(' '.join(tmp))

    train_setTrigrams = sorted(list(set(train_setTrigrams)))
    dict_train_setTrigrams = dict()
    for i, product in enumerate(train_setTrigrams):
        dict_train_setTrigrams[product] = i

    tfidf_vectorizerTrigrams = TfidfVectorizer()
    tfidf_matrix_trainTrigrams = tfidf_vectorizerTrigrams.fit_transform(train_setTrigrams)

    return [tfidf_matrix_train, dict_train, tfidf_matrix_trainTrigrams, dict_train_setTrigrams, 3]

def cos_n_grams(s1, s2, train_ngrams, tfidf_matrix_ngrams, grams_len = 3):
    """
            Returns the cosine n-grams for 2 given strings, s1 and s2

    :param s1:      the 1st string
    :param s2:      the 2nd string
    :param train_ngrams:     dictionary of n-grams to find indexes quickly
    :param tfidf_matrix_ngrams:     weights of n-grams
    :param grams_len:   length of n-grams (3 by default)
    :return:    cosine similarity (i.e. cosine between vectors)
    """
    #Pre-processing the 2 strings
    s1 = s1.lower().rstrip()
    s2 = s2.lower().rstrip()

    def get_useful(a):
        return[x for x in [a[i:i+grams_len] for i in range(len(a) - 1)] if len(x) == grams_len]

    a = ' '.join(get_useful(s1))
    b = ' '.join(get_useful(s2))

    index_s1 = train_ngrams[a]
    index_s2 = train_ngrams[b]

    score = cosine_similarity(tfidf_matrix_ngrams[index_s1:index_s1+1], tfidf_matrix_ngrams[index_s2:index_s2+1])
    return score

def cos_words(s1, s2, train_set, tfidf_matrix_ngrams):
    """

    :param s1:      The 1st string
    :param s2:      The 2nd string
    :param train_set:       The training set, as a dictionary to access the data quickly
    :param tfidf_matrix_ngrams:     weights of words
    :return:        cosine similarity (i.e. cosine between words)
    """
    index_s1 = train_set[s1.lower().rstrip()]
    index_s2 = train_set[s2.lower().rstrip()]

    score = cosine_similarity(tfidf_matrix_ngrams[index_s1:index_s1 + 1], tfidf_matrix_ngrams[index_s2:index_s2 + 1])
    return score

def lev_dist(s1, s2):
    return 1. - (lev.distance(s1, s2)*2/(len(s1) + len(s2)))

def get_dist_nlev(s1, s2, m):
    return 1. - dist.nlevenshtein(s1, s2, method=m)

def get_params(s1, s2, tfidf_matrix_train, dictTrain, tfidf_matrix_trainBigrams, dictTrainBigrams, lenGram, delete = []):
    """
            Returns the list of parameters required for the machine learning to work
    :param s1:          the 1st string
    :param s2:          the 2nd string
    :param tfidf_matrix_train:      the tfidf for the raw data
    :param dictTrain:               the dictionary with the raw data
    :param tfidf_matrix_trainBigram:    the tfidf for the trigrams
    :param dictTrainBigrams:            the dictionary with the trigrams
    :param lenGram:                     the length of an n-gram
    :param delete:                      the parameters we want to ignore
    :return:                            a list with the results after we applied a couple of specific functions
                                                            on our data
    """

    temp = [
        lev_dist(s1, s2),
        lev.jaro(s1, s2),
        lev.jaro_winkler(s1, s2),
        dist.sorensen(s1, s2),
        jaccard(set(s1), set(s2)),
        get_dist_nlev(s1, s2, 2),
        get_dist_nlev(s1, s2, 3),
        dice_coeff(s1, s2, grams_len=2),
        dice_coeff(s1, s2, grams_len=3),
        dice_coeff(s1, s2, grams_len=4),
        cos_words(s1, s2, dictTrain, tfidf_matrix_train),
        cos_n_grams(s1, s2, dictTrainBigrams, tfidf_matrix_trainBigrams, lenGram)
    ]

    if len(delete) != 0:
        for x in delete:
            temp[x] = 0.

    return temp


def train(tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,delete = []):
    allTrainX = []
    allTrainY = []

    with open("./data/train.csv") as input_file:
         for i, line in enumerate(input_file):
             if i != 1 and line.count('_') < 3:
                data = line.split(";")
                if (len(data) == 3):
                    s1 = data[1].lower()
                    s2 = data[3].lower()

                    temp = get_params(s1, s2,
                                      tfidf_matrix_train, dictTrain,
                                      tfidf_matrix_trainBigrams, dictTrainBigrams,
                                      lenGram, delete)

                    allTrainX.append(temp)
                    allTrainY.append(int(data[2]))

    X = np.array(allTrainX, dtype=float)
    Y = np.array(allTrainY, dtype=float)

    clf = linear_model.LogisticRegression(C=1.,
                                          dual=False,
                                          penalty='l1')

    clf.fit(X, Y)

    return clf


def solve(tfidf_matrix_train, dictTrain, tfidf_matrix_trainBigrams, dictTrainBigrams, lenGram, in_file, out_file):

    delete = []

    clf = train(tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram, delete)

    found_duplicate = []
    product_names = []
    product_ids = []
    cnt_prods = 0

    #Getting the data from the raw file into the memory
    with open(in_file) as f:
        for i, line in enumerate(f):
            a = line.split(';')
            if len(a) == 2:
                s = a[1].rstrip().lower()
                if s.count('_') < 3:
                    product_ids[cnt_prods] = int(a[0])
                    product_names[cnt_prods] = s
                    found_duplicate[cnt_prods] = False
                    cnt_prods += 1

    with open(out_file, "w") as f:
        for i in range(cnt_prods):
            if not found_duplicate[i]:
                printed_i = False;
                for j in range(i+1, cnt_prods):
                    if not found_duplicate[j]:

                        temp = get_params(product_names[i], product_names[j],
                                          tfidf_matrix_train, dictTrain,
                                          tfidf_matrix_trainBigrams, dictTrainBigrams,
                                          lenGram, delete)

                        is_a_match = clf.predict(temp)

                        if is_a_match == 1:
                            found_duplicate[i] = True
                            found_duplicate[j] = True
                            if not printed_i :
                                printed_i = True
                                f.write("%s;%s", str(product_ids[i]), product_names[i])

                            f.write(";%s;%s", str(product_ids[j], product_names[j]))
                if printed_i:
                    f.write("\n")


def main():

    source = "./data/raw.csv"
    dest = "./data/solution.csv"

    tfidf_matrix_train, dictTrain, tfidf_matrix_trainBigrams, dictTrainBigrams, lenGram = get_TDIDF(source)
    solve(tfidf_matrix_train, dictTrain, tfidf_matrix_trainBigrams, dictTrainBigrams, lenGram, source, dest)

if __name__ == "__main__" :
    main()