import pandas as pd

import basic_cleanups
from sklearn.feature_extraction.text import CountVectorizer


import spacy


class basic_analysis(basic_cleanups.basic_cleanups):
    def __init__(self, source, url = None, file_name = None, stem = 'stemmer'):
        self.url = url
        self.file_name = file_name
        self.stem = stem
        basic_cleanups.basic_cleanups.__init__(self, source, url = None, file_name = None, stem = 'stemmer')

    def basic_info(self, df, col = 'Lemmatized'):
        # First we need to remove stop words
        from nltk.corpus import stopwords
        from nltk.tokenize import NLTKWordTokenizer, RegexpTokenizer
        from nltk import FreqDist
        def remove_stop(string):
            tok = NLTKWordTokenizer()
            s = stopwords.words('english')
            string = tok.tokenize(string)
            k = [k for k in string if k not in s]
            return ' '.join(k)
        vectorizer = CountVectorizer()
        df['Text_no_stop'] = df[col].apply(remove_stop)
        #x = vectorizer.fit_transform(df['Text_no_stop'].to_list())
        #bag_of_words = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

        #Tokenize except of punctuations
        tok = RegexpTokenizer(pattern = r'\w+')
        k = tok.tokenize(' '.join(df['Text_no_stop'].to_list()))
        allWordDist = FreqDist(w.lower() for w in k)
        mostCommon = allWordDist.most_common(10)
        return mostCommon




