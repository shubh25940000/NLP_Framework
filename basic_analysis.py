import pandas as pd
import spacy
model = spacy.load('en_core_web_sm')

import basic_cleanups
from sklearn.feature_extraction.text import CountVectorizer
import logging
import sys
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('my_log_info.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

class basic_analysis(basic_cleanups.basic_cleanups):
    def __init__(self, source, url = None, file_name = None, stem = 'stemmer'):
        basic_cleanups.basic_cleanups.__init__(self, source, url=None, file_name=None, stem='stemmer')
        self.url = url
        self.source = source
        self.file_name = file_name
        self.stem = stem
        self.df = basic_cleanups.basic_cleanups.lemma(self)

    def remove_stop(self, col = 'Lemmatized'):
        # First we need to remove stop words
        df = self.df
        from nltk.corpus import stopwords
        from nltk.tokenize import NLTKWordTokenizer, RegexpTokenizer
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
        return df

    def mostcommon(self, col = 'Lemmatized'):
        df = self.remove_stop()
        from nltk.tokenize import RegexpTokenizer
        from nltk import FreqDist
        #Tokenize except of punctuations
        processed = model(' '.join(df[col].to_list()))
        tok = RegexpTokenizer(pattern = r'\w+')
        k = tok.tokenize(' '.join(df['Text_no_stop'].to_list()))
        allWordDist = FreqDist(w.lower() for w in k if w not in ['s'])
        mostCommon = allWordDist.most_common(5)
        mostCommon = pd.DataFrame(mostCommon, columns = ['word','number'])
        logger.info("Below are the top 5 words:")
        logger.info(mostCommon)
        k = [ent.text for ent in processed.ents]
        allWordDist = FreqDist(w.lower() for w in k if w not in ['s'])
        mostCommonents = allWordDist.most_common(5)
        mostCommonents = pd.DataFrame(mostCommonents, columns=['word', 'number'])
        logger.info("Below are the top 5 entities:")
        logger.info(mostCommonents)
        g = [(ent.text, ent.label_) for ent in processed.ents]
        print(set(g))
        return None




