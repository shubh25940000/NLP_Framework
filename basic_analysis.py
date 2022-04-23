import pandas as pd
import spacy
import gensim
from gensim.models import Word2Vec, KeyedVectors
import utils
model = spacy.load('en_core_web_sm')
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Input, GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import adam_v2

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
    def __init__(self, source = 'WEB', url = None, file_name = None, stem = 'stemmer', col = 'Text', label_col = 'label'):
        basic_cleanups.basic_cleanups.__init__(self, url = None, file_name = None, stem = 'stemmer',source = 'WEB', col = 'Text')
        self.url = url
        self.source = source
        self.file_name = file_name
        self.stem = stem
        self.col = col
        self.label_col = label_col
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
        k = [ent.text for ent in processed.ents if ent.label_ in ['NORP','ORG','GPE','PERSON','LANGUAGE','EVENT','LOC','MONEY','WORK_OF_ART']]
        allWordDist = FreqDist(w.lower() for w in k if w not in ['s'])
        mostCommonents = allWordDist.most_common(10)
        mostCommonents = pd.DataFrame(mostCommonents, columns=['word', 'number'])
        logger.info("Below are the top 5 entities:")
        logger.info(mostCommonents)
        g = [(ent.text, ent.label_) for ent in processed.ents if ent.label_ in ['NORP','ORG','GPE','PERSON','LANGUAGE','EVENT','LOC','MONEY','WORK_OF_ART']]
        print(set(g))
        return df

    def create_paddings(self, col = 'Text_cleaned'):
        df = self.mostcommon()
        label_col = self.label_col
        X_train, X_test, y_train, y_test = train_test_split(df[col], df[label_col], stratify=df[label_col], test_size= 0.25)
        tok = Tokenizer()
        tok.fit_on_texts(X_train.to_list())
        train_sequences = tok.texts_to_sequences(X_train.to_list())
        test_sequences = tok.texts_to_sequences(X_test.to_list())
        max_sequence_length = max([max(map(len, train_sequences)), max(map(len, test_sequences))])
        train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
        test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)
        return train_data, test_data, y_train, y_test, tok, max_sequence_length


    def binary_classification(self):
        glove_wiki = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False, no_header=True)
        train_data, test_data, y_train, y_test, tok, max_sequence_length = self.create_paddings()

        ##Creating the neural network
        embedding_layer = utils.make_embedding_layer(glove_wiki, tok, max_sequence_length)
        model = Sequential(
            [Input(shape=(max_sequence_length,), dtype='int32'),
             embedding_layer,
             GlobalAveragePooling1D(),
             Dense(128, activation='relu'),
             Dense(64, activation='relu'),
             Dense(1, activation='sigmoid')
             ])
        model.compile(loss='binary_crossentropy', optimizer= adam_v2, metrics=['accuracy'])
        model.fit(train_data, y_train, validation_data=(test_data, y_test), batch_size=32, epochs=20)
        return model





