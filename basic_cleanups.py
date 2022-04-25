# import packages

import pandas as pd
import numpy as np
import nltk

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class basic_cleanups:
    def __init__(self, url = None, file_name = None, test_file_name = None, stem = 'stemmer',source = 'WEB', col = 'Text'):
        self.source = source
        self.col = col
        self.url = url
        self.file_name = file_name
        self.stem = stem
        self.test_file_name = test_file_name

    def source_data(self):
        if self.source == 'WEB':
            from bs4 import BeautifulSoup
            import requests
            page = requests.get(self.url)
            soup = BeautifulSoup(page.content, 'html.parser')
            k = dict()
            result = soup.find_all('p')
            for i in range(1, len(result)):
                k[i] = result[i].get_text()
            df = pd.DataFrame(k.items())
            df.rename(columns = {0: 'Id', 1: 'Text'}, inplace = True)
            return df
        elif self.source == 'LOCAL':
            df = pd.read_csv(self.file_name)
            df_test = pd.read_csv(self.test_file_name)

            return df, df_test
    def clean_dataset(self):
        #Assuming the data is sourced from wikipedia
            #1. Remove numbers inside square brackets
            #2. Remove '\n' and '\t'
        df, df_test = self.source_data()
        col = self.col
        if self.source == 'WEB':
            def sub(string):
                import re
                p1 = re.compile("\[\d*\]")
                p2 = re.compile("(\n)|(\t)")
                return re.sub(p1, '', re.sub(p2, '', string))
            df['Text_cleaned'] = df['Text'].apply(sub)
            return df
        elif self.source == 'LOCAL':
            def sub(string):
                import re
                p1 = re.compile("\[\d*\]")
                p2 = re.compile("(\n)|(\t)")
                p3 = re.compile('#')
                p4 = re.compile('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
                return re.sub(p1, '', re.sub(p2, '', re.sub(p3, '', re.sub(p4, '', string))))
            df['Text_cleaned'] = df[col].apply(sub)
            df_test['Text_cleaned'] = df_test[col].apply(sub)

            return df, df_test
    def lemma(self, col = 'Text_cleaned'):
        df, df_test = self.clean_dataset()

        from nltk import NLTKWordTokenizer
        if self.stem == 'stemmer':
            from nltk.stem import PorterStemmer
            TOK = NLTKWordTokenizer()
            stem = PorterStemmer()
            def stemming(string):
                x = TOK.tokenize(string)
                x = ' '.join([stem.stem(i).lower() for i in x])
                return x
            df['Stemmed'] = df[col].apply(stemming)
            df_test['Stemmed'] = df_test[col].apply(stemming)
            return df, df_test
        elif self.stem == 'lemmatize':
            from nltk.stem import WordNetLemmatizer
            TOK = NLTKWordTokenizer()
            lemma = WordNetLemmatizer()
            def lemmatize(string):
                x = TOK.tokenize(string)
                t = nltk.pos_tag(x)
                j = []
                # Using POS Tags for precise lemmatization
                for word, tag in t:
                    if tag.startswith("NN"):
                        j.append(lemma.lemmatize(word, pos='n').lower())
                    elif tag.startswith('VB'):
                        j.append(lemma.lemmatize(word, pos='v').lower())
                    elif tag.startswith('JJ'):
                        j.append(lemma.lemmatize(word, pos='a').lower())
                    else:
                        j.append(word.lower())
                x = ' '.join(j)
                return x
            df['Lemmatized'] = df[col].apply(lemmatize)
            df_test['Lemmatized'] = df_test[col].apply(lemmatize)
            return df, df_test
