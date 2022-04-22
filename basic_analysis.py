# import packages

import pandas as pd
import numpy as np
import nltk

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class basic_analysis:
    def __init__(self, source, url = None, file_name = None, stem = 'stemmer'):
        self.source = source
        self.url = url
        self.file_name = file_name
        self.stem = stem

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
            return df
    def clean_dataset(self, df, col='Text'):
        #Assuming the data is sourced from wikipedia
            #1. Remove numbers inside square brackets
            #2. Remove '\n' and '\t'

        if self.source == 'WEB':
            def sub(string):
                import re
                p1 = re.compile("\[\d*\]")
                p2 = re.compile("(\n)|(\t)")
                return re.sub(p1, '', re.sub(p2, '', string))
            df['Text_cleamed'] = df['Text'].apply(sub)
            return df
        elif self.source == 'FILE':
            def sub(string):
                import re
                p1 = re.compile("\[\d*\]")
                p2 = re.compile("(\n)|(\t)")
                return re.sub(p1, '', re.sub(p2, '', string))
            df['Text_cleamed'] = df[col].apply(sub)
            return df
    def lemma(self, df, col = 'Text'):
        from nltk import NLTKWordTokenizer
        if self.stem == 'stemmer':
            from nltk.stem import PorterStemmer
            TOK = NLTKWordTokenizer()
            stem = PorterStemmer()
            def stemming(string):
                x = TOK.tokenize(string)
                x = ' '.join([stem.stem(i) for i in x])
                return x
            df['Stemmed'] = df[col].apply(stemming)
            return df
        elif self.stem == 'lemmatize':
            from nltk.stem import WordNetLemmatizer
            TOK = NLTKWordTokenizer()
            lemma = WordNetLemmatizer()
            def lemmatize(string):
                x = TOK.tokenize(string)
                x = ' '.join([lemma.lemmatize(i) for i in x])
                return x
            df['Lemmatized'] = df[col].apply(lemmatize)
            return df

