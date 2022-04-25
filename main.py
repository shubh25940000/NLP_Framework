#import packages

import basic_cleanups
import basic_analysis
import pandas as pd



if __name__ == '__main__':
    url = 'https://en.wikipedia.org/wiki/Avengers:_Infinity_War'
    x = basic_cleanups.basic_cleanups(url = url, file_name='train.csv',test_file_name='test.csv', stem = 'lemmatize',source = 'LOCAL',col = 'text')
    y = basic_analysis.basic_analysis(source ='LOCAL', url = url, file_name='train.csv',test_file_name='test.csv', stem = 'lemmatize', col = 'text', label_col='target')
    sub_out = y.binary_classification_logistic()
    sub_out[['id','target']].to_csv('Submission_output.csv', index = False)

