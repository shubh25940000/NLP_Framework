#import packages

import basic_cleanups
import basic_analysis
import pandas as pd



if __name__ == '__main__':
    url = 'https://en.wikipedia.org/wiki/Kolkata'
    x = basic_cleanups.basic_cleanups(source ='WEB', url = url, stem = 'lemmatize')
    y = basic_analysis.basic_analysis(source ='WEB', url = url, stem = 'lemmatize')
    y.mostcommon()

