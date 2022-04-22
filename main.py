#import packages

import basic_cleanups
import basic_analysis



if __name__ == '__main__':
    x = basic_cleanups.basic_cleanups(source ='WEB', url ='https://en.wikipedia.org/wiki/Barack_Obama', stem = 'lemmatize')
    y = basic_analysis.basic_analysis(source ='WEB', url ='https://en.wikipedia.org/wiki/Barack_Obama', stem = 'lemmatize')
    df = x.source_data()
    df = x.clean_dataset(df)
    df = x.lemma(df)
    vec = y.basic_info(df)
    df.to_csv('Output_check.csv', index = False)
    print(vec)
