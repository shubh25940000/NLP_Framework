#import packages

import basic_analysis


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = basic_analysis.basic_analysis(source ='WEB', url ='https://en.wikipedia.org/wiki/Barack_Obama')
    df = x.source_data()
    df = x.clean_dataset(df)
    df = x.lemma(df)
    print(df.head())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
