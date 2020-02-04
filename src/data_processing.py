import os
from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cats_processing(df):

    label = 'label'
    cats = [col for col in df.columns if  col!=label and df[col].dtypes =='object']
    for col in cats:
        df[col] = df[col].fillna('_NONE_')
        dict_=df[[label,col]].groupby(col).sum().to_dict()
        #print(dict_)
        dict_=df[[label, col]].groupby(col).sum().to_dict()['label']
        df[col] = df[col].map(dict_) #label by number of value in target
    return df

def fillna_numerics(df, col):

    value_counts = df[col].value_counts(dropna=False)
    if value_counts.iloc[0]/len(df)>.7:
        df[col] = df[col].fillna(value_counts.index[0])
    else:
        df[col] = df[col].fillna(df[col].median())
    return df


def num_processing(df):

    label = 'label'
    nums = [col for col in df.columns if df[col].dtypes!='object' and col!=label]
    for col in nums:
        df = fillna_numerics(df, col)
    return df


def min_max_scale(df):

    from sklearn.preprocessing import MinMaxScaler
    scale = MinMaxScaler()
    df1 = scale.fit(df).transform(df)
    df = pd.DataFrame(df1, index=df.index, columns= df.columns)
    print('To ensure that every feat are scaled: Min = %.2f, Max = %.2f'%(df.min().min(), df.max().max()))
    return df

def upsample(df):

    from sklearn.utils import resample
    df_0=df[df.label==0]
    df_1=df[df.label==1]
    # upsample minority
    if len(df_1) <len(df_0):
        df_1 =resample(df_1,
                        replace=True,  # sample with replacement
                        n_samples=len(df_0),  # match number in majority class
                        random_state=49)  # reproducible results
        return pd.concat([df_0,df_1])
    else:
        df_0=resample(df_0,   replace=True,  # sample with replacement
        n_samples=len(df_1),  # match number in majority class
        random_state=49)  # reproducible results
        return pd.concat([df_0,df_1])

def get_data_and_split_train_test(is_processing=False, train_test = 'train'):

    import gc
    from sklearn.model_selection import train_test_split
    path_to_store_train = path_to_file+'train_processed.csv'
    path_train = path_to_file+'train.csv'
    path_test = path_to_file+'test.csv'
    path_to_store_test = path_to_file+'test_processed.csv'
    if is_processing:
        print('processing data and store it to a file')
        df1 = pd.read_csv(path_train, index_col='id')
        train_size = len(df1)
        df2 = pd.read_csv(path_test, index_col='id')
        df = df1.append(df2)
        del df1, df2
        gc.collect()
        df = num_processing(df)
        df = cats_processing(df)
        df = min_max_scale(df)
        df1 = df.iloc[:train_size]
        df2 = df.iloc[train_size:]
        df1.to_csv(path_to_store_train)
        df2.to_csv(path_to_store_test)
    else:
        if train_test == 'train':
             print('reading train data processed')
             df = pd.read_csv(path_to_store_train, index_col='id')
             print('train shape=',df.shape)
             #print(df.head())
             print('Up sample to have balanced data set:')
             df = upsample(df)
             print('After upsample, df len=%s, percent of label 1=%.2f%%'%(len(df), df.label.sum()/len(df)*100))
             print('split train, test by 30-70')
             train, test, y_train, y_test = train_test_split(df.drop(['label'], axis=1), df.label,\
                                                             test_size=.3, random_state=49)
             return train, test, y_train, y_test
        else:
            print('reading test data processed')
            df = pd.read_csv(path_to_store_test, index_col='id')
            df = df.drop(['label'], axis=1)
            print('test shape=',df.shape)
            #print(df.head())
            return df

if __name__=='__main__':

    #get_data_and_split_train_test(is_processing=False, train_test='train')
    pass
