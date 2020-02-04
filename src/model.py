from config import *
from utils import *
from data_processing import *
########################################
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import _pickle as cPickle
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


def write_or_read_lgb_model(model=None):
    path=get_model_path()
    if model is not None:
        with open(path,'wb') as fid:
            cPickle.dump(model,fid)
    else:
        print('try to load and predict from saved model:')
        with open(path,'rb') as fid:
            model_loaded=cPickle.load(fid)
            return model_loaded


def lgb_model(num_boost_round=200):

    train,test,y_train,y_test = get_data_and_split_train_test()
    print('shape of train set=',train.shape)
    params_lgb['metric']=['binary_logloss','auc']
    params_lgb['max_depth']=15
    params_lgb['feature_fraction'] = .8
    import lightgbm as lgb
    s=time()
    print('shape of train set=',train.shape)
    evals_result={}
    lgb_train=lgb.Dataset(train,label=y_train)
    lgb_test=lgb.Dataset(test,label=y_test)
    model=lgb.train(params=params_lgb,train_set=lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_train,lgb_test],
                    evals_result=evals_result,
                    verbose_eval=num_boost_round//3)
    evaluate_model(model, train, test, y_train, y_test)
    predict_fill_sample(model, file='LGB_model')
    lgb.plot_metric(evals_result,'auc')
    plt.show()
    lgb.plot_importance(model,max_num_features=40)
    plt.show()
    print('-'*80)


def deep_learning_model():

    from keras.callbacks import ModelCheckpoint
    from keras.layers import Dropout,Dense
    from keras.models import Sequential,load_model
    from numpy.testing import assert_allclose
    import keras
    s=time()
    model_path=get_model_path()
    checkpoint=ModelCheckpoint(model_path,monitor='loss',verbose=0,save_best_only=True,mode='min')
    callbacks_list=[checkpoint]
    train, test, y_train, y_test = get_data_and_split_train_test()
    if is_retrain:
        classifier=load_model(model_path)
        print('*'*80)
        print('*'*20+' load model from history and train it again '+'*'*20)
        print('*'*80)
    else:
        print('*'*80)
        print('*'*20+' create new model '+'*'*20)
        print('*'*80)
        classifier=Sequential()
        input_dim=train.shape[1]
        classifier.add(Dense(output_dim=32,kernel_initializer='glorot_normal',activation='tanh',
                             input_dim=input_dim))
        classifier.add(Dense(output_dim=1,activation='sigmoid'))
        classifier.compile(keras.optimizers.Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

    history=classifier.fit(train,y_train,batch_size=1000,epochs=epochs,callbacks=callbacks_list,verbose=0)
    evaluate_model(classifier, train, test, y_train, y_test)
    plot_DL_acc(history)

def RF_model():

    train,test,y_train,y_test = get_data_and_split_train_test()
    print('shape of train set=',train.shape)
    model=RandomForestClassifier(max_depth=30,min_impurity_split=10**(-4),min_samples_split=50,max_features=.9)
    model.fit(train,y_train)
    evaluate_model(model,train,test,y_train,y_test)
    coef=model.feature_importances_
    coef=coef.reshape(-1,)
    coef=np.abs(coef)
    sort=np.argsort(coef)
    sort=sort[::-1]
    coef=coef[sort]
    cols_=train.columns[sort]
    plt.figure(figsize=(15,5))
    bar=plt.bar(range(len(coef)),coef)
    plt.xticks(range(len(coef)),cols_,rotation=90)
    for rect in bar:
        rect.set_color(pick_color())
        height=rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.0,height,round(height,2),ha='center',va='bottom',rotation=90)
    plt.title('Importance Feature Of Random Forest')
    plt.show()


def LR():

    from sklearn.linear_model import LogisticRegression
    train,test,y_train,y_test=get_data_and_split_train_test()
    print('shape of train set=',train.shape)
    model=LogisticRegression().fit(train,y_train)
    evaluate_model(model, train, test, y_train, y_test)
    predict_fill_sample(model,file='LR')


def predict_fill_sample(model, file='lgb_model_'):

    from datetime import datetime
    now=datetime.now()
    current_time=now.strftime("%d %m %y %H %M") # will add this time to the name of file distinct them
    path = path_to_submission +file + current_time+'.csv'
    print('path to save submission=', path)
    test = get_data_and_split_train_test(train_test = 'test')
    sample = pd.read_csv(path_to_file+'sample_submission.csv', index_col='id')
    pred = model.predict(test)
    if len(np.unique(pred))>2:
        pred = pred>.5
    sample['label'] = pred
    sample['label'] = sample.label.astype('int8')
    print('save result to file')
    sample.to_csv(path,  index_label='id')
    print('header of sample:')
    print(sample.head())

def augment(x,y,t=2):
    xs,xn=[],[]
    for i in range(t):
        mask=y>0
        x1=x[mask].copy()
        ids=np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c]=x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask=y==0
        x1=x[mask].copy()
        ids=np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c]=x1[ids][:,c]
        xn.append(x1)

    xs=np.vstack(xs)
    xn=np.vstack(xn)
    ys=np.ones(xs.shape[0])
    yn=np.zeros(xn.shape[0])
    x=np.vstack([x,xs,xn])
    y=np.concatenate([y,ys,yn])
    return x,y


def lgb_with_kfold():
    num_folds=5
    df,target=None,None
    folds=KFold(n_splits=num_folds,random_state=2319)
    oof=np.zeros(len(df))
    getVal=np.zeros(len(df))
    predictions=np.zeros(len(target))
    feature_importance_df=pd.DataFrame()
    print('memory use to store this df = %.0f MB'%(df.memory_usage().sum()/1024**2))
    print('Light GBM Model')
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(df.values,target.values)):
        X_train,y_train=df.iloc[trn_idx],target.iloc[trn_idx]
        X_valid,y_valid=df.iloc[val_idx],target.iloc[val_idx]

        X_tr,y_tr=augment(X_train.values,y_train.values)
        X_tr=pd.DataFrame(X_tr)

        print("Fold idx:{}".format(fold_+1))
        trn_data=lgb.Dataset(X_tr,label=y_tr)
        val_data=lgb.Dataset(X_valid,label=y_valid)

        clf=lgb.train(param,trn_data,500,valid_sets=[trn_data,val_data],verbose_eval=200,
                      early_stopping_rounds=200)
        oof[val_idx]=clf.predict(df.iloc[val_idx],num_iteration=clf.best_iteration)
        getVal[val_idx]+=clf.predict(df.iloc[val_idx],
                                     num_iteration=clf.best_iteration)/folds.n_splits
        path_to_file=model_kfold+str(fold_)+'on_110feat_with0001thresholdData'+'.pkl'
        print('save model to file at:')
        print(path_to_file)
        with open(path_to_file,'wb') as fid:
            cPickle.dump(clf,fid)

        fold_importance_df=pd.DataFrame()
        fold_importance_df["feature"]=features
        fold_importance_df["importance"]=clf.feature_importance()
        fold_importance_df["fold"]=fold_+1
        feature_importance_df=pd.concat([feature_importance_df,fold_importance_df],axis=0)

    print("CV score: {:<8.5f}".format(accuracy_score(target,oof>.5)))


if __name__=='__main__':
    lgb_model()
    #LR()

