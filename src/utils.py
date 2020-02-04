import numpy as np
def print_time(seconds):
    hours=int(seconds//(60*60))
    minutes=int((seconds-hours*60*60)//60)
    seconds=int(seconds-minutes*60-hours*60*60)
    if hours>0:
        return "%s hours %s mins %s secs"%(hours,minutes,seconds)
    elif minutes>0:
        return "%s mins, %s secs"%(minutes,seconds)
    else:
        return "%s secs"%seconds


def show_accuracy(y_true,y_pred):
    from sklearn.metrics import balanced_accuracy_score,accuracy_score,classification_report
    x=balanced_accuracy_score(y_true,y_pred)*100
    y=accuracy_score(y_true,y_pred)*100
    mess1='Balanced accuracy score = %.3f%%. Accuracy score = %.3f%%'%(x,y)
    return mess1


def memory_usage():
    import sys
    # These are the usual ipython objects, including this one you are creating
    x=10**9
    ipython_vars=['In','Out','exit','quit','get_ipython','ipython_vars']
    # Get a sorted list of the objects and their sizes
    return sorted([(x,sys.getsizeof(globals().get(x))) for x in dir() if\
                   not x.startswith('_') and x not in sys.modules and x not in ipython_vars],\
                  key=lambda x:x[1],reverse=True)


def extract_package_from_notebook():
    import os
    this_path=os.getcwd()
    if 'notebook' in this_path:
        this_path=this_path.split('\\')[:-1]
        this_path='\\'.join(this_path)
    this_path
    os.chdir(this_path)
    print('current path now is:')
    print(os.getcwd())


def plot_DL_acc(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.title('Model accuracy and Loss')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['acc','loss'],loc='upper left')
    plt.show()


def pick_color(n=1):
    import random
    colors=["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]
    random.shuffle(colors)
    if n==1:
        return colors[0]
    else:
        colors_ = []
        for i in range(n):
            colors_.append(random.choice(colors))
        return colors_


def evaluate_model(model,train,test,y_train,y_test):

    from sklearn.metrics import accuracy_score,balanced_accuracy_score
    pred_test=model.predict(test)
    pred_train=model.predict(train)
    if len(np.unique(pred_test))>2 or len(np.unique(pred_train))>2:
        pred_test=pred_test>.5
        pred_train=pred_train>.5
    is_balanced=(sum(y_train)+sum(y_test))/(len(y_train)+len(y_test))
    if np.abs(is_balanced-.5)<.02:
        is_balanced=True
    else:
        is_balanced=False
    print('*'*80)
    if not is_balanced:
        print('labels are not balanced:')
        acc_test=balanced_accuracy_score(y_test,pred_test)*100
        acc_train=balanced_accuracy_score(y_train,pred_train)*100
        print('balanced acc on test = %.2f %%'%acc_test,end=' | ')
        print('balanced acc on train  = %.2f%%'%acc_train)
    else:
        print('labels are balanced:')
    acc_test=accuracy_score(y_test,pred_test)*100
    acc_train=accuracy_score(y_train,pred_train)*100
    print('         acc on test = %.2f %%'%acc_test,end=' | ')
    print('         acc on train  = %.2f%%'%acc_train)
    print('*'*80)