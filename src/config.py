path_to_file = 'C:\\Users\\os_biennn\\Desktop\\bitbucket\\dmprepo\\dmprepo\\dmprepo\\'+\
'source-code\\machine_learning\\ML_Python\\credit_scoring\\data\\data\\'
path_to_submission = path_to_file+'submission\\'

params_lgb={}
params_lgb['objective']='binary'
params_lgb['learning_rate']=0.1
params_lgb['metric']='binary_logloss'