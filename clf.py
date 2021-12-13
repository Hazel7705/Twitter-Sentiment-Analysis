#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 13:47:46 2021

@author: nihz415
"""

dta=pd.read_csv('nctsticker.csv')
dta
dta.columns
#step TWO:CLEAN DATA
##test and drop duplicates
dta.duplicated()
dta=dta.drop_duplicates()
dta.duplicated(['Text'])
dta=dta.drop_duplicates(['Text'])
##clean
def clean_text(text):
    text=re.sub(r'@[A-Za-z0-9]+',' ' ,text)
    text = re.sub(r'https?:\/\/.*\/\w*',' ',text)
    text = re.sub(r'[^a-zA-Z#]',' ',text)
    text = re.sub(r'#',' ',text)
    text = re.sub(r'RT[\s]+',' ',text)
    return text

dta['tidy_tweet'] = dta['Text'].astype('U').apply(clean_text)  
dta['tidy_tweet']=dta['tidy_tweet'].str.lower()   
dta['tidy_tweet'] = dta['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("n't"," not")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("I'm","I am")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("'ll"," will")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("It's","It is")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("it's","It is")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("that's","that is")
##drop stopwords
stop = stopwords.words('english')
additional=['music core','music video','music bank','music','bank','core','award','nct','video','show','dream','127','wayv','nct','magic','carpet','ride','thwin']
stop+=additional
dta['tidy_tweet'] = dta['tidy_tweet'].str.split()
dta['tidy_tweet']=dta['tidy_tweet'].apply(lambda x:' '.join([item for item in x if item not in stop]))

##get 1000 for ML
dta_test=dta.sample(n=1000,replace=False,  random_state=None, axis=0)
outputpath='/Users/nihz415/Desktop/final project/datacanbeused/nctSTICKERtest.csv'
dta_test.to_csv(outputpath,index=True,header=True)
##import again
dta_test=pd.read_csv('nctSTICKERtest.csv')
dta_test.columns

dta_test['ML_group']  = np.random.randint(10,size=dta.shape[0])
dta_test['ML_group']  = (dta_test['ML_group']<=7)*0 + (dta_test['ML_group']==8)*1 +(dta_test['ML_group']==9)*2
corpus= dta_test['tidy_tweet'].to_list()   
vectorizer_count     = CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X                    = vectorizer_count.fit_transform(dta_test['tidy_tweet'].values.astype('U'))
features_frequency   = pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
#TVT
X_train = X[np.where(dta_test['ML_group']==0)[0],:]
X_valid = X[np.where(dta_test['ML_group']==1)[0],:]
X_test  = X[np.where(dta_test['ML_group']==2)[0],:]
y_train = dta_test.loc[dta_test['ML_group']==0,['score']]['score'].to_numpy()
y_valid = dta_test.loc[dta_test['ML_group']==1,['score']]['score'].to_numpy()
y_test  = dta_test.loc[dta_test['ML_group']==2,['score']]['score'].to_numpy()

#CLASSIFICATION
def logistic_reg_classifier_mult_labels(X_train,y_train,X_valid,y_valid,X_test,y_test):
    
    ' . '
    categories         = pd.DataFrame(np.sort(np.unique(y_train))).reset_index()
    categories.columns = ['index','label']
    

    ' . '    
    ccp_train_list = []
    ccp_valid_list = []
    ccp_test_list  = []
    for cat in categories['label'].to_list():
        y_train_c = 1*(y_train==cat)
        clf       = linear_model.LogisticRegression(tol          = 0.0001,
                                                    max_iter     = 10000,
                                                    random_state = None).fit(X_train, y_train_c)
        ccp_train_list.append(  clf.predict_proba(X_train)[:,1])
        ccp_valid_list.append(  clf.predict_proba(X_valid)[:,1])
        ccp_test_list.append(   clf.predict_proba(X_test)[:,1])
    
    ' . Topic probability matrix'
    ccp_train = pd.DataFrame(ccp_train_list).transpose()
    ccp_valid = pd.DataFrame(ccp_valid_list).transpose()
    ccp_test  = pd.DataFrame(ccp_test_list).transpose()
    
    'reset column index'
    ccp_train.columns = categories['label'].to_list()
    ccp_valid.columns = categories['label'].to_list()
    ccp_test.columns = categories['label'].to_list()
    
    'Choosing your predictive category for the y '
    ccp_train['label_hat'] =  ccp_train.idxmax(axis=1)
    ccp_valid['label_hat'] =  ccp_valid.idxmax(axis=1)
    ccp_test['label_hat']  =  ccp_test.idxmax(axis=1)    
    
    'caculate'
    confusionmatrix=confusion_matrix(y_test,ccp_test['label_hat'] )
    correct =  np.trace(confusionmatrix)
    total = confusionmatrix.sum()
    percent_accuracy = correct/total
        
    return(confusionmatrix,percent_accuracy)

logistic_reg_classifier_mult_labels(X_train,y_train,X_valid,y_valid,X_test,y_test)



#find best k
conf_matrices = []
def find_best_k(X_train,X_valid,y_train,y_valid):
    
    k            = 1;
    results_list = [];
    max_k_nn     = 100
    
    for k in range(1,max_k_nn+1):
       clf         = KNeighborsClassifier(n_neighbors=k).fit(X_train , y_train )
       y_hat_valid = clf.predict(X_valid)
       conf_matrices.append(confusion_matrix(y_valid, y_hat_valid ))
   
    for i in range(0,max_k_nn):
       confusionmatrix=conf_matrices[i]
       correct =  np.trace(confusionmatrix)
       total = confusionmatrix.sum()
       percent_accuracy = correct/total
       result_for_i=[i+1,percent_accuracy]
       results_list.append(result_for_i)
            
    for u in range(1,max_k_nn):
        if results_list[u][1]>results_list[u-1][1]:
            best=results_list[u]
            
    return best


def K_NN_clf_result(best,X_train,y_train,X_valid,y_valid,X_test,y_test):
    
    best=find_best_k(X_train,X_valid,y_train,y_valid)[0]
    
    Y_hat_train = []
    Y_hat_valid = []
    Y_hat_test  = []
    
    clf = KNeighborsClassifier(n_neighbors=best).fit(X_train , y_train )
    Y_hat_train.append(clf.predict(X_train))
    Y_hat_valid.append(clf.predict(X_valid))
    Y_hat_test.append(clf.predict(X_test))
    
    Y_hat_train = pd.DataFrame(Y_hat_train).transpose()
    Y_hat_valid = pd.DataFrame(Y_hat_valid).transpose()
    Y_hat_test = pd.DataFrame(Y_hat_test).transpose()
    
    Y_hat_train.columns = ['label_hat']
    Y_hat_valid.columns = ['label_hat']
    Y_hat_test.columns = ['label_hat']
    
    
    confusionmatrix=confusion_matrix(y_test,Y_hat_test['label_hat'] )
    correct =  np.trace(confusionmatrix)
    total = confusionmatrix.sum()
    percent_accuracy = correct/total

    return (confusionmatrix,percent_accuracy)

#use logistic to predict all
dta_pre=pd.merge(dta,dta_test,on='Unnamed: 0',how='left')
corpus= dta_pre['tidy_tweet_x'].to_list()   
vectorizer_count     = CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X                    = vectorizer_count.fit_transform(dta['tidy_tweet'].values.astype('U'))
features_frequency   = pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
dta_pre['ML_group'] = dta_pre['ML_group'].fillna(4)
#TVT
X_train = X[np.where(dta_pre['ML_group']==0)[0],:]
X_valid = X[np.where(dta_pre['ML_group']==1)[0],:]
X_test  = X[np.where(dta_pre['ML_group']==2)[0],:]
y_train = dta_pre.loc[dta_pre['ML_group']==0,['score']]['score'].to_numpy()
y_valid = dta_pre.loc[dta_pre['ML_group']==1,['score']]['score'].to_numpy()
y_test  = dta_pre.loc[dta_pre['ML_group']==2,['score']]['score'].to_numpy()
X_predict  =X

def logistic_prediction(X_train,y_train,X_valid,y_valid,X_test,y_test,X_predict):
    
    ' . '
    categories         = pd.DataFrame(np.sort(np.unique(y_train))).reset_index()
    categories.columns = ['index','label']
    

    ' . '    
    ccp_train_list = []
    ccp_valid_list = []
    ccp_test_list  = []
    predict_list=[]
    for cat in categories['label'].to_list():
        y_train_c = 1*(y_train==cat)
        clf       = linear_model.LogisticRegression(tol          = 0.0001,
                                                    max_iter     = 10000,
                                                    random_state = None).fit(X_train, y_train_c)
        ccp_train_list.append(  clf.predict_proba(X_train)[:,1])
        ccp_valid_list.append(  clf.predict_proba(X_valid)[:,1])
        ccp_test_list.append(   clf.predict_proba(X_test)[:,1])
        predict_list.append(   clf.predict_proba(X_predict)[:,1])
    
    ' . Topic probability matrix'
    ccp_train = pd.DataFrame(ccp_train_list).transpose()
    ccp_valid = pd.DataFrame(ccp_valid_list).transpose()
    ccp_test  = pd.DataFrame(ccp_test_list).transpose()
    predict  = pd.DataFrame(predict_list).transpose()
    
    
    'reset column index'
    ccp_train.columns = categories['label'].to_list()
    ccp_valid.columns = categories['label'].to_list()
    ccp_test.columns = categories['label'].to_list()
    predict.columns = categories['label'].to_list()
    
    'Choosing your predictive category for the y '
    ccp_train['label_hat'] =  ccp_train.idxmax(axis=1)
    ccp_valid['label_hat'] =  ccp_valid.idxmax(axis=1)
    ccp_test['label_hat']  =  ccp_test.idxmax(axis=1)
    predict['label_hat']  =  predict.idxmax(axis=1)
    
    'prediction'
    dta_pre['predict']=predict['label_hat']
        
    return(dta_pre)

dta_final=logistic_prediction(X_train,y_train,X_valid,y_valid,X_test,y_test,X_predict)
