#! /usr/bin/python
# -*- coding: utf-8 -*-

import csv
import logging
import random
import datetime, time
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier #random forest
from sklearn.ensemble import GradientBoostingClassifier, VotingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# from settings import DATA_DIR, LOG_DIR, RESULT_DIR
# import utils

from transformers import pipeline
import torch
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# EXPID = utils.get_expid()
EXPID = 'aa'
LOG_DIR='log'
RESULT_DIR='result'
DATA_DIR = 'data'
MAX_LEN = 512
# utils.set_logger('%s/%s.log' % (LOG_DIR, EXPID), 'DEBUG')
# TODO: log configurations (ex: parsing method etc) and/or commit id

import torch

def openfiles(filename):
    vrfilename = '%s/voteresult.csv' % DATA_DIR
    data = pd.read_csv(filename, sep='\t', header = 0)
    columns = ['fname', 'Ratio', 'flag']
    vr = pd.read_csv(vrfilename, sep=',', header = None)
    vr.columns = columns
    # print(vr.head())
    data = data.where((pd.notnull(data)), '')   # Replace np.nan with ''
    vr = vr.where((pd.notnull(vr)), '')  # Replace np.nan with ''

    logging.info("the number of total sentences: %s" % len(data['id']))
    flags = vr.loc[vr['flag']==1]['fname']
    new_data = data.loc[data['rfilename'].isin(flags)]
    logging.info("the number of unanimity sentences: %s" % len(new_data['id']))
    return new_data, len(new_data['id'])

def open_training_files(filename):
    vrfilename = '%s/voteresult.csv' % DATA_DIR
    data = pd.read_csv(filename, sep='\t', header = 0)
    columns = ['fname', 'Ratio', 'flag']
    vr = pd.read_csv(vrfilename, sep=',', header = None)
    vr.columns = columns

    data = data.where((pd.notnull(data)), '')   # Replace np.nan with ''
    vr = vr.where((pd.notnull(vr)), '')  # Replace np.nan with ''

    logging.info("the number of total sentences: %s" % len(data['id']))
    flags = vr.loc[vr['flag']==0]['fname']
    new_data = data.loc[data['rfilename'].isin(flags)]

    new_data.loc[new_data['sentiment']=="UP", 'sentiment'] =1
    new_data.loc[new_data['sentiment'] == "DOWN", 'sentiment'] = -1
    new_data.loc[new_data['sentiment'] == "STAY", 'sentiment'] = 0
    print(new_data[:5])
    logging.info("the number of training sentences: %s" % len(new_data['id']))
    return new_data, len(new_data['id'])


def delStay(X, y, flag):
    flag = int(flag)
    if flag == "3":
        idx_s = y[y == 'STAY'].sample(frac=0.1, random_state = 111).index.tolist()
        idx_u = y[y == 'UP'].index.tolist()
        idx_d = y[y == 'DOWN'].index.tolist()

        # idx2 = y[y == 'DOWN'].sample(frac=1, random_state=111).index.tolist()
        idx = idx_s.extend(idx_u).extend(idx_d)

        X = X[idx]
        y = y[idx]

    return X,y.to_numpy(), idx

def SamplingStay(data, flag):
    flag = int(flag)
    if flag == 3:
        tmp_frac =0.035
    else:
        tmp_frac = 0
    data_s = data.loc[data['sentiment'] == 'STAY'].sample(frac=tmp_frac, random_state = 111)
    data_u = data.loc[data['sentiment'] == 'UP'].sample(frac=1, random_state = 111)
    data_d = data.loc[data['sentiment'] == 'DOWN'].sample(frac=0.5, random_state = 111)
    new_data = pd.concat([data_s,data_u,data_d])
    # idx2 = y[y == 'DOWN'].sample(frac=1, random_state=111).index.tolist()
    print("length of new data: %d" % len(new_data))

    return new_data

def tokenizing(docs, mode=None, flag = 1,min_df = 0.01, max_df = 1.0):
    flag = int(flag)
    if flag == 1:#unigram
        if mode=='tf':
            vectorizer = CountVectorizer(min_df = min_df, max_df = max_df)
        elif mode=='tfidf':
            vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df)
        else:
            raise Exception('Invalid mode %s' % mode)
        matrix_td = vectorizer.fit_transform(docs) # term doc matrix
    else: # bigram
        if mode=='tf':
            vectorizer = CountVectorizer(ngram_range = (1,flag), min_df = min_df, max_df = max_df)
        elif mode=='tfidf':
            vectorizer = TfidfVectorizer(ngram_range = (1,flag), min_df = min_df, max_df = max_df)
        else:
            raise Exception('Invalid mode %s' % mode)
        matrix_td = vectorizer.fit_transform(docs) # term doc matrix
    return matrix_td.toarray(), vectorizer.get_feature_names()

def map_to_utf(data):
    for k, v in data.items():
        data[k] = ','.join(str(v)).decode('utf-8').encode('utf8') if isinstance(v, list) else str(v).decode('utf8').encode("utf-8")
    return data

def generate_LR(X_train, X_test, y_train, y_test, feature_names,selected_ch, r):
    selected_ch = int(selected_ch)
    if feature_names ==[]:
        flag = 1
    else:
        flag = 2
    #logreg = sklearn.preprocessing.LabelEncoder()
    logreg = linear_model.LogisticRegression(C=1e1, solver ='liblinear', class_weight='balanced', random_state=243*r)
    logging.info(logreg)
    if flag==1:
        fullmodel = logreg.fit(X_train, y_train)
        Remodel = SelectFromModel(fullmodel, prefit=True)
        X_train_new = Remodel.transform(X_train)
        X_test_new = Remodel.transform(X_test)
        model = logreg.fit(X_train_new, y_train)
        #logging.info(logreg.coef_)
        logging.info(X_train.shape)
        logging.info(X_train_new.shape)
    else:
        ch2 = SelectKBest(chi2, k=selected_ch)
        X_train_new = ch2.fit_transform(X_train, y_train)
        X_test_new = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names= np.asarray(feature_names)
        # logging.info(str(feature_names).encode('utf-8').decode('unicode_escape'))
        # logging.info(str(feature_names).encode('utf-8').decode('utf-8'))
        model = logreg.fit(X_train_new, y_train)
        #logging.info(logreg.coef_)
    train_predicted = model.predict(X_train_new)
    test_predicted = model.predict(X_test_new)
    probs = model.predict_proba(X_test_new)
    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)
    cm = metrics.confusion_matrix(y_test, test_predicted)
    report = metrics.classification_report(y_test, test_predicted)

    return train_accuracy, test_accuracy, cm, report

def generate_RF(X_train, X_test, y_train, y_test, feature_names, selected_ch, r):
    selected_ch = int(selected_ch)
    if feature_names ==[]:
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_depth=8, random_state=r*34)
        flag = 1
    else:
        flag = 1
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_depth=8, random_state=r*34)
    logging.info(rf)
    if flag == 1:
        rf.fit(X_train, y_train)
        train_predicted = rf.predict(X_train)
        test_predicted = rf.predict(X_test)
    elif flag ==2:
        # dimension reduction to selected_ch
        ch2 = SelectKBest(chi2,k=selected_ch)
        X_train_new = ch2.fit_transform(X_train, y_train)
        X_test_new = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names= np.asarray(feature_names)
        # logging.info(str(feature_names).encode('utf-8').decode('utf-8'))

        rf.fit(X_train_new, y_train)
        train_predicted = rf.predict(X_train_new)
        test_predicted = rf.predict(X_test_new)
    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)
    cm = confusion_matrix(y_test, test_predicted)

    return train_accuracy, test_accuracy, cm

def generate_NN(X_train, X_test, y_train, y_test,feature_names,selected_ch,r):
    selected_ch = int(selected_ch)
    if feature_names ==[]:
        flag = 1
    else:
        flag = 2
    hidden =100
    nn = MLPClassifier(solver='adam', alpha = 1e-2, hidden_layer_sizes =(hidden,2), random_state =124*r)
    logging.info(nn)
    if flag == 1:
        nn.fit(X_train, y_train)
        train_predicted = nn.predict(X_train)
        test_predicted = nn.predict(X_test)
    elif flag ==2:
        # dimension reduction to selected_ch
        ch2 = SelectKBest(chi2, k=selected_ch)
        X_train_new = ch2.fit_transform(X_train, y_train)
        X_test_new = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names = np.asarray(feature_names)
        # logging.info(str(feature_names).encode('utf-8').decode('utf-8'))

        nn.fit(X_train_new, y_train)
        train_predicted = nn.predict(X_train_new)
        test_predicted = nn.predict(X_test_new)

    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)

    cm = metrics.confusion_matrix(y_test, test_predicted)
    report = metrics.classification_report(y_test, test_predicted)
    return train_accuracy, test_accuracy, cm, report

def generate_BR(X_train, X_test, y_train, y_test, feature_names,selected_ch,r):
    # classification 인지 확인
    br = linear_model.BayesianRidge(compute_score=True, tol=1e-5)
    logging.info(br)

    # dimension reduction to selected_ch
    selected_ch = int(selected_ch)
    ch2 = SelectKBest(chi2, k=selected_ch)
    X_train_new = ch2.fit_transform(X_train, y_train)
    X_test_new = ch2.transform(X_test)
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    feature_names = np.asarray(feature_names)
    logging.info(str(feature_names).encode('utf-8').decode('utf-8'))

    br.fit(X_train_new, y_train)
    train_predicted = br.predict(X_train_new)
    test_predicted = br.predict(X_test_new)

    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)

    cm = metrics.confusion_matrix(y_test, test_predicted)
    report = metrics.classification_report(y_test, test_predicted)

    return train_accuracy, test_accuracy, cm, report

def generate_SVM(X_train, X_test, y_train, y_test, feature_names,selected_ch,r):

    svm = make_pipeline(StandardScaler(), SVC(kernel = 'sigmoid', C=1, random_state=r*24))
    #svm = make_pipeline(StandardScaler(), LinearSVC(penalty='l2', random_state=0, tol=1e-3))

    logging.info(svm)

    if feature_names ==[]:
        flag = 1
    else:
        flag = 2

    if flag == 1:
        svm.fit(X_train, y_train)
        train_predicted = svm.predict(X_train)
        test_predicted = svm.predict(X_test)

    elif flag == 2:

        # dimension reduction to selected_ch
        selected_ch = int(selected_ch)
        ch2 = SelectKBest(chi2, k=selected_ch)
        X_train_new = ch2.fit_transform(X_train, y_train)
        X_test_new = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names = np.asarray(feature_names)
        # logging.info(str(feature_names).encode('utf-8').decode('utf-8'))

        svm.fit(X_train_new, y_train)
        train_predicted = svm.predict(X_train_new)
        test_predicted = svm.predict(X_test_new)

    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)

    cm = metrics.confusion_matrix(y_test, test_predicted)
    report = metrics.classification_report(y_test, test_predicted)

    return train_accuracy, test_accuracy, cm, report

def generate_adaboost(X_train, X_test, y_train, y_test, feature_names, selected_ch, r):
    # base = linear_model.LinearRegression()
    selected_ch = int(selected_ch)
    if feature_names ==[]:
        flag = 1
        base = DecisionTreeClassifier(min_samples_leaf=2, max_depth=3, max_features='sqrt')
        #base = SVC(probability=True, kernel='linear', C=1)
    else:
        flag = 2
        base = DecisionTreeClassifier(min_samples_leaf=2, max_depth=3, max_features='sqrt')
        #base = SVC(probability=True, kernel='linear', C=1)
    ada = AdaBoostClassifier(base, n_estimators=400, algorithm="SAMME.R", learning_rate=0.1, random_state=33*r)
    logging.info(ada)

    if flag == 1:
        ada.fit(X_train, y_train)
        train_predicted = ada.predict(X_train)
        test_predicted = ada.predict(X_test)

    elif flag ==2:
        # dimension reduction to selected_ch
        ch2 = SelectKBest(chi2, k=selected_ch)
        X_train_new = ch2.fit_transform(X_train, y_train)
        X_test_new = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names = np.asarray(feature_names)
        # logging.info(str(feature_names).encode('utf-8').decode('utf-8'))

        ada.fit(X_train_new, y_train)
        train_predicted = ada.predict(X_train_new)
        test_predicted = ada.predict(X_test_new)

    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)

    cm = metrics.confusion_matrix(y_test, test_predicted)
    report = metrics.classification_report(y_test, test_predicted)

    return train_accuracy, test_accuracy, cm, report

def generate_gradientBoosting(X_train, X_test, y_train, y_test, feature_names, selected_ch, r):
    selected_ch = int(selected_ch)
    if feature_names ==[]:
        flag = 1
        params = {'n_estimators': 300, 'max_depth': 3, 'min_samples_leaf': 1, 'learning_rate': 0.8, 'loss': 'deviance',
                  'warm_start': True, 'random_state': 34*r}
    else:
        flag = 2
        params = {'n_estimators': 300, 'max_depth': 3, 'min_samples_leaf': 1, 'learning_rate': 0.8, 'loss': 'deviance',
                  'warm_start': False, 'random_state': 34*r, 'max_features': 'auto'}
    gb = GradientBoostingClassifier(**params)
    logging.info(gb)
    if flag == 1:
        gb.fit(X_train, y_train)
        train_predicted = gb.predict(X_train)
        test_predicted = gb.predict(X_test)
    elif flag == 2:
        # dimension reduction to selected_ch
        ch2 = SelectKBest(chi2, k=selected_ch)
        X_train_new = ch2.fit_transform(X_train, y_train)
        X_test_new = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names = np.asarray(feature_names)
        # logging.info(str(feature_names).encode('utf-8').decode('utf-8'))

        gb.fit(X_train_new, y_train)
        train_predicted = gb.predict(X_train_new)
        test_predicted = gb.predict(X_test_new)

    train_accuracy = metrics.accuracy_score(y_train, train_predicted)
    test_accuracy = metrics.accuracy_score(y_test, test_predicted)

    cm = metrics.confusion_matrix(y_test, test_predicted)
    report = metrics.classification_report(y_test, test_predicted)

    return train_accuracy, test_accuracy, cm, report

# def generate_vote(X_train, X_test, y_train, y_test, PARE, tolerance, max_f):
#     vis = False
#
#     if feature_names ==[]:
#         params = {'n_estimators': 300, 'max_depth': 7, 'min_samples_leaf': 1, 'learning_rate': 0.01, 'loss': 'lad',
#                   'warm_start': True, 'random_state': 1}
#         reg1 = GradientBoostingRegressor(**params)
#         reg2 = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_depth=7)
#         base = DecisionTreeRegressor(min_samples_leaf=1, max_depth=7)
#         flag = 1
#     else:
#         params = {'n_estimators': 300, 'max_depth': 7, 'min_samples_leaf': 1, 'learning_rate': 0.01, 'loss': 'lad',
#                   'max_features': max_f, 'warm_start': True, 'random_state': 1}
#         reg1 = GradientBoostingRegressor(**params)
#         reg2 = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_depth=7, max_features=max_f)
#         base = DecisionTreeRegressor(min_samples_leaf=1, max_depth=7, max_features=max_f)
#         flag = 2
#
#     # training
#     reg3 = AdaBoostRegressor(base, n_estimators=300, learning_rate=0.05)
#
#     #    reg3 = linear_model.LinearRegression()
#     ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('Ada', reg3)])
#
#     tolerance = float(tolerance)
#     logging.info(ereg)
#
#     print(X_train.shape)
#     print(X_test.shape)
#     ereg = ereg.fit(X_train, y_train)
#
#     train_predicted = ereg.predict(X_train)
#     test_predicted = ereg.predict(X_test)
#
#     if PARE == "True":
#         train_accuracy, train_error_list = pare_metrics(train_predicted, y_train, tolerance)
#         test_accuracy, test_error_list = pare_metrics(test_predicted, y_test, tolerance)
#
#     else:
#         print("PARE NO")
#         train_accuracy = metrics.mean_absolute_error(y_train, train_predicted)
#         test_accuracy = metrics.mean_absolute_error(y_test, test_predicted)
#
#     ## visualization, figure result 어디서 뜨는지 확인
#     if vis == True:
#         reg1.fit(X_train, y_train)
#         reg2.fit(X_train, y_train)
#         reg3.fit(X_train, y_train)
#         pred1 = reg1.predict(X_test)
#         pred2 = reg2.predict(X_test)
#         pred3 = reg3.predict(X_test)
#         pred4 = ereg.predict(X_test)
#
#         plt.figure()
#         plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
#         plt.plot(pred2, 'b^', label='RandomForestRegressor')
#         plt.plot(pred3, 'ys', label='LinearRegression')
#         plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')
#
#         plt.tick_params(axis='x', which='both', bottom=False, top=False,
#                         labelbottom=False)
#         plt.ylabel('predicted')
#         plt.xlabel('training samples')
#         plt.legend(loc="best")
#         plt.title('Regressor predictions and their average')
#
#         plt.show()
#
#     return train_accuracy, test_accuracy

def acc_write(model_name, accfile, args, train_acc, test_acc, k, mode, repeat_list):
    print(len(train_acc))
    with open(accfile, 'a', newline ='') as f:
        writer = csv.writer(f)
        for i in range(len(repeat_list)):
            writer.writerow([mode, model_name, args.number_variables, args.flag_unigram, repeat_list[i], i, train_acc[i], test_acc[i]])
    logging.info("%s: TOTAL Accuracy \n  train: %.4f, test: %.4f\n" % (model_name, sum(train_acc)/len(repeat_list),sum(test_acc)/len(repeat_list)))

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Get input parameters.',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-b', '--batch-size', dest='batch_size',
                        help='batch size', default=8, type=int)
    parser.add_argument('-u', '--unigram', required=True, dest='flag_unigram',
            help='unigram or not')
    parser.add_argument('-c', '--class', dest='flag_class',
            help='notice the number of class')
    parser.add_argument('-n', '--number', dest='number_variables',
            help='notice the number of variables')
    parser.add_argument('-m', '--mode', dest='mode',
            help='tf or tfidf or bert')
    args = parser.parse_args()
    # utils.set_logger('%s/%s_%s.log' % (LOG_DIR, EXPID,args), 'DEBUG')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    accfile = '%s/%s_acc_two_class_full_train_short_10_repeat.csv' % (RESULT_DIR, args.mode)

    if args.mode =='tfidf':
        tmp_modes = 'tf'
    else:
        tmp_modes = args.mode

    # extract unanimity sentences
    full_filename = '%s/%s_full_after_preprocessing_with_filename.txt' % (DATA_DIR, tmp_modes)
    short_filename = '%s/%s_short_after_preprocessing_with_filename.txt' % (DATA_DIR, tmp_modes)
    full_data, full_len= openfiles(full_filename)
    short_data, short_len = openfiles(short_filename)
    full_train_data, full_train_len=open_training_files(full_filename)

    # delete or sampling stay class
    full_new_data= SamplingStay(full_data, args.flag_class)
    short_new_data = SamplingStay(short_data, args.flag_class)
    full_new_len=len(full_new_data)
    short_new_len = len(short_new_data)

    # logging.info("After Delete \"STAY\" full text length: %d, short text length: %d\n" % (len(full_train_y), len(short_all_y)))
    logging.info(
        "After Delete or sampling \"STAY\" full text length: %d, short text length: %d\n" % (full_new_len, short_new_len))
    logging.info("Full train data length: %d\n" % full_train_len)

    all_data = full_new_data.append(short_new_data)
    all_len = len(all_data)

    ids = all_data['id']
    fnamelists = all_data['rfilename']
    y = all_data['sentiment']

    n_Stay = len(y[y=="STAY"])
    n_Down = len(y[y == "DOWN"])
    n_Up = len(y[y == "UP"])

    print("number of STAY: %d, number of Down: %d, number of Up: %d" % (n_Stay, n_Down, n_Up))

    if args.mode == 'tf' or args.mode == 'tfidf':
        docs, feature_names = tokenizing(list(all_data['sentence']), args.mode, args.flag_unigram)  # term doc matrix
        logging.info(docs.shape)

    elif args.mode =='bert':
        batch_size = args.batch_size

        # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')



        ########## Training ########

        train_data = []

        max_len = 0

        train_data = []
        for i in range(len(full_train_data)):
            train_data.append('[CLS] ' + full_train_data['sentence'].iloc[i] + ' [SEP]')

        x_train_tokenized = train_data
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        for i in tqdm(range(len(train_data))):
            x_train_tokenized[i] = tokenizer.tokenize(train_data[i])

        for i in tqdm(range(len(x_train_tokenized))):
            x_train_tokenized[i] = tokenizer.convert_tokens_to_ids(x_train_tokenized[i])

        max_len = 0
        for i in range(len(x_train_tokenized)):
            if len(x_train_tokenized[i]) > max_len:
                max_len = len(x_train_tokenized[i])

        print("max len: ", max_len)

        train_indexed_text = pad_sequences(x_train_tokenized, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        x_train_attention_mask = []
        for i in range(len(train_indexed_text)):
            seq_mask = [float(s > 0) for s in train_indexed_text[i]]
            x_train_attention_mask.append(seq_mask)

        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(list(train_indexed_text),
                                                                                            list(full_train_data['sentiment'].iloc[:]),
                                                                                            random_state=2018,
                                                                                            test_size=0.1)

        train_masks, validation_masks, _, _ = train_test_split(x_train_attention_mask,
                                                               train_indexed_text,
                                                               random_state=2018,
                                                               test_size=0.1)

        # train_labels = list(train_labels)
        # validation_labels = list(validation_labels)

        train_inputs = torch.tensor(train_inputs).to(torch.int64)
        train_labels = torch.tensor(train_labels).to(torch.int64) + 1
        train_masks = torch.tensor(train_masks).to(torch.int64)
        validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
        validation_labels = torch.tensor(validation_labels).to(torch.int64) + 1
        validation_masks = torch.tensor(validation_masks).to(torch.int64)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        logging.info("End of preparation for Training")

        total_loss = 0
        # model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)


        num_labels = 3

        class BertRNN(torch.nn.Module):

            def __init__(self):
                super(BertRNN, self).__init__()
                self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
                self.num_layers = 2
                self.gru = torch.nn.GRU(input_size=self.bert_model.config.hidden_size,
                                        hidden_size = self.bert_model.config.hidden_size,
                                        num_layers=self.num_layers, batch_first=True, bidirectional=False)

                if self.gru.bidirectional:
                    self.gru_output_dim = self.bert_model.config.hidden_size * 2
                    self.gru_num_direction = 2
                else:
                    self.gru_output_dim = self.bert_model.config.hidden_size
                    self.gru_num_direction = 1

                self.classifier = torch.nn.Sequential(torch.nn.Linear(self.gru_output_dim, self.gru_output_dim),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.gru_output_dim, num_labels))

                self.xe_loss_func = torch.nn.CrossEntropyLoss()

            def forward(self, batch):
                b_input_ids, b_input_mask, b_labels = batch
                # b_labels += 1
                outputs = self.bert_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                # outputs = model(b_input_ids)

                padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(outputs.last_hidden_state,
                                                                          lengths=b_input_mask.sum(1).cpu(),
                                                                          batch_first=True,
                                                                          enforce_sorted=False)
                # to remove cls token use padded_sequence[:, 1:]
                gru_output, gru_h_n = self.gru(padded_sequence)

                gru_h_n = gru_h_n.view(self.num_layers, self.gru_num_direction, -1, self.gru.hidden_size)

                gru_h_n_last_layer_hidden = gru_h_n[0].permute(1, 0, 2)
                gru_h_n_last_layer_hidden_flatten = gru_h_n_last_layer_hidden.flatten(1)

                logits = self.classifier(gru_h_n_last_layer_hidden_flatten)
                if b_labels is None:
                    loss = None
                else:
                    loss = self.xe_loss_func(logits, b_labels)


                return logits, loss, gru_h_n_last_layer_hidden_flatten

        model = BertRNN()
        model = model.to(device)



        optimizer = AdamW(model.parameters(),
                          lr=5e-5,
                          eps=1e-8
                          )
        epochs = 1
        total_steps = len(train_data) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        model.zero_grad()

        for epoch_i in range(0, epochs):
            t0 = time.time()
            model.train()
            for step, batch in enumerate(train_dataloader):
                # print(batch)
                if step % 500 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                batch = tuple(t.to(device) for t in batch)

                logits, loss, embedding = model(batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                total_loss += loss.sum().detach().cpu()


            avg_train_loss = total_loss / len(train_data)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    # outputs = model(b_input_ids)
                    outputs = model(batch)

                logits, loss, embedding = model(batch)

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1


            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Training complete!")


        ########### Evaluation ###############

        data_list = list(all_data['sentence'])
        #print(all_data['sentence'].iloc[0])
        test_data =[]
        max_len = 0
        for i in range(len(data_list)):
            # print(i, data_list[i])
            test_data.append('[CLS] ' + data_list[i] + ' [SEP]')
            data_list_tokenized = test_data
            indexed_text = test_data
            data_list_tokenized[i] = tokenizer.tokenize(test_data[i])
            indexed_text[i] = tokenizer.convert_tokens_to_ids(data_list_tokenized[i])

            if len(indexed_text[i]) > max_len:
                max_len = len(indexed_text[i])

        indexed_text = pad_sequences(indexed_text, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        docs =[]

        # model = BertModel.from_pretrained("bert-base-multilingual-cased", num_labels=3)
        # model = model.to(device)
        model.eval()

        for i in range(round(len(data_list) / batch_size)):
            print("Currently, %d/%d ing" % (batch_size*(i+1), all_len))
            tmp_indexed_text = indexed_text[i*batch_size:(i+1)*batch_size]
            tokens_tensor = torch.tensor(tmp_indexed_text).type(torch.LongTensor)
            tokens_tensor = tokens_tensor.to(device)
            input_mask = tokens_tensor != 0
            label = None
            logits, loss, embedding = model((tokens_tensor, input_mask, label))
            # output = model(tokens_tensor).pooler_output.detach().to('cpu').numpy()
            tmp_docs = embedding.detach().cpu().numpy()

            # tmp_docs = output.reshape(batch_size, 768)
            if docs == []:
                docs = tmp_docs
            else:
        #docs = [bert_embedding(i, device).reshape(768,) for i in list(all_data['sentence'])]
                docs = np.append(docs, np.array(tmp_docs), axis=0)
        if len(data_list) % batch_size !=0:
            tmp_indexed_text = indexed_text[(i+1) * batch_size:len(data_list)]
            tokens_tensor = torch.tensor(tmp_indexed_text).type(torch.LongTensor)
            tokens_tensor = tokens_tensor.to(device)
            output = model(tokens_tensor).pooler_output.detach().to('cpu').numpy()
            tmp_docs = output.reshape(len(data_list)-(i+1) * batch_size, 768)
            docs = np.append(docs, np.array(tmp_docs), axis=0)

        logging.info(docs.shape)
        columns = ['id', 'rfilename', 'sentence', 'sentiment']
        final_bert_data = pd.DataFrame(columns=columns)
        final_bert_data['id'] =ids
        final_bert_data['rfilename'] = fnamelists
        # final_bert_data['sentence'] = docs
        final_bert_data['sentiment'] = y

        for k in range(len(docs)):
            final_bert_data['sentence'].iloc[k] = np.array(docs[k])
        bert_embedding_filename = '%s/bert_embedding_data_%d_full_length_%d_all_length_fined_tuned.txt' % (DATA_DIR, full_new_len, short_new_len)
        final_bert_data.to_csv(bert_embedding_filename,  sep='\t', header=True, index=False, encoding='utf8')
        feature_names = []

    X = docs
    full_train_X = X[:full_new_len]
    short_all_X = X[full_new_len:]
    full_train_y = y[:full_new_len].to_numpy()
    short_all_y = y[full_new_len:].to_numpy()
    logging.info("full text length: %d, short text length: %d\n" % (full_new_len, short_new_len))

    lr_train_acc, lr_test_acc = [], []
    rf_train_acc, rf_test_acc = [], []
    nn_train_acc, nn_test_acc = [], []
    br_train_acc, br_test_acc = [], []
    svm_train_acc, svm_test_acc = [], []
    ada_train_acc, ada_test_acc = [], []
    gb_train_acc, gb_test_acc = [], []
    repeat_list =[]
    num_repeat = 10
    for r in range(num_repeat):
        num_cv = 5
        kf = StratifiedKFold(n_splits=num_cv)
        kf.get_n_splits(short_all_X, short_all_y)
        i = 0

        for train_index, test_index in kf.split(short_all_X, short_all_y):
            i += 1
            print("%d iteration............" % i)
            train_X = np.append(full_train_X, short_all_X[train_index], axis=0)
            train_y = np.append(full_train_y, short_all_y[train_index], axis=0)
            test_X = short_all_X[test_index]
            test_y = short_all_y[test_index]

            logging.info('Train sample: UP: %s, Down: %s, STAY: %s' % (len(train_y[train_y[:] == 'UP']), len(train_y[train_y[:] == 'DOWN']), len(train_y[train_y[:] == 'STAY'])))
            logging.info('Test sample: UP: %s, Down: %s, STAY: %s' % (len(test_y[test_y[:] == 'UP']), len(test_y[test_y[:] == 'DOWN']), len(train_y[train_y[:] == 'STAY'])))

            logging.info("Modeling of logistic regression...")
            train_tmp_acc, test_tmp_acc, tmp_cm, tmp_report = generate_LR(train_X, test_X, train_y, test_y, feature_names, args.number_variables, r) #logistic regression
            logging.info("Accuracy of Logistic Regression\n train: %.4f, test: %.4f\n" % (train_tmp_acc, test_tmp_acc))
            logging.info('\n%s' % str(tmp_cm))
            logging.info('\n%s' % str(tmp_report))
            lr_train_acc.append(train_tmp_acc)
            lr_test_acc.append(test_tmp_acc)

            logging.info("Modeling of random forest...")
            train_tmp_acc, test_tmp_acc, tmp_cm = generate_RF(train_X, test_X, train_y, test_y, feature_names,args.number_variables, r) # #random forest
            logging.info("Accuracy of Random Forest\n  train: %.4f, test: %.4f\n" % (train_tmp_acc, test_tmp_acc))
            logging.info('\n%s' % str(tmp_cm))
            rf_train_acc.append(train_tmp_acc)
            rf_test_acc.append(test_tmp_acc)

            logging.info("Modeling of neural network...")
            train_tmp_acc, test_tmp_acc, tmp_cm, tmp_report = generate_NN(train_X, test_X, train_y, test_y, feature_names, args.number_variables,r) #neural network
            logging.info("Accuracy of Neural network\n train: %.4f, test: %.4f\n" % (train_tmp_acc, test_tmp_acc))
            logging.info('\n%s' % str(tmp_cm))
            logging.info('\n%s' % str(tmp_report))
            nn_train_acc.append(train_tmp_acc)
            nn_test_acc.append(test_tmp_acc)

            #Support vector machine
            logging.info("Modeling of Support vector machine...")
            train_tmp_acc, test_tmp_acc, tmp_cm, tmp_report= generate_SVM(train_X, test_X, train_y, test_y, feature_names, args.number_variables, r)
            logging.info("Accuracy of Support vector machine\n train: %.4f, test: %.4f\n" % (train_tmp_acc, test_tmp_acc))
            logging.info('\n%s' % str(tmp_cm))
            logging.info('\n%s' % str(tmp_report))
            svm_train_acc.append(train_tmp_acc)
            svm_test_acc.append(test_tmp_acc)

            #adaboost
            logging.info("Modeling of AdaBoostClassifier with decision tree...")
            train_tmp_acc, test_tmp_acc, tmp_cm, tmp_report = generate_adaboost(train_X, test_X, train_y, test_y, feature_names, args.number_variables,r)
            logging.info("Accuracy of AdaBoostClassifier\n  train: %.4f, test: %.4f\n" % (train_tmp_acc, test_tmp_acc))
            logging.info('\n%s' % str(tmp_cm))
            logging.info('\n%s' % str(tmp_report))
            ada_train_acc.append(train_tmp_acc)
            ada_test_acc.append(test_tmp_acc)

            # # Gradient Boosting
            logging.info("Modeling of gradient boosting...")
            train_tmp_acc, test_tmp_acc, tmp_cm, tmp_report= generate_gradientBoosting(train_X, test_X, train_y, test_y, feature_names, args.number_variables, r) #neural network
            logging.info("Accuracy of gradient boosting\n train: %.4f, test: %.4f\n" % (train_tmp_acc, test_tmp_acc))
            logging.info('\n%s' % str(tmp_cm))
            logging.info('\n%s' % str(tmp_report))
            gb_train_acc.append(train_tmp_acc)
            gb_test_acc.append(test_tmp_acc)

            repeat_list.append(r)

    acc_write('lr', accfile, args, lr_train_acc, lr_test_acc, num_cv, args.mode, repeat_list)
    acc_write('rf', accfile, args, rf_train_acc, rf_test_acc, num_cv, args.mode, repeat_list)
    acc_write('nn', accfile, args, nn_train_acc, nn_test_acc, num_cv, args.mode, repeat_list)
    acc_write('svm', accfile, args, svm_train_acc, svm_test_acc, num_cv,args.mode, repeat_list)
    acc_write('ada', accfile, args, ada_train_acc, ada_test_acc, num_cv, args.mode, repeat_list)
    acc_write('gb', accfile, args, gb_train_acc, gb_test_acc, num_cv, args.mode, repeat_list)

