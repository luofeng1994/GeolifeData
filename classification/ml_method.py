import argparse
from main import getDate
from main import Logger
import logging
import os
import pandas as pd
from sklearn import preprocessing
from ml_models import *
from main import appendRecord
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='./Data_Geolife/train')
    parser.add_argument('--test_data_dir', default='./Data_Geolife/test')
    parser.add_argument('--classes', default='walk,bike,car,bus')
    parser.add_argument('--save_dir', default='./save')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--metrics_record_dir', default='./metrics_record')
    parser.add_argument('--metrics_record_file', default='./ml_record.csv')
    parser.add_argument('--section', default='ml')
    parser.add_argument('--is_training', default='train')
    parser.add_argument('--record', default=True)
    args = parser.parse_args()
    logger = Logger(logging.getLogger(), os.path.join(args.log_dir, 'ml', 'ml_{}.log'.format(getDate())))
    train_data_raw = pd.read_csv(os.path.join(args.train_data_dir, 'train.csv'))
    test_data_raw = pd.read_csv(os.path.join(args.test_data_dir, 'test.csv'))
    classes_to_handle = set(args.classes.split(','))
    num_classes = len(classes_to_handle)

    train_data = pd.DataFrame(columns=train_data_raw.columns.values)
    test_data = pd.DataFrame(columns=test_data_raw.columns.values)
    for c in classes_to_handle:
        train_data = train_data.append(train_data_raw[train_data_raw.label == c])
        test_data = test_data.append(test_data_raw[test_data_raw.label == c])
    le = preprocessing.LabelEncoder()
    le.fit(pd.concat([train_data.label, test_data.label], axis=0))
    train_y = le.transform(train_data.label)
    train_x = train_data.drop('label', axis=1)
    test_y = le.transform(test_data.label)
    test_x = test_data.drop('label', axis=1)

    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x_normalized = scaler.transform(train_x)
    test_x_normalized = scaler.transform(test_x)

    classifiers_to_run = ['NB',
                          'KNN',
                          'LR',
                          'RF',
                          'DT',
                          'SVM',
                          # 'SVMCV',
                          'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }
    for classifier in classifiers_to_run:
        logger.info('start training model: {}'.format(classifier))
        start_time = time.time()
        model = classifiers[classifier](train_x_normalized, train_y)
        logger.info('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x_normalized)
        # precision = metrics.precision_score(test_y, predict)
        # recall = metrics.recall_score(test_y, predict)
        # logger.info('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        logger.info('accuracy: %.2f%%' % (100 * accuracy))

        if args.record:
            record = {'section': classifier,
                      'accuracy': accuracy,
                      'metrics_record_dir': args.metrics_record_dir,
                      'metrics_record_file': args.metrics_record_file}
            appendRecord(record)

if __name__ == '__main__':
    main()