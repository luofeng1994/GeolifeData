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
    parser.add_argument('--train_data_dir', default='Data_Geolife/train')
    parser.add_argument('--train_file', default='train.csv')
    parser.add_argument('--test_data_dir', default='Data_Geolife/test')
    parser.add_argument('--test_file', default='test.csv')
    parser.add_argument('--classes', default='walk,bike,bus,car')
    parser.add_argument('--save_dir', default='./save')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--metrics_record_dir', default='./metrics_record')
    parser.add_argument('--metrics_record_file', default='./ml_record.csv')
    parser.add_argument('--section', default='ml')
    parser.add_argument('--is_training', default='train')
    parser.add_argument('--record', default=False)
    args = parser.parse_args()
    logger = Logger(logging.getLogger(), os.path.join(args.log_dir, 'ml', 'ml_{}.log'.format(getDate())))
    train_data_raw = pd.read_csv(os.path.join(args.train_data_dir, args.train_file))
    test_data_raw = pd.read_csv(os.path.join(args.test_data_dir, args.test_file))
    classes_to_handle = set(args.classes.split(','))

    train_data = pd.DataFrame(columns=train_data_raw.columns.values)
    test_data = pd.DataFrame(columns=test_data_raw.columns.values)

    for c in classes_to_handle:
        train_data = train_data.append(train_data_raw[train_data_raw.label == c])
        test_data = test_data.append(test_data_raw[test_data_raw.label == c])

    le = preprocessing.LabelEncoder()
    le.fit(pd.concat([train_data.label, test_data.label], axis=0))
    le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_y = le.transform(train_data.label)
    train_x = train_data.drop('label', axis=1)
    test_y = le.transform(test_data.label)
    test_x = test_data.drop('label', axis=1)


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
        record = {'section': classifier,
                  'metrics_record_dir': args.metrics_record_dir,
                  'metrics_record_file': args.metrics_record_file}
        model = classifiers[classifier](train_x, train_y)
        logger.info('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, predict)
        precision = metrics.precision_score(test_y, predict, average=None)
        recall = metrics.recall_score(test_y, predict, average=None)
        f1_score = metrics.f1_score(test_y, predict, average=None)
        record['accuracy'] = accuracy
        for cls in classes_to_handle:
            label_index = le_mapping[cls]
            record['%s_precision' % cls] = precision[label_index]
            record['%s_recll' % cls] = recall[label_index]
            record['%s_f1_score' % cls] = f1_score[label_index]
            logger.info('%s: precision: %.2f%%, recall: %.2f%%, f1_socre: %.2f%%' % (cls, 100 * precision[label_index], 100 * recall[label_index], 100*f1_score[label_index]))
        logger.info('accuracy: %.2f%%' % (100 * accuracy))

        if args.record:
            appendRecord(record)

if __name__ == '__main__':
    main()