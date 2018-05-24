import pandas as pd
import matplotlib.pyplot as plt


def draw_pic():
    dnn_result_path = './metrics_record/record.csv'
    ml_result_path = './metrics_record/ml_record.csv'
    measured_result_path = './metrics_record/measured_record.csv'
    dnn_result = pd.read_csv(dnn_result_path)
    ml_result = pd.read_csv(ml_result_path)
    measured_result = pd.read_csv(measured_result_path)


    dnn_result = dnn_result[['accuracy', 'epochs', 'is_training', 'learning_rate', 'num_layers', 'section']]
    ml_result = ml_result[['accuracy','section']]
    dnn_result_train = dnn_result[dnn_result['is_training']=='train']
    dnn_result_test = dnn_result[dnn_result['is_training']=='test']

    plt.figure()

    right = dnn_result_train.loc[(dnn_result_train['epochs'] == 50) & (dnn_result_train['learning_rate'] == 0.001)]
    index = right.index + 1
    left = dnn_result_test.loc[index]
    diff_num_layers = pd.merge(left[['accuracy', 'num_layers']], right[['accuracy', 'num_layers']], on='num_layers', suffixes=('_test', '_train'), sort=True)
    diff_num_layers.plot(x='num_layers', grid=True, title='epoches=50, learning_rate=0.001')


    right = dnn_result_train.loc[(dnn_result_train['num_layers'] == 1) & (dnn_result_train['learning_rate'] == 0.001)]
    index = right.index + 1
    left = dnn_result_test.loc[index]
    diff_epochs_1 = pd.merge(left[['accuracy', 'epochs']], right[['accuracy', 'epochs']], on='epochs', suffixes=('_test', '_train'), sort=True)
    diff_epochs_1.plot(x='epochs', grid=True, title='num_layers=1, learning_rate=0.001')


    right = dnn_result_train.loc[(dnn_result_train['num_layers'] == 4) & (dnn_result_train['learning_rate'] == 0.001)]
    index = right.index + 1
    left = dnn_result_test.loc[index]
    diff_epochs_2 = pd.merge(left[['accuracy', 'epochs']], right[['accuracy', 'epochs']], on='epochs', suffixes=('_test', '_train'), sort=True)
    diff_epochs_2.plot(x='epochs', grid=True, title='num_layers=4, learning_rate=0.001')



    right = dnn_result_train.loc[(dnn_result_train['epochs'] == 10) & (dnn_result_train['num_layers'] == 1)]
    index = right.index + 1
    left = dnn_result_test.loc[index]
    left.index = left.index-1
    diff_lr_1 = pd.merge(left[['accuracy']], right[['accuracy', 'learning_rate']], left_index=True, right_index=True, suffixes=('_test', '_train'), sort=True)
    diff_lr_1 = diff_lr_1.sort_values(by ='learning_rate', axis=0, ascending = True)
    diff_lr_1.plot(x='learning_rate', grid=True, title='num_layers=1, epochs=10')

    right = dnn_result_train.loc[(dnn_result_train['epochs'] == 50) & (dnn_result_train['num_layers'] == 1)]
    index = right.index + 1
    left = dnn_result_test.loc[index]
    left.index = left.index-1
    diff_lr_2 = pd.merge(left[['accuracy']], right[['accuracy', 'learning_rate']], left_index=True, right_index=True, suffixes=('_test', '_train'), sort=True)
    diff_lr_2 = diff_lr_2.sort_values(by='learning_rate', axis=0, ascending=True)
    diff_lr_2.plot(x='learning_rate', grid=True, title='num_layers=1, epochs=50')

    tmp = dnn_result_train[(dnn_result_train['epochs']==50) & (dnn_result_train['learning_rate']==0.001) & (dnn_result_train['num_layers']==3)]
    index = tmp.index
    DNN = dnn_result_test.loc[index+1]
    DNN = DNN[['accuracy', 'section']]
    ML = ml_result[['accuracy', 'section']]
    diff_algorithm = pd.concat([ML, DNN], axis=0)
    diff_algorithm.plot(kind='barh', x='section', grid=True, title='diff algorithms')

    measured_result = measured_result[['accuracy', 'test_data_dir']]
    measured_result.plot(kind='barh', x='test_data_dir', grid=True, title='measured data result')
    plt.show()
    aaa = 2

if __name__ == '__main__':
    draw_pic()
