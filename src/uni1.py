from logger import log
import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as tfk


def path_join(*args):
    return os.path.join(os.path.dirname(__file__), *args)


dftrain_raw = pd.read_csv(path_join('../data/titanic/train.csv'))
dftest_raw = pd.read_csv(path_join('../data/titanic/test.csv'))


def read_csv():
    '''
    读取csv文件
    '''
    log.info(dftest_raw.head(10))


# label分布情况


def show_survived():
    '''
    展示数据
    '''
    train_csv = pd.read_csv(path_join('../data/titanic/train.csv'))
    ax = train_csv['Survived'].value_counts().plot(kind='bar', figsize=(12, 8), fontsize=15, rot=0)
    ax.set_ylabel('Counts', fontsize=15)
    ax.set_xlabel('Survived', fontsize=15)
    plt.show()


# 年龄分布情况
def show_age():
    '''
    展示数据
    '''    
    train_csv = pd.read_csv(path_join('../data/titanic/train.csv'))
    ax = train_csv['Age'].plot(kind='hist', bins=20, color='purple', figsize=(12, 8), fontsize=15)
    ax.set_ylabel('Frequency', fontsize=15)
    ax.set_xlabel('Age', fontsize=15)
    plt.show()


# 年龄和label的相关性
def show_age_survived():
    '''
    展示数据
    '''
    train_csv = pd.read_csv(path_join('../data/titanic/train.csv'))
    ax = train_csv.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
    train_csv.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
    ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
    ax.set_ylabel('Density', fontsize=15)
    ax.set_xlabel('Age', fontsize=15)
    plt.show()


# 数据预处理
def preprocessing(dfdata):

    dfresult = pd.DataFrame()

    # Pclass Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return dfresult


# 定义模型
def def_models():
    tf.keras.backend.clear_session()

    model = tfk.models.Sequential()
    pass
    model.add(tfk.layers.Dense(20, activation='relu', input_shape=(15,)))
    model.add(tfk.layers.Dense(10, activation='relu'))
    model.add(tfk.layers.Dense(1, activation='sigmoid'))

    model.summary()

    return model


# 训练模型
def train_model(model, x_train, y_train):
    # 二分类问题选择二元交叉熵损失函数
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.2)  # 分割一部分训练数据用于验证
    return history


# 评估模型
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


def main():
    x_train = preprocessing(dftrain_raw)
    y_train = dftrain_raw['Survived'].values

    x_test = preprocessing(dftest_raw)
    y_test = dftest_raw['Survived'].values

    log.info(f'x_train = \n{x_train}')
    log.info(f"x_train.shape ={x_train.shape}")
    log.info(f"x_test.shape ={x_test.shape}")

    
    model = def_models()
    history = train_model(model, x_train, y_train)

    pass
    model_path = path_join('../my_data/keras_model.keras')
    model.save(model_path)
    plot_metric(history, "loss")
    plot_metric(history, "AUC")
    # model = models.load_model(model_path)

    # 测试模型
    log.info('测试模型')
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    # print(tf.config.list_physical_devices('GPU'))
    main()
    pass
