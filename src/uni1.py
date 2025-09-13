from logger import log
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def read_csv(path: str):
    '''
    读取csv文件
    '''
    return pd.read_csv(os.path.join(os.path.dirname(__file__), path))


def show_data():
    '''
    展示数据
    '''
    train_csv = read_csv('../data/titanic/train.csv')
    dftrain_raw = train_csv.head(10)
    log.info(dftrain_raw)
    ax = train_csv['Survived'].value_counts().plot(kind='bar', figsize=(12, 8), fontsize=15, rot=0)
    ax.set_ylabel('Counts', fontsize=15)
    ax.set_xlabel('Survived', fontsize=15)
    plt.show()


def main():
    show_data()


if __name__ == "__main__":
    main()
