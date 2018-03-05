import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = 'data/train_data.json'

SEED = 304


if __name__ == '__main__':
    train_data = pd.read_json(TRAIN_DATA_PATH)

    train_split, cv_split = train_test_split(train_data, test_size=0.1, random_state=SEED)

    print('training split')
    print(train_split.head())
    print('cross validation split')
    print(cv_split.head())

    train_split.to_json('data/train_data_train_split.json')
    cv_split.to_json('data/train_data_cv_split.json')