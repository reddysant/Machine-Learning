import pandas as pd

def drop_columns(data, columns_del_list):
    data.drop(columns_del_list, axis=1, inplace=True)
#  Dealing with null or na
def replce_na(data):
    data['v2a1'].fillna(0, inplace=True)
    data['v18q1'].fillna(0, inplace=True)
    data['rez_esc'].fillna(0, inplace=True)
    data['meaneduc'].fillna(data['meaneduc'].median(), inplace=True)
def get_cat_cols(data):
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    cat_columns = list(set(cols) - set(num_cols))
    return cat_columns
def encode_data(data):
    data['edjefe'] = data['edjefe'].replace(['yes'], 1)
    data['edjefe'] = data['edjefe'].replace(['no'], 0)
    data['edjefa'] = data['edjefa'].replace(['yes'], 1)
    data['edjefa'] = data['edjefa'].replace(['no'], 0)
def train_test_split(data):
    targ_1 = data[data.Target==1].sample(frac=0.8)
    targ_2 = data[data.Target==2].sample(frac=0.8)
    targ_3 = data[data.Target==3].sample(frac=0.8)
    targ_4 = data[data.Target==4].sample(frac=0.8)
    train  = pd.concat([targ_1, targ_2, targ_3, targ_4])
    train = train.sample(frac=1) # Shuffle data
    test = data.loc[~data.index.isin(train.index)]
    return train, test