import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, \
    BatchNormalization, LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import time
import matplotlib.pyplot as plt
import math
from numpy.random import seed
# seed(2018)
from tensorflow import set_random_seed
# set_random_seed(2018)
np.random.seed(123)

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000
n_threads = 3


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def rmsle(y, y_pred):
    """
    code from https://www.kaggle.com/marknagelberg/rmsle-function
    :type y_pred: list
    """
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5


def fill_missing_value(df):
    # HANDLE MISSING VALUES
    print("start fill_na with proper values")

    df.category_name.fillna(value="missing/missing/missing", inplace=True)

    df["brand_name"] = df["brand_name"].fillna("unknown")
    pop_brands = df["brand_name"].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

    df.item_description.fillna(value="missing", inplace=True)

    print(df.memory_usage(deep=True))
    return df
    # df.brand_name.fillna(value="missing", inplace=True)


def category_split(df):
    def list_2_str(l):
        cat_name = ""
        for piece in l:
            cat_name += piece
        return cat_name

    # split category into 3 levels
    df['level1'] = [a.split("/")[0] for a in df['category_name']]
    df['level2'] = [a.split("/")[1] for a in df['category_name']]
    df['level3'] = [list_2_str(a.split("/")[2:]) for a in df['category_name']]
    # df = df.drop(['category_name'], axis=1)
    return df


# get name and description lengths
def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0


def label_encoder(train, test, col_name):
    # PROCESS CATEGORICAL DATA
    print("Handling categorical variables...")
    le = LabelEncoder()
    le.fit(np.hstack([train[col_name], test[col_name]]))
    train[col_name] = le.transform(train[col_name])
    test[col_name] = le.transform(test[col_name])

    return train, test


def flatten(l):
    return [item for sublist in l for item in sublist]


def _get_sentiment(data):
    sia = SentimentIntensityAnalyzer()
    scores = []
    for s in data:
        scores.append(sia.polarity_scores(s)['compound'])
    return np.array(scores)


def get_sentiment(data, mode="single"):
    if mode == "multi":
        p = Pool(processes=n_threads)
        n = math.ceil(len(data) / n_threads)
        scores = p.map(_get_sentiment, [data[i:i + n] for i in range(0, len(data), n)])
    if mode == "single":
        scores = []
        for i in range(len(data)):
            scores.append(_get_sentiment(data[i]))

    return np.array(flatten(scores))


def _get_lemma_desc(args):
    data, index = args
    lmtzr = WordNetLemmatizer()
    lemmas = []
    for s in data:
        words = word_tokenize(s)
        lemmas.append(' '.join([lmtzr.lemmatize(w).lower() for w in words if w.isalpha()]))
    return pd.Series(lemmas, index=index)


def get_lemma_desc(data, index, mode="single"):
    if mode == "multi":
        p = Pool(processes=n_threads)
        n = math.ceil(len(data) / n_threads)
        lemmas = p.map(_get_lemma_desc, [(data[i:i + n], index[i:i + n]) for i in range(0, len(data), n)])
    if mode == "single":
        lemmas = []
        for i in range(len(data)):
            lemmas.append(_get_lemma_desc((data[i], index[i])))
    return np.array(flatten(lemmas))


def get_re(key):
    re_pic = []
    # re_pic.append(r'(see(n)?) ( in| the| my) (picture(s)?|photo(s)?)')
    re_pic.append(r'(see|in|seen|) ( all| the| my| each) (picture(s)?|photo(s)?)')
    re_pic.append(r'(more|item|above|show on|same to|different from|clear(er)?|better) (picture(s)?|photo(s)?)')
    re_pic.append(r'(picture(s)?|photo(s)?) show(s|ed)')
    re_pic.append(r'as (picture(s|d)?|photo(s|ed)?)')
    re_pic.append(
        r'(1th|2nd|3rd|\dth|first|last|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) (picture(s)?|photo(s)?)')

    re_mat = []
    re_mat.append(r'material(s)')

    re_without_box = []
    re_without_box.append(r'(no|without)( a| any| original| the)? box')

    re_with_box = []
    re_with_box.append(r'(new in|with|has|have)( original| the)? box')

    re_ins = []
    re_ins.append(r'[^(with out|no)] (instruction(s)?|specification(s)?|receipt(s)?)')

    re_unused = []
    re_unused.append(r'(brand new|unopened)')
    re_unused.append(r'(never|haven\'t|hasn\'t|isn\'t|aren\'t)( been| be)? (used|opened)')

    re_new = []
    re_new.append(r'(only|barely|hardly) use(d)')
    re_new.append(r'use(d) once')

    if key == "pic":
        return re_pic
    if key == "mat":
        return re_mat
    if key == "without_box":
        return re_without_box
    if key == "with_box":
        return re_with_box
    if key == "ins":
        return re_ins
    if key == "unused":
        return re_unused
    if key == "new":
        return re_new


def may_used(descriptions):
    re_unused = get_re("unused")
    unused_re = [re.compile(r, re.IGNORECASE) for r in re_unused]

    re_new = get_re("new")
    new_re = [re.compile(r, re.IGNORECASE) for r in re_new]

    matches = []
    for desc in descriptions:
        match = 0
        for r in unused_re:
            if r.search(desc) is not None:
                match = 1
                continue
        for r in new_re:
            if r.search(desc) is not None:
                match = 0.5
                continue
        matches.append(match)
    return np.array(matches)


def may_have_ins(descriptions):
    re_ins = get_re("ins")
    ins_re = [re.compile(r, re.IGNORECASE) for r in re_ins]

    matches = []
    for desc in descriptions:
        match = 0
        for r in ins_re:
            if r.search(desc) is not None:
                match = 1
                continue
        matches.append(match)
    return np.array(matches)


def may_have_box(descriptions):
    re_box = get_re("with_box")
    box_re = [re.compile(r, re.IGNORECASE) for r in re_box]

    re_no_box = get_re("without_box")
    no_box_re = [re.compile(r, re.IGNORECASE) for r in re_no_box]

    matches = []
    for desc in descriptions:
        match = 0.5
        for r in no_box_re:
            if r.search(desc) is not None:
                match = 0
                continue
        for r in box_re:
            if r.search(desc) is not None:
                match = 1
                continue
        matches.append(match)
    return np.array(matches)


def may_have_materials(descriptions):
    re_mat = get_re("mat")
    mat_word_re = [re.compile(r, re.IGNORECASE) for r in re_mat]

    matches = []
    for desc in descriptions:
        match = 0
        for r in mat_word_re:
            if r.search(desc) is not None:
                match = 1
                continue
        matches.append(match)
    return np.array(matches)


def may_have_pictures(descriptions):
    re_pic = get_re("pic")
    pic_word_re = [re.compile(r, re.IGNORECASE) for r in re_pic]

    matches = []
    for desc in descriptions:
        match = 0
        for r in pic_word_re:
            if r.search(desc) is not None:
                match = 1
                continue
        matches.append(match)
    return np.array(matches)


def add_additional_feature(train, test):
    # Add picture
    train.loc[:, 'may_have_pictures'] = pd.Series(may_have_pictures(train.item_description), index=train.index).astype('category')
    test.loc[:, 'may_have_pictures'] = pd.Series(may_have_pictures(test.item_description), index=test.index).astype('category')

    # # Add box
    # train.loc[:, 'may_have_box'] = pd.Series(may_have_box(train.item_description), index=train.index).astype('category')
    # test.loc[:, 'may_have_box'] = pd.Series(may_have_box(test.item_description), index=test.index).astype('category')
    #
    # # Add inscription
    # train.loc[:, 'may_have_ins'] = pd.Series(may_have_ins(train.item_description), index=train.index).astype('category')
    # test.loc[:, 'may_have_ins'] = pd.Series(may_have_ins(test.item_description), index=test.index).astype('category')
    #
    # # Add picture
    # train.loc[:, 'may_used'] = pd.Series(may_used(train.item_description), index=train.index).astype('category')
    # test.loc[:, 'may_used'] = pd.Series(may_used(test.item_description), index=test.index).astype('category')
    #
    # # Add picture
    # train.loc[:, 'may_have_mat'] = pd.Series(may_have_materials(train.item_description), index=train.index).astype('category')
    # test.loc[:, 'may_have_mat'] = pd.Series(may_have_materials(test.item_description), index=test.index).astype('category')

    return train, test


def add_mean_category_price(train, test):
    mean_price1 = train.groupby(['level1'])['price'].mean()
    df_mp1 = pd.DataFrame(mean_price1)
    df_mp1.rename(columns={'price': 'mean_price1'}, inplace=True)
    df_mp1.reset_index(inplace=True)
    train = train.join(df_mp1.set_index('level1'), on='level1')
    test = test.join(df_mp1.set_index('level1'), on='level1')

    mean_price2 = train.groupby(['level1', 'level2'])['price'].mean()
    df_mp2 = pd.DataFrame(mean_price2)
    df_mp2.rename(columns={'price': 'mean_price2'}, inplace=True)
    df_mp2.reset_index(inplace=True)
    train = train.join(df_mp2.set_index(['level1', 'level2']), on=['level1', 'level2'])
    test = test.join(df_mp2.set_index(['level1', 'level2']), on=['level1', 'level2'])

    mean_price3 = train.groupby(['level1', 'level2', 'level3'])['price'].mean()
    df_mp3 = pd.DataFrame(mean_price3)
    df_mp3.rename(columns={'price': 'mean_price3'}, inplace=True)
    df_mp3.reset_index(inplace=True)
    train = train.join(df_mp3.set_index(['level1', 'level2', 'level3']), on=['level1', 'level2', 'level3'])
    test = test.join(df_mp3.set_index(['level1', 'level2', 'level3']), on=['level1', 'level2', 'level3'])

    test.loc[np.isnan(test['mean_price2']) == True, 'mean_price2'] = test.loc[np.isnan(test['mean_price2']) == True, 'mean_price1']
    test.loc[np.isnan(test['mean_price3']) == True, 'mean_price3'] = test.loc[np.isnan(test['mean_price3']) == True, 'mean_price2']

    idx_split = len(train.level1)
    scaler = MinMaxScaler()
    all_scaled = scaler.fit_transform(np.concatenate([train['mean_price1'].values.reshape(-1, 1), test['mean_price1'].values.reshape(-1, 1)]))
    train['mean_price1'], test['mean_price1'] = all_scaled[:idx_split], all_scaled[idx_split:]

    all_scaled = scaler.fit_transform(np.concatenate([train['mean_price2'].values.reshape(-1, 1), test['mean_price2'].values.reshape(-1, 1)]))
    train['mean_price2'], test['mean_price2'] = all_scaled[:idx_split], all_scaled[idx_split:]

    all_scaled = scaler.fit_transform(np.concatenate([train['mean_price3'].values.reshape(-1, 1), test['mean_price3'].values.reshape(-1, 1)]))
    train['mean_price3'], test['mean_price3'] = all_scaled[:idx_split], all_scaled[idx_split:]

    return train, test


def data_preprocessing(train, test):
    # fill nan value
    train = fill_missing_value(train)
    test = fill_missing_value(test)
    print("filling missing value complete")

    # transform descriptions and names to lower cases
    train.loc[:, 'item_description'] = train.item_description.str.lower()
    test.loc[:, 'item_description'] = test.item_description.str.lower()
    train.loc[:, 'name'] = train.name.str.lower()
    test.loc[:, 'name'] = test.name.str.lower()
    train.loc[:, 'category_name'] = train.category_name.str.lower()
    test.loc[:, 'category_name'] = test.category_name.str.lower()
    print("Text to lower cases finished")

    return train, test


# KERAS DATA DEFINITION
def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        , 'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        , 'brand_name': np.array(dataset.brand_name)
        # , 'category_name': pad_sequences(dataset.seq_category_name, maxlen=MAX_CATE_SEQ)
        , 'category1_name': np.array(dataset.level1)
        , 'category2_name': np.array(dataset.level2)
        , 'category3_name': np.array(dataset.level3)
        , 'item_condition': np.array(dataset.item_condition_id)
        # , 'num_vars': np.array(dataset[["shipping", "desc_len", "may_have_pictures", "may_have_box", "may_have_ins",
        #                                 "may_used", "may_have_mat",
        #                                 # "desc_sentiment",
        #                                 "mean_price1", "mean_price2", "mean_price3"]])
        , 'shipping': np.array(dataset[["shipping"]])
        , 'desc_len': np.array(dataset[["desc_len"]])
        # , 'name_len': np.array(dataset[["name_len"]])
        # , 'may_have_vars': np.array(dataset[["may_have_pictures", "may_have_box", "may_have_ins", "may_used", "may_have_mat",]])
        # , 'price_vars': np.array(dataset[["mean_price1", "mean_price2", "mean_price3"]])
    }
    return X


def get_model():
    # params
    dr_r = 0.1

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    # category_name = Input(shape=[X_train["category_name"].shape[1]], name="category_name")
    category1_name = Input(shape=[1], name="category1_name")
    category2_name = Input(shape=[1], name="category2_name")
    category3_name = Input(shape=[1], name="category3_name")
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[1], name="shipping")
    desc_len = Input(shape=[1], name="desc_len")
    # name_len = Input(shape=[1], name="name_len")

    # num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    # may_have_vars = Input(shape=[X_train["may_have_vars"].shape[1]], name="may_have_vars")
    # price_vars = Input(shape=[X_train["price_vars"].shape[1]], name="price_vars")

    # Embeddings layers
    emb_size = 60
    emb_name = Embedding(MAX_TEXT, emb_size//3)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, emb_size//6)(brand_name)
    # emb_category_name = Embedding(MAX_TEXT, emb_size//3)(category_name)
    emb_category1_name = Embedding(MAX_CATEGORY1, 10)(category1_name)
    emb_category2_name = Embedding(MAX_CATEGORY2, 10)(category2_name)
    emb_category3_name = Embedding(MAX_CATEGORY3, 10)(category3_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_shipping = Embedding(MAX_CONDITION, 5)(shipping)
    emb_desc_len = Embedding(MAX_CONDITION, 5)(desc_len)

    # gru layer
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)
    # rnn_layer3 = GRU(8)(emb_category_name)

    # main layer
    main_l = concatenate([Flatten()(emb_brand_name),
                          Flatten()(emb_category1_name),
                          Flatten()(emb_category2_name),
                          Flatten()(emb_category3_name),
                          Flatten()(emb_item_condition),
                          Flatten()(emb_shipping),
                          Flatten()(emb_desc_len),
                          rnn_layer1,
                          rnn_layer2,
                          # rnn_layer3,
                          # num_vars
                          # , may_have_vars
                          # , price_vars
                          ])

    # main_l = Dropout(dr_r)(Dense(256)(main_l))
    main_l = Dropout(dr_r)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(dr_r)(Dense(256, activation='relu')(main_l))
    main_l = Dropout(dr_r)(Dense(128, activation='relu')(main_l))
    main_l = Dropout(dr_r)(Dense(64, activation='relu')(main_l))
    # main_l = Dropout(dr_r)(Dense(16, activation='relu')(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model_gru = Model([name, item_desc, brand_name,
                       # category_name,
                       category1_name, category2_name, category3_name,
                       item_condition
                       ,shipping
                       ,desc_len
                       # ,num_vars
                       # ,may_have_vars
                       #    , price_vars
                       ], output)
    model_gru.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])
    return model_gru


# LOAD DATA

timettl1 = time.time()
# path = "H:/Project/PriceSuggestion/"
path = "D:/Project/Price/"
train = pd.read_table(path + "train.tsv", sep=None, engine='python')
test = pd.read_table(path + "test.tsv", sep=None, engine='python')
# train = pd.read_table("../input/train.tsv", sep=None, engine='python')
# test = pd.read_table("../input/test.tsv", sep=None, engine='python')
print(train.shape)
print(test.shape)
time1 = time.time()


# drop low price data  (ADDNEW)
train = train.drop(train[(train.price < 3.0)].index)
train, test = data_preprocessing(train, test)


# split category to 3 levels
train = category_split(train)
test = category_split(test)
print("category spliting complete")
time2 = time.time()

# add mean price for 3 categories
train, test = add_mean_category_price(train, test)
time3 = time.time()

# Lemmatize
# train['desc_lemmas'] = get_lemma_desc(train.item_description, train.index)
# print("Train lemmalization finished")
time4 = time.time()
# test['desc_lemmas'] = get_lemma_desc(test.item_description, test.index)
# print("Test lemmalization finished")
time5 = time.time()

# label encoder
train, test = label_encoder(train, test, "level1")
train, test = label_encoder(train, test, "level2")
train, test = label_encoder(train, test, "level3")
train, test = label_encoder(train, test, "brand_name")
# train, test = label_encoder(train, test, "category_name")
print("Label encoding finished")
time6 = time.time()

# PROCESS TEXT: RAW
# raw_text = np.hstack([train.desc_lemmas, test.desc_lemmas, train.name, test.name])
raw_text = np.hstack([train.item_description, test.item_description, train.name, test.name])
print("Text to seq process finished")
time7 = time.time()

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("Fitting tokenizer finished")
time8 = time.time()

# train["seq_item_description"] = tok_raw.texts_to_sequences(train.desc_lemmas)
# test["seq_item_description"] = tok_raw.texts_to_sequences(test.desc_lemmas)
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description)
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description)
train["seq_name"] = tok_raw.texts_to_sequences(train.name)
test["seq_name"] = tok_raw.texts_to_sequences(test.name)
# train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name)
# test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name)
# train.head(3)
print("texts_to_sequences finished")
time9 = time.time()

# SEQUENCES VARIABLES ANALYSIS
max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(test.seq_name.apply(lambda x: len(x)))])
max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x))),
                                   np.max(test.seq_item_description.apply(lambda x: len(x)))])
print("max name seq " + str(max_name_seq))
print("max item desc seq " + str(max_seq_item_description))
time10 = time.time()

# Add description length
# idx_split = len(train.item_description)
# train['desc_len'] = np.array(list(map(lambda d: len(d), train['seq_item_description'])))
# test['desc_len'] = np.array(list(map(lambda d: len(d), test['seq_item_description'])))
# scaler = MinMaxScaler()
# all_scaled = scaler.fit_transform(np.concatenate([train['desc_len'].values.reshape(-1, 1),
#                                                   test['desc_len'].values.reshape(-1, 1)]))
# train['desc_len'], test['desc_len'] = all_scaled[:idx_split], all_scaled[idx_split:]
# print("desc length calculation finish")
# time11 = time.time()

train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))
# train['name_len'] = train['name'].apply(lambda x: wordCount(x))
# test['name_len'] = test['name'].apply(lambda x: wordCount(x))

idx_split = len(train.item_description)
scaler = MinMaxScaler()
all_scaled = scaler.fit_transform(np.concatenate([train['desc_len'].values.reshape(-1, 1), test['desc_len'].values.reshape(-1, 1)]))
train['desc_len'], test['desc_len'] = all_scaled[:idx_split], all_scaled[idx_split:]
# all_scaled = scaler.fit_transform(np.concatenate([train['name_len'].values.reshape(-1, 1), test['name_len'].values.reshape(-1, 1)]))
# train['name_len'], test['name_len'] = all_scaled[:idx_split], all_scaled[idx_split:]

print("desc length calculation finish")
time11 = time.time()

# Add sentiment scores
# train['desc_sentiment'] = get_sentiment(train.item_description, mode="single")
# print('item description sentiment for train set finished.')
time12 = time.time()

# test['desc_sentiment'] = get_sentiment(test.item_description, mode="single")
# print('item description sentiment for test set finished.')
time13 = time.time()

# train, test = add_additional_feature(train, test)
# print('add_additional_feature finished.')
time14 = time.time()

print("time 1 {0}".format(time2-time1))
print("time 2 {0}".format(time3-time2))
print("time 3 {0}".format(time4-time3))
print("time 4 {0}".format(time5-time4))
print("time 5 {0}".format(time6-time5))
print("time 6 {0}".format(time7-time6))
print("time 7 {0}".format(time8-time7))
print("time 8 {0}".format(time9-time8))
print("time 9 {0}".format(time10-time9))
print("time 10 {0}".format(time11-time10))
print("time 11 {0}".format(time12-time11))
print("time 12 {0}".format(time13-time12))
print("time 13 {0}".format(time14-time13))


# time 1 6.036165952682495
# time 2 4.854521036148071
# time 3 283.12333250045776
# time 4 133.56249594688416
# time 5 37.67669367790222
# time 6 0.37105298042297363
# time 7 92.57919239997864
# time 8 64.97585272789001
# time 9 1.916412591934204
# time 10 0.6348559856414795
# time 11 170.5714032649994
# time 12 77.60580372810364
# time 13 339.11557030677795

# EMBEDDINGS MAX VALUE
# Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_CATE_SEQ = 8
# MAX_TEXT = np.unique(flatten(np.concatenate([train.seq_item_description, test.seq_item_description, test.seq_name, train.seq_name, test.seq_category_name, train.seq_category_name]))).shape[0] + 1
MAX_TEXT = np.unique(flatten(np.concatenate([train.seq_item_description, test.seq_item_description, train.seq_name, test.seq_name]))).shape[0] + 1
MAX_CATEGORY1 = np.unique(np.concatenate([train.level1, test.level1])).shape[0] + 1
MAX_CATEGORY2 = np.unique(np.concatenate([train.level2, test.level2])).shape[0] + 1
MAX_CATEGORY3 = np.unique(np.concatenate([train.level3, test.level3])).shape[0] + 1
MAX_BRAND = np.unique(np.concatenate([train.brand_name, test.brand_name])).shape[0] + 1
MAX_CONDITION = np.unique(np.concatenate([train.item_condition_id, test.item_condition_id])).shape[0] + 1

# SCALE target variable
train["target"] = np.log(train.price + 1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train["target"] = target_scaler.fit_transform(train.target.values.reshape(-1, 1))

# train.to_csv(path + "clean_train.csv", encoding="utf8")

# EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)

# FITTING THE MODEL
BATCH_SIZE = 512*3
epochs = 2

model = get_model()
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01, mode='min')
model.fit(X_train, dtrain.target, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(X_valid, dvalid.target),
          verbose=1)

print('model fitting finished.')
time1 = time.time()

# EVLUEATE THE MODEL ON DEV TEST: What is it doing?
val_preds = model.predict(X_valid)
val_preds = target_scaler.inverse_transform(val_preds)
val_preds = np.exp(val_preds) + 1
print(len(val_preds))
print('val predict calculated.')
time2 = time.time()

# mean_absolute_error, mean_squared_log_error
y_true = np.array(dvalid.price.values)
y_pred = val_preds[:, 0]
v_rmsle = rmsle(y_true, y_pred)
print(" RMSLE error on dev test: " + str(v_rmsle))
time3 = time.time()

# CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = target_scaler.inverse_transform(preds)
preds = np.exp(preds) - 1
print(len(preds))
print('test predict calculated.')
time4 = time.time()

submission = pd.DataFrame({'test_id': test["test_id"]})
submission.loc[:, "price"] = preds
print('test dataframe complete.')
time5 = time.time()

# submission.to_csv(path + "submission.csv", index=False)
submission.to_csv("./submission.csv", index=False)
print('file saved')
time6 = time.time()
print("time 1{0}".format(time2-time1))
print("time 2{0}".format(time3-time2))
print("time 3{0}".format(time4-time3))
print("time 4{0}".format(time5-time4))
print("time 5{0}".format(time6-time5))

timettl2 = time.time()
print(timettl2 - timettl1)


preds_train = model.predict(X_train, batch_size=BATCH_SIZE)
preds_valid = model.predict(X_valid, batch_size=BATCH_SIZE)
preds_train = target_scaler.inverse_transform(preds_train)
preds_valid = target_scaler.inverse_transform(preds_valid)
preds_train = np.exp(preds_train) - 1
preds_valid = np.exp(preds_valid) - 1

dtrain.loc[:, "price_rnn"] = preds_train
dvalid.loc[:, "price_rnn"] = preds_valid
test.loc[:, "price_rnn"] = preds

dtrain.to_csv("./xgbtrain.csv", index=False)
dvalid.to_csv("./xgbval.csv", index=False)
test.to_csv("./xgbtest.csv", index=False)


dtrain = pd.read_csv("./xgbtrain.csv")
dvalid = pd.read_csv("./xgbval.csv")
test = pd.read_csv("./xgbtest.csv")

xgb_train = dtrain["level1", "level2", "level3", "brand_name", "desc_len", "item_condition_id", "shipping", "price_rnn"]
xgb_train_y = dtrain['target']
xgb_valid = dvalid["level1", "level2", "level3", "brand_name", "desc_len", "item_condition_id", "shipping", "price_rnn"]
xgb_valid_y = dvalid['target']
xgb_test = test["level1", "level2", "level3", "brand_name", "desc_len", "item_condition_id", "shipping", "price_rnn"]

d_train = xgb.DMatrix(xgb_train, xgb_train_y)
d_valid = xgb.DMatrix(xgb_valid, xgb_valid_y)
d_test1 = xgb.DMatrix(xgb_valid)
d_test2 = xgb.DMatrix(xgb_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

params = {
    'booster': 'gbtree',
    'min_child_weight': 7,  # CV   # 越高越可以避免overfitting，其值需要随着max_depth增大而增大。其本身大小也和train set size有关。
    'objective': 'binary:logistic',  # 0/1结果准确率
    'eval_metric': 'logloss',
    'max_depth': 8,  # CV  # 越高越容易overfitting，树越深越容易受到噪音、错误数据、样本误差的干扰
    # 'max_delta_step': 1.8,  # CV
    'max_delta_step': 3,  # CV  5
    'colsample_bytree': 0.8,
    'subsample': 0.8,  # CV
    'eta': 0.025,
    'gamma': 1,  # CV
    # 'scale_pos_weight': 2,
    'alpha': 3,
    'lambda': 3,
    'seed': 44
    }


mdl = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=20, verbose_eval=1)
joblib.dump(mdl, "./submission_xgb.csv", compress=3)

# 1466844/1466844 [==============================] - 586s 400us/step - loss: 0.0186 - mean_absolute_error: 0.1033 - rmsle_cust: 0.0043 - val_loss: 0.0202 - val_mean_absolute_error: 0.1073 - val_rmsle_cust: 0.0041
# RMSLE error on dev test: 0.45720729563135387
# online 0.44947

# 采用了新的长度计算方式，运行了一个epoch，大约在13min左右
# 1465344/1466844 [============================>.] - ETA: 0s - loss: 0.0791 - mean_absolute_error: 0.1863 - rmsle_cust: 0.0071
# 1466844/1466844 [==============================] - 645s 440us/step - loss: 0.0790 - mean_absolute_error: 0.1862 - rmsle_cust: 0.0071 - val_loss: 0.0248 - val_mean_absolute_error: 0.1207 - val_rmsle_cust: 0.0048