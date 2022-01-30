import re

import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from urlextract import URLExtract


class ClassImbalanceCorrection:
    def __init__(self):
        return

    def correct_class_imabalance(self, train, labels, num_classes=2, ids_train=None, is_df=False):
        classes = range(num_classes)
        smaller_class_sample_size = 1e9
        smaller_class = 0
        for classification_class in classes:
            num_samples_in_class = labels[labels == classification_class].shape[0]
            if num_samples_in_class < smaller_class_sample_size:
                smaller_class_sample_size = num_samples_in_class
                smaller_class = classification_class

        new_train = train[labels == smaller_class].copy()
        new_labels = labels[labels == smaller_class].copy()

        if ids_train is not None:
            new_ids_train = ids_train[labels == smaller_class].copy()

        for classification_class in classes:
            if is_df:
                if classification_class != smaller_class:
                    new_train = pd.concat([
                        new_train.copy(),
                        train.loc[labels == classification_class][:smaller_class_sample_size].copy()],
                        axis=0)
                    new_labels = pd.concat([
                        new_labels.copy(),
                        labels.loc[labels == classification_class][:smaller_class_sample_size].copy()],
                        axis=0)

                    if ids_train is not None:
                        new_ids_train = pd.concat([
                            new_ids_train.copy(),
                            ids_train.loc[labels == classification_class][:smaller_class_sample_size].copy()],
                            axis=0)
                else:
                    continue
            else:
                if classification_class != smaller_class:
                    new_train = np.concatenate([
                        new_train, train[labels == classification_class][:smaller_class_sample_size]],
                        axis=0)
                    new_labels = np.concatenate([
                        new_labels,
                        labels[labels == classification_class][:smaller_class_sample_size]],
                        axis=0)
                else:
                    continue

        return new_train, pd.DataFrame(new_labels), ids_train


class TextSantization:
    def __init__(self):
        self.extractor = URLExtract()
        self.lemmatizer = WordNetLemmatizer()
        return

    def __lower_casing(self, text):
        return text.lower()

    def __replace_usernames(self, text):
        regex_user = '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'
        replace_flag = 'USER'
        usernames = re.findall(regex_user, text)

        for username in usernames:
            text = text.replace(username, replace_flag)

        return text

    def __process_special_symbols(self, text):
        text = text.replace('<br />', '')
        text = text.replace('...', ' ')
        text = text.replace('.', ' . ')
        text = text.replace('?', ' ? ')
        text = text.replace(':', ' : ')
        text = text.replace(';', ' ; ')
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('!', ' ? ')
        text = text.replace("'", "")
        text = text.replace('"', "")

        text = ' '.join([val.strip() for val in text.split() if val != ' '])

        return text

    def __replace_urls(self, text):
        url_flag = 'URL'
        urls = self.extractor.find_urls(text)
        for url in urls:
            text = text.replace(url, url_flag)

        return text

    def __remove_stopwords(self, text):
        text = ' '.join([val for val in text.split() if val not in stopwords.words('english')])
        return text

    def __lemmatize(self, text):
        text = ' '.join([self.lemmatizer.lemmatize(val, pos='n') for val in text.split()])
        text = ' '.join([self.lemmatizer.lemmatize(val, pos='a') for val in text.split()])
        text = ' '.join([self.lemmatizer.lemmatize(val, pos='v') for val in text.split()])
        text = ' '.join([self.lemmatizer.lemmatize(val, pos='r') for val in text.split()])
        text = ' '.join([self.lemmatizer.lemmatize(val, pos='s') for val in text.split()])
        return text

    def sanitize_text(self, texts, is_df=False):
        for idx in range(len(texts)):
            if is_df:
                text = texts.loc[idx]
            else:
                text = texts[idx]

            text = self.__process_special_symbols(text)
            text = self.__lower_casing(text)
            text = self.__replace_urls(text)
            text = self.__replace_usernames(text)
            text = self.__remove_stopwords(text)
            text = self.__lemmatize(text)

            if is_df:
                texts.loc[idx] = text
            else:
                texts[idx] = text

        return texts


class CustomTokenizer(TextSantization, ClassImbalanceCorrection):
    def __init__(self,
                 train: pd.DataFrame,
                 test: pd.DataFrame):

        super(CustomTokenizer, self).__init__()

        self.train = train
        self.test = test
        self.tokenizer = TweetTokenizer()
        return

    def __tokenize(self, texts):
        tokenized = []
        max_length = 0
        for idx in range(texts.size):
            text = texts.loc[idx].lower()
            tokenized_tweet = self.tokenizer.tokenize(text)
            if max_length < len(tokenized_tweet):
                max_length = len(tokenized_tweet)
            tokenized.append(tokenized_tweet)

        return tokenized, max_length

    def __tokenize_train(self):
        ids = self.train['id'].tolist()
        texts = self.train['text']
        labels = self.train['target']

        texts = self.sanitize_text(texts.copy(), True)

        tokenized_train, max_length = self.__tokenize(texts)
        return ids, tokenized_train, labels, max_length

    def __tokenize_test(self):
        ids = self.test['id'].tolist()
        texts = self.test['text']

        texts = self.sanitize_text(texts.copy(), True)

        tokenized_test, max_length = self.__tokenize(texts)
        return ids, tokenized_test, max_length

    def __padding(self, tokenized, max_length, pad_token='[PAD]'):
        for idx in range(len(tokenized)):
            tokenized_tweet = tokenized[idx]
            diff = max_length - len(tokenized_tweet)
            if diff > 0:
                to_pad = [pad_token] * diff
                tokenized_tweet.extend(to_pad)
            tokenized[idx] = tokenized_tweet

        return tokenized

    def tokenizer_func(self, pad_token='[PAD]', save_path=None):
        ids_train, tokenized_train, labels, max_length_train = self.__tokenize_train()
        ids_test, tokenized_test, max_length_test = self.__tokenize_test()

        if max_length_train > max_length_test:
            tokenized_train = self.__padding(tokenized_train, max_length_train, pad_token=pad_token)
            tokenized_test = self.__padding(tokenized_test, max_length_train, pad_token=pad_token)
        else:
            tokenized_train = self.__padding(tokenized_train, max_length_test, pad_token=pad_token)
            tokenized_test = self.__padding(tokenized_test, max_length_test, pad_token=pad_token)

        train = pd.DataFrame(tokenized_train)
        test = pd.DataFrame(tokenized_test)

        train, labels, ids_train = self.correct_class_imabalance(train.copy(),
                                                                 labels.copy(), ids_train=pd.DataFrame(ids_train),
                                                                 is_df=True)

        train.insert(0, column='id', value=ids_train)
        labels.insert(0, column='id', value=ids_train)
        test.insert(0, column='id', value=ids_test)

        if save_path is not None:
            train.to_csv(save_path + 'train.csv', index=False)
            labels.to_csv(save_path + 'labels.csv', index=False)
            test.to_csv(save_path + 'test.csv', index=False)

        return train, labels, test


class TokenToId:
    def __init__(self):
        self.train = None
        self.test = None
        self.vocab = {}
        return

    def token_to_id_from_path(self, train_path, test_path):
        self.train = pd.read_csv(train_path).to_numpy()
        self.test = pd.read_csv(test_path).to_numpy()

        train, test = self.__get_token_ids()
        print("Vocab Length is:", len(self.vocab.keys()))
        return train, test

    def token_to_id_from_df(self, train, test):
        self.train = train.to_numpy()[:, 1:]
        self.test = test.to_numpy()[:, 1:]

        train, test = self.__get_token_ids()
        return train, test

    def __build_vocab(self):
        n_rows_train = self.train.shape[0]
        n_rows_test = self.test.shape[0]
        n_cols = self.train.shape[1]
        curr_id = 0

        for row_idx in range(n_rows_train):
            for col_idx in range(n_cols):
                token = self.train[row_idx, col_idx]
                if token not in self.vocab.keys():
                    self.vocab[token] = curr_id
                    curr_id += 1
                else:
                    continue

        for row_idx in range(n_rows_test):
            for col_idx in range(n_cols):
                token = self.test[row_idx, col_idx]
                if token not in self.vocab.keys():
                    self.vocab[token] = curr_id
                    curr_id += 1
                else:
                    continue

    def __token_to_id(self):
        n_rows_train = self.train.shape[0]
        n_rows_test = self.test.shape[0]
        n_cols = self.train.shape[1]

        train_ = np.zeros(shape=self.train.shape, dtype=np.int)
        test_ = np.zeros(shape=self.test.shape, dtype=np.int)

        for row_idx in range(n_rows_train):
            for col_idx in range(n_cols):
                token = self.train[row_idx, col_idx]
                token_id = self.vocab[token]
                train_[row_idx, col_idx] = token_id

        for row_idx in range(n_rows_test):
            for col_idx in range(n_cols):
                token = self.test[row_idx, col_idx]
                token_id = self.vocab[token]
                test_[row_idx, col_idx] = token_id

        return train_, test_

    def __get_token_ids(self):
        self.__build_vocab()
        train_, test_ = self.__token_to_id()
        return train_, test_


class BERTTransformerTokenizer(TextSantization, ClassImbalanceCorrection):
    def __init__(self,
                 train: pd.DataFrame,
                 test: pd.DataFrame,
                 max_length=128,
                 pretrained_model='vinai/bertweet-large'
                 ):
        super(BERTTransformerTokenizer, self).__init__()

        # labels = train['target'].copy()
        # self.train, _, _ = self.correct_class_imabalance(train, labels, train['id'], is_df=True)

        self.train = train
        self.test = test
        self.max_length = max_length

        self.bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_model, normalization=True, use_fast=False)

    def __tokenize_train(self):
        ids = self.train['id'].tolist()
        texts = self.train['text'].tolist()
        labels = self.train['target'].tolist()

        texts = self.sanitize_text(texts)

        train_tokenized = self.bert_tokenizer(texts, padding='max_length', max_length=self.max_length)

        input_ids = train_tokenized['input_ids']
        # token_type_ids = train_tokenized['token_type_ids']
        attention_mask = train_tokenized['attention_mask']

        return ids, input_ids, attention_mask, labels

    def __tokenize_test(self):
        ids = self.train['id'].tolist()
        texts = self.train['text'].tolist()

        texts = self.sanitize_text(texts)

        test_tokenized = self.bert_tokenizer(texts, padding='max_length', max_length=self.max_length)
        input_ids = test_tokenized['input_ids']
        # token_type_ids = test_tokenized['token_type_ids']
        attention_mask = test_tokenized['attention_mask']

        return ids, input_ids, attention_mask

    def tokenize(self):
        train_ids, train_input_ids, train_attention_mask, labels = self.__tokenize_train()
        test_ids, test_input_ids, test_attention_mask = self.__tokenize_test()

        return train_ids, train_input_ids, train_attention_mask, labels, \
               test_ids, test_input_ids, test_attention_mask

    def tokenize_to_pretrain(self, train_texts, test_texts):
        train_texts = self.sanitize_text(texts=train_texts)
        test_texts = self.sanitize_text(texts=test_texts)

        train_tokenized = self.bert_tokenizer(train_texts, padding='max_length', max_length=self.max_length,
                                              truncation=True)
        test_tokenized = self.bert_tokenizer(test_texts, padding='max_length', max_length=self.max_length,
                                             truncation=True)

        train_input_ids = train_tokenized['input_ids']
        train_attention_mask = train_tokenized['attention_mask']

        test_input_ids = test_tokenized['input_ids']
        test_attention_mask = test_tokenized['attention_mask']

        return train_input_ids, train_attention_mask, test_input_ids, test_attention_mask
