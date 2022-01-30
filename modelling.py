import torch
import yaml
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification


class LSTMModel(torch.nn.Module):
    def __init__(
            self
    ):

        super(LSTMModel, self).__init__()

        with open('config/model_config.yaml', 'r') as train_config_stream:
            train_config = yaml.safe_load(train_config_stream)

        lstmmodel_config = train_config['lstmmodel']

        self.embedding_module = torch.nn.Embedding(
            num_embeddings=lstmmodel_config['embedding_module']['num_embedding'],
            embedding_dim=lstmmodel_config['embedding_module']['embedding_dim']
        )
        if lstmmodel_config['lstm_module']['num_layers'] == 1:
            dropout_lstm = 0
        else:
            dropout_lstm = lstmmodel_config['lstm_module']['dropout']

        self.lstm_module = torch.nn.LSTM(
            input_size=lstmmodel_config['lstm_module']['input_size'],
            hidden_size=lstmmodel_config['lstm_module']['hidden_size'],
            num_layers=lstmmodel_config['lstm_module']['num_layers'],
            batch_first=True,
            dropout=dropout_lstm,
        )

        linear_features = lstmmodel_config['linear_module']['hidden_features']
        linear_features.insert(0, lstmmodel_config['linear_module']['in_features'])
        linear_features.append(lstmmodel_config['linear_module']['out_features'])
        num_linear_layers = len(linear_features)

        self.linear_module = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=linear_features[idx],
                out_features=linear_features[idx+1]
            )
            for idx in range(num_linear_layers - 1)
        ])

        if lstmmodel_config['linear_module']['activation'] == 'relu':
            self.linear_activation = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(lstmmodel_config['linear_module']['dropout'])

        return

    def forward(self, x):
        x = self.embedding_module(x)
        x, _ = self.lstm_module(x)
        for _, layer in enumerate(self.linear_module):
            x = layer(x)
            x = self.linear_activation(x)
            x = self.dropout(x)

        x = torch.mean(x, dim=1)
        return x


class BERTTweetModel(torch.nn.Module):
    def __init__(self):
        super(BERTTweetModel, self).__init__()
        self.bert_tweet_model = AutoModel.from_pretrained('vinai/bertweet-large')

        with open('config/model_config.yaml', 'r') as train_config_stream:
            train_config = yaml.safe_load(train_config_stream)

        berttweet_config = train_config['berttweet_config']

        linear_features = berttweet_config['linear_module']['hidden_features']
        linear_features.insert(0, berttweet_config['linear_module']['in_features'])
        linear_features.append(berttweet_config['linear_module']['out_features'])
        num_linear_layers = len(linear_features)

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=[128, 1024])

        self.linear_module = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=linear_features[idx],
                out_features=linear_features[idx+1]
            )
            for idx in range(num_linear_layers - 1)
        ])

        if berttweet_config['linear_module']['activation'] == 'relu':
            self.linear_activation = torch.nn.ReLU()

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        self.dropout = torch.nn.Dropout(berttweet_config['linear_module']['dropout'])

    def forward(self, x, attention_mask=None):
        x = self.bert_tweet_model(x, attention_mask=attention_mask)
        x = x.last_hidden_state
        x = self.layer_norm(x)
        for _, layer in enumerate(self.linear_module):
            x = layer(x)
            x = self.linear_activation(x)
            x = self.dropout(x)

        x = self.log_softmax(x)
        x = torch.mean(x, dim=1)
        return x


class FrozenBERTModel(torch.nn.Module):
    def __init__(self):
        super(FrozenBERTModel, self).__init__()

        with open('config/model_config.yaml', 'r') as train_config_stream:
            train_config = yaml.safe_load(train_config_stream)

        frozen_bert_config = train_config['frozen_bert']

        self.bert_tweet_model = AutoModel.from_pretrained(frozen_bert_config['bert_module']['pretrained_model'])

        if frozen_bert_config['lstm_module']['num_layers'] == 1:
            dropout_lstm = 0
        else:
            dropout_lstm = frozen_bert_config['lstm_module']['dropout']

        self.lstm_module = torch.nn.LSTM(
            input_size=frozen_bert_config['lstm_module']['input_size'],
            hidden_size=frozen_bert_config['lstm_module']['hidden_size'],
            num_layers=frozen_bert_config['lstm_module']['num_layers'],
            batch_first=True,
            dropout=dropout_lstm,
        )

        linear_features = frozen_bert_config['linear_module']['hidden_features']
        linear_features.insert(0, frozen_bert_config['linear_module']['in_features'])
        linear_features.append(frozen_bert_config['linear_module']['out_features'])
        num_linear_layers = len(linear_features)

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=[128, 1024])

        self.linear_module = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=linear_features[idx],
                out_features=linear_features[idx+1]
            )
            for idx in range(num_linear_layers - 1)
        ])

        if frozen_bert_config['linear_module']['activation'] == 'relu':
            self.linear_activation = torch.nn.ReLU()
        elif frozen_bert_config['linear_module']['activation'] == 'leakyrelu':
            self.linear_activation = torch.nn.LeakyReLU()

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        self.dropout = torch.nn.Dropout(frozen_bert_config['linear_module']['dropout'])

    def forward(self, x, attention_mask=None):
        with torch.no_grad():
            x = self.bert_tweet_model(x, attention_mask=attention_mask).last_hidden_state

        x, _ = self.lstm_module(x)
        x = self.layer_norm(x)

        for _, layer in enumerate(self.linear_module):
            x = layer(x)
            x = self.linear_activation(x)
            x = self.dropout(x)

        x = torch.mean(x, dim=1)
        return x


class BERTRegModel(torch.nn.Module):

    def __init__(self):
        super(BERTRegModel, self).__init__()

        with open('config/model_config.yaml', 'r') as train_config_stream:
            train_config = yaml.safe_load(train_config_stream)

        bert_reg_config = train_config['bert_reg']

        self.bert_tweet_model = AutoModel.from_pretrained(bert_reg_config['bert_module']['pretrained_model'])

        linear_features = bert_reg_config['linear_module']['hidden_features']
        linear_features.insert(0, bert_reg_config['linear_module']['in_features'])
        linear_features.append(bert_reg_config['linear_module']['out_features'])
        num_linear_layers = len(linear_features)

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=[512, 768])

        self.linear_module = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=linear_features[idx],
                out_features=linear_features[idx + 1]
            )
            for idx in range(num_linear_layers - 1)
        ])

        if bert_reg_config['linear_module']['activation'] == 'relu':
            self.linear_activation = torch.nn.ReLU()
        elif bert_reg_config['linear_module']['activation'] == 'leakyrelu':
            self.linear_activation = torch.nn.LeakyReLU()

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        self.dropout = torch.nn.Dropout(bert_reg_config['linear_module']['dropout'])

    def forward(self, x, attention_mask=None):
        x = self.bert_tweet_model(x, attention_mask=attention_mask)
        x = x.last_hidden_state
        x = self.layer_norm(x)
        for _, layer in enumerate(self.linear_module):
            x = layer(x)
            x = self.linear_activation(x)
            x = self.dropout(x)

        x = self.log_softmax(x)
        x = torch.mean(x, dim=1)
        return x


class BERTTweetClassifier(torch.nn.Module):
    def __init__(self):
        super(BERTTweetClassifier, self).__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-large')
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, attention_mask=None):
        x = self.model(x, attention_mask=attention_mask)
        x = self.log_softmax(x)
        x = torch.mean(x, dim=1)

        return x
