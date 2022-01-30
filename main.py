
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn import CrossEntropyLoss

from preprocessing import CustomTokenizer, TokenToId, BERTTransformerTokenizer
from datasethandler import CorpusDataset
from modelling import LSTMModel, BERTTweetModel, FrozenBERTModel, BERTTweetClassifier
from trainer import Train


def preprocessing():
    train = pd.read_csv('/Users/kartikrajeshwaran/Documents/Datasets/nlp_disaster_tweets/train.csv')
    test = pd.read_csv('/Users/kartikrajeshwaran/Documents/Datasets/nlp_disaster_tweets/test.csv')

    tokenizer = BERTTransformerTokenizer(train, test)

    train_ids, train_input_ids, train_attention_mask, labels, \
    test_ids, test_input_ids, test_attention_mask = tokenizer.tokenize()

    train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
        train_input_ids,
        train_attention_mask, labels,
        test_size=0.2, random_state=42)

    return train_input_ids, train_labels, train_attention_mask, val_input_ids, val_labels, val_attention_mask, test_ids


def model_training():
    train_input_ids, train_labels, train_attention_mask, val_input_ids, val_labels, val_attention_mask, _ = preprocessing()
    train_dataset = CorpusDataset(input_ids=train_input_ids, attention_mask=train_attention_mask, labels=train_labels)
    val_dataset = CorpusDataset(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)

    model = BERTTweetClassifier()

    with open('config/train_config.yaml', 'r') as train_config_stream:
        train_config = yaml.safe_load(train_config_stream)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_config['berttweet_config']['batch_size'])
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1)

    trainer = Train(
        is_checkpoint=train_config['berttweet_config']['is_checkpoint'],
        save_path=None,
        model_name=train_config['berttweet_config']['model_name'],
        model=model,
        optimizer=AdamW,
        scheduler=LinearLR,
        loss_fn=CrossEntropyLoss,
        lr=train_config['berttweet_config']['lr'],
        start_factor=0.5,
        total_iters=train_config['berttweet_config']['epochs'],
        verbose=True
    )

    trainer.train(train_config['berttweet_config']['epochs'], train_loader=train_loader, val_loader=val_loader)


if __name__ == '__main__':
    model_training()
