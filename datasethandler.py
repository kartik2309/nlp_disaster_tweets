import torch.cuda
from torch.utils.data import Dataset
from torch import tensor


class CorpusDataset(Dataset):
    def __init__(self, input_ids, attention_mask=None, labels=None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.input_ids = tensor(input_ids, dtype=torch.long, device=device)
        if attention_mask is not None:
            self.attention_mask = tensor(attention_mask, dtype=torch.long, device=device)
        if labels is not None:
            self.labels = tensor(labels, dtype=torch.long, device=device)

    def __getitem__(self, idx):
        item = {'input_ids': self.input_ids[idx]}

        if 'attention_mask' in self.__dict__:
            item['attention_mask'] = self.attention_mask[idx]

        if 'labels' in self.__dict__:
            item['labels'] = self.labels[idx]

        return item

    def __len__(self):
        return len(self.input_ids)
