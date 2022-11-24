import torch

import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

data = torch.load('cached_train_features')

data = torch.load('cached_train_features')
this = None
for idx, tmp in enumerate(data):
    if tmp.guid == 'train-PMUL0054.json-8':
        this = tmp
        print(idx)
        break



text = tokenizer.decode(getattr(data[idx], 'input_ids'))
print(text)
for value in ['start_pos', 'end_pos', 'values']:
    print(value, getattr(data[idx], value)['restaurant-name'])
    print()