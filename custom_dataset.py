import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.input_texts = list(dataset['combined'])
        self.max_len = max_len
        
        target_list = list(dataset.columns)[:-1]
        self.targets = self.dataset[target_list].values

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        input_text = str(self.input_texts[index])
        input_text = " ".join(input_text.split())
        
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'input_str': input_text
        }
