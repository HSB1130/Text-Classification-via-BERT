import torch
from torch import nn
from transformers import BertModel, AdamW
from evaluate import Evaluator
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class FullParamModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.classifier(output_dropout)
        return output


class FullParamTuner:
    def __init__(self, full_param_model, lr, epoch, train_DL, val_DL):
        self.model = full_param_model
        self.lr = lr
        self.epoch = epoch
        self.train_DL = train_DL
        self.evaluator = Evaluator(val_DL=val_DL, test_DL=None)

    def get_model(self):
        return self.model
        
    def train(self):
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(self.epoch):
            train_loss = 0

            for batch in tqdm(self.train_DL):
                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.float)

                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_acc, val_loss = self.evaluator.evaluate(self.model)

            print(f'Epoch      : {ep+1}')
            print(f'Train Loss : {train_loss/len(self.train_DL):.4f}')
            print(f'Val   Loss : {val_loss:.4f}')
            print(f'Val   ACC  : {val_acc:.4f}', end='\n\n')
