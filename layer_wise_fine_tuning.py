import torch
from torch import nn
from transformers import BertModel, AdamW
from evaluate import Evaluator
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LayerWiseModel(nn.Module):
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
        cls_output = output.last_hidden_state[:, 0, :]
        output_dropout = self.dropout(cls_output)
        output = self.classifier(output_dropout)
        return output


class LayerWiseTuner:
    def __init__(self, layer_wise_model, lr, epoch, train_DL, val_DL, start_unfreeze_epoch=2, unfreeze_per_epoch=1):
        self.model = layer_wise_model
        self.lr = lr
        self.epoch = epoch
        self.train_DL = train_DL
        self.evaluator = Evaluator(val_DL=val_DL, test_DL=None)
        self.start_unfreeze_epoch = start_unfreeze_epoch
        self.unfreeze_per_epoch = unfreeze_per_epoch

    def get_model(self):
        return self.model
    
    def _unfreeze_layers(self, num_layers_to_unfreeze):
        tot_layers = len(list(self.model.bert_model.encoder.layer))
        unfreeze_from = tot_layers - num_layers_to_unfreeze

        for idx, layer in enumerate(self.model.bert_model.encoder.layer):
            if idx >= unfreeze_from:  # 언프리징할 레이어
                for param in layer.parameters():
                    param.requires_grad = True

    def train(self):
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for param in self.model.bert_model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        for ep in range(self.epoch):
            train_loss = 0

            if ep >= self.start_unfreeze_epoch:
                self._unfreeze_layers(num_layers_to_unfreeze=(ep - self.start_unfreeze_epoch + 1) * self.unfreeze_per_epoch)

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
