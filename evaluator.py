import torch
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Evaluator:
    def __init__(self, val_DL, test_DL):
        self.val_DL = val_DL
        self.test_DL = test_DL

    def evaluate(self, model, use_test_dl=False):
        model.eval()
        criterion = nn.CrossEntropyLoss()

        if use_test_dl:
            DataLoader = self.test_DL
        else:
            DataLoader = self.val_DL

        total_loss = 0
        correct_predictions = 0
        num_samples = 0

        with torch.no_grad():
            for batch in DataLoader:
                input_ids = batch['input_ids'].to(device, dtype = torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
                targets = batch['targets'].to(device, dtype = torch.float)
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                pred_values, preds = torch.max(outputs, dim=1)
                trg_values, trgs = torch.max(targets, dim=1)
                num_samples += len(trgs)
                correct_predictions += torch.sum(preds==trgs)

        return float(correct_predictions)/num_samples, total_loss/len(DataLoader)
