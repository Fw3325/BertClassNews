import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import sklearn.model_selection
from torch.nn import CrossEntropyLoss
import time
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader, config):
    model.to(self.device)
    model.eval()
    final_pred = []
    total = 0
    correct = 0
    pred = {i:[] for i in range(len(config.lblEncode))}
    # pred = {i:[] for i in [0,2,3]}
    with torch.no_grad():

        for batch in dataloader:
            inputs = torch.stack([t for t in batch['input_ids']]).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            attention_mask =torch.stack([t for t in batch['attention_mask']]).to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask,
                      labels=labels)
            predictions = outputs[1].argmax(dim=1)
            for i in range(len(predictions)):
                predRes = predictions[i].item()
                pred[predRes].append((predictions[i] == labels[i]).item())
                final_pred.append((config.reverse_lblEncode[predRes], config.reverse_lblEncode[labels[i].item()]))
                # final_pred.append((predRes, labels[i].item()))

            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
      
    accuracy = correct/total
    res = {config.reverse_lblEncode[i]:sum(pred[i])/len(pred[i]) if len(pred[i]) > 0 else np.nan for i in pred }
    return accuracy, res, final_pred


def model_train(model, train_dataloader, val_dataloader, num_epochs = config.num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(num_epochs):
      start = time.time()
      losses = 0
      model.train()
      idx = 0
      for batch in train_dataloader:
        inputs, token_type_ids, attention_mask, labels = batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']

        input_ids = torch.stack([t for t in batch['input_ids']]).to(device)

        attention_mask =torch.stack([t for t in batch['attention_mask']]).to(device)
        labels = torch.tensor(batch['labels']).to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        loss = scrierion(outputs['logits'],labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item()
        acc = (outputs['logits'].argmax(1) == labels).float().mean()
        idx += 1
        curr = time.time()
        timeConsume = curr - start
        if idx % 300 == 0:
            print (f'Epoch {epoch+1}, batch {idx}, so far takes {timeConsume}')
          # los  =outputs['loss']
            print (f'Loss: {losses}')

      print (acc)
      val_acc, val_cat_acc, val_final_pred= evaluate(model, val_dataloader)
      print ('val acc:', val_acc)
        # if idx % 10 == 0:
        #     logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
        #                   f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")

      print(f'Epoch {epoch+1} complete, so far takes {timeConsume}')
    return model


