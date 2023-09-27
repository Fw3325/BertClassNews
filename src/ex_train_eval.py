import torch
from torch.utils.data import DataLoader

from models.bert_model import BertClassifier
from utils.data import BERTDataset

# Load training data
train_data = BERTDataset(...)

# Create model, optimizer, criterion
model = BertClassifier() 
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Create DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Training loop
for epoch in range(5):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
from models.bert_model import BertClassifier
from utils.data import BERTDataset
from utils.metrics import accuracy

# Load validation data
val_data = BERTDataset(...) 

# Create model
model = BertClassifier()

# Evaluate
val_loader = DataLoader(val_data, batch_size=16) 
accuracy, predictions = evaluate(model, val_loader)
print("Validation Accuracy: ", accuracy)