import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report

TAGS = [
    'handfootpreventative',
    'cartoxarrhythmia',
    'cartoxheartfailure',
    'cartoxvalvularcomplications',
    'drugsofinterest'
]

SPLIT_DIR = r"D:\\Github\\MedSDoH\\data\\train_cape\\split_datasets"
MODEL_DIR = "bert_uncased_model" # emilyalsentzer/Bio_ClinicalBERT
MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
EARLY_STOPPING_DELTA = 0.1
EARLY_STOPPING_PATIENCE = 2

os.makedirs(MODEL_DIR, exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe['sentence']
        self.targets = dataframe['target']
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BERTBinaryClassifier(nn.Module):
    def __init__(self):
        super(BERTBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        return self.output(self.dropout(pooled_output))

def train_model(tag):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} for tag: {tag}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'train.csv'))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))

    train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BERTBinaryClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    tag_dir = os.path.join(MODEL_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    checkpoint_path = os.path.join(tag_dir, 'checkpoint.pt')
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in train_loader:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                targets = batch['targets'].to(device).unsqueeze(1)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(ids, mask, token_type_ids)
                    loss = loss_fn(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

            model.eval()
            val_targets = []
            val_outputs = []
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch['ids'].to(device)
                    mask = batch['mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    targets = batch['targets'].to(device).unsqueeze(1)
                    with autocast():
                        outputs = model(ids, mask, token_type_ids)
                    val_targets.extend(targets.cpu().numpy())
                    val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

            preds = (np.array(val_outputs) >= 0.5).astype(int)
            val_targets = np.array(val_targets)
            print(classification_report(val_targets, preds, digits=4))

            with open(os.path.join(MODEL_DIR, f'{tag}_report.txt'), 'a') as f:
                f.write(f"Epoch {epoch+1}\n")
                f.write(classification_report(val_targets, preds, digits=4))
                f.write("\n\n")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(tag_dir, 'bert_interrupted_checkpoint.pt'))
        print("Checkpoint saved.")

# === Run ===
if __name__ == "__main__":
    train_model(TAGS[4])
