import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

examples = []
labels = []
label2id = {}
id2label = {}

for i, command in enumerate(data):
    label2id[command["command"]] = i
    id2label[i] = command["command"]
    for ex in command["examples"]:
        examples.append(ex["text"])
        labels.append(i)  # ID команды

# test 
train_texts, test_texts, train_labels, test_labels = train_test_split(
    examples, labels, test_size=0.2, random_state=42
)

# токенизатор

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class CommandDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = CommandDataset(train_texts, train_labels, tokenizer)
test_dataset = CommandDataset(test_texts, test_labels, tokenizer)

# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(label2id)
)
model.config.id2label = id2label
model.config.label2id = label2id

# Настройки тренировки
training_args = TrainingArguments(
    output_dir="./command_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# тренировка
trainer.train()

# Сохранение модели
model.save_pretrained("./command_model")
tokenizer.save_pretrained("./command_model")
