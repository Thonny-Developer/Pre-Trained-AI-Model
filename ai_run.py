# import os

# os.system('cls')
# os.system('Remove-Module PSReadLine')
# os.system('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8')
# os.system('powershell -noprofile')

import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели
model_path = "./command_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Функция предсказания команды
def predict_command(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return model.config.id2label[predicted_label]

# Функция для извлечения аргументов
def extract_arguments(command, text):
    args = {}

    if command == "set_alarm":
        match = re.search(r"(\d{1,2}[:.]?\d{0,2})\s*(утра|вечера|am|pm)?", text)
        if match:
            time = match.group(1).replace(".", ":")
            period = match.group(2)
            if period:
                time += " " + period
            args["time"] = time

    elif command == "set_volume":
        match = re.search(r"(\d{1,3})\s*%", text)
        if match:
            args["volume"] = match.group(1) + "%"

    elif command == "open_website":
        match = re.search(r"(ютуб|реддит|твитч|роблокс|почту|gmail|mail|тикток|tiktok|twich|roblox)", text, re.IGNORECASE)
        if match:
            args["website"] = match.group(1).lower()

    elif command == "set_timer":
        match = re.search(r"(\d+)\s*(секунд|минут|часов)?", text)
        if match:
            args["duration"] = match.group(1) + " " + (match.group(2) if match.group(2) else "секунд")

    return args

def process_input(text):
    command = predict_command(text)
    arguments = extract_arguments(command, text)
    return {"command": command, "arguments": arguments}

while True:
    inp = input("Введите запрос: ")
    print(process_input(inp))
