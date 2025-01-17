import torch
import torch.nn as nn
from torch.utils.data import Dataset
# from rouge import Rouge
from rouge_score import rouge_scorer

class FtDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids, self.attention_mask = self.tokenize_input()
        self.output_ids = self.tokenize_output()

    def tokenize_input(self):
        inputs = self.data['article']
        input_encoding = self.tokenizer(inputs.tolist(), truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return input_encoding['input_ids'], input_encoding['attention_mask']
    
    def tokenize_output(self):
        outputs = self.data['highlights']
        output_encoding = self.tokenizer(outputs.tolist(), truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return output_encoding['input_ids']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'output_ids': self.output_ids[idx]}
    
def train(model, train_loader, val_loader, optimizer, n_epochs, device, tokenizer):
    train_loss = []
    val_loss = []
    torch.cuda.reset_max_memory_allocated(device)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output_ids = batch['output_ids'].to(device)
            for j in range(output_ids.size(0)):
                output_ids[j][output_ids[j] == tokenizer.pad_token_id] = -100
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            outputs = model(input_ids=input_ids, labels=output_ids, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Average Training Loss: {avg_train_loss}")

        current_memory = torch.cuda.memory_allocated(device)
        print(f"GPU Memory Used After Training: {current_memory / (1024 ** 2):.2f} MB")

        max_memory = torch.cuda.max_memory_allocated(device)
        print(f"Max GPU Memory Used during training: {max_memory / (1024 ** 2):.2f} MB")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                output_ids = batch['output_ids'].to(device)
                for j in range(output_ids.size(0)):
                    output_ids[j][output_ids[j] == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=input_ids, labels=output_ids, attention_mask=attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
            avg_val_loss = total_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{n_epochs}, Average Validation Loss: {avg_val_loss}")

        current_memory = torch.cuda.memory_allocated(device)
        print(f"GPU Memory Used After Validation: {current_memory / (1024 ** 2):.2f} MB")

        max_memory = torch.cuda.max_memory_allocated(device)
        print(f"Max GPU Memory Used during Validation: {max_memory / (1024 ** 2):.2f} MB")

    return model, train_loss, val_loss

def test_ft(model, test_dataloader, tokenizer, original_answers, device):
    model.eval()
    predictions = []
    test_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['output_ids'].to(device)

            for j in range(labels.size(0)):
                label = labels[j]
                label[label == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item()

            for j in range(input_ids.size(0)):
                predicted_ids = torch.argmax(logits[j], dim=-1)
                predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                predictions.append(predicted_text)

    test_loss /= len(test_dataloader)

    print(f"Test Loss: {test_loss}")

    # rouge = Rouge()
    # rouge_scores = rouge.get_scores(predictions, original_answers, avg=True)
    # print(f"Rouge Scores: {rouge_scores}")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    total_scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
    }

    for reference, generated in zip(original_answers, predictions):
        scores = scorer.score(reference, generated)
        for metric, score in scores.items():
            total_scores[metric]['precision'] += score.precision
            total_scores[metric]['recall'] += score.recall
            total_scores[metric]['fmeasure'] += score.fmeasure

    num_samples = len(original_answers)
    average_scores = {metric: {k: v / num_samples for k, v in score.items()} for metric, score in total_scores.items()}

    for metric, score in average_scores.items():
        print(f"{metric}:")
        print(f"  Average Precision: {score['precision']:.4f}")
        print(f"  Average Recall: {score['recall']:.4f}")
        print(f"  Average F1 Measure: {score['fmeasure']:.4f}")