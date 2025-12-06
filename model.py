import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
try:
    from transformers import BertTokenizer, BertForSequenceClassification
except Exception as e:
    raise RuntimeError(
        "Failed to import 'transformers'. If this error relates to 'google.protobuf', run:"
        "\n  python -m pip install --upgrade protobuf==4.23.4"
        "\nThen restart your Python process and try again." ) from e

import pandas as pd
import numpy as np
import re

class CyberbullyingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BullyingDetector:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            print("BERT model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Initializing new BERT model...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            self.model.to(self.device)

    def save_model(self, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}/")

    def update_confidence(self, text, label):
        feedback_path = 'dataset/feedback.csv'
        dataset_path = 'dataset/cyberbullying_dataset.csv'
        os.makedirs('dataset', exist_ok=True)

        # Save new feedback
        new_row = pd.DataFrame({'text': [text], 'label': [1 if label == 'Bullying' else 0]})
        if os.path.exists(feedback_path):
            fb = pd.read_csv(feedback_path)
            fb = pd.concat([fb, new_row], ignore_index=True)
        else:
            fb = new_row
        fb.to_csv(feedback_path, index=False)

        # Clean feedback data
        fb['text'] = fb['text'].astype(str).str.strip()
        fb = fb[fb['text'] != '']
        fb['label'] = pd.to_numeric(fb['label'], errors='coerce')
        fb = fb[fb['label'].notna()]
        fb['label'] = fb['label'].astype(int)

        # Combine with original dataset
        if os.path.exists(dataset_path):
            orig_data = pd.read_csv(dataset_path)
            orig_data['text'] = orig_data['text'].astype(str).str.strip()
            orig_data = orig_data[orig_data['text'] != '']
            orig_data['label'] = pd.to_numeric(orig_data['label'], errors='coerce')
            orig_data = orig_data[orig_data['label'].notna()]
            orig_data['label'] = orig_data['label'].astype(int)
            
            # Duplicate feedback 3 times for stronger influence
            combined_data = pd.concat([orig_data] + [fb] * 3, ignore_index=True)
        else:
            combined_data = fb

        # Create datasets
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(combined_data, test_size=0.1, random_state=42)
        
        train_dataset = CyberbullyingDataset(
            train_data['text'].values,
            train_data['label'].values,
            self.tokenizer
        )
        val_dataset = CyberbullyingDataset(
            val_data['text'].values,
            val_data['label'].values,
            self.tokenizer
        )

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=8
        )

        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        num_training_steps = len(train_loader) * 3  # 3 epochs
        
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        best_val_loss = float('inf')
        patience = 2
        patience_counter = 0

        # Training loop
        for epoch in range(3):  # 3 epochs
            # Training
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break
            
            # Calculate validation accuracy
            from sklearn.metrics import accuracy_score
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            
            print(f"Epoch {epoch + 1}")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")
        
        print(f"Feedback training completed. Total feedback samples: {len(fb)}")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()

    def analyze(self, text):
        if not text or text.strip() == '':
            return {
                'prediction': 'Invalid Input',
                'confidence': 0,
                'severity': 'none',
                'message': 'Please enter some text to analyze'
            }
        processed_text = self.preprocess_text(text)
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence_scores = probabilities[0].cpu().numpy()
        confidence = float(max(confidence_scores)) * 100
        is_bullying = bool(prediction)
        if is_bullying:
            if confidence >= 80:
                severity = 'high'
            elif confidence >= 60:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            severity = 'none'
        result = {
            'prediction': 'Bullying' if is_bullying else 'Non-Bullying',
            'confidence': float(round(confidence, 2)),
            'severity': severity,
            'is_bullying': is_bullying,
            'probabilities': {
                'non_bullying': float(round(confidence_scores[0] * 100, 2)),
                'bullying': float(round(confidence_scores[1] * 100, 2))
            }
        }
        return result

    def batch_analyze(self, texts):
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results