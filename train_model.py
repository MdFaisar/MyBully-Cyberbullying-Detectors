import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import re
from tqdm import tqdm

class CyberbullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Accept pandas Series, numpy arrays or lists and ensure integer indexing works
        if isinstance(texts, pd.Series):
            texts = texts.reset_index(drop=True)
        if isinstance(labels, pd.Series):
            labels = labels.reset_index(drop=True)

        # Convert to plain Python lists for safe integer indexing in __getitem__
        self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
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


class CyberbullyingDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.to(self.device)
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              batch_size=16, epochs=3, learning_rate=2e-5):

        print("Preparing training data...")
        train_dataset = CyberbullyingDataset(
            train_texts,
            train_labels,
            self.tokenizer
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = CyberbullyingDataset(val_texts, val_labels, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        print(f"Training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            
            for batch in progress_bar:
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
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'\nEpoch {epoch + 1} - Average loss: {avg_loss:.4f}')
            
            if val_texts is not None and val_labels is not None:
                val_accuracy = self.evaluate(val_texts, val_labels)
                print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        print("\nTraining completed!")
    
    def evaluate(self, texts, labels):
        """Evaluate the model"""
        self.model.eval()
        dataset = CyberbullyingDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=32)
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(true_labels, predictions) * 100
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                 target_names=['Non-Bullying', 'Bullying']))
        
        return accuracy
    
    def predict(self, text):
        """Predict if text is bullying or not"""
        self.model.eval()
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
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence_scores = probabilities[0].cpu().numpy()
        
        return {
            'prediction': 'Bullying' if prediction == 1 else 'Non-Bullying',
            'confidence': float(max(confidence_scores)) * 100
        }
    
    def save_model(self, model_dir='models'):
        """Save the trained model and tokenizer"""
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Load a trained model and tokenizer"""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at {model_dir}")

        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        print("Model loaded successfully!")


def main():
    """Main training function"""
    print("Initializing Cyberbullying Detector...")
    detector = CyberbullyingDetector()
    
    print("\nTraining dataset...")

    # Note: For production, load a real dataset like:
    path = 'dataset/cyberbullying_dataset.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please place your CSV there.")

    df = pd.read_csv(path)

    # Expecting 'text' and 'label' columns in this CSV
    if 'text' not in df.columns:
        raise ValueError("The dataset CSV must contain a 'text' column with the message/comment text.")
    if 'label' not in df.columns:
        raise ValueError("The dataset CSV must contain a 'label' column with 0 (non-bullying) or 1 (bullying).")

    # Clean text column: ensure string and drop empty rows
    df['text'] = df['text'].astype(str).str.strip()
    before_text = len(df)
    df = df[df['text'] != '']
    dropped_text = before_text - len(df)
    if dropped_text > 0:
        print(f"Dropped {dropped_text} rows with empty/missing text")

    # Normalize labels: convert to numeric, drop NaNs
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    before_labels = len(df)
    df = df[df['label'].notna()]
    dropped_labels = before_labels - len(df)
    if dropped_labels > 0:
        print(f"Dropped {dropped_labels} rows with non-numeric/missing labels")

    # Cast to int and keep only 0/1
    df['label'] = df['label'].astype(int)
    before_filter = len(df)
    df = df[df['label'].isin([0, 1])]
    dropped_invalid = before_filter - len(df)
    if dropped_invalid > 0:
        print(f"Dropped {dropped_invalid} rows with labels outside 0/1")

    print(f"Dataset size: {len(df)} samples")
    print(f"Bullying samples: {int(sum(df['label'] == 1))}")
    print(f"Non-bullying samples: {int(sum(df['label'] == 0))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Train model
    detector.train(X_train, y_train)
    
    # Evaluate model
    detector.evaluate(X_test, y_test)
    
    # Save model
    detector.save_model()
    
    # Test predictions
    print("\n" + "="*50)
    print("Testing predictions:")
    print("="*50)
    
    # Use test.csv for testing predictions. Expected columns: 'text' (required) and optional 'label' (0 or 1).
    test_path = 'dataset/cyberbullying_dataset_test.csv'
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if 'text' not in test_df.columns:
            print("test.csv must contain a 'text' column.")
        else:
            has_label = 'label' in test_df.columns
            y_true = test_df['label'].tolist() if has_label else None
            y_pred = []

            for idx, row in test_df.iterrows():
                text = str(row['text'])
                result = detector.predict(text)
                print(f"\nText: '{text}'")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                if has_label:
                    y_pred.append(1 if result['prediction'] == 'Bullying' else 0)

            if has_label:
                acc = accuracy_score(y_true, y_pred)
                print(f"\nTest CSV Accuracy: {acc * 100:.2f}%")
    else:
        print("test.csv not found in project root. Create 'test.csv' with a 'text' column and optional 'label' column.")


if __name__ == "__main__":
    main()