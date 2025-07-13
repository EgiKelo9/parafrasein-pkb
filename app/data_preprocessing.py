import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import difflib

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class IndonesianTextPreprocessor:
    def __init__(self, tokenizer_name="Wikidepia/IndoT5-base-paraphrase"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = 512 
        self.abbreviation_dict = {
            'yg': 'yang',
            'dg': 'dengan',
            'tdk': 'tidak',
            'hrs': 'harus',
            'krn': 'karena',
            'utk': 'untuk',
            'shg': 'sehingga',
            'dll': 'dan lain-lain',
            'dst': 'dan seterusnya',
            'dsb': 'dan sebagainya',
            'dgn': 'dengan',
            'pd': 'pada',
            'trs': 'terus',
            'krg': 'kurang',
            'lbh': 'lebih'
        }
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+62|62|0)(\d{2,3}-?\d{3,4}-?\d{3,4})')
        
    def _tokenize_text(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_text(self, text):
        text = text.lower()
        words = text.split()
        expanded_words = []
        for word in words:
            if word in self.abbreviation_dict:
                expanded_words.append(self.abbreviation_dict[word])
            else:
                expanded_words.append(word)
        text = ' '.join(expanded_words)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        return text
    
    def calculate_similarity(self, text1, text2):
        words1 = set(self._tokenize_text(text1.lower()))
        words2 = set(self._tokenize_text(text2.lower()))
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def analyze_word_changes(self, original_text, paraphrased_text):
        original_words = self._tokenize_text(original_text.lower())
        paraphrased_words = self._tokenize_text(paraphrased_text.lower())
        matcher = difflib.SequenceMatcher(None, original_words, paraphrased_words)
        changes = []
        original_index = 0
        paraphrased_index = 0
        for match in matcher.get_matching_blocks():
            if original_index < match.a or paraphrased_index < match.b:
                original_segment = original_words[original_index:match.a]
                paraphrased_segment = paraphrased_words[paraphrased_index:match.b]
                if original_segment and paraphrased_segment:
                    changes.append({
                        'type': 'substitution',
                        'original': ' '.join(original_segment),
                        'paraphrased': ' '.join(paraphrased_segment),
                        'original_pos': (original_index, match.a),
                        'paraphrased_pos': (paraphrased_index, match.b)
                    })
                elif original_segment:
                    changes.append({
                        'type': 'deletion',
                        'original': ' '.join(original_segment),
                        'paraphrased': '',
                        'original_pos': (original_index, match.a),
                        'paraphrased_pos': (paraphrased_index, paraphrased_index)
                    })
                elif paraphrased_segment:
                    changes.append({
                        'type': 'addition',
                        'original': '',
                        'paraphrased': ' '.join(paraphrased_segment),
                        'original_pos': (original_index, original_index),
                        'paraphrased_pos': (paraphrased_index, match.b)
                    })
            original_index = match.a + match.size
            paraphrased_index = match.b + match.size
        return changes
    
    def categorize_paraphrase(self, original, paraphrase):
        similarity = self.calculate_similarity(original, paraphrase)
        if similarity > 0.8:
            return 'safe'
        elif similarity > 0.6:
            return 'balanced'
        else:
            return 'creative'
    
    def check_text_quality(self, text):
        if not text or len(text.strip()) == 0:
            return {'quality_score': 0, 'issues': ['empty_text']}
        issues = []
        quality_score = 100
        words = text.split()
        if len(words) < 3:
            issues.append('too_short')
            quality_score -= 30
        if len(words) > 100:
            issues.append('too_long')
            quality_score -= 20
        if len(text) > 0:
            special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
            if special_char_ratio > 0.3:
                issues.append('too_many_special_chars')
                quality_score -= 25
        unique_words = set(words)
        if len(words) > 0 and len(unique_words) / len(words) < 0.5:
            issues.append('too_repetitive')
            quality_score -= 20
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'word_count': len(words),
            'unique_word_ratio': len(unique_words) / len(words) if words else 0
        }
    
    def preprocess_paraphrase_pair(self, original, paraphrase):
        clean_original = self.normalize_text(self.clean_text(original))
        clean_paraphrase = self.normalize_text(self.clean_text(paraphrase))
        original_quality = self.check_text_quality(clean_original)
        paraphrase_quality = self.check_text_quality(clean_paraphrase)
        category = self.categorize_paraphrase(clean_original, clean_paraphrase)
        similarity_score = self.calculate_similarity(clean_original, clean_paraphrase)
        original_tokens = self.tokenizer.encode(clean_original, add_special_tokens=True, max_length=self.max_length, truncation=True)
        paraphrase_tokens = self.tokenizer.encode(clean_paraphrase, add_special_tokens=True, max_length=self.max_length, truncation=True)
        word_changes = self.analyze_word_changes(clean_original, clean_paraphrase)
        return {
            'original_text': clean_original,
            'paraphrase_text': clean_paraphrase,
            'category': category,
            'similarity_score': similarity_score,
            'original_quality': original_quality,
            'paraphrase_quality': paraphrase_quality,
            'original_token_count': len(original_tokens),
            'paraphrase_token_count': len(paraphrase_tokens),
            'word_changes_count': len(word_changes),
            'substitutions_count': len([c for c in word_changes if c['type'] == 'substitution']),
            'additions_count': len([c for c in word_changes if c['type'] == 'addition']),
            'deletions_count': len([c for c in word_changes if c['type'] == 'deletion']),
            'is_valid': (original_quality['quality_score'] >= 60 and 
                        paraphrase_quality['quality_score'] >= 60 and
                        similarity_score < 0.95)
        }

class ParaphraseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, category_filter=None):
        self.data = data
        if category_filter:
            self.data = data[data['category'] == category_filter].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_text = f"paraphrase: {item['original_text']}"
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            item['paraphrase_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        temperature_map = {
            'safe': 0.7,
            'balanced': 1.0,
            'creative': 1.3
        }
        creativity_level = temperature_map.get(item.get('category', 'balanced'), 1.0)
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'temperature': creativity_level,
            'category': item.get('category', 'balanced')
        }

def tokenize_data_for_training(examples, tokenizer, max_length=512):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_length,
        truncation=True,
        padding=True
    )
    labels = tokenizer(
        examples["target_text"],
        max_length=max_length,
        truncation=True,
        padding=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def create_data_splits(data, test_size=0.2, val_size=0.1, random_state=42):
    try:
        train_data, temp_data = train_test_split(
            data, 
            test_size=(test_size + val_size), 
            random_state=random_state,
            stratify=data['category'] if 'category' in data.columns else None
        )
        val_ratio = val_size / (test_size + val_size)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio),
            random_state=random_state,
            stratify=temp_data['category'] if 'category' in temp_data.columns else None
        )
    except ValueError:
        train_data, temp_data = train_test_split(
            data, 
            test_size=(test_size + val_size), 
            random_state=random_state
        )
        val_ratio = val_size / (test_size + val_size)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio),
            random_state=random_state
        )
    print(f"Data split selesai:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    if 'category' in data.columns:
        print("\nDistribusi kategori:")
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            dist = split_data['category'].value_counts()
            print(f"  {split_name}: {dist.to_dict()}")
    return train_data, val_data, test_data

def prepare_training_data_like_streamlit(data, tokenizer):
    train_data = []
    for _, row in data.iterrows():
        train_data.append({
            "input_text": f"paraphrase: {row['original_text']}",
            "target_text": row["paraphrase_text"],
            "category": row.get("category", "balanced")
        })
    return pd.DataFrame(train_data)

# Contoh penggunaan
if __name__ == "__main__":
    try:
        data = pd.read_csv("processed_paraphrase_data.csv")
        print(f"Data loaded: {len(data)} samples")
    except FileNotFoundError:
        print("File processed_paraphrase_data.csv tidak ditemukan. Jalankan data_collection.py terlebih dahulu.")
        exit()
    preprocessor = IndonesianTextPreprocessor(tokenizer_name="Wikidepia/IndoT5-base-paraphrase")
    processed_pairs = []
    print("Memproses pasangan parafrase...")
    for idx, row in data.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(data)}")
        processed = preprocessor.preprocess_paraphrase_pair(
            row['original'], 
            row['paraphrase']
        )
        processed['source'] = row.get('source', 'unknown')
        processed_pairs.append(processed)
    processed_df = pd.DataFrame(processed_pairs)
    valid_data = processed_df[processed_df['is_valid']].reset_index(drop=True)
    print(f"\nData valid: {len(valid_data)} dari {len(processed_df)} total")
    if not valid_data.empty:
        category_dist = valid_data['category'].value_counts()
        print("\nDistribusi kategori:")
        for cat, count in category_dist.items():
            print(f"  {cat}: {count} samples")
    train_data, val_data, test_data = create_data_splits(valid_data)
    train_formatted = prepare_training_data_like_streamlit(train_data, preprocessor.tokenizer)
    val_formatted = prepare_training_data_like_streamlit(val_data, preprocessor.tokenizer)
    test_formatted = prepare_training_data_like_streamlit(test_data, preprocessor.tokenizer)
    processed_df.to_csv("all_processed_data.csv", index=False)
    valid_data.to_csv("valid_processed_data.csv", index=False)
    train_data.to_csv("train_data.csv", index=False)
    val_data.to_csv("val_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    train_formatted.to_csv("train_formatted.csv", index=False)
    val_formatted.to_csv("val_formatted.csv", index=False)
    test_formatted.to_csv("test_formatted.csv", index=False)
    print("\nFile yang tersimpan:")
    print("- all_processed_data.csv: Semua data yang diproses")
    print("- valid_processed_data.csv: Data valid saja")
    print("- train_data.csv, val_data.csv, test_data.csv: Split data")
    print("- train_formatted.csv, val_formatted.csv, test_formatted.csv: Format training")
    dataset = ParaphraseDataset(valid_data, preprocessor.tokenizer)
    print(f"\nDataset siap untuk training: {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample input shape: {sample['input_ids'].shape}")
        print(f"Sample category: {sample['category']}")
        print(f"Sample temperature: {sample['temperature']}")
    print("\nPreprocessing selesai!")