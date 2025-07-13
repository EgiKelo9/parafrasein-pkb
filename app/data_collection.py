import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import json
import os
import difflib
import re
from collections import defaultdict

class DataCollector:
    def __init__(self):
        self.datasets = []
        self.combined_data = None
        self.tokenizer = None
        self.model = None
        
    def initialize_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoT5-base-paraphrase")
            self.model = T5ForConditionalGeneration.from_pretrained("Wikidepia/IndoT5-base-paraphrase")
            print("Model dan tokenizer berhasil dimuat: Wikidepia/IndoT5-base-paraphrase")
        except Exception as e:
            print(f"Gagal memuat model: {e}")
        
    def load_indonesian_paraphrase_datasets(self):
        print("Mengumpulkan dataset parafrase bahasa Indonesia...")
        try:
            paraphrase_dataset = load_dataset("jakartaresearch/id-paraphrase-detection")
            print(f"Dataset paraphrase detection loaded: {len(paraphrase_dataset['train'])} samples")
            self.datasets.append(('paraphrase_detection', paraphrase_dataset))
        except Exception as e:
            print(f"Gagal memuat dataset paraphrase detection: {e}")
    
    def create_paraphrase_pairs(self):
        paraphrase_pairs = []
        
        for dataset_name, dataset in self.datasets:
            if dataset_name == 'paraphrase_detection':
                for item in dataset['train']:
                    if item['label'] == 1:
                        similarity = self._calculate_similarity(item['sentence1'], item['sentence2'])
                        if similarity > 0.8:
                            category = 'safe'
                        elif similarity > 0.6:
                            category = 'balanced'
                        else:
                            category = 'creative'
                        paraphrase_pairs.append({
                            'original': item['sentence1'],
                            'paraphrase': item['sentence2'],
                            'category': category,
                            'source': dataset_name,
                            'similarity_score': similarity
                        })
        self.combined_data = pd.DataFrame(paraphrase_pairs)
        print(f"Total pasangan parafrase yang terkumpul: {len(self.combined_data)}")
        if not self.combined_data.empty:
            category_dist = self.combined_data['category'].value_counts()
            print("Distribusi kategori:")
            for cat, count in category_dist.items():
                print(f"  {cat}: {count} samples")
        return self.combined_data
    
    def _calculate_similarity(self, text1, text2):
        words1 = set(self._tokenize_text(text1.lower()))
        words2 = set(self._tokenize_text(text2.lower()))
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def _tokenize_text(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
    
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
    
    def generate_paraphrases_with_categories(self, text):
        if not self.model or not self.tokenizer:
            self.initialize_model()
        if not self.model or not self.tokenizer:
            return {"error": "Model tidak tersedia"}
        temperature_settings = {
            "safe": 0.7,
            "balanced": 1.0,
            "creative": 1.3
        }
        results = {}
        for style, temp in temperature_settings.items():
            try:
                inputs = self.tokenizer.encode(
                    f"paraphrase: {text}", 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=512,
                        temperature=temp,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results[style] = paraphrase              
            except Exception as e:
                results[style] = f"Error: {str(e)}"
        return results
    
    def prepare_training_data_from_dataset(self):
        if self.combined_data is None or self.combined_data.empty:
            print("Tidak ada data untuk dipersiapkan. Jalankan create_paraphrase_pairs() terlebih dahulu.")
            return None
        train_data = []
        for _, row in self.combined_data.iterrows():
            train_data.append({
                "input_text": f"paraphrase: {row['original']}",
                "target_text": row["paraphrase"],
                "category": row["category"]
            })
        return Dataset.from_pandas(pd.DataFrame(train_data))
    
    def tokenize_data(self, examples):
        if not self.tokenizer:
            self.initialize_model()
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding=True
        )
        labels = self.tokenizer(
            examples["target_text"],
            max_length=512,
            truncation=True,
            padding=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def save_processed_data(self, filepath="processed_paraphrase_data.csv"):
        if self.combined_data is not None and not self.combined_data.empty:
            self.combined_data.to_csv(filepath, index=False)
            print(f"Data tersimpan di: {filepath}")
        else:
            print("Tidak ada data untuk disimpan. Jalankan create_paraphrase_pairs() terlebih dahulu.")
    
    def load_processed_data(self, filepath="processed_paraphrase_data.csv"):
        try:
            self.combined_data = pd.read_csv(filepath)
            print(f"Data berhasil dimuat dari: {filepath}")
            print(f"Total data: {len(self.combined_data)}")
            return self.combined_data
        except Exception as e:
            print(f"Gagal memuat data: {e}")
            return None
    
    def get_statistics(self):
        if self.combined_data is None or self.combined_data.empty:
            return "Tidak ada data tersedia"
        stats = {
            "total_samples": len(self.combined_data),
            "category_distribution": self.combined_data['category'].value_counts().to_dict(),
            "average_similarity": self.combined_data['similarity_score'].mean() if 'similarity_score' in self.combined_data.columns else 0,
            "sources": self.combined_data['source'].value_counts().to_dict()
        }
        return stats

# Contoh penggunaan
if __name__ == "__main__":
    collector = DataCollector()
    collector.initialize_model()
    collector.load_indonesian_paraphrase_datasets()
    data = collector.create_paraphrase_pairs()
    collector.save_processed_data()
    stats = collector.get_statistics()
    print("\nStatistik Data:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if collector.model and collector.tokenizer:
        sample_text = "Saya sedang belajar pemrograman Python"
        paraphrases = collector.generate_paraphrases_with_categories(sample_text)
        print(f"\nContoh parafrase untuk: '{sample_text}'")
        for style, result in paraphrases.items():
            print(f"{style}: {result}")