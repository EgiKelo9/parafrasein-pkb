# model_architecture.py (Diperbaiki berdasarkan teststreamlit.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
)
from datasets import Dataset
import pandas as pd
import numpy as np
import json
import os
import gc
import difflib
import re
from typing import Dict, List, Optional, Tuple

class ParaphraseModel:
    """
    Model parafrase berdasarkan T5 seperti di teststreamlit.py
    Menggunakan arsitektur yang sudah terbukti bekerja dengan baik
    """
    
    def __init__(self, model_name="Wikidepia/IndoT5-base-paraphrase"):
        """
        Inisialisasi model dengan pre-trained T5 Indonesia
        
        Args:
            model_name: Nama model pre-trained yang akan digunakan
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.user_preferences = []
        
        # Temperature settings untuk setiap style seperti di teststreamlit.py
        self.temperature_settings = {
            "safe": 0.7,
            "balanced": 1.0,
            "creative": 1.3
        }
        
        # Mapping dari nama ke index untuk konsistensi
        self.style_to_index = {
            "safe": 0,
            "balanced": 1,
            "creative": 2
        }
        
        self.index_to_style = {v: k for k, v in self.style_to_index.items()}
    
    def generate_paraphrases(self, text, style="safe"):
        """
        Generate parafrase dengan style tertentu
        Mengadopsi logika dari teststreamlit.py
        
        Args:
            text: Teks input yang akan diparafrase
            style: Style parafrase (safe, balanced, creative)
            
        Returns:
            str: Hasil parafrase
        """
        if not text.strip():
            return "Please enter text to paraphrase"
        
        # Ambil temperature berdasarkan style
        temp = self.temperature_settings.get(style, 0.7)
        
        # Format input seperti di teststreamlit.py
        inputs = self.tokenizer.encode(
            f"paraphrase: {text}", 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        try:
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
            return paraphrase
            
        except Exception as e:
            return f"Error generating paraphrase: {str(e)}"
    
    def generate_all_styles(self, text):
        """
        Generate parafrase untuk semua style sekaligus
        
        Args:
            text: Teks input
            
        Returns:
            dict: Dictionary berisi hasil untuk setiap style
        """
        results = {}
        for style in ["safe", "balanced", "creative"]:
            results[style] = self.generate_paraphrases(text, style)
        return results
    
    def analyze_word_changes(self, original_text, paraphrased_text):
        """
        Analisis perubahan kata antara teks asli dan parafrase
        Mengadopsi fungsi dari teststreamlit.py
        """
        # Clean dan tokenize teks
        original_words = self._tokenize_text(original_text.lower())
        paraphrased_words = self._tokenize_text(paraphrased_text.lower())
        
        # Gunakan difflib untuk mencari perbedaan
        matcher = difflib.SequenceMatcher(None, original_words, paraphrased_words)
        
        changes = []
        original_index = 0
        paraphrased_index = 0
        
        # Dapatkan matching blocks dari difflib
        for match in matcher.get_matching_blocks():
            # Handle kata-kata sebelum matching block (ini adalah perubahan)
            if original_index < match.a or paraphrased_index < match.b:
                original_segment = original_words[original_index:match.a]
                paraphrased_segment = paraphrased_words[paraphrased_index:match.b]
                
                if original_segment and paraphrased_segment:
                    # Word substitution
                    changes.append({
                        'type': 'substitution',
                        'original': ' '.join(original_segment),
                        'paraphrased': ' '.join(paraphrased_segment),
                        'original_pos': (original_index, match.a),
                        'paraphrased_pos': (paraphrased_index, match.b)
                    })
                elif original_segment:
                    # Word deletion
                    changes.append({
                        'type': 'deletion',
                        'original': ' '.join(original_segment),
                        'paraphrased': '',
                        'original_pos': (original_index, match.a),
                        'paraphrased_pos': (paraphrased_index, paraphrased_index)
                    })
                elif paraphrased_segment:
                    # Word addition
                    changes.append({
                        'type': 'addition',
                        'original': '',
                        'paraphrased': ' '.join(paraphrased_segment),
                        'original_pos': (original_index, original_index),
                        'paraphrased_pos': (paraphrased_index, match.b)
                    })
            
            # Pindahkan indeks ke akhir matching block
            original_index = match.a + match.size
            paraphrased_index = match.b + match.size
        
        return changes
    
    def _tokenize_text(self, text):
        """
        Tokenisasi sederhana yang mempertahankan tanda baca
        Sama seperti di teststreamlit.py
        """
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
    
    def get_word_mappings(self, original_text, paraphrased_text):
        """
        Buat mapping perubahan kata yang disederhanakan untuk display
        Mengadopsi dari teststreamlit.py
        """
        changes = self.analyze_word_changes(original_text, paraphrased_text)
        
        # Filter untuk substitusi dan perubahan signifikan
        mappings = []
        for change in changes:
            if change['type'] == 'substitution':
                original = change['original'].strip()
                paraphrased = change['paraphrased'].strip()
                
                # Hanya sertakan perubahan yang bermakna
                if len(original) > 2 and len(paraphrased) > 2:
                    mappings.append({
                        'original': original,
                        'paraphrased': paraphrased
                    })
        
        return mappings
    
    def save_preference(self, original_text, paraphrase, style):
        """
        Simpan preferensi user untuk fine-tuning
        Mengadopsi dari teststreamlit.py
        """
        preference = {
            "original": original_text,
            "paraphrase": paraphrase,
            "style": style
        }
        self.user_preferences.append(preference)
        
        # Simpan ke file
        try:
            with open("user_preferences.json", "w", encoding='utf-8') as f:
                json.dump(self.user_preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving preferences to file: {e}")
        
        return f"Preference saved! Total preferences: {len(self.user_preferences)}"
    
    def load_preferences(self):
        """
        Load preferensi yang tersimpan
        """
        try:
            with open("user_preferences.json", "r", encoding='utf-8') as f:
                self.user_preferences = json.load(f)
                return f"Loaded {len(self.user_preferences)} preferences"
        except FileNotFoundError:
            self.user_preferences = []
            return "No saved preferences found"
        except Exception as e:
            return f"Error loading preferences: {e}"
    
    def prepare_training_data(self):
        """
        Persiapkan data training dari user preferences
        Mengadopsi dari teststreamlit.py
        """
        if not self.user_preferences:
            return "No user preferences found"
        
        train_data = []
        for pref in self.user_preferences:
            train_data.append({
                "input_text": f"paraphrase: {pref['original']}",
                "target_text": pref["paraphrase"]
            })
        
        return Dataset.from_pandas(pd.DataFrame(train_data))
    
    def tokenize_data(self, examples):
        """
        Tokenisasi data untuk training
        Mengadopsi dari teststreamlit.py
        """
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
    
    def fine_tune_model(self, epochs=3, learning_rate=5e-5, output_dir="./fine_tuned_model"):
        """
        Fine-tune model berdasarkan user preferences
        Mengadopsi dari teststreamlit.py
        """
        if not self.user_preferences:
            return "No user preferences for fine-tuning"
        
        # Persiapkan dataset
        train_dataset = self.prepare_training_data()
        train_dataset = train_dataset.map(
            self.tokenize_data, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Training arguments seperti di teststreamlit.py
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            save_steps=500,
            logging_steps=100,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            prediction_loss_only=True
        )
        
        # Inisialisasi trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        
        # Training
        trainer.train()
        
        # Clear cache dan save model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            trainer.save_model(output_dir)
        except Exception as e:
            trainer.args.save_safetensors = False
            trainer.save_model(output_dir)
        
        return f"Fine-tuning completed! Model saved with {len(self.user_preferences)} preference examples"
    
    def load_fine_tuned_model(self, model_path="./fine_tuned_model"):
        """
        Load model yang sudah di-fine-tune
        Mengadopsi dari teststreamlit.py
        """
        if os.path.exists(model_path):
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            return "Fine-tuned model loaded successfully"
        return "No fine-tuned model found"
    
    def get_style_statistics(self):
        """
        Dapatkan statistik preferensi style user
        """
        if not self.user_preferences:
            return {}
        
        style_counts = {"safe": 0, "balanced": 0, "creative": 0}
        for pref in self.user_preferences:
            style = pref.get("style", "balanced")
            if style in style_counts:
                style_counts[style] += 1
        
        total = len(self.user_preferences)
        style_percentages = {
            style: (count / total) * 100 
            for style, count in style_counts.items()
        }
        
        return {
            "counts": style_counts,
            "percentages": style_percentages,
            "total": total
        }

class CreativityControlledParaphraser(nn.Module):
    """
    Wrapper untuk model T5 dengan kontrol kreativitas
    Dibangun di atas model dasar yang sudah bekerja
    """
    
    def __init__(self, base_model_name="Wikidepia/IndoT5-base-paraphrase"):
        super(CreativityControlledParaphraser, self).__init__()
        
        # Gunakan model yang sama seperti di teststreamlit.py
        self.paraphrase_model = ParaphraseModel(base_model_name)
        
        # Embedding untuk creativity control
        self.creativity_embedding = nn.Embedding(3, 768)  # 3 styles, 768 dim
        
        # Layer untuk mengontrol diversity
        self.diversity_controller = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_text, creativity_level="balanced"):
        """
        Forward pass dengan kontrol kreativitas
        
        Args:
            input_text: Teks input
            creativity_level: Level kreativitas (safe/balanced/creative)
            
        Returns:
            str: Hasil parafrase
        """
        return self.paraphrase_model.generate_paraphrases(input_text, creativity_level)
    
    def generate_with_analysis(self, input_text, creativity_level="balanced"):
        """
        Generate dengan analisis perubahan kata
        
        Returns:
            dict: Hasil parafrase beserta analisis
        """
        paraphrase = self.paraphrase_model.generate_paraphrases(input_text, creativity_level)
        word_mappings = self.paraphrase_model.get_word_mappings(input_text, paraphrase)
        changes = self.paraphrase_model.analyze_word_changes(input_text, paraphrase)
        
        return {
            'paraphrase': paraphrase,
            'word_mappings': word_mappings,
            'changes': changes,
            'style': creativity_level
        }

class TextDifferenceAnalyzer:
    """
    Kelas untuk menganalisis perbedaan teks
    Menggunakan metode yang sama seperti di teststreamlit.py
    """
    
    def __init__(self):
        pass
    
    def get_word_level_changes(self, original_text, paraphrased_text):
        """
        Analisis perubahan pada level kata
        Mengadopsi dari teststreamlit.py
        """
        # Tokenisasi sederhana berdasarkan spasi dan tanda baca
        original_words = re.findall(r'\w+|[^\w\s]', original_text.lower())
        paraphrased_words = re.findall(r'\w+|[^\w\s]', paraphrased_text.lower())
        
        changes = {
            'added_words': [],
            'removed_words': [],
            'changed_words': [],
            'unchanged_words': [],
            'similarity_score': 0.0
        }
        
        # Hitung kata-kata yang sama
        original_set = set(original_words)
        paraphrased_set = set(paraphrased_words)
        
        changes['unchanged_words'] = list(original_set.intersection(paraphrased_set))
        changes['removed_words'] = list(original_set - paraphrased_set)
        changes['added_words'] = list(paraphrased_set - original_set)
        
        # Hitung similarity score
        if len(original_set.union(paraphrased_set)) > 0:
            changes['similarity_score'] = len(original_set.intersection(paraphrased_set)) / len(original_set.union(paraphrased_set))
        
        return changes
    
    def generate_highlighted_comparison(self, original_text, paraphrased_text):
        """
        Generate comparison dengan highlighting untuk display
        """
        changes = self.get_word_level_changes(original_text, paraphrased_text)
        
        # Highlight untuk teks asli
        original_words = original_text.split()
        original_highlighted = []
        
        for word in original_words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in changes['removed_words']:
                original_highlighted.append(f"~~{word}~~")  # Strikethrough
            elif clean_word in changes['unchanged_words']:
                original_highlighted.append(word)
            else:
                original_highlighted.append(f"**{word}**")  # Bold
        
        # Highlight untuk parafrase
        paraphrased_words = paraphrased_text.split()
        paraphrased_highlighted = []
        
        for word in paraphrased_words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in changes['added_words']:
                paraphrased_highlighted.append(f"*{word}*")  # Italic
            elif clean_word in changes['unchanged_words']:
                paraphrased_highlighted.append(word)
            else:
                paraphrased_highlighted.append(f"**{word}**")  # Bold
        
        return " ".join(original_highlighted), " ".join(paraphrased_highlighted)

class PreferenceTracker:
    """
    Kelas untuk melacak preferensi user
    Mengadopsi dari teststreamlit.py
    """
    
    def __init__(self, preference_file="user_preferences.json"):
        self.preference_file = preference_file
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
        """Load preferensi yang sudah tersimpan"""
        try:
            with open(self.preference_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error loading preferences: {e}")
            return []
    
    def save_preferences(self):
        """Simpan preferensi ke file"""
        try:
            with open(self.preference_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def record_preference(self, original_text, paraphrase_result, chosen_level, user_rating=None):
        """
        Record preferensi user
        """
        preference_data = {
            'original': original_text,
            'paraphrase': paraphrase_result,
            'style': chosen_level,
            'rating': user_rating
        }
        
        self.preferences.append(preference_data)
        
        # Simpan hanya 1000 contoh terakhir
        if len(self.preferences) > 1000:
            self.preferences = self.preferences[-1000:]
        
        self.save_preferences()
    
    def get_preference_statistics(self):
        """Dapatkan statistik preferensi user"""
        if not self.preferences:
            return {"safe": 0.0, "balanced": 0.0, "creative": 0.0}
        
        style_counts = {"safe": 0, "balanced": 0, "creative": 0}
        for pref in self.preferences:
            style = pref.get('style', 'balanced')
            if style in style_counts:
                style_counts[style] += 1
        
        total = len(self.preferences)
        return {
            style: (count / total) * 100 
            for style, count in style_counts.items()
        }

# Factory function untuk mudah digunakan
def create_paraphrase_model(model_name="Wikidepia/IndoT5-base-paraphrase"):
    """
    Factory function untuk membuat model parafrase
    """
    return ParaphraseModel(model_name)

def create_creativity_controlled_model(model_name="Wikidepia/IndoT5-base-paraphrase"):
    """
    Factory function untuk membuat model dengan kontrol kreativitas
    """
    return CreativityControlledParaphraser(model_name)

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi model
    model = create_paraphrase_model()
    
    # Load preferensi jika ada
    model.load_preferences()
    
    # Test generate parafrase
    test_text = "Saya sedang belajar pemrograman Python"
    
    print("Testing paraphrase generation:")
    for style in ["safe", "balanced", "creative"]:
        result = model.generate_paraphrases(test_text, style)
        print(f"{style.capitalize()}: {result}")
        
        # Analisis perubahan kata
        mappings = model.get_word_mappings(test_text, result)
        print(f"Word mappings for {style}: {mappings}")
        print()
    
    # Test analyzer
    analyzer = TextDifferenceAnalyzer()
    original = "Saya sedang belajar pemrograman Python"
    paraphrased = "Aku sedang mempelajari coding Python"
    
    changes = analyzer.get_word_level_changes(original, paraphrased)
    print("Word changes analysis:")
    print(f"Added: {changes['added_words']}")
    print(f"Removed: {changes['removed_words']}")
    print(f"Similarity: {changes['similarity_score']:.2f}")
    
    # Statistik preferensi
    stats = model.get_style_statistics()
    print(f"\nPreference statistics: {stats}")