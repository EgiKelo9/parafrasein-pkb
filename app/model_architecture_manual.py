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
    def __init__(self, model_name="Wikidepia/IndoT5-base-paraphrase"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.user_preferences = []
        self.temperature_settings = {
            "safe": 0.7,
            "balanced": 1.0,
            "creative": 1.3
        }
        self.style_to_index = {
            "safe": 0,
            "balanced": 1,
            "creative": 2
        }
        self.index_to_style = {v: k for k, v in self.style_to_index.items()}
    
    def generate_paraphrases(self, text, style="safe", show_manual_calculation=False):
        if not text.strip():
            if show_manual_calculation:
                return {"error": "Please enter text to paraphrase", "steps": []}
            return "Please enter text to paraphrase"
        if show_manual_calculation:
            return self._generate_with_manual_calculation(text, style)
        temp = self.temperature_settings.get(style, 0.7)
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
    
    def generate_all_styles(self, text, show_manual_calculation=False):
        results = {}
        for style in ["safe", "balanced", "creative"]:
            results[style] = self.generate_paraphrases(text, style, show_manual_calculation)
        return results
    def demonstrate_complete_process(self, text, style="balanced"):
        print("="*80)
        print("DEMONSTRASI PERHITUNGAN MANUAL PARAFRASE")
        print("="*80)
        result = self.generate_paraphrases(text, style, show_manual_calculation=True)
        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown error')}")
            return result
        print(f"Input Text: '{result['input']}'")
        print(f"Output Text: '{result['output']}'")
        print(f"Style: {result['style']}")
        print(f"Temperature: {result['summary']['temperature']}")
        print(f"Similarity Score: {result['summary']['similarity']:.4f}")
        print()
        for step in result['steps']:
            print(f"STEP {step['step']}: {step['description']}")
            print("-" * 50)
            print(f"Explanation: {step['explanation']}")
            if step['step'] == 1:  # Input preprocessing
                print(f"  Original Input: '{step['input']}'")
                print(f"  Formatted Input: '{step['output']}'")
            elif step['step'] == 2:  # Tokenization
                print(f"  Input Text: '{step['input']}'")
                print(f"  Number of Tokens: {len(step['tokens'])}")
                print(f"  Tokens: {step['tokens'][:10]}{'...' if len(step['tokens']) > 10 else ''}")
                print(f"  Token IDs: {step['token_ids'][:10]}{'...' if len(step['token_ids']) > 10 else ''}")
                print(f"  Vocabulary Size: {step['vocab_size']:,}")
            elif step['step'] == 3:  # Input encoding
                print(f"  Input Tensor Shape: {step['input_shape']}")
                print(f"  Max Length: {step['max_length']}")
            elif step['step'] == 4:  # Temperature
                print(f"  Style: {step['style']}")
                print(f"  Temperature: {step['temperature']}")
            elif step['step'] == 5:  # Model config
                print(f"  Model: {step['model_name']}")
                print(f"  Vocabulary Size: {step['vocab_size']:,}")
                print(f"  Model Dimension: {step['d_model']}")
                print(f"  Number of Layers: {step['num_layers']}")
                print(f"  Number of Attention Heads: {step['num_heads']}")
            elif step['step'] == 6:  # Generation
                print(f"  Output Shape: {step['output_shape']}")
                print(f"  Generation Config: {step['generation_config']}")
            elif step['step'] == 7:  # Token details
                print(f"  Input Length: {step['input_length']} tokens")
                print(f"  Generated Length: {step['generated_length']} tokens")
                print("  Generated Tokens:")
                for token_detail in step['token_details'][:10]:  # Show first 10 tokens
                    print(f"    Position {token_detail['position']}: ID={token_detail['token_id']}, Text='{token_detail['token_text']}', Special={token_detail['is_special']}")
                if len(step['token_details']) > 10:
                    print(f"    ... and {len(step['token_details']) - 10} more tokens")
            elif step['step'] == 8:  # Decoding
                print(f"  Raw Output: '{step['raw_output']}'")
                print(f"  Final Output: '{step['final_output']}'")
            elif step['step'] == 9:  # Analysis
                print(f"  Original: '{step['original_text']}'")
                print(f"  Paraphrased: '{step['paraphrased_text']}'")
                print(f"  Similarity Score: {step['similarity_score']:.4f}")
                print(f"  Number of Changes: {len(step['word_changes'])}")
            print()
        print("="*80)
        print("SUMMARY")
        print("="*80)
        summary = result['summary']
        print(f"Total Processing Steps: {summary['total_steps']}")
        print(f"Input Tokens: {summary['input_tokens']}")
        print(f"Output Tokens: {summary['output_tokens']}")
        print(f"Temperature Used: {summary['temperature']}")
        print(f"Final Similarity: {summary['similarity']:.4f}")
        print("="*80)
        return result
    
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
    
    def _tokenize_text(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
    
    def get_word_mappings(self, original_text, paraphrased_text):
        changes = self.analyze_word_changes(original_text, paraphrased_text)
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
        preference = {
            "original": original_text,
            "paraphrase": paraphrase,
            "style": style
        }
        self.user_preferences.append(preference)
        try:
            with open("user_preferences.json", "w", encoding='utf-8') as f:
                json.dump(self.user_preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving preferences to file: {e}")
        return f"Preference saved! Total preferences: {len(self.user_preferences)}"
    
    def load_preferences(self):
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
        if not self.user_preferences:
            return "No user preferences for fine-tuning"
        train_dataset = self.prepare_training_data()
        train_dataset = train_dataset.map(
            self.tokenize_data, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
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
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
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
        if os.path.exists(model_path):
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            return "Fine-tuned model loaded successfully"
        return "No fine-tuned model found"
    
    def get_style_statistics(self):
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
    
    def _generate_with_manual_calculation(self, text, style="safe"):
        steps = []
        # Step 1: Input preprocessing
        formatted_input = f"paraphrase: {text}"
        steps.append({
            "step": 1,
            "description": "Input Preprocessing",
            "input": text,
            "output": formatted_input,
            "explanation": "Menambahkan prefix 'paraphrase:' ke input text untuk memberikan instruksi ke model"
        })
        # Step 2: Tokenization
        tokens = self.tokenizer.tokenize(formatted_input)
        token_ids = self.tokenizer.encode(formatted_input, add_special_tokens=True)
        steps.append({
            "step": 2,
            "description": "Tokenization",
            "input": formatted_input,
            "tokens": tokens,
            "token_ids": token_ids,
            "vocab_size": self.tokenizer.vocab_size,
            "explanation": f"Memecah text menjadi {len(tokens)} token menggunakan tokenizer. Setiap token dikonversi ke ID numerik."
        })
        # Step 3: Input encoding
        inputs = self.tokenizer.encode(
            formatted_input, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        steps.append({
            "step": 3,
            "description": "Input Encoding",
            "input_shape": list(inputs.shape),
            "input_tensor": inputs.tolist(),
            "max_length": 512,
            "explanation": f"Mengkonversi token IDs ke tensor PyTorch dengan shape {list(inputs.shape)}"
        })
        # Step 4: Temperature setting
        temp = self.temperature_settings.get(style, 0.7)
        steps.append({
            "step": 4,
            "description": "Temperature Configuration",
            "style": style,
            "temperature": temp,
            "explanation": f"Setting temperature {temp} untuk style '{style}'. Higher temperature = more creative/random output."
        })
        # Step 5: Model inference
        try:
            with torch.no_grad():
                model_config = self.model.config
                steps.append({
                    "step": 5,
                    "description": "Model Configuration",
                    "model_name": self.model_name,
                    "vocab_size": model_config.vocab_size,
                    "d_model": model_config.d_model,
                    "num_layers": model_config.num_layers,
                    "num_heads": model_config.num_heads,
                    "explanation": "Konfigurasi model T5 yang digunakan untuk generation"
                })
                # Step 6: Generation process
                outputs = self.model.generate(
                    inputs,
                    max_length=512,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                generated_ids = outputs.sequences[0]
                steps.append({
                    "step": 6,
                    "description": "Generation Process",
                    "generation_config": {
                        "max_length": 512,
                        "temperature": temp,
                        "do_sample": True,
                        "num_return_sequences": 1
                    },
                    "output_shape": list(generated_ids.shape),
                    "generated_ids": generated_ids.tolist(),
                    "explanation": "Model melakukan autoregressive generation, memprediksi token satu per satu"
                })
                # Step 7: Token-by-token generation detail
                input_length = inputs.shape[1]
                generated_tokens_only = generated_ids[input_length:]
                token_details = []
                for i, token_id in enumerate(generated_tokens_only.tolist()):
                    token_text = self.tokenizer.decode([token_id])
                    token_details.append({
                        "position": i + 1,
                        "token_id": token_id,
                        "token_text": token_text,
                        "is_special": token_id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
                    })
                steps.append({
                    "step": 7,
                    "description": "Token-by-Token Generation Detail",
                    "input_length": input_length,
                    "generated_length": len(generated_tokens_only),
                    "token_details": token_details,
                    "explanation": "Detail setiap token yang di-generate oleh model"
                })
                # Step 8: Decoding
                final_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                steps.append({
                    "step": 8,
                    "description": "Decoding",
                    "raw_output": self.tokenizer.decode(generated_ids, skip_special_tokens=False),
                    "final_output": final_output,
                    "explanation": "Mengkonversi token IDs kembali ke text dan menghilangkan special tokens"
                })
                # Step 9: Analysis
                word_changes = self.analyze_word_changes(text, final_output)
                similarity_score = self._calculate_similarity(text, final_output)
                steps.append({
                    "step": 9,
                    "description": "Output Analysis",
                    "original_text": text,
                    "paraphrased_text": final_output,
                    "word_changes": word_changes,
                    "similarity_score": similarity_score,
                    "explanation": "Analisis perubahan antara input dan output"
                })
                return {
                    "success": True,
                    "input": text,
                    "output": final_output,
                    "style": style,
                    "steps": steps,
                    "summary": {
                        "total_steps": len(steps),
                        "input_tokens": len(tokens),
                        "output_tokens": len(generated_tokens_only),
                        "temperature": temp,
                        "similarity": similarity_score
                    }
                }
        except Exception as e:
            steps.append({
                "step": "ERROR",
                "description": "Error in generation",
                "error": str(e),
                "explanation": "Terjadi error during generation process"
            })
            return {
                "success": False,
                "error": str(e),
                "steps": steps
            }
    
    def _calculate_similarity(self, text1, text2):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def show_model_architecture(self):
        config = self.model.config
        architecture_info = {
            "model_type": "T5ForConditionalGeneration",
            "model_name": self.model_name,
            "parameters": {
                "vocab_size": config.vocab_size,
                "d_model": config.d_model,
                "d_kv": config.d_kv,
                "d_ff": config.d_ff,
                "num_layers": config.num_layers,
                "num_decoder_layers": config.num_decoder_layers,
                "num_heads": config.num_heads,
                "relative_attention_num_buckets": config.relative_attention_num_buckets,
                "dropout_rate": config.dropout_rate,
                "layer_norm_epsilon": config.layer_norm_epsilon,
                "initializer_factor": config.initializer_factor,
                "feed_forward_proj": config.feed_forward_proj
            },
            "special_tokens": {
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token,
                "unk_token": self.tokenizer.unk_token,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "unk_token_id": self.tokenizer.unk_token_id
            }
        }
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        architecture_info["parameter_count"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_parameters_formatted": f"{total_params:,}",
            "trainable_parameters_formatted": f"{trainable_params:,}"
        }
        return architecture_info
    
    def analyze_attention_weights(self, text, style="safe"):
        formatted_input = f"paraphrase: {text}"
        inputs = self.tokenizer.encode(formatted_input, return_tensors="pt", max_length=512, truncation=True)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=512,
                    temperature=self.temperature_settings.get(style, 0.7),
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    output_scores=True
                )
                input_tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])
                return {
                    "input_tokens": input_tokens,
                    "input_length": len(input_tokens),
                    "attention_available": outputs.attentions is not None,
                    "num_layers": len(outputs.attentions) if outputs.attentions else 0,
                    "explanation": "Attention weights menunjukkan bagian mana dari input yang difokuskan model saat generating setiap token output"
                }
        except Exception as e:
            return {
                "error": str(e),
                "explanation": "Error in attention analysis"
            }
            
class CreativityControlledParaphraser(nn.Module):
    def __init__(self, base_model_name="Wikidepia/IndoT5-base-paraphrase"):
        super(CreativityControlledParaphraser, self).__init__()
        self.paraphrase_model = ParaphraseModel(base_model_name)
        self.creativity_embedding = nn.Embedding(3, 768) 
        self.diversity_controller = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_text, creativity_level="balanced", show_calculation=False):
        if show_calculation:
            return self.paraphrase_model.generate_paraphrases(input_text, creativity_level, show_manual_calculation=True)
        return self.paraphrase_model.generate_paraphrases(input_text, creativity_level)
    
    def generate_with_analysis(self, input_text, creativity_level="balanced", show_calculation=False):
        if show_calculation:
            detailed_result = self.paraphrase_model.generate_paraphrases(input_text, creativity_level, show_manual_calculation=True)
            if detailed_result.get("success", False):
                paraphrase = detailed_result["output"]
                word_mappings = self.paraphrase_model.get_word_mappings(input_text, paraphrase)
                changes = self.paraphrase_model.analyze_word_changes(input_text, paraphrase)
                detailed_result.update({
                    'word_mappings': word_mappings,
                    'changes': changes,
                    'style': creativity_level
                })
            return detailed_result
        else:
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
    def __init__(self):
        pass
    
    def get_word_level_changes(self, original_text, paraphrased_text):
        original_words = re.findall(r'\w+|[^\w\s]', original_text.lower())
        paraphrased_words = re.findall(r'\w+|[^\w\s]', paraphrased_text.lower())
        changes = {
            'added_words': [],
            'removed_words': [],
            'changed_words': [],
            'unchanged_words': [],
            'similarity_score': 0.0
        }
        original_set = set(original_words)
        paraphrased_set = set(paraphrased_words)
        changes['unchanged_words'] = list(original_set.intersection(paraphrased_set))
        changes['removed_words'] = list(original_set - paraphrased_set)
        changes['added_words'] = list(paraphrased_set - original_set)
        if len(original_set.union(paraphrased_set)) > 0:
            changes['similarity_score'] = len(original_set.intersection(paraphrased_set)) / len(original_set.union(paraphrased_set))
        return changes
    
    def generate_highlighted_comparison(self, original_text, paraphrased_text):
        changes = self.get_word_level_changes(original_text, paraphrased_text)
        original_words = original_text.split()
        original_highlighted = []
        for word in original_words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in changes['removed_words']:
                original_highlighted.append(f"~~{word}~~")
            elif clean_word in changes['unchanged_words']:
                original_highlighted.append(word)
            else:
                original_highlighted.append(f"**{word}**")
        paraphrased_words = paraphrased_text.split()
        paraphrased_highlighted = []
        for word in paraphrased_words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in changes['added_words']:
                paraphrased_highlighted.append(f"*{word}*")
            elif clean_word in changes['unchanged_words']:
                paraphrased_highlighted.append(word)
            else:
                paraphrased_highlighted.append(f"**{word}**")
        
        return " ".join(original_highlighted), " ".join(paraphrased_highlighted)

class PreferenceTracker:
    def __init__(self, preference_file="user_preferences.json"):
        self.preference_file = preference_file
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
        try:
            with open(self.preference_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error loading preferences: {e}")
            return []
    
    def save_preferences(self):
        try:
            with open(self.preference_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def record_preference(self, original_text, paraphrase_result, chosen_level, user_rating=None):
        preference_data = {
            'original': original_text,
            'paraphrase': paraphrase_result,
            'style': chosen_level,
            'rating': user_rating
        }
        self.preferences.append(preference_data)
        if len(self.preferences) > 1000:
            self.preferences = self.preferences[-1000:]
        self.save_preferences()
    
    def get_preference_statistics(self):
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

def create_paraphrase_model(model_name="Wikidepia/IndoT5-base-paraphrase"):
    return ParaphraseModel(model_name)

def create_creativity_controlled_model(model_name="Wikidepia/IndoT5-base-paraphrase"):
    return CreativityControlledParaphraser(model_name)

# Contoh penggunaan
if __name__ == "__main__":
    model = create_paraphrase_model()
    model.load_preferences()
    test_text = "Saya sedang belajar pemrograman Python"
    print("Testing paraphrase generation:")
    for style in ["safe", "balanced", "creative"]:
        result = model.generate_paraphrases(test_text, style)
        print(f"{style.capitalize()}: {result}")
        mappings = model.get_word_mappings(test_text, result)
        print(f"Word mappings for {style}: {mappings}")
        print()
    print("\n" + "="*80)
    print("DEMONSTRASI PERHITUNGAN MANUAL DETAIL")
    print("="*80)
    detailed_result = model.demonstrate_complete_process(test_text, "balanced")
    print("\n" + "="*80)
    print("ARSITEKTUR MODEL")
    print("="*80)
    architecture = model.show_model_architecture()
    print(f"Model Type: {architecture['model_type']}")
    print(f"Model Name: {architecture['model_name']}")
    print(f"Total Parameters: {architecture['parameter_count']['total_parameters_formatted']}")
    print(f"Trainable Parameters: {architecture['parameter_count']['trainable_parameters_formatted']}")
    print("\nModel Configuration:")
    for key, value in architecture['parameters'].items():
        print(f"  {key}: {value}")
    print("\nSpecial Tokens:")
    for key, value in architecture['special_tokens'].items():
        print(f"  {key}: {value}")
    print("\n" + "="*80)
    print("ANALISIS ATTENTION (Simplified)")
    print("="*80)
    attention_analysis = model.analyze_attention_weights(test_text, "balanced")
    if "error" not in attention_analysis:
        print(f"Input Length: {attention_analysis['input_length']} tokens")
        print(f"Attention Available: {attention_analysis['attention_available']}")
        print(f"Number of Layers: {attention_analysis['num_layers']}")
        print(f"Input Tokens: {attention_analysis['input_tokens'][:10]}...")
    else:
        print(f"Error in attention analysis: {attention_analysis['error']}")
    print("\n" + "="*80)
    print("TEXT DIFFERENCE ANALYSIS")
    print("="*80)
    analyzer = TextDifferenceAnalyzer()
    original = "Saya sedang belajar pemrograman Python"
    paraphrased = "Aku sedang mempelajari coding Python"
    changes = analyzer.get_word_level_changes(original, paraphrased)
    print(f"Original: '{original}'")
    print(f"Paraphrased: '{paraphrased}'")
    print(f"Added: {changes['added_words']}")
    print(f"Removed: {changes['removed_words']}")
    print(f"Unchanged: {changes['unchanged_words']}")
    print(f"Similarity: {changes['similarity_score']:.2f}")
    orig_highlighted, para_highlighted = analyzer.generate_highlighted_comparison(original, paraphrased)
    print(f"\nHighlighted Original: {orig_highlighted}")
    print(f"Highlighted Paraphrased: {para_highlighted}")
    stats = model.get_style_statistics()
    print(f"\nPreference statistics: {stats}")
    print("\n" + "="*80)
    print("CREATIVITY CONTROLLED MODEL TEST")
    print("="*80)
    creativity_model = create_creativity_controlled_model()
    detailed_creative_result = creativity_model.generate_with_analysis(
        test_text, 
        "creative", 
        show_calculation=True
    )
    if detailed_creative_result.get("success", False):
        print(f"Creativity Model Input: '{detailed_creative_result['input']}'")
        print(f"Creativity Model Output: '{detailed_creative_result['output']}'")
        print(f"Style: {detailed_creative_result['style']}")
        print(f"Total Steps: {detailed_creative_result['summary']['total_steps']}")
        print(f"Word Mappings: {detailed_creative_result.get('word_mappings', [])}")
    else:
        print(f"Error in creativity model: {detailed_creative_result.get('error', 'Unknown error')}")
    print("\n" + "="*80)
    print("COMPLETE PROCESS DEMONSTRATION FINISHED")
    print("="*80)