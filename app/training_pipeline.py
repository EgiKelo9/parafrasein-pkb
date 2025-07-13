import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from sklearn.model_selection import train_test_split 
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from data_collection import DataCollector
from data_preprocessing import IndonesianTextPreprocessor, ParaphraseDataset, create_data_splits
from model_architecture import ParaphraseModel, create_paraphrase_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimplifiedParaphraseTrainer:
    def __init__(self, config):
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        self.data_dir = self.config.get('data_dir', 'data')
        self.log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.max_epochs = self.config.get('max_epochs', 3)
        self.batch_size = self.config.get('batch_size', 2)
        self.learning_rate = self.config.get('learning_rate', 5e-5)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 2)
        self.save_steps = self.config.get('save_steps', 500)
        self.logging_steps = self.config.get('logging_steps', 100)
        self.model_name = self.config.get('model_name', 'Wikidepia/IndoT5-base-paraphrase')
        self.model = None
        self.tokenizer = None
        self.use_mixed_precision = self.config.get('use_mixed_precision', True)
        self.data_sample_ratio = self.config.get('data_sample_ratio', 0.3)
        self.max_steps_per_epoch = self.config.get('max_steps_per_epoch', 100)
        self.global_step = 0
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': []
        }
        self.user_preferences = []
    
    def setup_model(self):
        logger.info("Setting up model using ParaphraseModel architecture...")
        self.paraphrase_model = create_paraphrase_model(self.model_name)
        self.model = self.paraphrase_model.model
        self.tokenizer = self.paraphrase_model.tokenizer
        self.model = self.model.to(self.device)
        logger.info(f"Model initialized: {self.model_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def setup_data_from_preferences(self, preferences_file="user_preferences.json"):
        logger.info("Setting up data from user preferences...")
        try:
            with open(preferences_file, 'r', encoding='utf-8') as f:
                self.user_preferences = json.load(f)
            logger.info(f"Loaded {len(self.user_preferences)} user preferences")
        except FileNotFoundError:
            logger.warning("No user preferences file found. Using dummy data.")
            self.user_preferences = [
                {"original": "Saya sedang belajar", "paraphrase": "Aku sedang mempelajari", "style": "safe"},
                {"original": "Ini adalah contoh", "paraphrase": "Ini merupakan contoh", "style": "balanced"},
                {"original": "Teknologi berkembang pesat", "paraphrase": "Perkembangan teknologi sangat cepat", "style": "creative"}
            ] * 10
        if not self.user_preferences:
            logger.error("No training data available")
            return
        train_data = []
        for pref in self.user_preferences:
            train_data.append({
                "input_text": f"paraphrase: {pref['original']}",
                "target_text": pref["paraphrase"],
                "style": pref.get("style", "balanced")
            })
        df = pd.DataFrame(train_data)
        if len(df) > 1:
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        else:
            train_df = df
            val_df = df
        if self.data_sample_ratio < 1.0:
            train_df = train_df.sample(frac=self.data_sample_ratio, random_state=42)
            val_df = val_df.sample(frac=self.data_sample_ratio, random_state=42)
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)
        self.train_dataset = self.train_dataset.map(
            self.tokenize_data, 
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        self.val_dataset = self.val_dataset.map(
            self.tokenize_data, 
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        logger.info("Data setup completed")
    
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
    
    def train_with_transformers(self):
        logger.info("Starting training with Transformers Trainer...")
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            prediction_loss_only=True,
            fp16=self.use_mixed_precision,
            max_steps=self.max_steps_per_epoch * self.max_epochs
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer
        )
        start_time = time.time()
        try:
            trainer.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                trainer.save_model(os.path.join(self.checkpoint_dir, "final_model"))
            except Exception as e:
                logger.warning(f"Error saving with safetensors: {e}")
                trainer.args.save_safetensors = False
                trainer.save_model(os.path.join(self.checkpoint_dir, "final_model"))
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return f"Training completed! Model saved with {len(self.user_preferences)} preference examples"
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return f"Training failed: {str(e)}"
    
    def fine_tune_from_preferences(self, preferences_file="user_preferences.json"):
        logger.info("Fine-tuning model from user preferences...")
        self.setup_model()
        self.setup_data_from_preferences(preferences_file)
        result = self.train_with_transformers()
        return result
    
    def generate_paraphrase_test(self, text, style="balanced"):
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        temperature_settings = {
            "safe": 0.7,
            "balanced": 1.0,
            "creative": 1.3
        }
        temp = temperature_settings.get(style, 1.0)
        inputs = self.tokenizer.encode(
            f"paraphrase: {text}", 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        try:
            inputs = inputs.to(self.device)
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
    
    def load_fine_tuned_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.checkpoint_dir, "final_model")
        if os.path.exists(model_path):
            try:
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = self.model.to(self.device)
                return "Fine-tuned model loaded successfully"
            except Exception as e:
                return f"Error loading model: {str(e)}"
        return "No fine-tuned model found"
    
    def save_training_history(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        history_file = os.path.join(self.log_dir, f"training_history_{timestamp}.json")
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Training history saved to {history_file}")
        except Exception as e:
            logger.error(f"Error saving training history: {e}")

class LegacyParaphraseTrainer:
    def __init__(self, config):
        pass

def create_default_config():
    return {
        "max_epochs": 3,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "gradient_accumulation_steps": 2,
        "save_steps": 500,
        "logging_steps": 100,
        "model_name": "Wikidepia/IndoT5-base-paraphrase",
        "checkpoint_dir": "checkpoints",
        "data_dir": "data",
        "log_dir": "logs",
        "use_mixed_precision": True,
        "data_sample_ratio": 0.3,
        "max_steps_per_epoch": 100
    }

def main():
    parser = argparse.ArgumentParser(description="ParafraseIn Training Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--preferences", type=str, default="user_preferences.json", help="Path to user preferences file")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    parser.add_argument("--test_text", type=str, default="Saya sedang belajar pemrograman", help="Text for testing")
    parser.add_argument("--style", type=str, choices=["safe", "balanced", "creative"], default="balanced", help="Style for testing")
    args = parser.parse_args()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        logger.warning(f"Config file {args.config} not found. Using default configuration.")
        config = create_default_config()
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Default config saved to {args.config}")
    trainer = SimplifiedParaphraseTrainer(config)
    if args.mode == "train":
        result = trainer.fine_tune_from_preferences(args.preferences)
        logger.info(f"Training result: {result}")
        trainer.save_training_history()
        logger.info("Testing trained model...")
        test_result = trainer.generate_paraphrase_test(args.test_text, args.style)
        logger.info(f"Test result ({args.style}): {test_result}")
    elif args.mode == "test":
        trainer.setup_model()
        load_result = trainer.load_fine_tuned_model()
        logger.info(f"Load result: {load_result}")
        if "successfully" in load_result:
            for style in ["safe", "balanced", "creative"]:
                result = trainer.generate_paraphrase_test(args.test_text, style)
                logger.info(f"Test result ({style}): {result}")
        else:
            logger.error("Failed to load model for testing")

if __name__ == "__main__":
    main()