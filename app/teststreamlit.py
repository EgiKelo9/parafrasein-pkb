import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import json
import os
import gc
import difflib
from collections import defaultdict
import re

class ParaphraseApp:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoT5-base-paraphrase")
        self.model = T5ForConditionalGeneration.from_pretrained("Wikidepia/IndoT5-base-paraphrase")
        self.dataset = None
        self.user_preferences = []
        
    def load_dataset(self):
        self.dataset = load_dataset("jakartaresearch/id-paraphrase-detection")
        return f"Dataset loaded: {len(self.dataset['train'])} training samples"
    
    def generate_paraphrases(self, text, style="safe"):
        if not text.strip():
            return ["Please enter text to paraphrase"]
        
        # Different temperature settings for different styles
        temperature_settings = {
            "safe": 0.7,
            "balanced": 1.0,
            "creative": 1.3
        }
        
        temp = temperature_settings.get(style, 0.7)
        
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
    
    def analyze_word_changes(self, original_text, paraphrased_text):
        """
        Analyze the differences between original and paraphrased text.
        This function identifies changed, added, and removed words.
        """
        # Clean and tokenize the texts
        original_words = self._tokenize_text(original_text.lower())
        paraphrased_words = self._tokenize_text(paraphrased_text.lower())
        
        # Use difflib to find the differences
        # SequenceMatcher helps us find the longest matching subsequences
        matcher = difflib.SequenceMatcher(None, original_words, paraphrased_words)
        
        changes = []
        original_index = 0
        paraphrased_index = 0
        
        # Get the matching blocks from difflib
        for match in matcher.get_matching_blocks():
            # Handle words before the matching block (these are changes)
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
            
            # Move indices to the end of the matching block
            original_index = match.a + match.size
            paraphrased_index = match.b + match.size
        
        return changes
    
    def _tokenize_text(self, text):
        """
        Simple tokenization that preserves punctuation and handles Indonesian text well.
        """
        # Use regex to split on whitespace while preserving punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
    
    def get_word_mappings(self, original_text, paraphrased_text):
        """
        Create a simplified mapping of key word changes for display.
        This focuses on the most significant substitutions.
        """
        changes = self.analyze_word_changes(original_text, paraphrased_text)
        
        # Filter for substitutions and significant changes
        mappings = []
        for change in changes:
            if change['type'] == 'substitution':
                original = change['original'].strip()
                paraphrased = change['paraphrased'].strip()
                
                # Only include meaningful changes (not just punctuation or very short words)
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
        
        # Save to session state and sync with app instance
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = []
        st.session_state.user_preferences.append(preference)
        
        # Update app instance preferences from session state
        app = st.session_state.get('paraphrase_app')
        if app:
            app.user_preferences = st.session_state.user_preferences
        
        # Save to file
        try:
            with open("user_preferences.json", "w", encoding='utf-8') as f:
                json.dump(st.session_state.user_preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving preferences to file: {e}")
        
        return f"Preference saved! Total preferences: {len(st.session_state.user_preferences)}"
    
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
    
    def fine_tune_model(self, epochs=3, learning_rate=5e-5):
        if not self.user_preferences:
            return "No user preferences for fine-tuning"
        
        train_dataset = self.prepare_training_data()
        train_dataset = train_dataset.map(
            self.tokenize_data, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
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
        
        # Clear cache and save model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            trainer.save_model("./fine_tuned_model")
        except Exception as e:
            trainer.args.save_safetensors = False
            trainer.save_model("./fine_tuned_model")
        
        return f"Fine-tuning completed! Model saved with {len(self.user_preferences)} preference examples"
    
    def load_fine_tuned_model(self):
        if os.path.exists("./fine_tuned_model"):
            self.model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_model")
            self.tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
            return "Fine-tuned model loaded successfully"
        return "No fine-tuned model found"

# Initialize the app
@st.cache_resource
def load_paraphrase_app():
    return ParaphraseApp()

def display_word_changes(original_text, paraphrased_text, app):
    """
    Display the word changes section with proper styling to match the interface.
    This function creates the visual representation of word transformations.
    """
    st.markdown('<div class="section-header">🔄 Perubahan Kata</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah kata-kata yang telah diparafrase:")
    
    word_mappings = app.get_word_mappings(original_text, paraphrased_text)
    
    if not word_mappings:
        st.info("Tidak ada perubahan kata yang signifikan terdeteksi.")
        return
    
    # Display word changes in a grid format like the reference image
    for i, mapping in enumerate(word_mappings[:5]):  # Limit to 5 most significant changes
        col1, col2, col3 = st.columns([1, 0.2, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: #e53e3e; color: white; padding: 1rem; border-radius: 5px; text-align: center; margin-bottom: 1rem;">
                <strong>"{mapping['original']}"</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem; color: #cccccc;">→</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: #48bb78; color: white; padding: 1rem; border-radius: 5px; text-align: center; margin-bottom: 1rem;">
                <strong>"{mapping['paraphrased']}"</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Show detailed analysis in an expandable section
    with st.expander("Lihat Analisis Lengkap Perubahan"):
        changes = app.analyze_word_changes(original_text, paraphrased_text)
        
        st.markdown("**Ringkasan Perubahan:**")
        substitutions = [c for c in changes if c['type'] == 'substitution']
        additions = [c for c in changes if c['type'] == 'addition']
        deletions = [c for c in changes if c['type'] == 'deletion']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Substitusi", len(substitutions))
        with col2:
            st.metric("Penambahan", len(additions))
        with col3:
            st.metric("Penghapusan", len(deletions))
        
        if changes:
            st.markdown("**Detail Perubahan:**")
            for change in changes:
                if change['type'] == 'substitution':
                    st.write(f"🔄 **{change['original']}** → **{change['paraphrased']}**")
                elif change['type'] == 'addition':
                    st.write(f"➕ Ditambahkan: **{change['paraphrased']}**")
                elif change['type'] == 'deletion':
                    st.write(f"➖ Dihapus: **{change['original']}**")

def main():
    st.set_page_config(
        page_title="ParafraseIn",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for styling (enhanced with word change styling)
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #cccccc;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
    }
    .style-card {
        background: #2d3748;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid transparent;
    }
    .style-card:hover {
        border-color: #4a5568;
    }
    .style-card.selected {
        border-color: #4299e1;
    }
    .success-message {
        background: #22543d;
        color: #68d391;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .word-change {
        background: #1a202c;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .word-change-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 1rem;
        align-items: center;
        margin-bottom: 1rem;
    }
    .original-word {
        background: #e53e3e;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .paraphrased-word {
        background: #48bb78;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .arrow {
        font-size: 1.5rem;
        color: #cccccc;
        text-align: center;
    }
    .stTextArea > div > div > textarea {
        background-color: #2d3748;
        color: #ffffff;
        border: 1px solid #4a5568;
    }
    .stButton > button {
        background-color: #e53e3e;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #c53030;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load the app
    app = load_paraphrase_app()
    
    # Store app instance in session state for preference syncing
    st.session_state.paraphrase_app = app
    
    # Load existing preferences if available
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = []
        # Try to load from file
        try:
            with open("user_preferences.json", "r", encoding='utf-8') as f:
                st.session_state.user_preferences = json.load(f)
                app.user_preferences = st.session_state.user_preferences
        except:
            pass
    
    # Initialize session state
    if 'paraphrase_results' not in st.session_state:
        st.session_state.paraphrase_results = {}
    if 'selected_style' not in st.session_state:
        st.session_state.selected_style = None
    if 'selected_paraphrase' not in st.session_state:
        st.session_state.selected_paraphrase = None
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = []
    
    # Header
    st.markdown('<div class="main-header">ParafraseIn</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sebuah platform parafrase online untuk membantu kamu mengubah teks mentah menjadi lebih baik dan lebih mudah dipahami. Dengan teknologi AI, ParafraseIn dapat membantu kamu dalam bidang akademik</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="section-header">Input Teks</div>', unsafe_allow_html=True)
    st.markdown("Masukkan teks yang ingin diparafrase:")
    
    input_text = st.text_area(
        "Input Text",
        placeholder="contoh: aku coba rumah tinggal",
        height=150,
        key="input_text",
        label_visibility="collapsed"
    )
    
    if st.button("Mulai Parafrase", key="generate_btn"):
        if input_text.strip():
            # Generate paraphrases for all styles
            styles = ["safe", "balanced", "creative"]
            results = {}
            
            with st.spinner("Generating paraphrases..."):
                for style in styles:
                    results[style] = app.generate_paraphrases(input_text, style)
            
            st.session_state.paraphrase_results = results
            st.session_state.selected_style = None  # Reset selection
            st.session_state.selected_paraphrase = None
            st.success("✅ Parafrase berhasil dihasilkan!")
        else:
            st.error("Please enter text to paraphrase")
    
    # Results section
    if st.session_state.paraphrase_results:
        st.markdown('<div class="section-header">Pilih Hasil Parafrase yang Kamu Sukai</div>', unsafe_allow_html=True)
        st.markdown("Pilih salah satu hasil parafrase di bawah ini:")
        
        # Style options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="style-card">
                <h4>🛡️ Hasil Safe</h4>
                <p><em>Parafrase dengan perubahan minimal</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.paraphrase_results.get("safe"):
                st.text_area(
                    "Safe Result",
                    value=st.session_state.paraphrase_results["safe"],
                    height=200,
                    key="safe_result",
                    disabled=True,
                    label_visibility="collapsed"
                )
                if st.button("Pilih Safe", key="select_safe"):
                    st.session_state.selected_style = "safe"
                    st.session_state.selected_paraphrase = st.session_state.paraphrase_results["safe"]
                    # Save the preference
                    result = app.save_preference(
                        st.session_state.get("input_text", ""),
                        st.session_state.paraphrase_results["safe"],
                        "safe"
                    )
                    st.success("Safe style selected and preference saved!")
                    st.rerun()  # Refresh to show word changes
        
        with col2:
            st.markdown("""
            <div class="style-card">
                <h4>⚖️ Hasil Balanced</h4>
                <p><em>Parafrase dengan keseimbangan perubahan</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.paraphrase_results.get("balanced"):
                st.text_area(
                    "Balanced Result",
                    value=st.session_state.paraphrase_results["balanced"],
                    height=200,
                    key="balanced_result",
                    disabled=True,
                    label_visibility="collapsed"
                )
                if st.button("Pilih Balanced", key="select_balanced"):
                    st.session_state.selected_style = "balanced"
                    st.session_state.selected_paraphrase = st.session_state.paraphrase_results["balanced"]
                    # Save the preference
                    result = app.save_preference(
                        st.session_state.get("input_text", ""),
                        st.session_state.paraphrase_results["balanced"],
                        "balanced"
                    )
                    st.success("Balanced style selected and preference saved!")
                    st.rerun()  # Refresh to show word changes
        
        with col3:
            st.markdown("""
            <div class="style-card">
                <h4>🎨 Hasil Creative</h4>
                <p><em>Parafrase dengan perubahan kreatif</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.paraphrase_results.get("creative"):
                st.text_area(
                    "Creative Result",
                    value=st.session_state.paraphrase_results["creative"],
                    height=200,
                    key="creative_result",
                    disabled=True,
                    label_visibility="collapsed"
                )
                if st.button("Pilih Creative", key="select_creative"):
                    st.session_state.selected_style = "creative"
                    st.session_state.selected_paraphrase = st.session_state.paraphrase_results["creative"]
                    # Save the preference
                    result = app.save_preference(
                        st.session_state.get("input_text", ""),
                        st.session_state.paraphrase_results["creative"],
                        "creative"
                    )
                    st.success("Creative style selected and preference saved!")
                    st.rerun()  # Refresh to show word changes
        
        # Word changes section - only show when a style is selected
        if st.session_state.selected_style and st.session_state.selected_paraphrase:
            original_text = st.session_state.get("input_text", "")
            if original_text.strip():
                display_word_changes(original_text, st.session_state.selected_paraphrase, app)
    
    # Sidebar for additional features
    with st.sidebar:
        st.header("Advanced Features")
        
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                result = app.load_dataset()
                st.success(result)
        
        st.subheader("Fine-tuning")
        epochs = st.slider("Training Epochs", 1, 10, 3)
        learning_rate = st.number_input("Learning Rate", value=5e-5, format="%.2e")
        
        if st.button("Start Fine-tuning"):
            # Sync preferences before fine-tuning
            app.user_preferences = st.session_state.user_preferences
            if st.session_state.user_preferences:
                with st.spinner("Fine-tuning model..."):
                    result = app.fine_tune_model(epochs, learning_rate)
                    st.success(result)
            else:
                st.error("No user preferences found for fine-tuning")
        
        if st.button("Load Fine-tuned Model"):
            result = app.load_fine_tuned_model()
            if "successfully" in result:
                st.success(result)
            else:
                st.error(result)
        
        # Display preferences count
        st.subheader("User Preferences")
        st.info(f"Total saved preferences: {len(st.session_state.user_preferences)}")
        
        if st.button("Clear Preferences"):
            st.session_state.user_preferences = []
            app.user_preferences = []
            # Clear the file too
            try:
                with open("user_preferences.json", "w", encoding='utf-8') as f:
                    json.dump([], f)
            except:
                pass
            st.success("Preferences cleared!")

if __name__ == "__main__":
    main()