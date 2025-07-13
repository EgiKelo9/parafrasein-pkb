import streamlit as st
import torch
import json
import os
import gc
import difflib
import re
from collections import defaultdict
import pandas as pd
from st_copy_to_clipboard import st_copy_to_clipboard

from model_architecture import ParaphraseModel, create_paraphrase_model, TextDifferenceAnalyzer, PreferenceTracker
from training_pipeline import SimplifiedParaphraseTrainer, create_default_config

class ParafraseInApp:
    def __init__(self):
        self.paraphrase_model = create_paraphrase_model("Wikidepia/IndoT5-base-paraphrase")
        self.trainer = None
        self.config = create_default_config()
        self.text_analyzer = TextDifferenceAnalyzer()
        self.preference_tracker = PreferenceTracker("user_preferences.json")
        self.user_preferences = self.preference_tracker.load_preferences()
        
    def load_fine_tuned_model_if_exists(self):
        fine_tuned_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'final_model')
        if os.path.exists(fine_tuned_path):
            result = self.paraphrase_model.load_fine_tuned_model(fine_tuned_path)
            return result
        return "No fine-tuned model found"
    
    def generate_paraphrases(self, text, style="safe"):
        if not text.strip():
            return "Please enter text to paraphrase"
        return self.paraphrase_model.generate_paraphrases(text, style)
    
    def generate_all_styles(self, text):
        return self.paraphrase_model.generate_all_styles(text)
    
    def analyze_word_changes(self, original_text, paraphrased_text):
        return self.paraphrase_model.analyze_word_changes(original_text, paraphrased_text)
    
    def get_word_mappings(self, original_text, paraphrased_text):
        return self.paraphrase_model.get_word_mappings(original_text, paraphrased_text)
    
    def save_preference(self, original_text, paraphrase, style):
        self.preference_tracker.record_preference(original_text, paraphrase, style)
        preference = {
            "original": original_text,
            "paraphrase": paraphrase,
            "style": style
        }
        self.paraphrase_model.user_preferences.append(preference)
        self.paraphrase_model.save_preference(original_text, paraphrase, style)
        return f"Preference saved! Total preferences: {len(self.preference_tracker.preferences)}"
    
    def get_preference_statistics(self):
        return self.preference_tracker.get_preference_statistics()
    
    def fine_tune_model(self, epochs=3, learning_rate=5e-5):
        try:
            if not self.trainer:
                self.trainer = SimplifiedParaphraseTrainer(self.config)
            self.paraphrase_model.user_preferences = [
                {"original": pref["original"], "paraphrase": pref["paraphrase"], "style": pref["style"]}
                for pref in self.preference_tracker.preferences
            ]
            self.config['max_epochs'] = epochs
            self.config['learning_rate'] = learning_rate
            result = self.trainer.fine_tune_from_preferences("user_preferences.json")
            if "completed" in result:
                load_result = self.load_fine_tuned_model_if_exists()
                if "successfully" in load_result:
                    return f"{result}\n{load_result}"
            return result
        except Exception as e:
            return f"Fine-tuning failed: {str(e)}"

# Initialize the app
@st.cache_resource
def load_parafrase_app():
    app = ParafraseInApp()
    app.load_fine_tuned_model_if_exists()
    return app

def display_word_changes(original_text, paraphrased_text, app):
    st.markdown('<div class="section-header">🔄 Perubahan Kata</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah kata-kata yang telah diparafrase:")
    word_mappings = app.get_word_mappings(original_text, paraphrased_text)
    if not word_mappings:
        st.info("Tidak ada perubahan kata yang signifikan terdeteksi.")
        return
    for i, mapping in enumerate(word_mappings[:5]): 
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

def display_preference_statistics(app):
    stats = app.get_preference_statistics()
    st.subheader("📊 Statistik Preferensi")
    if stats and any(stats.values()):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Safe", f"{stats.get('safe', 0):.1f}%")
        with col2:
            st.metric("Balanced", f"{stats.get('balanced', 0):.1f}%")
        with col3:
            st.metric("Creative", f"{stats.get('creative', 0):.1f}%")
    else:
        st.info("Belum ada data preferensi")

def main():
    st.set_page_config(
        page_title="ParafraseIn",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
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
    .copy-button {
        background-color: #4a5568;
        color: white;
        border: none;
        border-radius: 3px;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
        margin-top: 0.5rem;
        width: 100%;
    }
    .copy-button:hover {
        background-color: #2d3748;
    }
    </style>
    """, unsafe_allow_html=True)
    app = load_parafrase_app()
    if 'paraphrase_results' not in st.session_state:
        st.session_state.paraphrase_results = {}
    if 'selected_style' not in st.session_state:
        st.session_state.selected_style = None
    if 'selected_paraphrase' not in st.session_state:
        st.session_state.selected_paraphrase = None
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    st.markdown('<div class="main-header">ParafraseIn</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sebuah platform parafrase online untuk membantu kamu mengubah teks mentah menjadi lebih baik dan lebih mudah dipahami. Dengan teknologi AI, ParafraseIn dapat membantu kamu dalam bidang akademik. Menggunakan model yang dapat di-fine-tune berdasarkan preferensi Anda untuk hasil yang lebih personal dan akurat.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📝 Input Teks</div>', unsafe_allow_html=True)
    st.markdown("Masukkan teks yang ingin diparafrase:")
    input_text = st.text_area(
        "Input Text",
        placeholder="Contoh: Saya sedang belajar pemrograman Python untuk mengembangkan aplikasi web",
        height=150,
        key="text_input",
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_clicked = st.button("🚀 Mulai Parafrase", key="generate_btn")
    if generate_clicked:
        if input_text.strip():
            st.session_state.input_text = input_text
            with st.spinner("🔄 Generating paraphrases..."):
                results = app.generate_all_styles(input_text)
            st.session_state.paraphrase_results = results
            st.session_state.selected_style = None
            st.session_state.selected_paraphrase = None
            st.success("✅ Parafrase berhasil dihasilkan!")
        else:
            st.error("❌ Mohon masukkan teks yang akan diparafrase")
    if st.session_state.paraphrase_results:
        st.markdown('<div class="section-header">🎯 Pilih Hasil Parafrase yang Kamu Sukai</div>', unsafe_allow_html=True)
        st.markdown("Pilih salah satu hasil parafrase di bawah ini:")
        col1, col2, col3 = st.columns(3)
        styles_info = {
            "safe": {
                "icon": "🛡️",
                "title": "Hasil Safe",
                "desc": "Parafrase dengan perubahan minimal dan konservatif"
            },
            "balanced": {
                "icon": "⚖️", 
                "title": "Hasil Balanced",
                "desc": "Parafrase dengan keseimbangan perubahan yang optimal"
            },
            "creative": {
                "icon": "🎨",
                "title": "Hasil Creative", 
                "desc": "Parafrase dengan perubahan kreatif dan inovatif"
            }
        }
        columns = [col1, col2, col3]
        styles = ["safe", "balanced", "creative"]
        for i, (col, style) in enumerate(zip(columns, styles)):
            with col:
                info = styles_info[style]
                st.markdown(f"""
                <div class="style-card">
                    <h4>{info['icon']} {info['title']}</h4>
                    <p><em>{info['desc']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                if st.session_state.paraphrase_results.get(style):
                    result_text = st.session_state.paraphrase_results[style]
                    st.text_area(
                        f"{style.capitalize()} Result",
                        value=result_text,
                        height=200,
                        key=f"{style}_result",
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    st_copy_to_clipboard(
                        text=result_text,
                        before_copy_label=f"📋 Copy {style.capitalize()}",
                        after_copy_label=f"✅ Copied {style.capitalize()}!",
                        key=f"copy_{style}"
                    )
                    if st.button(f"✅ Pilih {style.capitalize()}", key=f"choose_{style}", use_container_width=True):
                        st.session_state.selected_style = style
                        st.session_state.selected_paraphrase = result_text
                        result = app.save_preference(
                            st.session_state.input_text,
                            result_text,
                            style
                        )
                        st.success(f"🎉 {style.capitalize()} style dipilih dan preferensi tersimpan!")
                        st.rerun()
        if st.session_state.selected_style and st.session_state.selected_paraphrase:
            if st.session_state.input_text.strip():
                display_word_changes(
                    st.session_state.input_text, 
                    st.session_state.selected_paraphrase, 
                    app
                )
    with st.sidebar:
        st.header("🔧 Advanced Features")
        st.subheader("📋 Model Status")
        model_status = app.load_fine_tuned_model_if_exists()
        if "successfully" in model_status:
            st.success("✅ Fine-tuned model aktif")
        else:
            st.info("ℹ️ Menggunakan base model")
        display_preference_statistics(app)
        st.subheader("🧠 Fine-tuning Model")
        st.markdown("Customize model berdasarkan preferensi Anda")
        epochs = st.slider("Training Epochs", 1, 10, 3, help="Jumlah epoch training")
        learning_rate = st.number_input(
            "Learning Rate", 
            value=5e-5, 
            format="%.2e",
            help="Learning rate untuk training"
        )
        total_prefs = len(app.preference_tracker.preferences)
        st.info(f"📊 Total preferensi tersimpan: {total_prefs}")
        if st.button("🚀 Mulai Fine-tuning"):
            if total_prefs >= 5:
                with st.spinner("🔄 Fine-tuning model... Mohon tunggu..."):
                    result = app.fine_tune_model(epochs, learning_rate)
                if "completed" in result:
                    st.success("🎉 Fine-tuning berhasil!")
                    st.balloons()
                else:
                    st.error(f"❌ Fine-tuning gagal: {result}")
            else:
                st.error(f"❌ Minimal 5 preferensi diperlukan untuk fine-tuning (saat ini: {total_prefs})")
        st.subheader("📁 Model Management")
        if st.button("🔄 Reset ke Base Model"):
            app.paraphrase_model = create_paraphrase_model("Wikidepia/IndoT5-base-paraphrase")
            st.success("✅ Model direset ke base model")
            st.rerun()
        if st.button("🗑️ Clear All Preferences"):
            app.preference_tracker.preferences = []
            app.preference_tracker.save_preferences()
            app.paraphrase_model.user_preferences = []
            try:
                model_pref_file = getattr(app.paraphrase_model, 'preference_file', 'user_preferences.json')
                if os.path.exists(model_pref_file):
                    with open(model_pref_file, 'w', encoding='utf-8') as f:
                        json.dump([], f)
            except Exception as e:
                st.warning(f"Warning: Could not clear model preference file: {e}")
            st.success("🗑️ Semua preferensi telah dihapus")
            st.rerun()
        st.subheader("💾 Data Management")
        if st.button("📥 Export Preferences"):
            if app.preference_tracker.preferences:
                prefs_json = json.dumps(app.preference_tracker.preferences, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download preferences.json",
                    data=prefs_json,
                    file_name="preferences.json",
                    mime="application/json"
                )
            else:
                st.error("❌ Tidak ada preferensi untuk diekspor")
        st.subheader("ℹ️ Model Information")
        st.markdown("""
        **Base Model:** Wikidepia/IndoT5-base-paraphrase
        
        **Temperature Settings:**
        - Safe: 0.7 (konservatif)
        - Balanced: 1.0 (seimbang)  
        - Creative: 1.3 (kreatif)
        
        **Features:**
        - Real-time paraphrasing
        - Word-level change analysis
        - User preference learning
        - Custom fine-tuning
        """)

if __name__ == "__main__":
    main()