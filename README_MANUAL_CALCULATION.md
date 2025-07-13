# Manual Calculation Paraphrase Model

## Deskripsi

Kode ini telah dimodifikasi untuk menampilkan perhitungan manual step-by-step dari proses parafrase, mulai dari input hingga output. Fitur ini memungkinkan Anda untuk memahami bagaimana model T5 memproses text dan menghasilkan parafrase.

## Fitur Baru

### 1. Perhitungan Manual Detail (`show_manual_calculation=True`)

Model sekarang dapat menampilkan:
- **Step 1: Input Preprocessing** - Penambahan prefix "paraphrase:"
- **Step 2: Tokenization** - Pemecahan text menjadi token dan konversi ke ID
- **Step 3: Input Encoding** - Konversi ke tensor PyTorch
- **Step 4: Temperature Configuration** - Setting parameter kreativitas
- **Step 5: Model Configuration** - Detail arsitektur model
- **Step 6: Generation Process** - Proses autoregressive generation
- **Step 7: Token-by-Token Details** - Detail setiap token yang dihasilkan
- **Step 8: Decoding** - Konversi kembali ke text
- **Step 9: Output Analysis** - Analisis perubahan dan similarity

### 2. Method Baru

#### `demonstrate_complete_process(text, style)`
```python
model = create_paraphrase_model()
result = model.demonstrate_complete_process("Saya belajar AI", "balanced")
```
Menampilkan seluruh proses dengan format yang mudah dibaca.

#### `show_model_architecture()`
```python
architecture = model.show_model_architecture()
print(f"Total Parameters: {architecture['parameter_count']['total_parameters_formatted']}")
```
Menampilkan detail arsitektur model T5.

#### `analyze_attention_weights(text, style)`
```python
attention_info = model.analyze_attention_weights("Contoh text", "balanced")
```
Analisis attention weights (versi simplified).

### 3. Parameter Baru

#### `generate_paraphrases(text, style, show_manual_calculation=False)`
```python
# Generasi biasa
result = model.generate_paraphrases("Text input", "balanced")

# Dengan perhitungan manual
detailed_result = model.generate_paraphrases("Text input", "balanced", show_manual_calculation=True)
```

#### `generate_with_analysis(input_text, creativity_level, show_calculation=False)`
```python
creativity_model = create_creativity_controlled_model()
result = creativity_model.generate_with_analysis("Text", "creative", show_calculation=True)
```

## Cara Penggunaan

### 1. Import Model
```python
from model_architecture_manual import create_paraphrase_model, create_creativity_controlled_model

# Inisialisasi model
model = create_paraphrase_model()
```

### 2. Generasi Sederhana
```python
# Generasi biasa
result = model.generate_paraphrases("Saya belajar Python", "balanced")
print(result)
```

### 3. Generasi dengan Perhitungan Manual
```python
# Dengan detail perhitungan
detailed_result = model.generate_paraphrases(
    "Saya belajar Python", 
    "balanced", 
    show_manual_calculation=True
)

if detailed_result["success"]:
    print(f"Input: {detailed_result['input']}")
    print(f"Output: {detailed_result['output']}")
    print(f"Total Steps: {detailed_result['summary']['total_steps']}")
    
    # Lihat detail setiap step
    for step in detailed_result['steps']:
        print(f"Step {step['step']}: {step['description']}")
        print(f"Explanation: {step['explanation']}")
```

### 4. Demonstrasi Lengkap
```python
# Demonstrasi dengan output yang rapi
model.demonstrate_complete_process("Contoh text", "balanced")
```

### 5. Analisis Arsitektur
```python
architecture = model.show_model_architecture()
print(f"Model: {architecture['model_name']}")
print(f"Parameters: {architecture['parameter_count']['total_parameters_formatted']}")
print(f"Layers: {architecture['parameters']['num_layers']}")
print(f"Attention Heads: {architecture['parameters']['num_heads']}")
```

### 6. Model dengan Kontrol Kreativitas
```python
creativity_model = create_creativity_controlled_model()

# Dengan analisis detail
result = creativity_model.generate_with_analysis(
    "Text input", 
    "creative", 
    show_calculation=True
)

if result.get("success"):
    print(f"Output: {result['output']}")
    print(f"Word Mappings: {result['word_mappings']}")
    print(f"Changes: {len(result['changes'])}")
```

## Menjalankan Demo

### 1. Demo File Utama
```bash
cd "d:\SEMESTER 4\PENGANTAR KECERDASAN BUATAN\parafrasein-trained"
python app/model_architecture_manual.py
```

### 2. Demo Interaktif
```bash
python demo_manual_calculation.py
```

## Output yang Dihasilkan

### Informasi Step-by-Step
```
STEP 1: Input Preprocessing
--------------------------------------------------
Explanation: Menambahkan prefix 'paraphrase:' ke input text
Original Input: 'Saya belajar Python'
Formatted Input: 'paraphrase: Saya belajar Python'

STEP 2: Tokenization
--------------------------------------------------
Explanation: Memecah text menjadi 6 token menggunakan tokenizer
Tokens: ['▁paraphrase', ':', '▁Say', 'a', '▁bel', 'ajar', '▁Python']
Token IDs: [32098, 10, 2206, 9, 493, 2305, 25334]
Vocabulary Size: 32,128

...dan seterusnya untuk 9 steps
```

### Summary Informasi
```
SUMMARY
================================================================================
Total Processing Steps: 9
Input Tokens: 7
Output Tokens: 8
Temperature Used: 1.0
Final Similarity: 0.7234
================================================================================
```

### Arsitektur Model
```
Model Type: T5ForConditionalGeneration
Model Name: Wikidepia/IndoT5-base-paraphrase
Total Parameters: 222,903,808
Trainable Parameters: 222,903,808

Model Configuration:
  vocab_size: 32128
  d_model: 768
  num_layers: 12
  num_heads: 12
  dropout_rate: 0.1
```

## Kegunaan Fitur Manual Calculation

1. **Educational Purpose**: Memahami cara kerja model transformer
2. **Debugging**: Mengidentifikasi masalah dalam proses generation
3. **Research**: Analisis detail untuk pengembangan model
4. **Optimization**: Memahami bottleneck dalam proses
5. **Validation**: Memverifikasi bahwa model bekerja sesuai ekspektasi

## Catatan Penting

- Fitur manual calculation akan memperlambat proses karena banyak logging
- Gunakan `show_manual_calculation=False` untuk production
- Memory usage akan lebih tinggi karena menyimpan detail setiap step
- Cocok untuk debugging dan educational purpose

## Dependencies

- torch
- transformers
- pandas
- numpy
- difflib (built-in)
- re (built-in)
- json (built-in)

## Contoh Lengkap

```python
# Contoh penggunaan lengkap
from model_architecture_manual import create_paraphrase_model

def example_usage():
    # 1. Inisialisasi
    model = create_paraphrase_model()
    
    # 2. Input text
    text = "Kecerdasan buatan sangat menarik untuk dipelajari"
    
    # 3. Generasi dengan manual calculation
    print("=== PERHITUNGAN MANUAL ===")
    model.demonstrate_complete_process(text, "balanced")
    
    # 4. Bandingkan semua style
    print("\n=== PERBANDINGAN STYLE ===")
    results = model.generate_all_styles(text, show_manual_calculation=False)
    for style, result in results.items():
        print(f"{style}: {result}")
    
    # 5. Analisis arsitektur
    print("\n=== ARSITEKTUR MODEL ===")
    arch = model.show_model_architecture()
    print(f"Total params: {arch['parameter_count']['total_parameters_formatted']}")

if __name__ == "__main__":
    example_usage()
```
