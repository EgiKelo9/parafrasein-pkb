# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contoh sederhana penggunaan manual calculation
Jalankan dari root directory: python test_manual_calculation.py
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
sys.path.insert(0, current_dir)
sys.path.insert(0, app_dir)

def test_manual_calculation():
    """Test fitur manual calculation"""
    print("Testing Manual Calculation Feature...")
    print("="*50)
    
    try:
        # Import model
        from app.model_architecture_manual import create_paraphrase_model
        
        # Inisialisasi model
        print("Initializing model...")
        model = create_paraphrase_model()
        
        # Test text
        test_text = "Saya sedang mempelajari AI"
        print(f"Input text: '{test_text}'")
        print()
        
        # Test 1: Generate biasa
        print("1. GENERASI BIASA:")
        print("-" * 30)
        simple_result = model.generate_paraphrases(test_text, "balanced")
        print(f"Result: '{simple_result}'")
        print()
        
        # Test 2: Generate dengan manual calculation
        print("2. GENERASI DENGAN MANUAL CALCULATION:")
        print("-" * 30)
        detailed_result = model.generate_paraphrases(
            test_text, 
            "balanced", 
            show_manual_calculation=True
        )
        
        if detailed_result.get("success", False):
            print(f"✓ Success!")
            print(f"Input: '{detailed_result['input']}'")
            print(f"Output: '{detailed_result['output']}'")
            print(f"Style: {detailed_result['style']}")
            print(f"Total Steps: {detailed_result['summary']['total_steps']}")
            print(f"Temperature: {detailed_result['summary']['temperature']}")
            print(f"Similarity: {detailed_result['summary']['similarity']:.4f}")
            print(f"Input Tokens: {detailed_result['summary']['input_tokens']}")
            print(f"Output Tokens: {detailed_result['summary']['output_tokens']}")
            
            print("\nStep Details:")
            for i, step in enumerate(detailed_result['steps'][:3]):  # Show first 3 steps
                print(f"  Step {step['step']}: {step['description']}")
                print(f"    {step['explanation']}")
        else:
            print(f"✗ Error: {detailed_result.get('error', 'Unknown error')}")
        
        print()
        
        # Test 3: Model architecture
        print("3. MODEL ARCHITECTURE:")
        print("-" * 30)
        arch_info = model.show_model_architecture()
        print(f"Model: {arch_info['model_name']}")
        print(f"Total Parameters: {arch_info['parameter_count']['total_parameters_formatted']}")
        print(f"Vocab Size: {arch_info['parameters']['vocab_size']:,}")
        print(f"Model Dimension: {arch_info['parameters']['d_model']}")
        print(f"Layers: {arch_info['parameters']['num_layers']}")
        print(f"Attention Heads: {arch_info['parameters']['num_heads']}")
        print()
        
        # Test 4: Demonstrate complete process
        print("4. COMPLETE PROCESS DEMONSTRATION:")
        print("-" * 30)
        print("Running complete process demonstration...")
        demo_result = model.demonstrate_complete_process(test_text, "creative")
        
        if demo_result and demo_result.get("success", False):
            print("✓ Complete process demonstration finished successfully!")
        else:
            print("✗ Error in complete process demonstration")
        
        print("\n" + "="*50)
        print("MANUAL CALCULATION TEST COMPLETED")
        print("="*50)
        
        return True
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("Make sure you're running from the correct directory and dependencies are installed.")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def simple_usage_example():
    """Contoh penggunaan sederhana tanpa dependencies eksternal"""
    print("\nSIMPLE USAGE EXAMPLE:")
    print("="*50)
    
    example_code = '''
# Cara menggunakan fitur manual calculation:

from app.model_architecture_manual import create_paraphrase_model

# 1. Inisialisasi model
model = create_paraphrase_model()

# 2. Generasi biasa
result = model.generate_paraphrases("Text input", "balanced")

# 3. Generasi dengan manual calculation
detailed = model.generate_paraphrases(
    "Text input", 
    "balanced", 
    show_manual_calculation=True
)

# 4. Demonstrasi lengkap
model.demonstrate_complete_process("Text input", "creative")

# 5. Info arsitektur
arch = model.show_model_architecture()
print(f"Parameters: {arch['parameter_count']['total_parameters_formatted']}")
'''
    
    print(example_code)

if __name__ == "__main__":
    print("MANUAL CALCULATION TEST")
    print("="*50)
    print("This script tests the manual calculation features of the paraphrase model.")
    print()
    
    # Test manual calculation
    success = test_manual_calculation()
    
    # Show usage example
    simple_usage_example()
    
    if success:
        print("\n✓ All tests completed successfully!")
        print("You can now use the manual calculation features in your paraphrase model.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    print("\nFor detailed documentation, see: README_MANUAL_CALCULATION.md")
