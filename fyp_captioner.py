import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BartTokenizer
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data
nltk.download('punkt', quiet=True)

# Set up the app
st.set_page_config(page_title="Chest X-Ray Report Generator", layout="wide")

# Main title
st.title("Chest X-Ray Image Caption Generator")
st.write("Upload chest X-ray image")

# Sidebar for reference captions
with st.sidebar:
    st.header("Reference Captions")
    st.write("Select reference caption:")
    
    reference_options = {
        "Normal Findings": "Heart size is normal. Mediastinal contours are unremarkable. No pneumothorax or pleural effusion.",
        "Pneumonia": "Right lower lobe consolidation consistent with pneumonia. No pleural effusion.",
        "Cardiomegaly": "Cardiomegaly present. Pulmonary vascular congestion noted.",
        "Clear Lungs": "Lungs are clear. No focal consolidation. No pleural effusion.",
        "Pneumothorax": "Right apical pneumothorax noted. Lung partially collapsed."
    }
    
    selected_reference = st.selectbox("Choose reference type:", list(reference_options.keys()))
    reference_caption = reference_options[selected_reference]
    
    st.write("Selected Reference:")
    st.text(reference_caption)

# Load model components
@st.cache_resource
def load_model_components():
    try:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", 
            "facebook/bart-base"
        )
        
        model_path = r"C:\Users\Saad Ahsin\Desktop\FYP-2 Reports\BART\Important Files\BART_image_captionerr.pth"
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        model.eval()
        return model, feature_extractor, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_caption(image, model, feature_extractor, tokenizer):
    try:
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
        
    except Exception as e:
        return f"Error: {e}"

def calculate_bleu(reference, candidate):
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], cand_tokens, 
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoothing)
    return round(score, 4)

# Load model
model, feature_extractor, tokenizer = load_model_components()

# Main layout - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Image Upload")
    
    uploaded_file = st.file_uploader("Choose chest X-ray image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)
        
        if st.button("Generate Report"):
            if model is not None:
                with st.spinner(""):
                    generated_caption = generate_caption(image, model, feature_extractor, tokenizer)
                    
                    # Store in session state
                    st.session_state.generated_caption = generated_caption
                    st.session_state.image_processed = True
                    
                    st.write("Generated Report:")
                    st.text(generated_caption)
            else:
                st.error("Model not loaded properly")

with col2:
    st.header("Report Comparison")
    
    if 'image_processed' in st.session_state and st.session_state.image_processed:
        generated_caption = st.session_state.generated_caption
        
        # Side by side comparison
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.write("Reference Caption:")
            st.text(reference_caption)
        
        with comp_col2:
            st.write("Generated Caption:")
            st.text(generated_caption)
        
        # BLEU Score Calculation
        st.subheader("BLEU Score")
        
        if st.button("Calculate BLEU Score"):
            bleu_score = calculate_bleu(reference_caption, generated_caption)
            
            col1_metric, col2_metric = st.columns(2)
            
            with col1_metric:
                st.write("BLEU-4 Score:")
                st.write(f"{bleu_score:.4f}")
            
            with col2_metric:
                if bleu_score > 0.4:
                    rating = "Good"
                elif bleu_score > 0.2:
                    rating = "Fair"
                else:
                    rating = "Poor"
                st.write("Rating:")
                st.write(rating)

