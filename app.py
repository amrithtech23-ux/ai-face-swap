import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import base64
import requests
import replicate
from io import BytesIO

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="AI Powered Face Swap",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size: 3rem !important;
    font-weight: bold !important;
    color: #6C63FF !important;
    text-align: center !important;
    margin-bottom: 10px !important;
}
.subtitle {
    font-size: 1.2rem !important;
    color: #666 !important;
    text-align: center !important;
    margin-bottom: 30px !important;
}
.result-image {
    border: 3px solid #6C63FF !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
}
.success-box {
    background-color: #d4edda !important;
    border-left: 5px solid #28a745 !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin: 20px 0 !important;
}
.info-box {
    background-color: #e8f0fe !important;
    border-left: 5px solid #6C63FF !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin: 20px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<h1 class="main-title">🎭 AI Powered Face Swap</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Swap faces in photos instantly using advanced AI technology</p>', unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================
def get_api_key():
    """Retrieve API key from multiple sources"""
    if st.session_state.get("replicate_api_key"):
        return st.session_state.replicate_api_key
    if st.secrets.get("REPLICATE_API_TOKEN"):
        return st.secrets["REPLICATE_API_TOKEN"]
    return None

def image_to_bytes(image):
    """Convert PIL Image to bytes"""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def base64_encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Key Configuration
    api_key_from_secrets = st.secrets.get("REPLICATE_API_TOKEN")
    
    replicate_api_key = st.text_input(
        "🔑 Replicate API Key",
        type="password",
        help="Get your API key from https://replicate.com/account/api-tokens",
        key="replicate_api_key_input",
        value=api_key_from_secrets if api_key_from_secrets else ""
    )
    
    if replicate_api_key:
        st.session_state.replicate_api_key = replicate_api_key
        st.success("✅ API Key configured!")
    elif api_key_from_secrets:
        st.session_state.replicate_api_key = api_key_from_secrets
        st.success("✅ API Key loaded from Secrets!")
    else:
        st.warning("⚠️ Please add your Replicate API Key")
    
    st.markdown("---")
    
    # Model Selection
    model_version = st.selectbox(
        "🤖 Face Swap Model",
        ["inswapper_128", "simswap_256"],
        index=0,
        help="inswapper_128: Faster, good quality\nsimswap_256: Higher quality, slower"
    )
    
    st.markdown("---")
    
    # Info Box
    st.markdown("""
    <div class="info-box">
    <strong>📌 How to Use:</strong>
    <ol>
        <li>Add your Replicate API Key</li>
        <li>Upload source image (face to use)</li>
        <li>Upload target image (face to replace)</li>
        <li>Click "Swap Faces" button</li>
        <li>Download your swapped image!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("- ✅ High-quality face detection")
    st.markdown("- ✅ Realistic face swapping")
    st.markdown("- ✅ Privacy-focused processing")
    st.markdown("- ✅ Fast results")
    
    st.markdown("---")
    st.caption("📄 MIT License | Powered by Replicate AI")

# ================= MAIN CONTENT =================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📷 Source Image (Face to Use)")
    source_file = st.file_uploader(
        "Upload source image",
        type=["jpg", "jpeg", "png"],
        key="source_uploader",
        help="This image contains the face you want to swap"
    )
    
    if source_file:
        source_image = Image.open(source_file).convert('RGB')
        st.image(source_image, caption="Source Image", use_container_width=True)
        st.success(f"✅ Uploaded: {source_file.name}")

with col2:
    st.markdown("### 🎯 Target Image (Face to Replace)")
    target_file = st.file_uploader(
        "Upload target image",
        type=["jpg", "jpeg", "png"],
        key="target_uploader",
        help="This image will have its face replaced"
    )
    
    if target_file:
        target_image = Image.open(target_file).convert('RGB')
        st.image(target_image, caption="Target Image", use_container_width=True)
        st.success(f"✅ Uploaded: {target_file.name}")

# ================= SWAP BUTTON =================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    swap_btn = st.button("🎭 Swap Faces", type="primary", use_container_width=True, key="swap_btn")

# ================= PROCESS FACE SWAP =================
if swap_btn:
    current_api_key = get_api_key()
    
    if not current_api_key:
        st.error("❌ Please add your Replicate API Key in the sidebar first!")
    elif source_file is None or target_file is None:
        st.warning("⚠️ Please upload both source and target images!")
    else:
        with st.spinner("🔄 Processing face swap... This may take 30-60 seconds..."):
            try:
                # Save images temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as src_tmp:
                    source_image.save(src_tmp, format='JPEG')
                    src_path = src_tmp.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tgt_tmp:
                    target_image.save(tgt_tmp, format='JPEG')
                    tgt_path = tgt_tmp.name
                
                # Set Replicate API token
                os.environ["REPLICATE_API_TOKEN"] = current_api_key
                
                # Choose model based on selection
                if model_version == "inswapper_128":
                    model_name = "roop-inswapper/inswapper_128:cdcf4910f92b38e3a6a98b70446f26b601a2593b0e5ce2a5203e5e24e646f307"
                else:
                    model_name = "batouresearch/simswap-256:84a16e34428a1e04f4310a14a2fcc2ae0b8f9b2b8c0e1e1f3c3e3e3e3e3e3e3e"
                
                # Run face swap using Replicate
                output = replicate.run(
                    model_name,
                    input={
                        "source_image": open(src_path, "rb"),
                        "target_image": open(tgt_path, "rb"),
                        "face_index": 0,
                        "face_weight": 1.0
                    }
                )
                
                # Output is typically a URL to the result image
                if output:
                    # Download the result image
                    result_url = output if isinstance(output, str) else output[0]
                    response = requests.get(result_url)
                    result_image = Image.open(BytesIO(response.content))
                    
                    st.session_state.swap_result = {
                        "image": result_image,
                        "success": True,
                        "message": "Face swap completed successfully!"
                    }
                    
                    st.success("✅ Face swap completed!")
                else:
                    st.error("❌ No output received from the API")
                
                # Clean up temp files
                os.unlink(src_path)
                os.unlink(tgt_path)
                
            except replicate.exceptions.ReplicateError as e:
                st.error(f"❌ Replicate API Error: {str(e)}")
                if "authentication" in str(e).lower():
                    st.error("🔑 Please check your API key is correct")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("💡 Make sure you have a valid Replicate API key")

# ================= DISPLAY RESULT =================
if 'swap_result' in st.session_state and st.session_state.swap_result.get('success'):
    st.markdown("---")
    st.markdown('<div class="success-box"><h3>🎉 Face Swap Complete!</h3><p>Your face swap has been processed successfully!</p></div>', unsafe_allow_html=True)
    
    # Display result
    st.markdown("### 🎨 Result")
    st.image(st.session_state.swap_result['image'], 
             caption="Swapped Face Result", 
             use_container_width=True,
             output_format="PNG")
    
    # Download button
    result_bytes = image_to_bytes(st.session_state.swap_result['image'])
    st.download_button(
        label="📥 Download Swapped Image",
        data=result_bytes,
        file_name="face_swap_result.png",
        mime="image/png",
        use_container_width=True
    )
    
    # Display comparison
    st.markdown("---")
    st.markdown("### 📸 Before & After Comparison")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(source_image, caption="Source Face", use_container_width=True)
    
    with col2:
        st.markdown("### +")
    
    with col3:
        st.image(target_image, caption="Target Image", use_container_width=True)
    
    st.markdown("### ⬇️")
    st.image(st.session_state.swap_result['image'], 
             caption="Final Result", 
             use_container_width=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🔒 Privacy Notice:</strong> Images are processed securely and not stored permanently.</p>
    <p><strong>⚠️ Disclaimer:</strong> Use responsibly. Do not use for misleading or harmful purposes.</p>
    <p>📄 MIT License | ☁️ Powered by Replicate AI</p>
</div>
""", unsafe_allow_html=True)
