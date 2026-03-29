import streamlit as st
import os
import tempfile
import requests
import time
import base64
from PIL import Image
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
.error-box {
    background-color: #f8d7da !important;
    border-left: 5px solid #dc3545 !important;
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

def encode_image_to_base64(image_path):
    """Encode image file to base64 data URI"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    # Determine mime type based on file extension
    if image_path.lower().endswith('.png'):
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{encoded_string}"

def run_face_swap_api(source_path, target_path, api_key):
    """Call Replicate API directly using requests with proper file handling"""
    
    # Using a stable faceswap model on Replicate (lucataco/faceswap)
    model_version = "9a4298548422074c3f57258c5d544497314ae4112df7870ca4211c7c9c3dd90d"
    
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    
    # Encode images to base64 data URIs
    source_uri = encode_image_to_base64(source_path)
    target_uri = encode_image_to_base64(target_path)
    
    # 1. Create the prediction
    payload = {
        "version": model_version,
        "input": {
            "target_image": target_uri,
            "swap_image": source_uri
        }
    }
    
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 201:
        raise Exception(f"Failed to start prediction: {response.text}")
        
    prediction = response.json()
    prediction_url = prediction["urls"]["get"]
    
    # 2. Poll for results
    max_attempts = 30
    attempt = 0
    while prediction["status"] not in ["succeeded", "failed", "canceled"] and attempt < max_attempts:
        time.sleep(2)  # Wait 2 seconds
        attempt += 1
        response = requests.get(prediction_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to check status: {response.text}")
        prediction = response.json()
    
    if prediction["status"] == "succeeded":
        return prediction["output"]
    elif prediction["status"] == "failed":
        raise Exception(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
    else:
        raise Exception("Prediction was canceled or timed out")

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
    st.caption("☁️ Powered by Replicate AI")

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
                
                # Call API
                result_url = run_face_swap_api(src_path, tgt_path, current_api_key)
                
                # Download result
                if result_url:
                    # Handle both single URL and list of URLs
                    if isinstance(result_url, list):
                        result_url = result_url[0]
                    
                    response = requests.get(result_url)
                    result_image = Image.open(BytesIO(response.content))
                    
                    st.session_state.swap_result = {
                        "image": result_image,
                        "success": True
                    }
                    st.success("✅ Face swap completed!")
                else:
                    st.error("❌ No result received from API")
                
                # Clean up temp files
                os.unlink(src_path)
                os.unlink(tgt_path)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if "authentication" in str(e).lower():
                    st.error("🔑 Please check your API key is correct")
                elif "credit" in str(e).lower():
                    st.error("💳 Insufficient credits. Please add credits to your Replicate account.")

# ================= DISPLAY RESULT =================
if 'swap_result' in st.session_state and st.session_state.swap_result.get('success'):
    st.markdown("---")
    st.markdown('<div class="success-box"><h3>🎉 Face Swap Complete!</h3><p>Your face swap has been processed successfully!</p></div>', unsafe_allow_html=True)
    
    # Display result
    st.markdown("### 🎨 Result")
    st.image(st.session_state.swap_result['image'], 
             caption="Swapped Face Result", 
             use_container_width=True)
    
    # Download button
    result_bytes = BytesIO()
    st.session_state.swap_result['image'].save(result_bytes, format="PNG")
    st.download_button(
        label="📥 Download Swapped Image",
        data=result_bytes.getvalue(),
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
        st.markdown("### ➡️")
    
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
