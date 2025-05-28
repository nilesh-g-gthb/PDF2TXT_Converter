import streamlit as st
import base64
import os
import fitz  # PyMuPDF
import requests
from mistralai import Mistral
from datetime import datetime

# Init Mistral
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# ImgBB API key from env
imgbb_api_key = os.getenv("IMGBB_API_KEY")


st.set_page_config(page_title="Mistral OCR", layout="centered")
st.title("📄 Mistral OCR - PDF/Image Text Extractor")
st.markdown("Choose a method:")

option = st.radio("Input Type", ["📤 Upload PDF (Full Auto)", "🌐 Paste Image URL"])

def convert_pdf_to_images(pdf_file, output_dir):
    """Convert PDF to images and save to folder"""
    images = []
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(image_path)
        images.append(image_path)
    return images

def upload_image_to_imgbb(image_path):
    """Upload local image file to ImgBB and return public URL"""
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": imgbb_api_key,
            "image": encoded_image,
        },
    )
    res_json = response.json()
    if not res_json.get("success"):
        raise Exception(f"Upload failed: {res_json}")
    return res_json["data"]["url"]

def run_ocr_on_image_url(url):
    """OCR for direct image URL"""
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "image_url", "image_url": url},
        include_image_base64=False,
    )
    return "\n\n".join([page.markdown for page in response.pages])

if option == "📤 Upload PDF (Full Auto)":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output_images", f"pdf_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        with st.spinner("🔄 Converting PDF to images..."):
            image_paths = convert_pdf_to_images(uploaded_pdf, output_dir)
        st.success(f"✅ Converted! Images saved in: `{output_dir}`")

        show_images = st.checkbox("Show scanned images", value=False)
        if show_images:
            for i, img_path in enumerate(image_paths):
                st.image(img_path, caption=f"Page {i+1}", use_container_width=True)
                st.code(f"Local path: {os.path.abspath(img_path)}", language="bash")

        full_text = ""
        with st.spinner("⬆️ Uploading images to ImgBB and running OCR..."):
            for i, img_path in enumerate(image_paths):
                try:
                    st.write(f"📤 Uploading page {i+1}...")
                    img_url = upload_image_to_imgbb(img_path)
                    st.write(f"✅ Uploaded: {img_url}")
                    st.write(f"🧠 Running OCR on page {i+1}...")
                    result = run_ocr_on_image_url(img_url)
                    full_text += f"\n\n--- Page {i+1} ---\n\n{result}"
                except Exception as e:
                    st.error(f"❌ Upload/OCR failed for Page {i+1}: {e}")
                    break  # stop on first failure

        if full_text:
            st.success("🎉 OCR Completed")
            st.text_area("📄 Extracted Text", full_text.strip(), height=300)
            st.download_button("⬇️ Download as .txt", full_text.strip(), file_name="ocr_output.txt")

elif option == "🌐 Paste Image URL":
    image_urls = st.text_area("Enter one or more HTTPS image URLs (separated by commas or newlines)")
if st.button("🧠 Run OCR"):
    urls = [url.strip() for part in image_urls.splitlines() for url in part.split(",") if url.strip().startswith("https://")]
    if urls:
        for idx, url in enumerate(urls, 1):
            with st.spinner(f"Running OCR on image {idx}..."):
                try:
                    extracted = run_ocr_on_image_url(url)
                    st.success(f"✅ OCR Completed for image {idx}")
                    st.text_area(f"📄 Extracted Text (Image {idx})", extracted, height=300)
                    st.download_button(f"⬇️ Download Image {idx} Text", extracted, file_name=f"ocr_output_{idx}.txt")
                except Exception as e:
                    st.error(f"❌ Error for image {idx}: {e}")
    else:
        st.warning("⚠️ Please enter at least one valid HTTPS image URL.")
