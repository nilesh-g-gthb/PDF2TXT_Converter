import streamlit as st
import base64
import os
import fitz  # PyMuPDF
import requests
from mistralai import Mistral
from datetime import datetime
import time

# Init Mistral
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# ImgBB API key from env
imgbb_api_key = os.getenv("IMGBB_API_KEY")

st.set_page_config(page_title="Mistral OCR", layout="centered")
st.title("📄 Mistral OCR - PDF/Image Text Extractor")

option = st.radio("Input Type", ["📤 Upload PDF (Full Auto)", "🌐 Paste Image URL"])

# ----- Retry Wrapper -----
def retry(func, max_attempts=3, delay=2):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            time.sleep(delay)

# ----- PDF to Image -----
def convert_pdf_to_images(pdf_file, output_dir):
    images = []
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(image_path)
        images.append(image_path)
    return images

# ----- Upload to ImgBB -----
def upload_image_to_imgbb(image_path):
    def task():
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

    return retry(task)

# ----- Run OCR -----
def run_ocr_on_image_url(url):
    def task():
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": url},
            include_image_base64=False,
        )
        return "\n\n".join([page.markdown for page in response.pages])

    return retry(task)

# ----- Init Session -----
if "images" not in st.session_state:
    st.session_state.images = []
if "imgbb_urls" not in st.session_state:
    st.session_state.imgbb_urls = []
if "ocr_texts" not in st.session_state:
    st.session_state.ocr_texts = []

# ----- Auto Resume Option -----
if st.session_state.images and st.session_state.ocr_texts:
    if st.button("🔁 Resume Previous Session"):
        full_text = ""
        for i, (url, text) in enumerate(zip(st.session_state.imgbb_urls, st.session_state.ocr_texts)):
            full_text += f"\n\n--- Page {i+1} ---\n\n{text}"
        st.success("🎉 Recovered OCR Results")
        st.text_area("📄 Extracted Text", full_text.strip(), height=300)
        st.download_button("⬇️ Download as .txt", full_text.strip(), file_name="ocr_output.txt")

# ----- Upload PDF Flow -----
if option == "📤 Upload PDF (Full Auto)":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output_images", f"pdf_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        with st.spinner("🔄 Converting PDF to images..."):
            st.session_state.images = convert_pdf_to_images(uploaded_pdf, output_dir)
        st.success(f"✅ Converted! Images saved in: `{output_dir}`")

        show_images = st.checkbox("Show scanned images", value=False)
        if show_images:
            for i, img_path in enumerate(st.session_state.images):
                st.image(img_path, caption=f"Page {i+1}", use_container_width=True)
                st.code(f"Local path: {os.path.abspath(img_path)}", language="bash")

        full_text = ""
        with st.spinner("⬆️ Uploading images to ImgBB and running OCR..."):
            for i, img_path in enumerate(st.session_state.images):
                if i < len(st.session_state.imgbb_urls):
                    img_url = st.session_state.imgbb_urls[i]
                    result = st.session_state.ocr_texts[i]
                else:
                    try:
                        st.write(f"📤 Uploading page {i+1}...")
                        img_url = upload_image_to_imgbb(img_path)
                        st.session_state.imgbb_urls.append(img_url)

                        st.write(f"🧠 Running OCR on page {i+1}...")
                        result = run_ocr_on_image_url(img_url)
                        st.session_state.ocr_texts.append(result)
                    except Exception as e:
                        st.error(f"❌ Upload/OCR failed for Page {i+1}: {e}")
                        break

                full_text += f"\n\n--- Page {i+1} ---\n\n{result}"

        if full_text:
            st.success("🎉 OCR Completed")
            st.text_area("📄 Extracted Text", full_text.strip(), height=300)
            st.download_button("⬇️ Download as .txt", full_text.strip(), file_name="ocr_output.txt")

# ----- Image URL Flow -----
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
