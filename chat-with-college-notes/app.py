import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF for PDF reading

# -----------------------
# Utility Functions
# -----------------------

def create_client(api_key):
    """Create Gemini API client safely."""
    if not api_key:
        st.error("❌ API key is missing! Please enter your Gemini API key.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai

def read_pdf(file):
    """Extract text from uploaded PDF using PyMuPDF."""
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def chunk_text(text, max_chars=1200, overlap=200):
    """Split text into memory-safe chunks."""
    chunks = []
    text_length = len(text)

    for i in range(0, text_length, max_chars - overlap):
        end = min(i + max_chars, text_length)
        chunks.append(text[i:end])

        # Safety check: stop if too many chunks (prevents MemoryError)
        if len(chunks) > 500:
            break  

    return chunks

def answer_question(model, chunks, question):
    """Send chunks + question to Gemini and return response."""
    context = "\n\n".join(chunks)
    prompt = f"You are a helpful study assistant. Use the following notes:\n{context}\n\nAnswer this question: {question}"

    response = model.generate_content(prompt)
    return response.text


# -----------------------
# Streamlit App
# -----------------------

st.set_page_config(page_title="Chat with College Notes", page_icon="📚", layout="wide")
st.title("📚 Chat with College Notes")
st.write("Upload your lecture notes (PDF) and ask questions about them!")

# API Key Input
api_key_input = st.text_input("🔑 Enter your Gemini API Key:", type="password")
if api_key_input:
    client = create_client(api_key_input)
    model = client.GenerativeModel("gemini-1.5-flash")
else:
    st.warning("⚠️ Please enter your API key to continue.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("📂 Upload your notes (PDF)", type=["pdf"])

if uploaded_file:
    text = read_pdf(uploaded_file)

    if len(text) > 1_000_000:  # ~1 MB text cutoff
        st.warning("⚠️ Large PDF detected! Only the first part will be processed.")
    
    chunks = chunk_text(text, max_chars=1200, overlap=200)

    st.success(f"✅ Notes loaded successfully! ({len(chunks)} chunks created)")

    # Question Input
    question = st.text_input("💬 Ask a question about your notes:")
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(model, chunks, question)
            st.subheader("📖 Answer")
            st.write(answer)
