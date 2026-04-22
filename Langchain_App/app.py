from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


from PyPDF2 import PdfReader
from docx import Document

def load_document(file_path):
    # TXT
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # PDF
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        return text

    # DOCX
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = ""

        for para in doc.paragraphs:
            text += para.text + "\n"

        return text

    else:
        print("❌ Unsupported file type (.txt, .pdf, .docx only)")
        return None


def main():
    print("=== RAG App (Upload TXT or PDF) ===")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # 📂 Ask only once
    file_path = input("Enter file path (.txt or .pdf): ")

    document = load_document(file_path)

    if not document:
        return

    print("✅ Document loaded successfully!")

    while True:
        user_input = input("\nAsk something (type 'exit' to quit): ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Answer ONLY using this document:

{document}

Question:
{user_input}
"""
                }
            ]
        )

        print("\nAI:", response.choices[0].message.content)
    
if __name__ == "__main__":
    main()