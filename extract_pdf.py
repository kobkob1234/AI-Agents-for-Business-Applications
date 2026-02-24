import sys
import subprocess

try:
    import fitz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

if __name__ == "__main__":
    text = extract_text(sys.argv[1])
    with open("project_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("PDF extracted to project_text.txt")
