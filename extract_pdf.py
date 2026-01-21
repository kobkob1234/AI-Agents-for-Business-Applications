from pdfminer.high_level import extract_text
import sys

def extract(pdf_path, output_path):
    try:
        text = extract_text(pdf_path)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract("Course Project.pdf", "course_project_text.txt")
