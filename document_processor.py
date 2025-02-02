import PyPDF2
import docx
import os

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        """Extract text from the document based on its file type."""
        file_extension = os.path.splitext(self.file_path)[1].lower()

        if file_extension == '.pdf':
            return self._extract_from_pdf()
        elif file_extension == '.txt':
            return self._extract_from_txt()
        elif file_extension in ['.doc', '.docx']:
            return self._extract_from_docx()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _extract_from_pdf(self):
        """Extract text from PDF file."""
        try:
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _extract_from_txt(self):
        """Extract text from TXT file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error extracting text from TXT: {str(e)}")
            return ""

    def _extract_from_docx(self):
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(self.file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return ""