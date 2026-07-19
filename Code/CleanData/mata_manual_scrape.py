# Author: Alden Porter
# This file extracts text from the mata manaul an puts it in a usable format.
import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    
    pdf_path = "../Input/m-2.pdf"  # Path to the Mata manual PDF
    output_path = "../Output/mata_manual.txt"  # Path to save the extracted text

    # Extract text from the PDF
    print("Starting Text Conversion...")
    text = convert_pdf_to_txt(pdf_path)

    # Save the extracted text to a file
    save_text_to_file(text, output_path)

    print(f"Extracted text from {pdf_path} and saved to {output_path}")
