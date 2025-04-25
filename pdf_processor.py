"""
pdf text and image processor
this script processes PDF files from the SBA_PDF folder by:
- extracting all regular text content
- finding and processing any images within the PDFs
- using OCR to get text from those images
- combining both types of text into organized output files
for each PDF, it creates a corresponding text file in the SBA_text_files folder,
named with the same number as the source PDF (e.g., "1.pdf" becomes "sba_1.txt").
the output files clearly separate regular text from image-extracted text.

note: processing images with OCR may take some time depending on the number and complexity of images in each PDF.
"""

# pdf to text 
import PyPDF2
from pathlib import Path
from paddleocr import PaddleOCR
import fitz  # pymupdf
from PIL import Image
import io
import numpy

### not convinced that this image extraction function works very well based on the results. 
### we may want to disccuss before chunking
def extract_images_from_pdf(pdf_path):
    """ extract images from PDF and return their text content using PaddleOCR """
    image_texts = []
    
    # initialize paddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # open pdf with pymupdf
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # convert image bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # perform ocr on the image
            result = ocr.ocr(numpy.array(image))
            
            # extract text from ocr result
            if result and result[0]:  # Check if result exists and has content
                text = "\n".join([line[1][0] for line in result[0] if line[1][0].strip()])
                if text:
                    image_texts.append(f"[Image {page_num + 1}.{img_index + 1}]:\n{text}\n")
    
    doc.close()
    return image_texts

def extract_text_from_pdf(pdf_path):
    """ extract regular text content from PDF """
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def process_pdf(pdf_path):
    """ process a PDF file to extract both regular text and text from images """
    print(f"\nProcessing file: {pdf_path}")
    
    # extract regular text
    print("- extracting regular text...")
    regular_text = extract_text_from_pdf(pdf_path)
    
    # extract text from images
    print("- extracting text from images (this may take a few moments)...")
    image_texts = extract_images_from_pdf(pdf_path)
    
    # combine the texts
    combined_text = "=== REGULAR TEXT ===\n\n"
    combined_text += regular_text
    
    if image_texts:
        combined_text += "\n\n=== TEXT FROM IMAGES ===\n\n"
        combined_text += "\n".join(image_texts)
    
    return combined_text

def main():
    """ process all PDFs in SBA_PDF folder and save extracted text to individual files """
    # create output directory if it doesn't exist
    output_dir = Path("SBA_text_files")
    output_dir.mkdir(exist_ok=True)

    # process all PDFs in the SBA_PDF folder
    pdf_folder = Path("SBA_PDF")
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("no PDF files found in the SBA_PDF folder!")
        return
        
    # sort the files to ensure consistent processing order
    pdf_files.sort()
    total_files = len(pdf_files)
    
    print(f"found {total_files} PDF files to process.")
    
    # process each PDF file
    for index, pdf_file in enumerate(pdf_files, 1):
        print(f"\nprocessing file {index} of {total_files}: {pdf_file.name}")
        
        # process the PDF (extract both regular text and image text)
        combined_text = process_pdf(pdf_file)
        
        # create output text file with the same number reference
        output_file = output_dir / f"sba_{pdf_file.stem}.txt"
        
        # save the combined text to a file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        
        print(f"created {output_file.name}")
        print(f"output file location: {output_file.absolute()}")

if __name__ == "__main__":
    main() 