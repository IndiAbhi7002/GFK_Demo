import easyocr

# Initialize the OCR reader once
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if you don't have a GPU

def extract_text_from_image(image_path):
    """
    Extracts text from a given image using EasyOCR.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text as a single string.
    """
    try:
        result = reader.readtext(image_path, detail=0)
        text = ' '.join(result).strip()
        return text
    except Exception as e:
        return f"Error reading image: {e}"
