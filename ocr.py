import io
import re

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def extract_text_from_bytes(image_bytes):
    """Extract text from raw image bytes using pytesseract + Pillow.

    Returns a cleaned string (whitespace-normalized). Raises RuntimeError with
    actionable instructions if required dependencies are missing.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed. Install with: pip install Pillow")
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract\nAlso ensure the Tesseract OCR engine is installed on your system (see README).")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    try:
        raw_text = pytesseract.image_to_string(image)
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")

    # Basic cleanup
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    return cleaned


def analyze_image_bytes(image_bytes, detector):
    """Extract text from image bytes and run the detector on the extracted text.

    Returns a dict { 'extracted_text': str, 'analysis': <detector.analyze() result> }.
    """
    extracted = extract_text_from_bytes(image_bytes)
    if not extracted:
        # Return a minimal analysis indicating no text
        return {
            'extracted_text': '',
            'analysis': {
                'prediction': 'Invalid Input',
                'confidence': 0,
                'severity': 'none',
                'message': 'No text found in image'
            },
            'success': True
        }

    analysis = detector.analyze(extracted)
    return {
        'extracted_text': extracted,
        'analysis': analysis,
        'success': True
    }
