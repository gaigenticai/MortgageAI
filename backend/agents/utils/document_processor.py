"""
Document Processing Utilities for MortgageAI

This module provides OCR and document parsing capabilities for mortgage applications,
supporting PDF, image formats, and structured form data extraction.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

import cv2
import numpy as np
import pytesseract
from PIL import Image
import pdfplumber
import tabula

from ..config import settings
from .ocr_service import OCRService


class DocumentProcessor:
    """
    Advanced document processing for mortgage applications.

    Supports:
    - OCR for scanned documents
    - PDF text extraction
    - Form field recognition
    - Table data extraction
    - Multi-language support
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Configure OCR
        if hasattr(settings, 'TESSERACT_DATA_PATH'):
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_DATA_PATH

        # Initialize OCR service
        self.ocr_service = None
        try:
            self.ocr_service = OCRService()
            self.logger.info("OCR.space API service initialized")
        except Exception as e:
            self.logger.warning(f"OCR.space API not available, falling back to Tesseract: {str(e)}")

        # Document type patterns for field extraction
        self.field_patterns = {
            'application_form': {
                'applicant_name': r'(?:name|naam)[\s:]+([A-Za-z\s]+)',
                'date_of_birth': r'(?:birth|geboren)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                'address': r'(?:address|adres)[\s:]+([^\n]+)',
                'mortgage_amount': r'(?:amount|bedrag)[\s:]+€?\s*(\d+(?:[.,]\d+)?)',
                'property_value': r'(?:value|waarde)[\s:]+€?\s*(\d+(?:[.,]\d+)?)',
                'income': r'(?:income|inkomen)[\s:]+€?\s*(\d+(?:[.,]\d+)?)',
                'loan_term': r'(?:term|looptijd)[\s:]+(\d+)\s*(?:years?|jaar|yr)',
            },
            'income_proof': {
                'employer_name': r'(?:employer|werkgever)[\s:]+([^\n]+)',
                'annual_income': r'(?:annual|jaar)[\s:]+€?\s*(\d+(?:[.,]\d+)?)',
                'tax_year': r'(?:tax|belasting)[\s:]+(\d{4})',
            },
            'property_documents': {
                'property_address': r'(?:property|pand)[\s:]+([^\n]+)',
                'valuation_amount': r'(?:valuation|taxatie)[\s:]+€?\s*(\d+(?:[.,]\d+)?)',
                'valuation_date': r'(?:date|datum)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            },
            'id_document': {
                'document_number': r'(?:number|nummer)[\s:]+([A-Z0-9]+)',
                'expiry_date': r'(?:expiry|verval)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                'issue_date': r'(?:issue|uitgifte)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            }
        }

    async def process_document(self, document_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process a document and extract relevant mortgage application data.

        Args:
            document_path: Path to the document file
            document_type: Type of document being processed

        Returns:
            Extracted data and metadata
        """
        try:
            file_path = Path(document_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")

            file_extension = file_path.suffix.lower()

            # Process based on file type
            if file_extension == '.pdf':
                extracted_data = await self._process_pdf(document_path)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                extracted_data = await self._process_image(document_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Extract structured data based on document type
            structured_data = self._extract_structured_data(extracted_data['text'], document_type)

            # Validate extracted data
            validation_results = self._validate_extracted_data(structured_data, document_type)

            result = {
                'raw_text': extracted_data['text'],
                'structured_data': structured_data,
                'validation_results': validation_results,
                'confidence_score': extracted_data.get('confidence', 0),
                'processing_method': extracted_data.get('method', 'unknown'),
                'document_metadata': {
                    'file_size': file_path.stat().st_size,
                    'pages': extracted_data.get('pages', 1),
                    'language_detected': extracted_data.get('language', 'unknown')
                }
            }

            self.logger.info(f"Successfully processed {document_type} document: {len(structured_data)} fields extracted")
            return result

        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise

    async def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF document for text extraction."""
        try:
            text_content = []
            confidence_scores = []

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)

                    # Try table extraction for structured data
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # Convert table to text representation
                            table_text = self._table_to_text(table)
                            text_content.append(table_text)

            full_text = '\n'.join(text_content)

            # Try OCR as fallback for scanned PDFs
            if not full_text.strip():
                full_text = await self._ocr_pdf(pdf_path)

            return {
                'text': full_text,
                'method': 'pdf_text_extraction',
                'pages': len(text_content),
                'confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            }

        except Exception as e:
            self.logger.warning(f"PDF processing failed, trying OCR: {str(e)}")
            # Fallback to OCR
            ocr_text = await self._ocr_pdf(pdf_path)
            return {
                'text': ocr_text,
                'method': 'pdf_ocr_fallback',
                'pages': 1,
                'confidence': 0.7  # Lower confidence for OCR
            }

    async def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image document using OCR.space API or fallback to Tesseract."""
        try:
            # Try OCR.space API first if available
            if self.ocr_service:
                try:
                    result = await self.ocr_service.process_document(image_path, 'auto')
                    if result['text'].strip():
                        return result
                except Exception as e:
                    self.logger.warning(f"OCR.space API failed, falling back to Tesseract: {str(e)}")

            # Fallback to Tesseract OCR
            self.logger.info("Using Tesseract OCR as fallback")

            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image_path)

            # Perform OCR
            ocr_data = pytesseract.image_to_data(processed_image, lang=settings.OCR_LANGUAGES, output_type=pytesseract.Output.DICT)

            # Extract text and confidence
            text_parts = []
            confidences = []

            for i, confidence in enumerate(ocr_data['conf']):
                if int(confidence) > 0:  # Only include recognized text
                    text_parts.append(ocr_data['text'][i])
                    confidences.append(int(confidence))

            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                'text': full_text,
                'method': 'image_ocr_tesseract',
                'pages': 1,
                'confidence': avg_confidence / 100,  # Convert to 0-1 scale
                'language': settings.OCR_LANGUAGES
            }

        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            raise

    async def _ocr_pdf(self, pdf_path: str) -> str:
        """Perform OCR on PDF by converting pages to images."""
        try:
            text_parts = []

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Convert page to image
                    page_image = page.to_image(resolution=300).original

                    # Preprocess and OCR
                    processed_image = self._preprocess_image_from_pil(page_image)
                    page_text = pytesseract.image_to_string(processed_image, lang=settings.OCR_LANGUAGES)

                    text_parts.append(page_text)

            return '\n'.join(text_parts)

        except Exception as e:
            self.logger.error(f"PDF OCR failed: {str(e)}")
            return ""

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)

            # Apply thresholding for better contrast
            _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return cv2.imread(image_path)

    def _preprocess_image_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Preprocess PIL image for OCR."""
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self._preprocess_image_from_cv(opencv_image)

    def _preprocess_image_from_cv(self, cv_image: np.ndarray) -> np.ndarray:
        """Preprocess OpenCV image array."""
        # Convert to grayscale if needed
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        # Apply noise reduction and thresholding
        denoised = cv2.medianBlur(gray, 3)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return threshold

    def _extract_structured_data(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract structured data from text using regex patterns."""
        extracted_data = {}
        patterns = self.field_patterns.get(document_type, {})

        text_lower = text.lower()

        for field, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip() if match.groups() else match.group(0).strip()

                # Post-process specific field types
                if field in ['mortgage_amount', 'property_value', 'income', 'annual_income', 'valuation_amount']:
                    value = self._clean_numeric_value(value)
                elif field in ['date_of_birth', 'valuation_date', 'expiry_date', 'issue_date']:
                    value = self._clean_date_value(value)

                extracted_data[field] = value

        return extracted_data

    def _validate_extracted_data(self, data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Validate extracted data for completeness and accuracy."""
        validation_results = {}
        required_fields = self.field_patterns.get(document_type, {})

        for field in required_fields.keys():
            if field in data and data[field]:
                validation_results[field] = {
                    'present': True,
                    'value': data[field],
                    'validation_status': 'extracted'
                }
            else:
                validation_results[field] = {
                    'present': False,
                    'value': None,
                    'validation_status': 'missing'
                }

        return validation_results

    def _clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean and parse numeric values from extracted text."""
        try:
            # Remove currency symbols and extra characters
            cleaned = re.sub(r'[€£$]', '', value)
            cleaned = re.sub(r'[^\d.,]', '', cleaned)

            # Handle European number format (comma as decimal separator)
            if ',' in cleaned and '.' in cleaned:
                # Assume last dot/comma is decimal separator
                parts = re.split(r'[,.]', cleaned)
                if len(parts) >= 2:
                    integer_part = ''.join(parts[:-1])
                    decimal_part = parts[-1]
                    cleaned = f"{integer_part}.{decimal_part}"
            elif ',' in cleaned:
                # Replace comma with dot for decimal
                cleaned = cleaned.replace(',', '.')

            return float(cleaned)

        except (ValueError, AttributeError):
            return None

    def _clean_date_value(self, value: str) -> Optional[str]:
        """Clean and standardize date values."""
        try:
            # Try different date formats
            date_patterns = [
                r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',  # DD-MM-YYYY or DD/MM/YYYY
                r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',  # YYYY-MM-DD
            ]

            for pattern in date_patterns:
                match = re.search(pattern, value)
                if match:
                    groups = match.groups()
                    if len(groups[0]) == 4:  # YYYY-MM-DD format
                        return f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
                    else:  # DD-MM-YYYY format
                        return f"{groups[2]}-{groups[1].zfill(2)}-{groups[0].zfill(2)}"

            return None

        except Exception:
            return None

    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to readable text format."""
        if not table:
            return ""

        text_rows = []
        for row in table:
            if row:
                # Clean and join row data
                clean_row = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                if clean_row:
                    text_rows.append(' | '.join(clean_row))

        return '\n'.join(text_rows)
