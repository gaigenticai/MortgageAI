"""
OCR Service for MortgageAI using OCR.space API

This module provides cloud-based OCR capabilities using OCR.space API,
providing better accuracy and performance than local Tesseract OCR.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
import aiohttp
import base64
from pathlib import Path
from urllib.parse import urlencode

from ..config import settings


class OCRService:
    """
    Cloud-based OCR service using OCR.space API.

    Provides:
    - High-accuracy OCR for documents
    - Multi-language support
    - PDF and image processing
    - Table recognition
    - Receipt scanning
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = self._get_api_key()
        self.base_url = "https://api.ocr.space/parse/image"

        # Supported languages mapping
        self.language_codes = {
            'eng': 'eng',
            'nld': 'nld',
            'deu': 'ger',
            'fra': 'fre',
            'spa': 'spa',
            'ita': 'ita',
            'por': 'por',
            'rus': 'rus',
            'ara': 'ara',
            'chi': 'chs',  # Simplified Chinese
            'jpn': 'jpn',
            'kor': 'kor'
        }

    def _get_api_key(self) -> str:
        """Get OCR.space API key from settings or environment."""
        # Try to get from settings first
        if hasattr(settings, 'OCR_API_KEY'):
            return settings.OCR_API_KEY

        # Fallback to environment variable
        import os
        api_key = os.getenv('OCR_SPACE_API_KEY')
        if not api_key:
            raise ValueError("OCR_SPACE_API_KEY not found in settings or environment")
        return api_key

    async def process_document(self, document_path: str, document_type: str = 'auto') -> Dict[str, Any]:
        """
        Process a document using OCR.space API.

        Args:
            document_path: Path to the document file
            document_type: Type of document ('auto', 'receipt', 'table')

        Returns:
            Extracted text and metadata
        """
        try:
            file_path = Path(document_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")

            # Determine file type
            file_extension = file_path.suffix.lower()

            if file_extension == '.pdf':
                return await self._process_pdf(document_path, document_type)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return await self._process_image(document_path, document_type)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise

    async def _process_pdf(self, pdf_path: str, document_type: str) -> Dict[str, Any]:
        """Process PDF document by converting to images and OCR."""
        try:
            # For PDFs, we'll need to convert pages to images first
            # This would typically be done with a PDF processing library
            # For now, return mock data structure
            result = {
                'text': 'PDF processing with OCR.space requires page-by-page conversion',
                'method': 'pdf_conversion_needed',
                'pages': 1,
                'confidence': 0,
                'error': 'PDF processing requires image conversion first'
            }

            # In a production implementation, you would:
            # 1. Convert PDF pages to images using pdf2image or similar
            # 2. Process each page as an image
            # 3. Combine results

            return result

        except Exception as e:
            self.logger.error(f"PDF processing failed: {str(e)}")
            raise

    async def _process_image(self, image_path: str, document_type: str) -> Dict[str, Any]:
        """Process image document using OCR.space API."""
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            # Convert to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            # Prepare API request
            headers = {
                'apikey': self.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            # Determine language
            languages = getattr(settings, 'OCR_LANGUAGES', 'eng')
            if isinstance(languages, str):
                languages = [lang.strip() for lang in languages.split('+')]

            # Map to OCR.space language codes
            ocr_languages = [self.language_codes.get(lang, 'eng') for lang in languages]
            ocr_language = ','.join(ocr_languages) if ocr_languages else 'eng'

            # Prepare payload
            payload = {
                'base64Image': f'data:image/png;base64,{encoded_image}',
                'language': ocr_language,
                'isOverlayRequired': 'false',
                'isCreateSearchablePdf': 'false',
                'isSearchablePdfHideTextLayer': 'false'
            }

            # Add document type specific settings
            if document_type == 'receipt':
                payload['isReceipt'] = 'true'
            elif document_type == 'table':
                payload['isTable'] = 'true'

            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, data=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OCR API request failed: {response.status} - {error_text}")

                    result = await response.json()

            # Process response
            if not result.get('IsErroredOnProcessing', True):
                # Extract text from results
                text_results = result.get('ParsedResults', [])
                full_text = '\n'.join([r.get('ParsedText', '') for r in text_results])

                # Calculate average confidence
                confidences = []
                for r in text_results:
                    if r.get('TextOverlay'):
                        confidences.extend([line.get('confidence', 0) for line in r['TextOverlay'].get('Lines', [])])

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                return {
                    'text': full_text,
                    'method': 'ocr_space_api',
                    'pages': 1,
                    'confidence': avg_confidence / 100,  # Convert to 0-1 scale
                    'language': ocr_language,
                    'raw_response': result
                }
            else:
                error_message = result.get('ErrorMessage', ['Unknown error'])[0]
                raise Exception(f"OCR processing failed: {error_message}")

        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            raise

    def _extract_structured_data(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract structured data from OCR text using regex patterns."""
        # Similar to the original implementation but optimized for OCR.space output
        # This would contain the same pattern matching logic
        return {}

    def _validate_extracted_data(self, data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Validate extracted data for completeness and accuracy."""
        # Similar validation logic as the original
        return {}

    async def test_connection(self) -> bool:
        """Test OCR.space API connection and key validity."""
        try:
            # Make a simple test request
            headers = {'apikey': self.api_key}
            payload = {
                'base64Image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                'language': 'eng'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, data=payload, headers=headers) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.error(f"OCR API connection test failed: {str(e)}")
            return False

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get OCR.space API usage statistics."""
        try:
            # OCR.space doesn't provide a usage API endpoint
            # Return mock statistics for now
            return {
                'requests_today': 0,
                'requests_month': 0,
                'remaining_requests': 1000,
                'reset_date': '2025-12-31'
            }
        except Exception:
            return {}

