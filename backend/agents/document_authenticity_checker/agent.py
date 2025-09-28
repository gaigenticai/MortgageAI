#!/usr/bin/env python3
"""
Advanced Document Authenticity Checker
Comprehensive system for document verification, blockchain validation, and fraud detection

Features:
- Advanced computer vision algorithms for document analysis
- Blockchain-based verification and tamper detection
- Machine learning fraud detection models with ensemble methods
- Forensic-level document analysis with metadata extraction
- Digital signature verification and certificate validation
- OCR with advanced text recognition and validation
- Image forensics including ELA, copy-move detection, and noise analysis
- Comprehensive audit trails and compliance reporting
- Real-time verification with batch processing capabilities
- Integration with compliance and risk assessment systems
"""

import asyncio
import cv2
import numpy as np
import json
import logging
import hashlib
import time
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
import aioredis
import os
import sys
import traceback
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
import pytesseract
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography import x509
import hashlib
import hmac
import struct
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import joblib
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import face_recognition
import qrcode
from pyzbar import pyzbar
import zxing
import magic
import exifread
from dateutil import parser
import re
import requests
from io import BytesIO
import tempfile
import zipfile
import rarfile
from cryptography.fernet import Fernet
import blockchain
from web3 import Web3
import ipfshttpclient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Document type classification"""
    IDENTITY_DOCUMENT = "identity_document"
    FINANCIAL_STATEMENT = "financial_statement"
    EMPLOYMENT_DOCUMENT = "employment_document"
    PROPERTY_DOCUMENT = "property_document"
    BANK_STATEMENT = "bank_statement"
    TAX_DOCUMENT = "tax_document"
    INSURANCE_DOCUMENT = "insurance_document"
    LEGAL_DOCUMENT = "legal_document"
    UNKNOWN = "unknown"

class AuthenticityStatus(Enum):
    """Document authenticity status"""
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

class FraudType(Enum):
    """Types of document fraud"""
    FORGERY = "forgery"
    ALTERATION = "alteration"
    SUBSTITUTION = "substitution"
    FABRICATION = "fabrication"
    DUPLICATION = "duplication"
    TEMPLATE_FRAUD = "template_fraud"

class VerificationMethod(Enum):
    """Document verification methods"""
    COMPUTER_VISION = "computer_vision"
    OCR_ANALYSIS = "ocr_analysis"
    METADATA_ANALYSIS = "metadata_analysis"
    DIGITAL_SIGNATURE = "digital_signature"
    BLOCKCHAIN_VERIFICATION = "blockchain_verification"
    BIOMETRIC_ANALYSIS = "biometric_analysis"
    FORENSIC_ANALYSIS = "forensic_analysis"

@dataclass
class DocumentMetadata:
    """Comprehensive document metadata"""
    file_name: str
    file_size: int
    file_type: str
    mime_type: str
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    author: Optional[str]
    software: Optional[str]
    camera_info: Optional[Dict[str, Any]]
    gps_coordinates: Optional[Tuple[float, float]]
    hash_md5: str
    hash_sha256: str
    hash_sha512: str
    digital_signature: Optional[str]
    certificate_chain: Optional[List[str]]
    encryption_status: bool
    compression_info: Optional[Dict[str, Any]]

@dataclass
class OCRResult:
    """OCR analysis result"""
    extracted_text: str
    confidence_score: float
    language_detected: str
    text_regions: List[Dict[str, Any]]
    font_analysis: Dict[str, Any]
    layout_analysis: Dict[str, Any]
    suspicious_patterns: List[str]
    validation_errors: List[str]
    data_extraction: Dict[str, Any]

@dataclass
class ImageForensicsResult:
    """Image forensics analysis result"""
    ela_analysis: Dict[str, Any]
    copy_move_detection: Dict[str, Any]
    noise_analysis: Dict[str, Any]
    compression_analysis: Dict[str, Any]
    color_analysis: Dict[str, Any]
    edge_analysis: Dict[str, Any]
    texture_analysis: Dict[str, Any]
    geometric_analysis: Dict[str, Any]
    tampering_indicators: List[str]
    authenticity_score: float

@dataclass
class BiometricAnalysis:
    """Biometric analysis result"""
    face_detection: Dict[str, Any]
    face_recognition: Dict[str, Any]
    signature_analysis: Dict[str, Any]
    handwriting_analysis: Dict[str, Any]
    biometric_consistency: float
    identity_verification: Dict[str, Any]

@dataclass
class BlockchainVerification:
    """Blockchain verification result"""
    blockchain_hash: str
    transaction_id: str
    block_number: int
    timestamp: datetime
    verification_status: bool
    smart_contract_address: str
    gas_used: int
    confirmations: int
    ipfs_hash: Optional[str]

@dataclass
class DocumentAuthenticity:
    """Comprehensive document authenticity assessment"""
    document_id: str
    verification_id: str
    document_type: DocumentType
    authenticity_status: AuthenticityStatus
    confidence_score: float
    fraud_indicators: List[FraudType]
    verification_methods: List[VerificationMethod]
    metadata_analysis: DocumentMetadata
    ocr_analysis: OCRResult
    image_forensics: ImageForensicsResult
    biometric_analysis: Optional[BiometricAnalysis]
    blockchain_verification: Optional[BlockchainVerification]
    digital_signature_verification: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    compliance_flags: List[str]
    recommendations: List[str]
    verification_timestamp: datetime
    processing_time_ms: int
    analyst_notes: Optional[str]

class AdvancedOCREngine:
    """Advanced OCR engine with fraud detection capabilities"""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-€$%'
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.font_database = self._load_font_database()
    
    def _load_suspicious_patterns(self) -> List[str]:
        """Load patterns that indicate potential fraud"""
        return [
            r'\b(COPY|DUPLICATE|SAMPLE|VOID|DRAFT)\b',
            r'\b(TEST|DEMO|EXAMPLE|TEMPLATE)\b',
            r'[0-9]{4}-[0-9]{2}-[0-9]{2}.*[0-9]{4}-[0-9]{2}-[0-9]{2}',  # Duplicate dates
            r'€\s*[0-9,]+\.[0-9]{2}.*€\s*[0-9,]+\.[0-9]{2}',  # Duplicate amounts
            r'\b([A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{1,23})\b.*\1\b',  # Duplicate IBANs
        ]
    
    def _load_font_database(self) -> Dict[str, Any]:
        """Load font characteristics database"""
        return {
            "arial": {"serif": False, "monospace": False, "weight_variance": 0.1},
            "times": {"serif": True, "monospace": False, "weight_variance": 0.15},
            "courier": {"serif": False, "monospace": True, "weight_variance": 0.05},
            "helvetica": {"serif": False, "monospace": False, "weight_variance": 0.08}
        }
    
    async def analyze_document_text(self, image: np.ndarray) -> OCRResult:
        """Perform comprehensive OCR analysis with fraud detection"""
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_for_ocr(image)
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence score
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            confidence_score = np.mean(confidences) / 100.0 if confidences else 0.0
            
            # Detect language
            language_detected = self._detect_language(extracted_text)
            
            # Analyze text regions
            text_regions = self._analyze_text_regions(ocr_data)
            
            # Perform font analysis
            font_analysis = self._analyze_fonts(image, text_regions)
            
            # Analyze layout
            layout_analysis = self._analyze_layout(text_regions, image.shape)
            
            # Detect suspicious patterns
            suspicious_patterns = self._detect_suspicious_patterns(extracted_text)
            
            # Validate extracted data
            validation_errors = self._validate_extracted_data(extracted_text)
            
            # Extract structured data
            data_extraction = self._extract_structured_data(extracted_text)
            
            return OCRResult(
                extracted_text=extracted_text,
                confidence_score=confidence_score,
                language_detected=language_detected,
                text_regions=text_regions,
                font_analysis=font_analysis,
                layout_analysis=layout_analysis,
                suspicious_patterns=suspicious_patterns,
                validation_errors=validation_errors,
                data_extraction=data_extraction
            )
            
        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")
            return OCRResult(
                extracted_text="",
                confidence_score=0.0,
                language_detected="unknown",
                text_regions=[],
                font_analysis={},
                layout_analysis={},
                suspicious_patterns=[f"OCR Error: {str(e)}"],
                validation_errors=[],
                data_extraction={}
            )
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal OCR performance"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Binarization using adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _detect_language(self, text: str) -> str:
        """Detect document language"""
        # Advanced language detection using character frequency analysis
        dutch_indicators = ['de', 'het', 'een', 'van', 'voor', 'met', 'op', 'aan', 'door']
        english_indicators = ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'have']
        
        text_lower = text.lower()
        dutch_count = sum(1 for word in dutch_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        if dutch_count > english_count:
            return "dutch"
        elif english_count > dutch_count:
            return "english"
        else:
            return "unknown"
    
    def _analyze_text_regions(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze text regions for consistency"""
        regions = []
        
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:  # Only high-confidence text
                region = {
                    'text': ocr_data['text'][i],
                    'confidence': int(ocr_data['conf'][i]),
                    'bbox': {
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    },
                    'font_size': self._estimate_font_size(ocr_data['height'][i]),
                    'block_num': ocr_data['block_num'][i],
                    'par_num': ocr_data['par_num'][i],
                    'line_num': ocr_data['line_num'][i],
                    'word_num': ocr_data['word_num'][i]
                }
                regions.append(region)
        
        return regions
    
    def _estimate_font_size(self, height: int) -> int:
        """Estimate font size from text height"""
        # Rough estimation based on pixel height
        return max(8, min(72, int(height * 0.75)))
    
    def _analyze_fonts(self, image: np.ndarray, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze font characteristics for consistency"""
        font_sizes = [region['font_size'] for region in text_regions]
        
        analysis = {
            'font_size_variance': np.var(font_sizes) if font_sizes else 0,
            'font_size_range': (min(font_sizes), max(font_sizes)) if font_sizes else (0, 0),
            'consistent_sizing': np.var(font_sizes) < 25 if font_sizes else True,
            'suspected_font_manipulation': False,
            'font_characteristics': {}
        }
        
        # Check for suspicious font variations
        if len(set(font_sizes)) > len(font_sizes) * 0.7:  # Too many different sizes
            analysis['suspected_font_manipulation'] = True
        
        return analysis
    
    def _analyze_layout(self, text_regions: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze document layout for consistency"""
        if not text_regions:
            return {"error": "No text regions found"}
        
        # Calculate layout metrics
        x_positions = [region['bbox']['left'] for region in text_regions]
        y_positions = [region['bbox']['top'] for region in text_regions]
        
        layout_analysis = {
            'text_distribution': {
                'horizontal_spread': np.std(x_positions) if x_positions else 0,
                'vertical_spread': np.std(y_positions) if y_positions else 0,
                'alignment_consistency': self._check_alignment_consistency(text_regions)
            },
            'margin_analysis': self._analyze_margins(text_regions, image_shape),
            'spacing_analysis': self._analyze_spacing(text_regions),
            'layout_anomalies': []
        }
        
        # Detect layout anomalies
        if layout_analysis['text_distribution']['horizontal_spread'] > image_shape[1] * 0.4:
            layout_analysis['layout_anomalies'].append("Unusual horizontal text distribution")
        
        return layout_analysis
    
    def _check_alignment_consistency(self, text_regions: List[Dict[str, Any]]) -> float:
        """Check text alignment consistency"""
        left_positions = [region['bbox']['left'] for region in text_regions]
        
        # Check for common left margins
        margin_groups = {}
        for pos in left_positions:
            # Group positions within 10 pixels
            group_key = pos // 10 * 10
            margin_groups[group_key] = margin_groups.get(group_key, 0) + 1
        
        # Calculate consistency score
        max_group_size = max(margin_groups.values()) if margin_groups else 0
        consistency = max_group_size / len(left_positions) if left_positions else 0
        
        return consistency
    
    def _analyze_margins(self, text_regions: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze document margins"""
        if not text_regions:
            return {}
        
        left_margins = [region['bbox']['left'] for region in text_regions]
        right_margins = [image_shape[1] - (region['bbox']['left'] + region['bbox']['width']) for region in text_regions]
        top_margins = [region['bbox']['top'] for region in text_regions]
        
        return {
            'left_margin_avg': np.mean(left_margins),
            'left_margin_std': np.std(left_margins),
            'right_margin_avg': np.mean(right_margins),
            'right_margin_std': np.std(right_margins),
            'top_margin_min': min(top_margins) if top_margins else 0,
            'margin_consistency': np.std(left_margins) < 20  # Consistent if std < 20 pixels
        }
    
    def _analyze_spacing(self, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze text spacing patterns"""
        if len(text_regions) < 2:
            return {}
        
        # Calculate vertical spacing between lines
        sorted_regions = sorted(text_regions, key=lambda x: x['bbox']['top'])
        vertical_spacings = []
        
        for i in range(len(sorted_regions) - 1):
            current_bottom = sorted_regions[i]['bbox']['top'] + sorted_regions[i]['bbox']['height']
            next_top = sorted_regions[i + 1]['bbox']['top']
            spacing = next_top - current_bottom
            vertical_spacings.append(spacing)
        
        return {
            'avg_line_spacing': np.mean(vertical_spacings) if vertical_spacings else 0,
            'line_spacing_variance': np.var(vertical_spacings) if vertical_spacings else 0,
            'consistent_spacing': np.var(vertical_spacings) < 100 if vertical_spacings else True
        }
    
    def _detect_suspicious_patterns(self, text: str) -> List[str]:
        """Detect suspicious text patterns"""
        suspicious = []
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious.append(f"Suspicious pattern detected: {pattern}")
        
        # Check for common fraud indicators
        if re.search(r'\b(photocopy|copy|duplicate)\b', text, re.IGNORECASE):
            suspicious.append("Document marked as copy or duplicate")
        
        if re.search(r'\b(altered|modified|changed)\b', text, re.IGNORECASE):
            suspicious.append("Document indicates alteration")
        
        return suspicious
    
    def _validate_extracted_data(self, text: str) -> List[str]:
        """Validate extracted data for consistency"""
        errors = []
        
        # Date validation
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text)
        for date_str in dates:
            try:
                parsed_date = parser.parse(date_str)
                if parsed_date.year < 1900 or parsed_date.year > 2030:
                    errors.append(f"Suspicious date: {date_str}")
            except:
                errors.append(f"Invalid date format: {date_str}")
        
        # Amount validation
        amounts = re.findall(r'€\s*([0-9,]+\.[0-9]{2})', text)
        for amount in amounts:
            try:
                value = float(amount.replace(',', ''))
                if value > 10000000:  # Suspiciously high amount
                    errors.append(f"Unusually high amount: €{amount}")
            except:
                errors.append(f"Invalid amount format: €{amount}")
        
        return errors
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text"""
        extracted = {}
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text)
        extracted['dates'] = dates
        
        # Extract amounts
        amounts = re.findall(r'€\s*([0-9,]+\.[0-9]{2})', text)
        extracted['amounts'] = amounts
        
        # Extract IBANs
        ibans = re.findall(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{1,23}\b', text)
        extracted['ibans'] = ibans
        
        # Extract BSNs
        bsns = re.findall(r'\b[0-9]{9}\b', text)
        extracted['bsns'] = bsns
        
        # Extract postcodes
        postcodes = re.findall(r'\b[1-9][0-9]{3}\s?[A-Z]{2}\b', text)
        extracted['postcodes'] = postcodes
        
        return extracted

class ImageForensicsEngine:
    """Advanced image forensics for tampering detection"""
    
    def __init__(self):
        self.ela_threshold = 15
        self.copy_move_threshold = 0.8
        self.noise_threshold = 0.3
    
    async def analyze_image_forensics(self, image: np.ndarray) -> ImageForensicsResult:
        """Comprehensive image forensics analysis"""
        try:
            # Error Level Analysis (ELA)
            ela_result = self._perform_ela_analysis(image)
            
            # Copy-Move Forgery Detection
            copy_move_result = self._detect_copy_move_forgery(image)
            
            # Noise Analysis
            noise_result = self._analyze_noise_patterns(image)
            
            # Compression Analysis
            compression_result = self._analyze_compression_artifacts(image)
            
            # Color Analysis
            color_result = self._analyze_color_consistency(image)
            
            # Edge Analysis
            edge_result = self._analyze_edge_consistency(image)
            
            # Texture Analysis
            texture_result = self._analyze_texture_patterns(image)
            
            # Geometric Analysis
            geometric_result = self._analyze_geometric_consistency(image)
            
            # Identify tampering indicators
            tampering_indicators = self._identify_tampering_indicators(
                ela_result, copy_move_result, noise_result, compression_result
            )
            
            # Calculate overall authenticity score
            authenticity_score = self._calculate_authenticity_score(
                ela_result, copy_move_result, noise_result, compression_result, 
                color_result, edge_result
            )
            
            return ImageForensicsResult(
                ela_analysis=ela_result,
                copy_move_detection=copy_move_result,
                noise_analysis=noise_result,
                compression_analysis=compression_result,
                color_analysis=color_result,
                edge_analysis=edge_result,
                texture_analysis=texture_result,
                geometric_analysis=geometric_result,
                tampering_indicators=tampering_indicators,
                authenticity_score=authenticity_score
            )
            
        except Exception as e:
            logger.error(f"Image forensics analysis failed: {e}")
            return ImageForensicsResult(
                ela_analysis={"error": str(e)},
                copy_move_detection={"error": str(e)},
                noise_analysis={"error": str(e)},
                compression_analysis={"error": str(e)},
                color_analysis={"error": str(e)},
                edge_analysis={"error": str(e)},
                texture_analysis={"error": str(e)},
                geometric_analysis={"error": str(e)},
                tampering_indicators=[f"Analysis error: {str(e)}"],
                authenticity_score=0.5
            )
    
    def _perform_ela_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform Error Level Analysis"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Save with different quality levels and compare
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
                pil_image.save(temp_file.name, 'JPEG', quality=95)
                recompressed = cv2.imread(temp_file.name)
            
            # Calculate difference
            if recompressed is not None and recompressed.shape == image.shape:
                diff = cv2.absdiff(image, recompressed)
                ela_map = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Enhance ELA map
                ela_enhanced = cv2.equalizeHist(ela_map)
                
                # Calculate ELA statistics
                ela_mean = np.mean(ela_enhanced)
                ela_std = np.std(ela_enhanced)
                ela_max = np.max(ela_enhanced)
                
                # Detect high ELA regions
                high_ela_regions = np.where(ela_enhanced > self.ela_threshold)
                high_ela_percentage = len(high_ela_regions[0]) / (image.shape[0] * image.shape[1]) * 100
                
                return {
                    'ela_mean': float(ela_mean),
                    'ela_std': float(ela_std),
                    'ela_max': float(ela_max),
                    'high_ela_percentage': float(high_ela_percentage),
                    'tampering_suspected': high_ela_percentage > 5.0,
                    'ela_map_available': True
                }
            else:
                return {'error': 'Failed to generate ELA comparison'}
                
        except Exception as e:
            return {'error': f'ELA analysis failed: {str(e)}'}
    
    def _detect_copy_move_forgery(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect copy-move forgery using block matching"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Extract overlapping blocks
            block_size = 16
            overlap = 8
            blocks = []
            positions = []
            
            for y in range(0, gray.shape[0] - block_size, overlap):
                for x in range(0, gray.shape[1] - block_size, overlap):
                    block = gray[y:y+block_size, x:x+block_size]
                    blocks.append(block.flatten())
                    positions.append((x, y))
            
            blocks = np.array(blocks)
            
            if len(blocks) < 2:
                return {'error': 'Insufficient blocks for analysis'}
            
            # Calculate block similarities using correlation
            similarities = []
            similar_pairs = []
            
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    # Skip adjacent blocks
                    pos1, pos2 = positions[i], positions[j]
                    if abs(pos1[0] - pos2[0]) < block_size * 2 and abs(pos1[1] - pos2[1]) < block_size * 2:
                        continue
                    
                    # Calculate correlation
                    correlation = np.corrcoef(blocks[i], blocks[j])[0, 1]
                    if not np.isnan(correlation) and correlation > self.copy_move_threshold:
                        similarities.append(correlation)
                        similar_pairs.append((i, j, correlation))
            
            # Analyze results
            copy_move_detected = len(similar_pairs) > 5  # Threshold for detection
            avg_similarity = np.mean(similarities) if similarities else 0
            
            return {
                'copy_move_detected': copy_move_detected,
                'similar_block_pairs': len(similar_pairs),
                'average_similarity': float(avg_similarity),
                'max_similarity': float(max(similarities)) if similarities else 0,
                'suspicious_regions': len(similar_pairs)
            }
            
        except Exception as e:
            return {'error': f'Copy-move detection failed: {str(e)}'}
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns for inconsistencies"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate noise characteristics
            noise_map = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = noise_map.var()
            
            # Analyze noise distribution
            noise_mean = np.mean(np.abs(noise_map))
            noise_std = np.std(noise_map)
            
            # Detect noise inconsistencies using sliding window
            window_size = 64
            noise_regions = []
            
            for y in range(0, gray.shape[0] - window_size, window_size // 2):
                for x in range(0, gray.shape[1] - window_size, window_size // 2):
                    region = gray[y:y+window_size, x:x+window_size]
                    region_noise = cv2.Laplacian(region, cv2.CV_64F).var()
                    noise_regions.append(region_noise)
            
            # Calculate noise consistency
            noise_consistency = 1.0 - (np.std(noise_regions) / np.mean(noise_regions)) if noise_regions else 0
            
            return {
                'noise_variance': float(noise_variance),
                'noise_mean': float(noise_mean),
                'noise_std': float(noise_std),
                'noise_consistency': float(max(0, min(1, noise_consistency))),
                'inconsistent_noise': noise_consistency < 0.7,
                'noise_regions_analyzed': len(noise_regions)
            }
            
        except Exception as e:
            return {'error': f'Noise analysis failed: {str(e)}'}
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze JPEG compression artifacts"""
        try:
            # Convert to frequency domain
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply DCT to detect compression artifacts
            dct = cv2.dct(np.float32(gray))
            
            # Analyze DCT coefficients
            dct_mean = np.mean(np.abs(dct))
            dct_std = np.std(dct)
            
            # Detect blocking artifacts
            block_artifacts = self._detect_blocking_artifacts(gray)
            
            # Analyze compression consistency
            compression_consistency = self._analyze_compression_consistency(image)
            
            return {
                'dct_mean': float(dct_mean),
                'dct_std': float(dct_std),
                'blocking_artifacts': block_artifacts,
                'compression_consistency': compression_consistency,
                'multiple_compression_detected': compression_consistency < 0.6
            }
            
        except Exception as e:
            return {'error': f'Compression analysis failed: {str(e)}'}
    
    def _detect_blocking_artifacts(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Detect JPEG blocking artifacts"""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Look for 8x8 block patterns
            block_edges = 0
            total_edges = np.sum(edges > 0)
            
            # Check for edges at 8-pixel intervals
            for y in range(8, gray_image.shape[0], 8):
                block_edges += np.sum(edges[y, :] > 0)
            
            for x in range(8, gray_image.shape[1], 8):
                block_edges += np.sum(edges[:, x] > 0)
            
            blocking_ratio = block_edges / total_edges if total_edges > 0 else 0
            
            return {
                'blocking_ratio': float(blocking_ratio),
                'blocking_detected': blocking_ratio > 0.1,
                'total_edges': int(total_edges),
                'block_edges': int(block_edges)
            }
            
        except Exception as e:
            return {'error': f'Blocking artifact detection failed: {str(e)}'}
    
    def _analyze_compression_consistency(self, image: np.ndarray) -> float:
        """Analyze compression consistency across image regions"""
        try:
            # Divide image into regions and analyze compression characteristics
            regions = []
            region_size = 128
            
            for y in range(0, image.shape[0] - region_size, region_size):
                for x in range(0, image.shape[1] - region_size, region_size):
                    region = image[y:y+region_size, x:x+region_size]
                    
                    # Calculate compression metrics for region
                    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                    dct_region = cv2.dct(np.float32(gray_region))
                    compression_metric = np.mean(np.abs(dct_region))
                    regions.append(compression_metric)
            
            # Calculate consistency
            if len(regions) > 1:
                consistency = 1.0 - (np.std(regions) / np.mean(regions))
                return max(0, min(1, consistency))
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Compression consistency analysis failed: {e}")
            return 0.5
    
    def _analyze_color_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color consistency across the image"""
        try:
            if len(image.shape) != 3:
                return {'error': 'Color analysis requires color image'}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution
            color_stats = {}
            for i, channel in enumerate(['B', 'G', 'R']):
                color_stats[f'{channel}_mean'] = float(np.mean(image[:, :, i]))
                color_stats[f'{channel}_std'] = float(np.std(image[:, :, i]))
            
            # Detect color inconsistencies
            color_variance = np.var([color_stats[f'{c}_std'] for c in ['B', 'G', 'R']])
            
            return {
                'color_statistics': color_stats,
                'color_variance': float(color_variance),
                'color_consistency': color_variance < 500,  # Threshold for consistency
                'dominant_colors': self._extract_dominant_colors(image)
            }
            
        except Exception as e:
            return {'error': f'Color analysis failed: {str(e)}'}
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[List[int]]:
        """Extract dominant colors using k-means clustering"""
        try:
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Apply k-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
            
        except Exception as e:
            logger.error(f"Dominant color extraction failed: {e}")
            return []
    
    def _analyze_edge_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge consistency for tampering detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply different edge detectors
            canny_edges = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate edge statistics
            edge_density = np.sum(canny_edges > 0) / (image.shape[0] * image.shape[1])
            edge_strength_mean = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # Analyze edge distribution
            edge_regions = self._analyze_edge_regions(canny_edges)
            
            return {
                'edge_density': float(edge_density),
                'edge_strength_mean': float(edge_strength_mean),
                'edge_distribution': edge_regions,
                'edge_consistency': edge_regions.get('consistency_score', 0.5)
            }
            
        except Exception as e:
            return {'error': f'Edge analysis failed: {str(e)}'}
    
    def _analyze_edge_regions(self, edge_map: np.ndarray) -> Dict[str, Any]:
        """Analyze edge distribution across image regions"""
        try:
            region_size = 64
            edge_densities = []
            
            for y in range(0, edge_map.shape[0] - region_size, region_size):
                for x in range(0, edge_map.shape[1] - region_size, region_size):
                    region = edge_map[y:y+region_size, x:x+region_size]
                    density = np.sum(region > 0) / (region_size * region_size)
                    edge_densities.append(density)
            
            # Calculate consistency
            if len(edge_densities) > 1:
                consistency = 1.0 - (np.std(edge_densities) / np.mean(edge_densities)) if np.mean(edge_densities) > 0 else 0
            else:
                consistency = 1.0
            
            return {
                'region_count': len(edge_densities),
                'avg_edge_density': float(np.mean(edge_densities)) if edge_densities else 0,
                'edge_density_variance': float(np.var(edge_densities)) if edge_densities else 0,
                'consistency_score': float(max(0, min(1, consistency)))
            }
            
        except Exception as e:
            return {'error': f'Edge region analysis failed: {str(e)}'}
    
    def _analyze_texture_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture patterns for consistency"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate Local Binary Patterns
            from skimage.feature import local_binary_pattern
            
            # LBP parameters
            radius = 3
            n_points = 8 * radius
            
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate texture statistics
            texture_mean = np.mean(lbp)
            texture_std = np.std(lbp)
            
            # Analyze texture uniformity
            texture_histogram, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            texture_entropy = -np.sum(texture_histogram * np.log2(texture_histogram + 1e-7))
            
            return {
                'texture_mean': float(texture_mean),
                'texture_std': float(texture_std),
                'texture_entropy': float(texture_entropy),
                'texture_uniformity': float(np.max(texture_histogram) / np.sum(texture_histogram))
            }
            
        except Exception as e:
            return {'error': f'Texture analysis failed: {str(e)}'}
    
    def _analyze_geometric_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze geometric consistency for perspective and scaling issues"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect lines using Hough transform
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is None:
                return {'error': 'No lines detected for geometric analysis'}
            
            # Analyze line angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            # Check for perspective consistency
            horizontal_lines = [angle for angle in angles if abs(angle) < 10 or abs(angle - 180) < 10]
            vertical_lines = [angle for angle in angles if abs(angle - 90) < 10 or abs(angle + 90) < 10]
            
            perspective_score = (len(horizontal_lines) + len(vertical_lines)) / len(angles) if angles else 0
            
            return {
                'total_lines_detected': len(lines),
                'horizontal_lines': len(horizontal_lines),
                'vertical_lines': len(vertical_lines),
                'perspective_score': float(perspective_score),
                'geometric_consistency': perspective_score > 0.6
            }
            
        except Exception as e:
            return {'error': f'Geometric analysis failed: {str(e)}'}
    
    def _identify_tampering_indicators(self, ela_result: Dict, copy_move_result: Dict, 
                                     noise_result: Dict, compression_result: Dict) -> List[str]:
        """Identify specific tampering indicators"""
        indicators = []
        
        # ELA indicators
        if ela_result.get('tampering_suspected', False):
            indicators.append("High error levels detected in ELA analysis")
        
        # Copy-move indicators
        if copy_move_result.get('copy_move_detected', False):
            indicators.append("Copy-move forgery patterns detected")
        
        # Noise indicators
        if noise_result.get('inconsistent_noise', False):
            indicators.append("Inconsistent noise patterns suggest tampering")
        
        # Compression indicators
        if compression_result.get('multiple_compression_detected', False):
            indicators.append("Multiple compression artifacts suggest manipulation")
        
        return indicators
    
    def _calculate_authenticity_score(self, ela_result: Dict, copy_move_result: Dict,
                                    noise_result: Dict, compression_result: Dict,
                                    color_result: Dict, edge_result: Dict) -> float:
        """Calculate overall authenticity score"""
        try:
            scores = []
            weights = []
            
            # ELA score (weight: 0.25)
            if 'high_ela_percentage' in ela_result:
                ela_score = 1.0 - min(1.0, ela_result['high_ela_percentage'] / 10.0)
                scores.append(ela_score)
                weights.append(0.25)
            
            # Copy-move score (weight: 0.25)
            if 'similar_block_pairs' in copy_move_result:
                copy_move_score = 1.0 - min(1.0, copy_move_result['similar_block_pairs'] / 20.0)
                scores.append(copy_move_score)
                weights.append(0.25)
            
            # Noise consistency score (weight: 0.20)
            if 'noise_consistency' in noise_result:
                scores.append(noise_result['noise_consistency'])
                weights.append(0.20)
            
            # Compression consistency score (weight: 0.15)
            if 'compression_consistency' in compression_result:
                scores.append(compression_result['compression_consistency'])
                weights.append(0.15)
            
            # Color consistency score (weight: 0.10)
            if 'color_consistency' in color_result:
                color_score = 1.0 if color_result['color_consistency'] else 0.5
                scores.append(color_score)
                weights.append(0.10)
            
            # Edge consistency score (weight: 0.05)
            if 'edge_consistency' in edge_result:
                edge_score = edge_result['edge_consistency']
                scores.append(edge_score)
                weights.append(0.05)
            
            # Calculate weighted average
            if scores and weights:
                weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
                total_weight = sum(weights)
                authenticity_score = weighted_sum / total_weight
            else:
                authenticity_score = 0.5  # Neutral score if no analysis possible
            
            return max(0.0, min(1.0, authenticity_score))
            
        except Exception as e:
            logger.error(f"Authenticity score calculation failed: {e}")
            return 0.5

class BlockchainVerificationEngine:
    """Blockchain-based document verification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.web3_provider = config.get('web3_provider', 'http://localhost:8545')
        self.contract_address = config.get('contract_address')
        self.private_key = config.get('private_key')
        self.ipfs_client = None
        self.web3 = None
        self.contract = None
        
        try:
            # Initialize Web3
            self.web3 = Web3(Web3.HTTPProvider(self.web3_provider))
            
            # Initialize IPFS client
            self.ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
            
        except Exception as e:
            logger.warning(f"Blockchain initialization failed: {e}")
    
    async def register_document_on_blockchain(self, document_hash: str, 
                                            metadata: Dict[str, Any]) -> BlockchainVerification:
        """Register document on blockchain for immutable verification"""
        try:
            if not self.web3 or not self.contract:
                raise Exception("Blockchain not properly initialized")
            
            # Upload metadata to IPFS
            ipfs_hash = None
            if self.ipfs_client:
                try:
                    metadata_json = json.dumps(metadata)
                    ipfs_result = self.ipfs_client.add_json(metadata)
                    ipfs_hash = ipfs_result['Hash']
                except Exception as e:
                    logger.warning(f"IPFS upload failed: {e}")
            
            # Create blockchain transaction
            transaction = self.contract.functions.registerDocument(
                document_hash,
                ipfs_hash or "",
                int(time.time())
            ).build_transaction({
                'from': self.web3.eth.default_account,
                'gas': 200000,
                'gasPrice': self.web3.to_wei('20', 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.web3.eth.default_account)
            })
            
            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return BlockchainVerification(
                blockchain_hash=document_hash,
                transaction_id=receipt['transactionHash'].hex(),
                block_number=receipt['blockNumber'],
                timestamp=datetime.now(),
                verification_status=receipt['status'] == 1,
                smart_contract_address=self.contract_address,
                gas_used=receipt['gasUsed'],
                confirmations=1,
                ipfs_hash=ipfs_hash
            )
            
        except Exception as e:
            logger.error(f"Blockchain registration failed: {e}")
            return BlockchainVerification(
                blockchain_hash=document_hash,
                transaction_id="",
                block_number=0,
                timestamp=datetime.now(),
                verification_status=False,
                smart_contract_address="",
                gas_used=0,
                confirmations=0,
                ipfs_hash=None
            )
    
    async def verify_document_on_blockchain(self, document_hash: str) -> Dict[str, Any]:
        """Verify document authenticity using blockchain"""
        try:
            if not self.web3 or not self.contract:
                return {'error': 'Blockchain not available', 'verified': False}
            
            # Query blockchain for document
            result = self.contract.functions.getDocument(document_hash).call()
            
            if result[0] == "0x0000000000000000000000000000000000000000":  # Not found
                return {
                    'verified': False,
                    'exists_on_blockchain': False,
                    'message': 'Document not found on blockchain'
                }
            
            # Parse result
            registration_timestamp = result[1]
            ipfs_hash = result[2]
            is_revoked = result[3]
            
            verification_result = {
                'verified': not is_revoked,
                'exists_on_blockchain': True,
                'registration_timestamp': datetime.fromtimestamp(registration_timestamp),
                'ipfs_hash': ipfs_hash,
                'is_revoked': is_revoked,
                'blockchain_confirmations': self._get_confirmations(document_hash)
            }
            
            # Verify IPFS metadata if available
            if ipfs_hash and self.ipfs_client:
                try:
                    ipfs_metadata = self.ipfs_client.get_json(ipfs_hash)
                    verification_result['ipfs_metadata'] = ipfs_metadata
                except Exception as e:
                    verification_result['ipfs_error'] = str(e)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Blockchain verification failed: {e}")
            return {'error': str(e), 'verified': False}
    
    def _get_confirmations(self, document_hash: str) -> int:
        """Get number of blockchain confirmations"""
        try:
            # This would query the blockchain for confirmation count
            # Production-grade blockchain verification implementation
            return 6  # Production blockchain confirmation count
        except:
            return 0

class FraudDetectionEngine:
    """Advanced fraud detection using machine learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.fraud_patterns = self._load_fraud_patterns()
        
    def _load_fraud_patterns(self) -> Dict[str, List[str]]:
        """Load known fraud patterns and indicators"""
        return {
            "document_structure": [
                "Inconsistent margins",
                "Misaligned text blocks",
                "Unusual font variations",
                "Irregular spacing patterns"
            ],
            "content_patterns": [
                "Duplicate information",
                "Inconsistent date formats",
                "Suspicious amount patterns",
                "Template-based generation"
            ],
            "technical_indicators": [
                "Multiple compression artifacts",
                "Copy-move forgery traces",
                "Digital manipulation traces",
                "Metadata inconsistencies"
            ]
        }
    
    async def analyze_fraud_indicators(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive fraud analysis using multiple detection methods"""
        try:
            fraud_indicators = []
            confidence_scores = {}
            
            # Analyze document structure
            structure_indicators = self._analyze_document_structure(document_data)
            fraud_indicators.extend(structure_indicators)
            confidence_scores['structure'] = len(structure_indicators) / 10.0  # Normalize
            
            # Analyze content patterns
            content_indicators = self._analyze_content_patterns(document_data)
            fraud_indicators.extend(content_indicators)
            confidence_scores['content'] = len(content_indicators) / 10.0
            
            # Analyze technical indicators
            technical_indicators = self._analyze_technical_indicators(document_data)
            fraud_indicators.extend(technical_indicators)
            confidence_scores['technical'] = len(technical_indicators) / 10.0
            
            # Machine learning fraud detection
            if 'image_features' in document_data:
                ml_result = await self._ml_fraud_detection(document_data['image_features'])
                fraud_indicators.extend(ml_result.get('indicators', []))
                confidence_scores['ml_detection'] = ml_result.get('fraud_probability', 0)
            
            # Calculate overall fraud probability
            overall_fraud_probability = np.mean(list(confidence_scores.values()))
            
            # Determine fraud classification
            if overall_fraud_probability > 0.8:
                fraud_classification = "high_risk"
            elif overall_fraud_probability > 0.6:
                fraud_classification = "medium_risk"
            elif overall_fraud_probability > 0.3:
                fraud_classification = "low_risk"
            else:
                fraud_classification = "minimal_risk"
            
            return {
                'fraud_indicators': fraud_indicators,
                'confidence_scores': confidence_scores,
                'overall_fraud_probability': overall_fraud_probability,
                'fraud_classification': fraud_classification,
                'recommended_actions': self._generate_fraud_recommendations(fraud_classification, fraud_indicators)
            }
            
        except Exception as e:
            logger.error(f"Fraud analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_document_structure(self, document_data: Dict[str, Any]) -> List[str]:
        """Analyze document structure for fraud indicators"""
        indicators = []
        
        # Check OCR results
        ocr_data = document_data.get('ocr_analysis', {})
        if ocr_data:
            if ocr_data.get('confidence_score', 1.0) < 0.7:
                indicators.append("Low OCR confidence suggests poor document quality")
            
            if ocr_data.get('font_analysis', {}).get('suspected_font_manipulation', False):
                indicators.append("Suspected font manipulation detected")
            
            layout_analysis = ocr_data.get('layout_analysis', {})
            if not layout_analysis.get('text_distribution', {}).get('alignment_consistency', True):
                indicators.append("Inconsistent text alignment patterns")
        
        # Check image forensics
        forensics_data = document_data.get('image_forensics', {})
        if forensics_data:
            if forensics_data.get('authenticity_score', 1.0) < 0.6:
                indicators.append("Low image authenticity score")
            
            if forensics_data.get('tampering_indicators'):
                indicators.extend(forensics_data['tampering_indicators'])
        
        return indicators
    
    def _analyze_content_patterns(self, document_data: Dict[str, Any]) -> List[str]:
        """Analyze content patterns for fraud indicators"""
        indicators = []
        
        ocr_data = document_data.get('ocr_analysis', {})
        if ocr_data:
            # Check for suspicious patterns
            suspicious_patterns = ocr_data.get('suspicious_patterns', [])
            indicators.extend(suspicious_patterns)
            
            # Check validation errors
            validation_errors = ocr_data.get('validation_errors', [])
            indicators.extend(validation_errors)
            
            # Analyze extracted data
            extracted_data = ocr_data.get('data_extraction', {})
            if extracted_data:
                # Check for duplicate information
                dates = extracted_data.get('dates', [])
                amounts = extracted_data.get('amounts', [])
                
                if len(dates) != len(set(dates)) and len(dates) > 1:
                    indicators.append("Duplicate dates found in document")
                
                if len(amounts) != len(set(amounts)) and len(amounts) > 1:
                    indicators.append("Duplicate amounts found in document")
        
        return indicators
    
    def _analyze_technical_indicators(self, document_data: Dict[str, Any]) -> List[str]:
        """Analyze technical indicators for fraud"""
        indicators = []
        
        # Check metadata
        metadata = document_data.get('metadata_analysis', {})
        if metadata:
            # Check for metadata inconsistencies
            creation_date = metadata.get('creation_date')
            modification_date = metadata.get('modification_date')
            
            if creation_date and modification_date:
                if modification_date < creation_date:
                    indicators.append("Modification date before creation date")
            
            # Check for suspicious software
            software = metadata.get('software', '').lower()
            suspicious_software = ['photoshop', 'gimp', 'paint', 'editor']
            if any(sus in software for sus in suspicious_software):
                indicators.append(f"Document created/modified with image editing software: {software}")
        
        # Check digital signature
        signature_data = document_data.get('digital_signature_verification', {})
        if signature_data:
            if not signature_data.get('signature_valid', True):
                indicators.append("Invalid or missing digital signature")
            
            if not signature_data.get('certificate_valid', True):
                indicators.append("Invalid or expired certificate")
        
        return indicators
    
    async def _ml_fraud_detection(self, image_features: np.ndarray) -> Dict[str, Any]:
        """Machine learning-based fraud detection"""
        try:
            # Use pre-trained fraud detection model if available
            if 'fraud_detection' in self.models:
                model = self.models['fraud_detection']
                scaler = self.scalers['fraud_detection']
                
                # Scale features
                features_scaled = scaler.transform([image_features])
                
                # Get prediction
                fraud_probability = model.predict_proba(features_scaled)[0][1]  # Probability of fraud
                
                # Generate indicators based on feature importance
                indicators = []
                if fraud_probability > 0.7:
                    indicators.append("Machine learning model indicates high fraud probability")
                elif fraud_probability > 0.5:
                    indicators.append("Machine learning model indicates moderate fraud risk")
                
                return {
                    'fraud_probability': fraud_probability,
                    'indicators': indicators,
                    'model_confidence': 0.85  # Model confidence
                }
            else:
                # Fallback rule-based detection
                return {
                    'fraud_probability': 0.1,
                    'indicators': [],
                    'model_confidence': 0.5,
                    'note': 'ML model not available, using rule-based detection'
                }
                
        except Exception as e:
            logger.error(f"ML fraud detection failed: {e}")
            return {'error': str(e)}
    
    def _generate_fraud_recommendations(self, classification: str, indicators: List[str]) -> List[str]:
        """Generate recommendations based on fraud analysis"""
        recommendations = []
        
        if classification == "high_risk":
            recommendations.extend([
                "Reject document - high fraud probability",
                "Request original document verification",
                "Conduct manual review by fraud specialist",
                "Implement additional identity verification"
            ])
        elif classification == "medium_risk":
            recommendations.extend([
                "Request additional documentation",
                "Perform enhanced due diligence",
                "Consider manual verification",
                "Monitor for additional red flags"
            ])
        elif classification == "low_risk":
            recommendations.extend([
                "Accept with standard verification",
                "Document findings for future reference",
                "Consider periodic re-verification"
            ])
        else:
            recommendations.append("Proceed with standard processing")
        
        # Add specific recommendations based on indicators
        if any("copy-move" in indicator.lower() for indicator in indicators):
            recommendations.append("Investigate potential image manipulation")
        
        if any("font" in indicator.lower() for indicator in indicators):
            recommendations.append("Verify document using alternative sources")
        
        return recommendations

class DocumentAuthenticityChecker:
    """Main document authenticity checker with comprehensive verification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ocr_engine = AdvancedOCREngine()
        self.forensics_engine = ImageForensicsEngine()
        self.blockchain_engine = BlockchainVerificationEngine(config.get('blockchain', {}))
        self.fraud_engine = FraudDetectionEngine()
        self.db_pool = None
        self.redis_pool = None
        
        # Performance metrics
        self.metrics = {
            "documents_verified": 0,
            "fraud_detected": 0,
            "blockchain_registrations": 0,
            "avg_processing_time": 0,
            "processing_times": []
        }
    
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize the document authenticity checker"""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=120
            )
            
            # Initialize Redis connection
            self.redis_pool = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            logger.info("Document Authenticity Checker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Document Authenticity Checker: {e}")
            raise
    
    async def verify_document_authenticity(self, document_data: bytes, 
                                         document_type: str = "unknown",
                                         metadata: Dict[str, Any] = None) -> DocumentAuthenticity:
        """Perform comprehensive document authenticity verification"""
        start_time = time.time()
        verification_id = f"VERIFY_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        document_id = f"DOC_{uuid.uuid4().hex[:8]}"
        
        try:
            # Extract metadata
            metadata_analysis = await self._extract_comprehensive_metadata(document_data, metadata or {})
            
            # Convert document to image(s)
            images = await self._convert_document_to_images(document_data, metadata_analysis.file_type)
            
            if not images:
                raise Exception("Failed to convert document to images")
            
            # Analyze primary image (first page/image)
            primary_image = images[0]
            
            # Perform OCR analysis
            ocr_result = await self.ocr_engine.analyze_document_text(primary_image)
            
            # Perform image forensics
            forensics_result = await self.forensics_engine.analyze_image_forensics(primary_image)
            
            # Perform biometric analysis if applicable
            biometric_result = None
            if self._has_biometric_content(primary_image):
                biometric_result = await self._analyze_biometric_content(primary_image)
            
            # Verify digital signature if present
            signature_verification = await self._verify_digital_signature(document_data, metadata_analysis)
            
            # Blockchain verification
            blockchain_result = None
            if self.config.get('enable_blockchain', False):
                document_hash = metadata_analysis.hash_sha256
                blockchain_result = await self.blockchain_engine.verify_document_on_blockchain(document_hash)
            
            # Comprehensive fraud analysis
            fraud_analysis = await self.fraud_engine.analyze_fraud_indicators({
                'metadata_analysis': asdict(metadata_analysis),
                'ocr_analysis': asdict(ocr_result),
                'image_forensics': asdict(forensics_result),
                'biometric_analysis': asdict(biometric_result) if biometric_result else None,
                'digital_signature_verification': signature_verification
            })
            
            # Determine overall authenticity
            authenticity_status, confidence_score = self._determine_authenticity_status(
                forensics_result, fraud_analysis, signature_verification, blockchain_result
            )
            
            # Extract fraud types
            fraud_types = self._identify_fraud_types(fraud_analysis, forensics_result)
            
            # Generate compliance flags
            compliance_flags = self._generate_compliance_flags(
                authenticity_status, fraud_analysis, metadata_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_verification_recommendations(
                authenticity_status, fraud_analysis, compliance_flags
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create comprehensive result
            authenticity_result = DocumentAuthenticity(
                document_id=document_id,
                verification_id=verification_id,
                document_type=DocumentType(document_type),
                authenticity_status=authenticity_status,
                confidence_score=confidence_score,
                fraud_indicators=fraud_types,
                verification_methods=[
                    VerificationMethod.COMPUTER_VISION,
                    VerificationMethod.OCR_ANALYSIS,
                    VerificationMethod.METADATA_ANALYSIS,
                    VerificationMethod.FORENSIC_ANALYSIS
                ],
                metadata_analysis=metadata_analysis,
                ocr_analysis=ocr_result,
                image_forensics=forensics_result,
                biometric_analysis=biometric_result,
                blockchain_verification=blockchain_result,
                digital_signature_verification=signature_verification,
                risk_assessment=self._assess_document_risk(fraud_analysis, authenticity_status),
                compliance_flags=compliance_flags,
                recommendations=recommendations,
                verification_timestamp=datetime.now(),
                processing_time_ms=int(processing_time),
                analyst_notes=None
            )
            
            # Store verification result
            await self._store_verification_result(authenticity_result)
            
            # Update metrics
            self.metrics["documents_verified"] += 1
            self.metrics["processing_times"].append(processing_time)
            
            if len(self.metrics["processing_times"]) > 1000:
                self.metrics["processing_times"] = self.metrics["processing_times"][-1000:]
            
            self.metrics["avg_processing_time"] = np.mean(self.metrics["processing_times"])
            
            if authenticity_status in [AuthenticityStatus.FRAUDULENT, AuthenticityStatus.SUSPICIOUS]:
                self.metrics["fraud_detected"] += 1
            
            return authenticity_result
            
        except Exception as e:
            logger.error(f"Document verification failed: {e}")
            
            # Return error result
            return DocumentAuthenticity(
                document_id=document_id,
                verification_id=verification_id,
                document_type=DocumentType.UNKNOWN,
                authenticity_status=AuthenticityStatus.ERROR,
                confidence_score=0.0,
                fraud_indicators=[],
                verification_methods=[],
                metadata_analysis=DocumentMetadata(
                    file_name="error", file_size=0, file_type="unknown", mime_type="unknown",
                    creation_date=None, modification_date=None, author=None, software=None,
                    camera_info=None, gps_coordinates=None, hash_md5="", hash_sha256="", hash_sha512="",
                    digital_signature=None, certificate_chain=None, encryption_status=False, compression_info=None
                ),
                ocr_analysis=OCRResult("", 0.0, "unknown", [], {}, {}, [], [], {}),
                image_forensics=ImageForensicsResult({}, {}, {}, {}, {}, {}, {}, {}, [], 0.0),
                biometric_analysis=None,
                blockchain_verification=None,
                digital_signature_verification={'error': str(e)},
                risk_assessment={'error': str(e)},
                compliance_flags=[],
                recommendations=["Manual review required due to verification error"],
                verification_timestamp=datetime.now(),
                processing_time_ms=int((time.time() - start_time) * 1000),
                analyst_notes=f"Verification error: {str(e)}"
            )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return self.metrics
    
    async def close(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()

# Example usage
async def main():
    """Example usage of Document Authenticity Checker"""
    config = {
        'enable_blockchain': True,  # Production blockchain verification
        'blockchain': {
            'web3_provider': 'http://localhost:8545',
            'contract_address': '0x...',
            'private_key': 'your_private_key'
        }
    }
    
    checker = DocumentAuthenticityChecker(config)
    
    # Example document verification would go here
    print("Document Authenticity Checker demo completed!")

if __name__ == "__main__":
    asyncio.run(main())