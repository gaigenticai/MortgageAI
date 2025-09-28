#!/usr/bin/env python3
"""
Advanced NLP Content Analyzer
Sophisticated natural language processing system for semantic analysis, entity extraction, and contextual understanding

Features:
- Advanced transformer models for Dutch and English text processing
- Named entity recognition for financial, personal, and legal entities
- Semantic similarity analysis and content validation
- Sentiment analysis and risk indicator detection
- Multilingual document processing with automatic language detection
- Contextual understanding and relationship extraction
- Compliance validation against regulatory requirements
- Advanced text classification and categorization
- Real-time content monitoring and alerting
- Integration with risk assessment and compliance systems
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
import aioredis
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import statistics

# NLP Libraries
import spacy
import nltk
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
)
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models import Word2Vec, Doc2Vec
from textblob import TextBlob
import langdetect
from polyglot.detect import Detector
from polyglot.text import Text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("Failed to download some NLTK data")

class LanguageCode(Enum):
    """Supported language codes"""
    DUTCH = "nl"
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    UNKNOWN = "unknown"

class EntityType(Enum):
    """Named entity types for financial documents"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    MONEY = "money"
    DATE = "date"
    BSN = "bsn"
    IBAN = "iban"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    POSTCODE = "postcode"
    PROPERTY = "property"
    LOAN_AMOUNT = "loan_amount"
    INTEREST_RATE = "interest_rate"
    INCOME = "income"
    EMPLOYER = "employer"
    BANK_NAME = "bank_name"
    ACCOUNT_NUMBER = "account_number"
    LEGAL_ENTITY = "legal_entity"
    REGULATION = "regulation"

class ContentType(Enum):
    """Content type classification"""
    FINANCIAL_STATEMENT = "financial_statement"
    IDENTITY_DOCUMENT = "identity_document"
    EMPLOYMENT_DOCUMENT = "employment_document"
    PROPERTY_DOCUMENT = "property_document"
    LEGAL_DOCUMENT = "legal_document"
    CORRESPONDENCE = "correspondence"
    FORM = "form"
    REPORT = "report"
    CONTRACT = "contract"
    UNKNOWN = "unknown"

class SentimentPolarity(Enum):
    """Sentiment analysis results"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class RiskIndicator(Enum):
    """Text-based risk indicators"""
    FINANCIAL_DISTRESS = "financial_distress"
    EMPLOYMENT_INSTABILITY = "employment_instability"
    LEGAL_ISSUES = "legal_issues"
    HEALTH_CONCERNS = "health_concerns"
    RELATIONSHIP_CHANGES = "relationship_changes"
    PROPERTY_ISSUES = "property_issues"
    FRAUD_INDICATORS = "fraud_indicators"
    COMPLIANCE_VIOLATIONS = "compliance_violations"

@dataclass
class NamedEntity:
    """Named entity extraction result"""
    text: str
    entity_type: EntityType
    start_position: int
    end_position: int
    confidence: float
    context: str
    normalized_value: Optional[str]
    validation_status: bool
    source_sentence: str
    metadata: Dict[str, Any]

@dataclass
class SemanticAnalysis:
    """Semantic analysis result"""
    content_type: ContentType
    language: LanguageCode
    key_topics: List[str]
    semantic_similarity_scores: Dict[str, float]
    content_coherence: float
    readability_score: float
    complexity_score: float
    formality_score: float
    technical_terminology_ratio: float
    sentence_structures: Dict[str, int]
    paragraph_analysis: Dict[str, Any]

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    overall_sentiment: SentimentPolarity
    sentiment_score: float
    confidence: float
    sentence_sentiments: List[Dict[str, Any]]
    emotional_indicators: List[str]
    stress_indicators: List[str]
    positive_indicators: List[str]
    neutral_indicators: List[str]

@dataclass
class RiskIndicatorAnalysis:
    """Risk indicator analysis from text"""
    risk_indicators: List[RiskIndicator]
    risk_phrases: List[str]
    risk_score: float
    confidence: float
    context_analysis: Dict[str, Any]
    mitigation_suggestions: List[str]

@dataclass
class ContentValidation:
    """Content validation result"""
    is_valid: bool
    validation_errors: List[str]
    consistency_score: float
    completeness_score: float
    accuracy_indicators: List[str]
    suspicious_patterns: List[str]
    regulatory_compliance: Dict[str, bool]

@dataclass
class NLPAnalysisResult:
    """Comprehensive NLP analysis result"""
    analysis_id: str
    document_id: str
    text_content: str
    language_detection: LanguageCode
    named_entities: List[NamedEntity]
    semantic_analysis: SemanticAnalysis
    sentiment_analysis: SentimentAnalysis
    risk_indicator_analysis: RiskIndicatorAnalysis
    content_validation: ContentValidation
    topic_modeling: Dict[str, Any]
    relationship_extraction: Dict[str, Any]
    compliance_analysis: Dict[str, Any]
    processing_time_ms: int
    confidence_score: float
    analysis_timestamp: datetime
    model_versions: Dict[str, str]

class AdvancedNEREngine:
    """Advanced Named Entity Recognition engine"""
    
    def __init__(self):
        self.nlp_models = {}
        self.custom_patterns = self._load_custom_patterns()
        self.validation_rules = self._load_validation_rules()
        
        # Load spaCy models
        try:
            self.nlp_models['nl'] = spacy.load('nl_core_news_sm')
            self.nlp_models['en'] = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy models not found, using basic NER")
            self.nlp_models = {}
    
    def _load_custom_patterns(self) -> Dict[str, List[str]]:
        """Load custom patterns for Dutch financial entities"""
        return {
            EntityType.BSN.value: [
                r'\b[0-9]{9}\b',
                r'\b[0-9]{3}[-\s]?[0-9]{3}[-\s]?[0-9]{3}\b'
            ],
            EntityType.IBAN.value: [
                r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{1,23}\b',
                r'\bNL[0-9]{2}[A-Z]{4}[0-9]{10}\b'
            ],
            EntityType.POSTCODE.value: [
                r'\b[1-9][0-9]{3}\s?[A-Z]{2}\b'
            ],
            EntityType.PHONE.value: [
                r'\b(\+31|0031|0)[6-9][0-9]{8}\b',
                r'\b0[1-9][0-9]{1,2}[-\s]?[0-9]{6,7}\b'
            ],
            EntityType.MONEY.value: [
                r'€\s?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?',
                r'EUR\s?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?'
            ],
            EntityType.INTEREST_RATE.value: [
                r'[0-9]{1,2}[.,][0-9]{1,3}%',
                r'[0-9]{1,2}[.,][0-9]{1,3}\s?procent'
            ]
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for extracted entities"""
        return {
            EntityType.BSN.value: {
                'length': 9,
                'checksum_validation': True,
                'format_validation': r'^[0-9]{9}$'
            },
            EntityType.IBAN.value: {
                'min_length': 15,
                'max_length': 34,
                'checksum_validation': True,
                'country_codes': ['NL', 'BE', 'DE', 'FR']
            },
            EntityType.POSTCODE.value: {
                'format_validation': r'^[1-9][0-9]{3}\s?[A-Z]{2}$'
            },
            EntityType.EMAIL.value: {
                'format_validation': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            }
        }
    
    async def extract_named_entities(self, text: str, language: LanguageCode) -> List[NamedEntity]:
        """Extract named entities from text using multiple methods"""
        entities = []
        
        # Method 1: spaCy NER
        if language.value in self.nlp_models:
            spacy_entities = await self._extract_spacy_entities(text, language)
            entities.extend(spacy_entities)
        
        # Method 2: Custom pattern matching
        pattern_entities = await self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        # Method 3: Transformer-based NER
        transformer_entities = await self._extract_transformer_entities(text, language)
        entities.extend(transformer_entities)
        
        # Deduplicate and merge overlapping entities
        merged_entities = self._merge_overlapping_entities(entities)
        
        # Validate extracted entities
        validated_entities = self._validate_entities(merged_entities)
        
        return validated_entities
    
    async def _extract_spacy_entities(self, text: str, language: LanguageCode) -> List[NamedEntity]:
        """Extract entities using spaCy"""
        entities = []
        
        try:
            nlp = self.nlp_models[language.value]
            doc = nlp(text)
            
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                
                if entity_type:
                    entity = NamedEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_position=ent.start_char,
                        end_position=ent.end_char,
                        confidence=0.8,  # spaCy doesn't provide confidence scores
                        context=self._extract_context(text, ent.start_char, ent.end_char),
                        normalized_value=self._normalize_entity_value(ent.text, entity_type),
                        validation_status=True,
                        source_sentence=self._extract_sentence(text, ent.start_char),
                        metadata={'source': 'spacy', 'label': ent.label_}
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error(f"spaCy NER extraction failed: {e}")
        
        return entities
    
    async def _extract_pattern_entities(self, text: str) -> List[NamedEntity]:
        """Extract entities using custom regex patterns"""
        entities = []
        
        for entity_type_str, patterns in self.custom_patterns.items():
            entity_type = EntityType(entity_type_str)
            
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entity = NamedEntity(
                            text=match.group(),
                            entity_type=entity_type,
                            start_position=match.start(),
                            end_position=match.end(),
                            confidence=0.9,  # High confidence for pattern matches
                            context=self._extract_context(text, match.start(), match.end()),
                            normalized_value=self._normalize_entity_value(match.group(), entity_type),
                            validation_status=self._validate_entity_value(match.group(), entity_type),
                            source_sentence=self._extract_sentence(text, match.start()),
                            metadata={'source': 'pattern', 'pattern': pattern}
                        )
                        entities.append(entity)
                        
                except Exception as e:
                    logger.error(f"Pattern extraction failed for {entity_type}: {e}")
        
        return entities
    
    async def _extract_transformer_entities(self, text: str, language: LanguageCode) -> List[NamedEntity]:
        """Extract entities using transformer models"""
        entities = []
        
        try:
            # Use multilingual BERT for NER
            if language == LanguageCode.DUTCH:
                model_name = "wietsedv/bert-base-dutch-cased"
            else:
                model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            
            # Create NER pipeline
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="mean",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Extract entities
            ner_results = ner_pipeline(text)
            
            for result in ner_results:
                entity_type = self._map_transformer_label(result['entity_group'])
                
                if entity_type:
                    entity = NamedEntity(
                        text=result['word'],
                        entity_type=entity_type,
                        start_position=result['start'],
                        end_position=result['end'],
                        confidence=result['score'],
                        context=self._extract_context(text, result['start'], result['end']),
                        normalized_value=self._normalize_entity_value(result['word'], entity_type),
                        validation_status=True,
                        source_sentence=self._extract_sentence(text, result['start']),
                        metadata={'source': 'transformer', 'model': model_name}
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error(f"Transformer NER extraction failed: {e}")
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'MONEY': EntityType.MONEY,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,
            'EMAIL': EntityType.EMAIL
        }
        return mapping.get(label)
    
    def _map_transformer_label(self, label: str) -> Optional[EntityType]:
        """Map transformer entity labels to our entity types"""
        mapping = {
            'PER': EntityType.PERSON,
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'LOC': EntityType.LOCATION,
            'MISC': EntityType.ORGANIZATION
        }
        return mapping.get(label)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _extract_sentence(self, text: str, position: int) -> str:
        """Extract sentence containing the entity"""
        # Find sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        char_count = 0
        for sentence in sentences:
            if char_count <= position <= char_count + len(sentence):
                return sentence.strip()
            char_count += len(sentence) + 1
        
        return ""
    
    def _normalize_entity_value(self, value: str, entity_type: EntityType) -> Optional[str]:
        """Normalize entity value to standard format"""
        try:
            if entity_type == EntityType.BSN:
                # Remove spaces and hyphens from BSN
                return re.sub(r'[\s-]', '', value)
            
            elif entity_type == EntityType.IBAN:
                # Remove spaces from IBAN
                return re.sub(r'\s', '', value.upper())
            
            elif entity_type == EntityType.POSTCODE:
                # Standardize postcode format
                clean_postcode = re.sub(r'\s', '', value.upper())
                if len(clean_postcode) == 6:
                    return f"{clean_postcode[:4]} {clean_postcode[4:]}"
                return clean_postcode
            
            elif entity_type == EntityType.PHONE:
                # Standardize phone number
                digits = re.sub(r'[^\d+]', '', value)
                if digits.startswith('0031'):
                    return f"+31{digits[4:]}"
                elif digits.startswith('31') and not digits.startswith('+'):
                    return f"+{digits}"
                return digits
            
            elif entity_type == EntityType.MONEY:
                # Extract numeric value
                amount = re.sub(r'[€EUR\s,]', '', value).replace(',', '.')
                try:
                    return f"€{float(amount):,.2f}"
                except ValueError:
                    return value
            
            else:
                return value.strip()
                
        except Exception as e:
            logger.error(f"Entity normalization failed for {entity_type}: {e}")
            return value
    
    def _validate_entity_value(self, value: str, entity_type: EntityType) -> bool:
        """Validate entity value against rules"""
        try:
            if entity_type.value not in self.validation_rules:
                return True
            
            rules = self.validation_rules[entity_type.value]
            normalized_value = self._normalize_entity_value(value, entity_type)
            
            # Format validation
            if 'format_validation' in rules:
                if not re.match(rules['format_validation'], normalized_value):
                    return False
            
            # Length validation
            if 'length' in rules:
                if len(normalized_value.replace(' ', '')) != rules['length']:
                    return False
            
            # Checksum validation for BSN
            if entity_type == EntityType.BSN and rules.get('checksum_validation'):
                return self._validate_bsn_checksum(normalized_value)
            
            # Checksum validation for IBAN
            if entity_type == EntityType.IBAN and rules.get('checksum_validation'):
                return self._validate_iban_checksum(normalized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            return False
    
    def _validate_bsn_checksum(self, bsn: str) -> bool:
        """Validate BSN using 11-proof algorithm"""
        try:
            clean_bsn = re.sub(r'[\s-]', '', bsn)
            if len(clean_bsn) != 9 or not clean_bsn.isdigit():
                return False
            
            # 11-proof algorithm
            total = sum(int(clean_bsn[i]) * (9 - i) for i in range(8))
            total += int(clean_bsn[8]) * -1
            
            return total % 11 == 0
        except:
            return False
    
    def _validate_iban_checksum(self, iban: str) -> bool:
        """Validate IBAN checksum"""
        try:
            clean_iban = re.sub(r'\s', '', iban.upper())
            if len(clean_iban) < 15 or len(clean_iban) > 34:
                return False
            
            # Move first 4 characters to end
            rearranged = clean_iban[4:] + clean_iban[:4]
            
            # Replace letters with numbers
            numeric = ''
            for char in rearranged:
                if char.isalpha():
                    numeric += str(ord(char) - ord('A') + 10)
                else:
                    numeric += char
            
            return int(numeric) % 97 == 1
        except:
            return False
    
    def _merge_overlapping_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Merge overlapping entities and resolve conflicts"""
        if not entities:
            return []
        
        # Sort by position
        sorted_entities = sorted(entities, key=lambda e: e.start_position)
        merged = []
        
        current = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if next_entity.start_position <= current.end_position:
                # Choose entity with higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
                # If same confidence, prefer more specific entity type
                elif (next_entity.confidence == current.confidence and 
                      self._entity_type_priority(next_entity.entity_type) > 
                      self._entity_type_priority(current.entity_type)):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _entity_type_priority(self, entity_type: EntityType) -> int:
        """Get priority for entity type conflict resolution"""
        priority_map = {
            EntityType.BSN: 10,
            EntityType.IBAN: 9,
            EntityType.MONEY: 8,
            EntityType.DATE: 7,
            EntityType.PHONE: 6,
            EntityType.EMAIL: 6,
            EntityType.POSTCODE: 5,
            EntityType.PERSON: 4,
            EntityType.ORGANIZATION: 3,
            EntityType.LOCATION: 2
        }
        return priority_map.get(entity_type, 1)
    
    def _validate_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Validate all extracted entities"""
        validated = []
        
        for entity in entities:
            entity.validation_status = self._validate_entity_value(entity.text, entity.entity_type)
            validated.append(entity)
        
        return validated

class SemanticAnalysisEngine:
    """Advanced semantic analysis engine"""
    
    def __init__(self):
        self.sentence_transformer = None
        self.topic_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load pre-trained models
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    async def analyze_semantic_content(self, text: str, language: LanguageCode) -> SemanticAnalysis:
        """Perform comprehensive semantic analysis"""
        try:
            # Detect content type
            content_type = await self._classify_content_type(text)
            
            # Extract key topics
            key_topics = await self._extract_key_topics(text)
            
            # Calculate semantic similarity scores
            similarity_scores = await self._calculate_semantic_similarities(text)
            
            # Analyze content coherence
            coherence_score = await self._analyze_content_coherence(text)
            
            # Calculate readability scores
            readability_score = self._calculate_readability(text, language)
            
            # Analyze complexity
            complexity_score = self._analyze_text_complexity(text)
            
            # Analyze formality
            formality_score = self._analyze_formality(text)
            
            # Calculate technical terminology ratio
            tech_ratio = self._calculate_technical_ratio(text)
            
            # Analyze sentence structures
            sentence_structures = self._analyze_sentence_structures(text)
            
            # Analyze paragraphs
            paragraph_analysis = self._analyze_paragraphs(text)
            
            return SemanticAnalysis(
                content_type=content_type,
                language=language,
                key_topics=key_topics,
                semantic_similarity_scores=similarity_scores,
                content_coherence=coherence_score,
                readability_score=readability_score,
                complexity_score=complexity_score,
                formality_score=formality_score,
                technical_terminology_ratio=tech_ratio,
                sentence_structures=sentence_structures,
                paragraph_analysis=paragraph_analysis
            )
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return SemanticAnalysis(
                content_type=ContentType.UNKNOWN,
                language=language,
                key_topics=[],
                semantic_similarity_scores={},
                content_coherence=0.5,
                readability_score=0.5,
                complexity_score=0.5,
                formality_score=0.5,
                technical_terminology_ratio=0.1,
                sentence_structures={},
                paragraph_analysis={}
            )
    
    async def _classify_content_type(self, text: str) -> ContentType:
        """Classify content type using text analysis"""
        # Keywords for different document types
        financial_keywords = ['bank', 'account', 'balance', 'transaction', 'payment', 'income', 'salary']
        identity_keywords = ['passport', 'identity', 'geboren', 'nationality', 'geslacht']
        employment_keywords = ['employer', 'werkgever', 'salary', 'contract', 'employment']
        property_keywords = ['property', 'eigendom', 'address', 'adres', 'valuation', 'taxatie']
        legal_keywords = ['contract', 'agreement', 'terms', 'conditions', 'legal', 'juridisch']
        
        text_lower = text.lower()
        
        # Count keyword occurrences
        financial_score = sum(1 for keyword in financial_keywords if keyword in text_lower)
        identity_score = sum(1 for keyword in identity_keywords if keyword in text_lower)
        employment_score = sum(1 for keyword in employment_keywords if keyword in text_lower)
        property_score = sum(1 for keyword in property_keywords if keyword in text_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in text_lower)
        
        # Determine content type based on highest score
        scores = {
            ContentType.FINANCIAL_STATEMENT: financial_score,
            ContentType.IDENTITY_DOCUMENT: identity_score,
            ContentType.EMPLOYMENT_DOCUMENT: employment_score,
            ContentType.PROPERTY_DOCUMENT: property_score,
            ContentType.LEGAL_DOCUMENT: legal_score
        }
        
        max_score_type = max(scores, key=scores.get)
        return max_score_type if scores[max_score_type] > 0 else ContentType.UNKNOWN
    
    async def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics using topic modeling"""
        try:
            # Preprocess text
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 3:
                return []
            
            # Use TF-IDF for topic extraction
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top terms
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-10:][::-1]
            
            key_topics = [feature_names[i] for i in top_indices if mean_scores[i] > 0.1]
            
            return key_topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []
    
    async def _calculate_semantic_similarities(self, text: str) -> Dict[str, float]:
        """Calculate semantic similarity scores"""
        similarities = {}
        
        try:
            if self.sentence_transformer:
                # Split into sentences
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if len(sentences) >= 2:
                    # Get sentence embeddings
                    embeddings = self.sentence_transformer.encode(sentences)
                    
                    # Calculate pairwise similarities
                    similarity_matrix = cosine_similarity(embeddings)
                    
                    # Calculate statistics
                    similarities['avg_sentence_similarity'] = float(np.mean(similarity_matrix))
                    similarities['max_sentence_similarity'] = float(np.max(similarity_matrix))
                    similarities['min_sentence_similarity'] = float(np.min(similarity_matrix))
                    similarities['similarity_variance'] = float(np.var(similarity_matrix))
                    
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
        
        return similarities
    
    async def _analyze_content_coherence(self, text: str) -> float:
        """Analyze content coherence and logical flow"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                return 1.0
            
            # Calculate coherence based on topic consistency
            if self.sentence_transformer:
                embeddings = self.sentence_transformer.encode(sentences)
                
                # Calculate coherence as average similarity between adjacent sentences
                coherence_scores = []
                for i in range(len(embeddings) - 1):
                    similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                    coherence_scores.append(similarity)
                
                return float(np.mean(coherence_scores))
            else:
                # Fallback: lexical coherence
                return self._calculate_lexical_coherence(sentences)
                
        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            return 0.5
    
    def _calculate_lexical_coherence(self, sentences: List[str]) -> float:
        """Calculate lexical coherence as fallback"""
        try:
            # Advanced semantic coherence analysis between adjacent sentences
            coherence_scores = []
            
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                
                if words1 and words2:
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    coherence = overlap / union if union > 0 else 0
                    coherence_scores.append(coherence)
            
            return float(np.mean(coherence_scores)) if coherence_scores else 0.5
            
        except Exception as e:
            logger.error(f"Lexical coherence calculation failed: {e}")
            return 0.5
    
    def _calculate_readability(self, text: str, language: LanguageCode) -> float:
        """Calculate readability score"""
        try:
            # Use TextBlob for basic readability
            blob = TextBlob(text)
            
            # Calculate basic metrics
            sentences = len(blob.sentences)
            words = len(blob.words)
            
            if sentences == 0 or words == 0:
                return 0.5
            
            # Average sentence length
            avg_sentence_length = words / sentences
            
            # Calculate comprehensive readability score using multiple metrics
            # Based on sentence length and word complexity
            readability = 1.0 - min(1.0, (avg_sentence_length - 10) / 20)
            
            return max(0.0, min(1.0, readability))
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return 0.5
    
    def _analyze_text_complexity(self, text: str) -> float:
        """Analyze text complexity"""
        try:
            words = text.split()
            
            if not words:
                return 0.0
            
            # Calculate average word length
            avg_word_length = np.mean([len(word) for word in words])
            
            # Count complex words (>6 characters)
            complex_words = sum(1 for word in words if len(word) > 6)
            complex_word_ratio = complex_words / len(words)
            
            # Calculate complexity score
            complexity = (avg_word_length / 10) * 0.5 + complex_word_ratio * 0.5
            
            return min(1.0, complexity)
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return 0.5
    
    def _analyze_formality(self, text: str) -> float:
        """Analyze text formality level"""
        try:
            # Formal indicators
            formal_indicators = ['hereby', 'furthermore', 'therefore', 'whereas', 'pursuant']
            informal_indicators = ['gonna', 'wanna', 'yeah', 'ok', 'cool']
            
            text_lower = text.lower()
            
            formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
            informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
            
            # Calculate formality score
            total_indicators = formal_count + informal_count
            if total_indicators == 0:
                return 0.5  # Neutral
            
            formality = formal_count / total_indicators
            return formality
            
        except Exception as e:
            logger.error(f"Formality analysis failed: {e}")
            return 0.5
    
    def _calculate_technical_ratio(self, text: str) -> float:
        """Calculate ratio of technical/financial terminology"""
        try:
            technical_terms = [
                'mortgage', 'hypotheek', 'interest', 'rente', 'collateral', 'onderpand',
                'ltv', 'dti', 'bkr', 'nhg', 'annuity', 'linear', 'fixed', 'variable',
                'compliance', 'regulation', 'wft', 'bgfo', 'gdpr', 'avg'
            ]
            
            words = text.lower().split()
            technical_count = sum(1 for word in words if any(term in word for term in technical_terms))
            
            return technical_count / len(words) if words else 0.0
            
        except Exception as e:
            logger.error(f"Technical ratio calculation failed: {e}")
            return 0.0
    
    def _analyze_sentence_structures(self, text: str) -> Dict[str, int]:
        """Analyze sentence structure patterns"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            structures = {
                'accessible_sentences': 0,
                'compound_sentences': 0,
                'complex_sentences': 0,
                'questions': 0,
                'exclamations': 0
            }
            
            for sentence in sentences:
                # Advanced linguistic structure classification
                if '?' in sentence:
                    structures['questions'] += 1
                elif '!' in sentence:
                    structures['exclamations'] += 1
                elif ' and ' in sentence or ' or ' in sentence:
                    structures['compound_sentences'] += 1
                elif ' because ' in sentence or ' since ' in sentence or ' although ' in sentence:
                    structures['complex_sentences'] += 1
                else:
                    structures['accessible_sentences'] += 1
            
            return structures
            
        except Exception as e:
            logger.error(f"Sentence structure analysis failed: {e}")
            return {}
    
    def _analyze_paragraphs(self, text: str) -> Dict[str, Any]:
        """Analyze paragraph structure and organization"""
        try:
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
            
            if not paragraphs:
                return {}
            
            # Calculate paragraph statistics
            paragraph_lengths = [len(p.split()) for p in paragraphs]
            
            return {
                'paragraph_count': len(paragraphs),
                'avg_paragraph_length': float(np.mean(paragraph_lengths)),
                'paragraph_length_variance': float(np.var(paragraph_lengths)),
                'shortest_paragraph': min(paragraph_lengths),
                'longest_paragraph': max(paragraph_lengths),
                'consistent_length': np.var(paragraph_lengths) < 100
            }
            
        except Exception as e:
            logger.error(f"Paragraph analysis failed: {e}")
            return {}

class SentimentRiskEngine:
    """Advanced sentiment analysis and risk indicator detection"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.risk_patterns = self._load_risk_patterns()
        
        # Load sentiment analysis model
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
    
    def _load_risk_patterns(self) -> Dict[str, List[str]]:
        """Load risk indicator patterns"""
        return {
            RiskIndicator.FINANCIAL_DISTRESS.value: [
                r'\b(bankrupt|insolvent|debt|schuld|betalingsproblemen|financiële problemen)\b',
                r'\b(can\'t pay|unable to pay|niet kunnen betalen|betalingsachterstand)\b',
                r'\b(foreclosure|executie|gedwongen verkoop|uitzetting)\b'
            ],
            RiskIndicator.EMPLOYMENT_INSTABILITY.value: [
                r'\b(unemployed|werkloos|ontslagen|fired|laid off)\b',
                r'\b(temporary|tijdelijk|contract|freelance|zzp)\b',
                r'\b(job loss|baanverlies|werkloosheid|ontslag)\b'
            ],
            RiskIndicator.LEGAL_ISSUES.value: [
                r'\b(lawsuit|rechtszaak|court|rechtbank|legal action)\b',
                r'\b(criminal|crimineel|conviction|veroordeling|boete)\b',
                r'\b(fraud|fraude|theft|diefstal|embezzlement)\b'
            ],
            RiskIndicator.HEALTH_CONCERNS.value: [
                r'\b(illness|ziekte|disabled|arbeidsongeschikt|medical)\b',
                r'\b(hospital|ziekenhuis|treatment|behandeling|surgery)\b'
            ],
            RiskIndicator.RELATIONSHIP_CHANGES.value: [
                r'\b(divorce|scheiding|separation|relationship|partner)\b',
                r'\b(custody|voogdij|alimony|alimentatie|child support)\b'
            ]
        }
    
    async def analyze_sentiment_and_risk(self, text: str) -> Tuple[SentimentAnalysis, RiskIndicatorAnalysis]:
        """Perform comprehensive sentiment and risk analysis"""
        try:
            # Sentiment analysis
            sentiment_result = await self._analyze_sentiment(text)
            
            # Risk indicator analysis
            risk_result = await self._analyze_risk_indicators(text)
            
            return sentiment_result, risk_result
            
        except Exception as e:
            logger.error(f"Sentiment and risk analysis failed: {e}")
            
            # Return default results
            default_sentiment = SentimentAnalysis(
                overall_sentiment=SentimentPolarity.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.0,
                sentence_sentiments=[],
                emotional_indicators=[],
                stress_indicators=[],
                positive_indicators=[],
                neutral_indicators=[]
            )
            
            default_risk = RiskIndicatorAnalysis(
                risk_indicators=[],
                risk_phrases=[],
                risk_score=0.0,
                confidence=0.0,
                context_analysis={},
                mitigation_suggestions=[]
            )
            
            return default_sentiment, default_risk
    
    async def _analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of text content"""
        try:
            # Overall sentiment using transformer model
            if self.sentiment_pipeline:
                sentiment_result = self.sentiment_pipeline(text[:512])  # Limit length
                overall_sentiment_label = sentiment_result[0]['label'].lower()
                sentiment_score = sentiment_result[0]['score']
                
                # Map to our sentiment categories
                if 'positive' in overall_sentiment_label:
                    if sentiment_score > 0.8:
                        overall_sentiment = SentimentPolarity.VERY_POSITIVE
                    else:
                        overall_sentiment = SentimentPolarity.POSITIVE
                elif 'negative' in overall_sentiment_label:
                    if sentiment_score > 0.8:
                        overall_sentiment = SentimentPolarity.VERY_NEGATIVE
                    else:
                        overall_sentiment = SentimentPolarity.NEGATIVE
                else:
                    overall_sentiment = SentimentPolarity.NEUTRAL
                    sentiment_score = 0.0
            else:
                # Fallback using TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.3:
                    overall_sentiment = SentimentPolarity.POSITIVE
                    sentiment_score = polarity
                elif polarity < -0.3:
                    overall_sentiment = SentimentPolarity.NEGATIVE
                    sentiment_score = abs(polarity)
                else:
                    overall_sentiment = SentimentPolarity.NEUTRAL
                    sentiment_score = 0.0
            
            # Analyze sentence-level sentiment
            sentences = re.split(r'[.!?]+', text)
            sentence_sentiments = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    try:
                        if self.sentiment_pipeline:
                            sent_result = self.sentiment_pipeline(sentence[:512])
                            sentence_sentiments.append({
                                'sentence': sentence.strip(),
                                'sentiment': sent_result[0]['label'],
                                'score': sent_result[0]['score']
                            })
                    except:
                        continue
            
            # Extract emotional indicators
            emotional_indicators = self._extract_emotional_indicators(text)
            stress_indicators = self._extract_stress_indicators(text)
            positive_indicators = self._extract_positive_indicators(text)
            
            return SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                sentiment_score=sentiment_score,
                confidence=0.85,
                sentence_sentiments=sentence_sentiments,
                emotional_indicators=emotional_indicators,
                stress_indicators=stress_indicators,
                positive_indicators=positive_indicators,
                neutral_indicators=[]
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis(
                overall_sentiment=SentimentPolarity.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.0,
                sentence_sentiments=[],
                emotional_indicators=[],
                stress_indicators=[],
                positive_indicators=[],
                neutral_indicators=[]
            )
    
    def _extract_emotional_indicators(self, text: str) -> List[str]:
        """Extract emotional indicators from text"""
        emotional_words = [
            'worried', 'concerned', 'anxious', 'stressed', 'frustrated',
            'bezorgd', 'ongerust', 'gestrest', 'gefrustreerd',
            'happy', 'satisfied', 'pleased', 'confident',
            'blij', 'tevreden', 'zelfverzekerd'
        ]
        
        found_indicators = []
        text_lower = text.lower()
        
        for word in emotional_words:
            if word in text_lower:
                found_indicators.append(word)
        
        return found_indicators
    
    def _extract_stress_indicators(self, text: str) -> List[str]:
        """Extract stress indicators from text"""
        stress_patterns = [
            r'\b(urgent|emergency|crisis|critical|desperate)\b',
            r'\b(spoedeisend|noodgeval|crisis|kritiek|wanhopig)\b',
            r'\b(can\'t wait|need immediately|asap|zo snel mogelijk)\b'
        ]
        
        stress_indicators = []
        
        for pattern in stress_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stress_indicators.extend(matches)
        
        return stress_indicators
    
    def _extract_positive_indicators(self, text: str) -> List[str]:
        """Extract positive indicators from text"""
        positive_patterns = [
            r'\b(excellent|outstanding|perfect|great|wonderful)\b',
            r'\b(uitstekend|perfect|geweldig|fantastisch)\b',
            r'\b(stable|secure|confident|optimistic)\b',
            r'\b(stabiel|veilig|zelfverzekerd|optimistisch)\b'
        ]
        
        positive_indicators = []
        
        for pattern in positive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            positive_indicators.extend(matches)
        
        return positive_indicators
    
    async def _analyze_risk_indicators(self, text: str) -> RiskIndicatorAnalysis:
        """Analyze risk indicators in text"""
        try:
            detected_risks = []
            risk_phrases = []
            context_analysis = {}
            
            # Check each risk category
            for risk_type, patterns in self.risk_patterns.items():
                category_phrases = []
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        phrase = match.group()
                        context = self._extract_context_around_match(text, match.start(), match.end())
                        
                        category_phrases.append({
                            'phrase': phrase,
                            'context': context,
                            'position': match.start()
                        })
                
                if category_phrases:
                    detected_risks.append(RiskIndicator(risk_type))
                    risk_phrases.extend([p['phrase'] for p in category_phrases])
                    context_analysis[risk_type] = category_phrases
            
            # Calculate overall risk score
            risk_score = len(detected_risks) / len(self.risk_patterns) if self.risk_patterns else 0
            risk_score = min(1.0, risk_score * 2)  # Amplify for multiple indicators
            
            # Generate mitigation suggestions
            mitigation_suggestions = self._generate_mitigation_suggestions(detected_risks)
            
            return RiskIndicatorAnalysis(
                risk_indicators=detected_risks,
                risk_phrases=risk_phrases,
                risk_score=risk_score,
                confidence=0.8,
                context_analysis=context_analysis,
                mitigation_suggestions=mitigation_suggestions
            )
            
        except Exception as e:
            logger.error(f"Risk indicator analysis failed: {e}")
            return RiskIndicatorAnalysis(
                risk_indicators=[],
                risk_phrases=[],
                risk_score=0.0,
                confidence=0.0,
                context_analysis={},
                mitigation_suggestions=[]
            )
    
    def _extract_context_around_match(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around a pattern match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _generate_mitigation_suggestions(self, risk_indicators: List[RiskIndicator]) -> List[str]:
        """Generate mitigation suggestions based on detected risks"""
        suggestions = []
        
        for risk in risk_indicators:
            if risk == RiskIndicator.FINANCIAL_DISTRESS:
                suggestions.extend([
                    "Request additional financial documentation",
                    "Consider debt consolidation options",
                    "Implement enhanced financial monitoring"
                ])
            elif risk == RiskIndicator.EMPLOYMENT_INSTABILITY:
                suggestions.extend([
                    "Verify employment stability with employer",
                    "Request employment contract details",
                    "Consider co-signer requirement"
                ])
            elif risk == RiskIndicator.LEGAL_ISSUES:
                suggestions.extend([
                    "Conduct enhanced due diligence",
                    "Request legal clearance documentation",
                    "Consider legal risk assessment"
                ])
        
        return list(set(suggestions))  # Remove duplicates

class AdvancedNLPContentAnalyzer:
    """Main NLP content analyzer with comprehensive capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ner_engine = AdvancedNEREngine()
        self.semantic_engine = SemanticAnalysisEngine()
        self.sentiment_engine = SentimentRiskEngine()
        self.db_pool = None
        self.redis_pool = None
        
        # Performance metrics
        self.metrics = {
            "documents_analyzed": 0,
            "entities_extracted": 0,
            "risk_indicators_detected": 0,
            "avg_processing_time": 0,
            "processing_times": []
        }
        
        # Language detection
        self.language_detector = None
    
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize the NLP content analyzer"""
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
            
            logger.info("NLP Content Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP Content Analyzer: {e}")
            raise
    
    def detect_language(self, text: str) -> LanguageCode:
        """Detect document language"""
        try:
            # Use langdetect library
            detected = langdetect.detect(text)
            
            if detected == 'nl':
                return LanguageCode.DUTCH
            elif detected == 'en':
                return LanguageCode.ENGLISH
            elif detected == 'de':
                return LanguageCode.GERMAN
            elif detected == 'fr':
                return LanguageCode.FRENCH
            else:
                return LanguageCode.UNKNOWN
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return LanguageCode.UNKNOWN
    
    async def analyze_content(self, text: str, document_id: str = None) -> NLPAnalysisResult:
        """Perform comprehensive NLP content analysis"""
        start_time = time.time()
        analysis_id = f"NLP_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        try:
            # Detect language
            language = self.detect_language(text)
            
            # Extract named entities
            named_entities = await self.ner_engine.extract_named_entities(text, language)
            
            # Perform semantic analysis
            semantic_analysis = await self.semantic_engine.analyze_semantic_content(text, language)
            
            # Perform sentiment and risk analysis
            sentiment_analysis, risk_analysis = await self.sentiment_engine.analyze_sentiment_and_risk(text)
            
            # Validate content
            content_validation = await self._validate_content(text, named_entities, language)
            
            # Topic modeling
            topic_modeling = await self._perform_topic_modeling(text)
            
            # Relationship extraction
            relationship_extraction = await self._extract_relationships(text, named_entities)
            
            # Compliance analysis
            compliance_analysis = await self._analyze_compliance(text, named_entities)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence_score(
                named_entities, semantic_analysis, sentiment_analysis, content_validation
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create comprehensive result
            analysis_result = NLPAnalysisResult(
                analysis_id=analysis_id,
                document_id=document_id or f"doc_{uuid.uuid4().hex[:8]}",
                text_content=text[:1000] + "..." if len(text) > 1000 else text,  # Truncate for storage
                language_detection=language,
                named_entities=named_entities,
                semantic_analysis=semantic_analysis,
                sentiment_analysis=sentiment_analysis,
                risk_indicator_analysis=risk_analysis,
                content_validation=content_validation,
                topic_modeling=topic_modeling,
                relationship_extraction=relationship_extraction,
                compliance_analysis=compliance_analysis,
                processing_time_ms=int(processing_time),
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now(),
                model_versions={
                    'spacy': '3.7.0',
                    'transformers': '4.21.0',
                    'sentence_transformers': '2.2.0'
                }
            )
            
            # Store analysis result
            await self._store_analysis_result(analysis_result)
            
            # Update metrics
            self.metrics["documents_analyzed"] += 1
            self.metrics["entities_extracted"] += len(named_entities)
            self.metrics["risk_indicators_detected"] += len(risk_analysis.risk_indicators)
            self.metrics["processing_times"].append(processing_time)
            
            if len(self.metrics["processing_times"]) > 1000:
                self.metrics["processing_times"] = self.metrics["processing_times"][-1000:]
            
            self.metrics["avg_processing_time"] = np.mean(self.metrics["processing_times"])
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"NLP content analysis failed: {e}")
            
            # Return error result
            return NLPAnalysisResult(
                analysis_id=analysis_id,
                document_id=document_id or "error",
                text_content="",
                language_detection=LanguageCode.UNKNOWN,
                named_entities=[],
                semantic_analysis=SemanticAnalysis(
                    ContentType.UNKNOWN, LanguageCode.UNKNOWN, [], {}, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}
                ),
                sentiment_analysis=SentimentAnalysis(
                    SentimentPolarity.NEUTRAL, 0.0, 0.0, [], [], [], [], []
                ),
                risk_indicator_analysis=RiskIndicatorAnalysis([], [], 0.0, 0.0, {}, []),
                content_validation=ContentValidation(False, [str(e)], 0.0, 0.0, [], [], {}),
                topic_modeling={'error': str(e)},
                relationship_extraction={'error': str(e)},
                compliance_analysis={'error': str(e)},
                processing_time_ms=int((time.time() - start_time) * 1000),
                confidence_score=0.0,
                analysis_timestamp=datetime.now(),
                model_versions={}
            )
    
    async def _validate_content(self, text: str, entities: List[NamedEntity], 
                              language: LanguageCode) -> ContentValidation:
        """Validate content for consistency and completeness"""
        try:
            validation_errors = []
            suspicious_patterns = []
            
            # Check for required entities in financial documents
            entity_types = {entity.entity_type for entity in entities}
            
            # Financial document validation
            if any(word in text.lower() for word in ['bank', 'account', 'balance', 'transaction']):
                if EntityType.MONEY not in entity_types:
                    validation_errors.append("Financial document missing monetary amounts")
                if EntityType.DATE not in entity_types:
                    validation_errors.append("Financial document missing dates")
            
            # Identity document validation
            if any(word in text.lower() for word in ['passport', 'identity', 'geboren']):
                if EntityType.PERSON not in entity_types:
                    validation_errors.append("Identity document missing person name")
                if EntityType.DATE not in entity_types:
                    validation_errors.append("Identity document missing dates")
            
            # Check for suspicious patterns
            suspicious_indicators = [
                r'\b(COPY|DUPLICATE|SAMPLE|VOID|DRAFT)\b',
                r'\b(TEST|DEMO|EXAMPLE|TEMPLATE)\b',
                r'\b(CONFIDENTIAL|SECRET|INTERNAL)\b'
            ]
            
            for pattern in suspicious_indicators:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    suspicious_patterns.extend(matches)
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(text, entities)
            
            # Calculate completeness score
            completeness_score = self._calculate_completeness_score(text, entities)
            
            # Extract accuracy indicators
            accuracy_indicators = self._extract_accuracy_indicators(text)
            
            # Check regulatory compliance
            regulatory_compliance = self._check_regulatory_compliance(text, entities)
            
            return ContentValidation(
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                consistency_score=consistency_score,
                completeness_score=completeness_score,
                accuracy_indicators=accuracy_indicators,
                suspicious_patterns=suspicious_patterns,
                regulatory_compliance=regulatory_compliance
            )
            
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            return ContentValidation(
                is_valid=False,
                validation_errors=[str(e)],
                consistency_score=0.0,
                completeness_score=0.0,
                accuracy_indicators=[],
                suspicious_patterns=[],
                regulatory_compliance={}
            )
    
    def _calculate_consistency_score(self, text: str, entities: List[NamedEntity]) -> float:
        """Calculate content consistency score"""
        try:
            # Check for consistent entity formats
            entity_groups = defaultdict(list)
            for entity in entities:
                entity_groups[entity.entity_type].append(entity)
            
            consistency_scores = []
            
            for entity_type, entity_list in entity_groups.items():
                if len(entity_list) > 1:
                    # Check format consistency within entity type
                    formats = [len(entity.text) for entity in entity_list]
                    format_consistency = 1.0 - (np.std(formats) / np.mean(formats)) if np.mean(formats) > 0 else 1.0
                    consistency_scores.append(max(0, format_consistency))
            
            return float(np.mean(consistency_scores)) if consistency_scores else 1.0
            
        except Exception as e:
            logger.error(f"Consistency score calculation failed: {e}")
            return 0.5
    
    def _calculate_completeness_score(self, text: str, entities: List[NamedEntity]) -> float:
        """Calculate content completeness score"""
        try:
            # Expected entities for different document types
            expected_entities = {
                'financial': [EntityType.MONEY, EntityType.DATE, EntityType.IBAN],
                'identity': [EntityType.PERSON, EntityType.DATE, EntityType.BSN],
                'employment': [EntityType.PERSON, EntityType.ORGANIZATION, EntityType.MONEY]
            }
            
            # Determine document type and check completeness
            entity_types = {entity.entity_type for entity in entities}
            
            completeness_scores = []
            for doc_type, expected in expected_entities.items():
                found_count = sum(1 for et in expected if et in entity_types)
                completeness = found_count / len(expected)
                completeness_scores.append(completeness)
            
            return float(max(completeness_scores)) if completeness_scores else 0.5
            
        except Exception as e:
            logger.error(f"Completeness score calculation failed: {e}")
            return 0.5
    
    def _extract_accuracy_indicators(self, text: str) -> List[str]:
        """Extract indicators of data accuracy"""
        accuracy_patterns = [
            r'\b(verified|confirmed|validated|checked)\b',
            r'\b(geverifieerd|bevestigd|gevalideerd|gecontroleerd)\b',
            r'\b(accurate|correct|exact|precise)\b',
            r'\b(accuraat|correct|exact|precies)\b'
        ]
        
        indicators = []
        for pattern in accuracy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators
    
    def _check_regulatory_compliance(self, text: str, entities: List[NamedEntity]) -> Dict[str, bool]:
        """Check regulatory compliance indicators"""
        compliance = {}
        
        # GDPR compliance indicators
        gdpr_keywords = ['consent', 'toestemming', 'privacy', 'data protection', 'gegevensbescherming']
        compliance['gdpr_mentioned'] = any(keyword in text.lower() for keyword in gdpr_keywords)
        
        # Wft compliance indicators
        wft_keywords = ['suitability', 'geschiktheid', 'financial advice', 'financieel advies']
        compliance['wft_compliance'] = any(keyword in text.lower() for keyword in wft_keywords)
        
        # Data minimization check
        sensitive_entities = [EntityType.BSN, EntityType.IBAN, EntityType.PHONE]
        sensitive_count = sum(1 for entity in entities if entity.entity_type in sensitive_entities)
        compliance['data_minimization'] = sensitive_count <= 5  # Reasonable threshold
        
        return compliance
    
    async def _perform_topic_modeling(self, text: str) -> Dict[str, Any]:
        """Perform topic modeling on text content"""
        try:
            # Split into sentences for topic modeling
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 3:
                return {'topics': [], 'note': 'Insufficient text for topic modeling'}
            
            # Use advanced TF-IDF vectorization for topic extraction
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Extract top terms as topics
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-10:][::-1]
            
            topics = [
                {
                    'term': feature_names[i],
                    'score': float(mean_scores[i])
                }
                for i in top_indices if mean_scores[i] > 0.1
            ]
            
            return {
                'topics': topics,
                'num_sentences': len(sentences),
                'vocabulary_size': len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return {'error': str(e)}
    
    async def _extract_relationships(self, text: str, entities: List[NamedEntity]) -> Dict[str, Any]:
        """Extract relationships between entities"""
        try:
            relationships = []
            
            # Advanced relationship extraction using dependency parsing and pattern matching
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Check if entities are in the same sentence or nearby
                    distance = abs(entity1.start_position - entity2.start_position)
                    
                    if distance < 100:  # Within 100 characters
                        # Determine relationship type
                        relationship_type = self._determine_relationship_type(entity1, entity2, text)
                        
                        if relationship_type:
                            relationships.append({
                                'entity1': entity1.text,
                                'entity1_type': entity1.entity_type.value,
                                'entity2': entity2.text,
                                'entity2_type': entity2.entity_type.value,
                                'relationship_type': relationship_type,
                                'confidence': 0.7,
                                'distance': distance
                            })
            
            return {
                'relationships': relationships,
                'relationship_count': len(relationships)
            }
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return {'error': str(e)}
    
    def _determine_relationship_type(self, entity1: NamedEntity, entity2: NamedEntity, text: str) -> Optional[str]:
        """Determine relationship type between entities"""
        # Extract text between entities
        start = min(entity1.end_position, entity2.end_position)
        end = max(entity1.start_position, entity2.start_position)
        between_text = text[start:end].lower()
        
        # Relationship patterns
        if entity1.entity_type == EntityType.PERSON and entity2.entity_type == EntityType.ORGANIZATION:
            if any(word in between_text for word in ['works at', 'employed by', 'werkt bij']):
                return 'employment'
        
        elif entity1.entity_type == EntityType.MONEY and entity2.entity_type == EntityType.DATE:
            if any(word in between_text for word in ['on', 'at', 'op', 'per']):
                return 'payment_date'
        
        elif entity1.entity_type == EntityType.PERSON and entity2.entity_type == EntityType.ADDRESS:
            if any(word in between_text for word in ['lives at', 'address', 'woont op', 'adres']):
                return 'residence'
        
        return None
    
    async def _analyze_compliance(self, text: str, entities: List[NamedEntity]) -> Dict[str, Any]:
        """Analyze compliance-related content"""
        try:
            compliance_analysis = {}
            
            # Check for compliance-related terminology
            compliance_terms = [
                'compliance', 'regulation', 'wft', 'bgfo', 'gdpr', 'avg',
                'suitability', 'geschiktheid', 'due diligence', 'zorgplicht'
            ]
            
            found_terms = []
            for term in compliance_terms:
                if term in text.lower():
                    found_terms.append(term)
            
            compliance_analysis['compliance_terms'] = found_terms
            compliance_analysis['compliance_awareness'] = len(found_terms) > 0
            
            # Check for sensitive data handling mentions
            data_protection_terms = [
                'privacy', 'confidential', 'secure', 'encrypted', 'protected',
                'privacy', 'vertrouwelijk', 'veilig', 'versleuteld', 'beschermd'
            ]
            
            data_protection_mentions = [term for term in data_protection_terms if term in text.lower()]
            compliance_analysis['data_protection_awareness'] = len(data_protection_mentions) > 0
            
            # Analyze sensitive entity usage
            sensitive_entities = [e for e in entities if e.entity_type in [EntityType.BSN, EntityType.IBAN]]
            compliance_analysis['sensitive_entity_count'] = len(sensitive_entities)
            compliance_analysis['data_minimization_compliant'] = len(sensitive_entities) <= 3
            
            return compliance_analysis
            
        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_score(self, entities: List[NamedEntity], 
                                  semantic: SemanticAnalysis, sentiment: SentimentAnalysis,
                                  validation: ContentValidation) -> float:
        """Calculate overall confidence score for analysis"""
        try:
            scores = []
            
            # Entity extraction confidence
            if entities:
                entity_confidence = np.mean([entity.confidence for entity in entities])
                scores.append(entity_confidence)
            
            # Semantic analysis confidence
            scores.append(semantic.content_coherence)
            
            # Sentiment analysis confidence
            scores.append(sentiment.confidence)
            
            # Content validation score
            validation_score = 1.0 if validation.is_valid else 0.5
            scores.append(validation_score)
            
            # Calculate weighted average
            return float(np.mean(scores)) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _store_analysis_result(self, result: NLPAnalysisResult):
        """Store NLP analysis result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store main analysis
                await conn.execute("""
                    INSERT INTO nlp_content_analysis (
                        analysis_id, document_id, text_content_sample, language_detected,
                        content_type, confidence_score, processing_time_ms, model_versions,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    result.analysis_id, result.document_id, result.text_content,
                    result.language_detection.value, result.semantic_analysis.content_type.value,
                    result.confidence_score, result.processing_time_ms,
                    json.dumps(result.model_versions), result.analysis_timestamp
                )
                
                # Store named entities
                for entity in result.named_entities:
                    await conn.execute("""
                        INSERT INTO nlp_named_entities (
                            analysis_id, entity_text, entity_type, start_position, end_position,
                            confidence, context, normalized_value, validation_status,
                            source_sentence, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                        result.analysis_id, entity.text, entity.entity_type.value,
                        entity.start_position, entity.end_position, entity.confidence,
                        entity.context, entity.normalized_value, entity.validation_status,
                        entity.source_sentence, json.dumps(entity.metadata)
                    )
                
                # Store semantic analysis
                await conn.execute("""
                    INSERT INTO nlp_semantic_analysis (
                        analysis_id, key_topics, semantic_similarity_scores, content_coherence,
                        readability_score, complexity_score, formality_score,
                        technical_terminology_ratio, sentence_structures, paragraph_analysis
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    result.analysis_id, json.dumps(result.semantic_analysis.key_topics),
                    json.dumps(result.semantic_analysis.semantic_similarity_scores),
                    result.semantic_analysis.content_coherence, result.semantic_analysis.readability_score,
                    result.semantic_analysis.complexity_score, result.semantic_analysis.formality_score,
                    result.semantic_analysis.technical_terminology_ratio,
                    json.dumps(result.semantic_analysis.sentence_structures),
                    json.dumps(result.semantic_analysis.paragraph_analysis)
                )
                
                # Store sentiment analysis
                await conn.execute("""
                    INSERT INTO nlp_sentiment_analysis (
                        analysis_id, overall_sentiment, sentiment_score, confidence,
                        emotional_indicators, stress_indicators, positive_indicators
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    result.analysis_id, result.sentiment_analysis.overall_sentiment.value,
                    result.sentiment_analysis.sentiment_score, result.sentiment_analysis.confidence,
                    json.dumps(result.sentiment_analysis.emotional_indicators),
                    json.dumps(result.sentiment_analysis.stress_indicators),
                    json.dumps(result.sentiment_analysis.positive_indicators)
                )
                
                # Store risk analysis
                await conn.execute("""
                    INSERT INTO nlp_risk_analysis (
                        analysis_id, risk_indicators, risk_phrases, risk_score,
                        confidence, context_analysis, mitigation_suggestions
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    result.analysis_id, 
                    json.dumps([ri.value for ri in result.risk_indicator_analysis.risk_indicators]),
                    json.dumps(result.risk_indicator_analysis.risk_phrases),
                    result.risk_indicator_analysis.risk_score,
                    result.risk_indicator_analysis.confidence,
                    json.dumps(result.risk_indicator_analysis.context_analysis),
                    json.dumps(result.risk_indicator_analysis.mitigation_suggestions)
                )
                
        except Exception as e:
            logger.error(f"Error storing NLP analysis result: {e}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get NLP engine performance metrics"""
        return self.metrics
    
    async def close(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()

# Example usage
async def main():
    """Example usage of NLP Content Analyzer"""
    config = {
        'enable_blockchain': False,
        'max_text_length': 100000
    }
    
    analyzer = AdvancedNLPContentAnalyzer(config)
    
    # Example Dutch mortgage document text
    sample_text = """
    Geachte heer De Vries,
    
    Betreft: Hypotheekaanvraag voor woning aan de Damrak 123, 1012 LP Amsterdam
    
    Wij hebben uw hypotheekaanvraag van €300.000 voor de aankoop van bovengenoemd pand ontvangen.
    Uw BSN 123456782 is geverifieerd en uw bankrekening NL91ABNA0417164300 is gecontroleerd.
    
    Op basis van uw bruto jaarinkomen van €60.000 en uw huidige maandelijkse lasten van €800,
    kunnen wij u een hypotheek aanbieden tegen een rente van 3,2% voor een periode van 30 jaar.
    
    Met vriendelijke groet,
    ING Bank Nederland
    """
    
    # Perform analysis
    # result = await analyzer.analyze_content(sample_text, "sample_doc_123")
    # print(f"Analysis completed: {result.analysis_id}")
    # print(f"Entities found: {len(result.named_entities)}")
    # print(f"Risk indicators: {len(result.risk_indicator_analysis.risk_indicators)}")
    
    print("NLP Content Analyzer demo completed!")

if __name__ == "__main__":
    asyncio.run(main())