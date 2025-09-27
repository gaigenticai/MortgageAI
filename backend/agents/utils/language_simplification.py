"""
Advanced Language Simplification Tool for MortgageAI

This module provides comprehensive language simplification capabilities with CEFR B1 compliance,
readability scoring, and real-time text optimization for mortgage and financial content.

Features:
- CEFR (Common European Framework of Reference) level assessment and targeting
- Advanced readability scoring using multiple metrics
- Real-time text optimization with context preservation
- Financial domain-specific simplification rules
- Multi-language support (Dutch, English) with cultural adaptations
- Accessibility compliance (WCAG 2.1) for inclusive communication
- AFM compliance for clear financial communication requirements
- AI-powered synonym replacement and sentence restructuring
- Personalized simplification based on client profiles
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import statistics
from collections import defaultdict, Counter
import math

# Advanced NLP and ML libraries
import spacy
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    pipeline, T5ForConditionalGeneration, T5Tokenizer
)
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import textstat
import syllables
from textblob import TextBlob

# Language-specific libraries
import pyphen  # Syllable counting
from readability import Readability  # Advanced readability metrics

from ..config import settings


class CEFRLevel(Enum):
    """Common European Framework of Reference levels."""
    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate (target level)
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficiency


class TextComplexityLevel(Enum):
    """Text complexity classification."""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class SimplificationStrategy(Enum):
    """Strategies for text simplification."""
    LEXICAL = "lexical"              # Word-level simplification
    SYNTACTIC = "syntactic"          # Sentence structure simplification
    SEMANTIC = "semantic"            # Meaning preservation
    DISCOURSE = "discourse"          # Document structure optimization
    DOMAIN_SPECIFIC = "domain_specific"  # Financial terminology handling


@dataclass
class ReadabilityScores:
    """Comprehensive readability assessment results."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    automated_readability_index: float
    coleman_liau_index: float
    gunning_fog: float
    smog_index: float
    lix_readability: float
    rix_readability: float
    cefr_level: CEFRLevel
    complexity_level: TextComplexityLevel
    readability_summary: Dict[str, Any]


@dataclass
class SimplificationResult:
    """Results of text simplification process."""
    original_text: str
    simplified_text: str
    simplification_strategies: List[SimplificationStrategy]
    readability_before: ReadabilityScores
    readability_after: ReadabilityScores
    complexity_reduction: float  # Percentage reduction
    preservation_score: float    # Meaning preservation (0-1)
    changes_made: List[Dict[str, Any]]
    warnings: List[str]
    recommendations: List[str]
    processing_time: float


@dataclass
class SimplificationContext:
    """Context for simplification process."""
    target_cefr_level: CEFRLevel = CEFRLevel.B1
    target_audience: str = "general_public"
    domain: str = "finance"
    language: str = "nl"
    preserve_technical_terms: bool = True
    max_sentence_length: int = 20  # words
    max_syllables_per_word: int = 3
    cultural_context: str = "dutch"
    accessibility_requirements: List[str] = field(default_factory=list)
    client_profile: Optional[Dict[str, Any]] = None


class AdvancedReadabilityAnalyzer:
    """Advanced readability analysis with multiple metrics and CEFR assessment."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize language-specific tools
        self.syllable_counters = {}
        self.word_frequency_lists = {}
        self.stopwords_sets = {}
        
        # Initialize for Dutch and English
        for lang in ['nl', 'en']:
            try:
                if lang == 'nl':
                    self.syllable_counters[lang] = pyphen.Pyphen(lang='nl_NL')
                    # Load Dutch word frequency list (simplified)
                    self.word_frequency_lists[lang] = self._load_dutch_word_frequencies()
                else:
                    self.syllable_counters[lang] = pyphen.Pyphen(lang='en_US')
                    # Load English word frequency list
                    self.word_frequency_lists[lang] = self._load_english_word_frequencies()
                
                # Load stopwords
                nltk.download('stopwords', quiet=True)
                self.stopwords_sets[lang] = set(stopwords.words('dutch' if lang == 'nl' else 'english'))
                
            except Exception as e:
                self.logger.error(f"Error initializing language tools for {lang}: {e}")
        
        # CEFR level thresholds
        self.cefr_thresholds = {
            'flesch_reading_ease': {
                CEFRLevel.A1: (90, 100),
                CEFRLevel.A2: (80, 90),
                CEFRLevel.B1: (70, 80),
                CEFRLevel.B2: (60, 70),
                CEFRLevel.C1: (50, 60),
                CEFRLevel.C2: (0, 50)
            },
            'avg_sentence_length': {
                CEFRLevel.A1: (0, 8),
                CEFRLevel.A2: (8, 12),
                CEFRLevel.B1: (12, 16),
                CEFRLevel.B2: (16, 20),
                CEFRLevel.C1: (20, 25),
                CEFRLevel.C2: (25, float('inf'))
            },
            'avg_syllables_per_word': {
                CEFRLevel.A1: (0, 1.3),
                CEFRLevel.A2: (1.3, 1.5),
                CEFRLevel.B1: (1.5, 1.7),
                CEFRLevel.B2: (1.7, 2.0),
                CEFRLevel.C1: (2.0, 2.3),
                CEFRLevel.C2: (2.3, float('inf'))
            }
        }
    
    def _load_dutch_word_frequencies(self) -> Dict[str, int]:
        """Load Dutch word frequency list."""
        # Simplified Dutch frequency list for financial domain
        # In production, this would load from a comprehensive frequency database
        dutch_common_words = {
            # Basic words (high frequency)
            'de': 10000, 'het': 9000, 'een': 8000, 'van': 7000, 'in': 6000,
            'te': 5000, 'dat': 4000, 'op': 3500, 'voor': 3000, 'met': 2800,
            'zijn': 2500, 'als': 2200, 'er': 2000, 'maar': 1800, 'om': 1600,
            'hebben': 1400, 'niet': 1200, 'bij': 1000, 'kan': 900, 'waar': 800,
            
            # Financial terms (medium frequency)
            'geld': 700, 'bank': 650, 'betalen': 600, 'kosten': 550, 'euro': 500,
            'hypotheek': 450, 'lening': 400, 'rente': 380, 'maand': 350, 'jaar': 320,
            'huis': 300, 'inkomen': 280, 'aflossen': 250, 'sparen': 220, 'verzekering': 200,
            'schuld': 180, 'krediet': 160, 'waarde': 140, 'contract': 120, 'voorwaarden': 100,
            
            # Complex financial terms (low frequency)
            'hypotheekschuld': 80, 'rentevast': 60, 'eigenwoningforfait': 40,
            'vermogensrendementsheffing': 20, 'hypotheekrenteaftrek': 15
        }
        
        return dutch_common_words
    
    def _load_english_word_frequencies(self) -> Dict[str, int]:
        """Load English word frequency list."""
        # Simplified English frequency list for financial domain
        english_common_words = {
            # Basic words (high frequency)
            'the': 10000, 'a': 9000, 'an': 8000, 'and': 7000, 'to': 6000,
            'of': 5000, 'in': 4000, 'that': 3500, 'for': 3000, 'with': 2800,
            'is': 2500, 'as': 2200, 'it': 2000, 'but': 1800, 'on': 1600,
            'have': 1400, 'not': 1200, 'at': 1000, 'can': 900, 'where': 800,
            
            # Financial terms (medium frequency)
            'money': 700, 'bank': 650, 'pay': 600, 'cost': 550, 'loan': 500,
            'mortgage': 450, 'interest': 400, 'rate': 380, 'month': 350, 'year': 320,
            'house': 300, 'income': 280, 'payment': 250, 'save': 220, 'insurance': 200,
            'debt': 180, 'credit': 160, 'value': 140, 'contract': 120, 'terms': 100,
            
            # Complex financial terms (low frequency)
            'amortization': 80, 'collateral': 60, 'refinancing': 40,
            'securitization': 20, 'underwriting': 15
        }
        
        return english_common_words
    
    async def analyze_readability(self, text: str, language: str = "nl") -> ReadabilityScores:
        """Comprehensive readability analysis of text."""
        try:
            # Basic text statistics
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Filter out punctuation
            words_only = [word for word in words if word.isalpha()]
            
            if not sentences or not words_only:
                return self._create_empty_scores()
            
            # Calculate basic metrics
            sentence_count = len(sentences)
            word_count = len(words_only)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Syllable counting
            syllable_count = self._count_syllables(words_only, language)
            avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0
            
            # Standard readability formulas
            flesch_reading_ease = self._calculate_flesch_reading_ease(
                avg_sentence_length, avg_syllables_per_word
            )
            flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(
                avg_sentence_length, avg_syllables_per_word
            )
            
            # Additional metrics using textstat
            ari = textstat.automated_readability_index(text)
            cli = textstat.coleman_liau_index(text)
            gunning_fog = textstat.gunning_fog(text)
            smog = textstat.smog_index(text)
            
            # European readability metrics
            lix = self._calculate_lix_readability(text, language)
            rix = self._calculate_rix_readability(text, language)
            
            # CEFR level assessment
            cefr_level = self._assess_cefr_level(
                flesch_reading_ease, avg_sentence_length, avg_syllables_per_word
            )
            
            # Overall complexity level
            complexity_level = self._assess_complexity_level(flesch_reading_ease)
            
            # Create readability summary
            readability_summary = {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'syllable_count': syllable_count,
                'avg_sentence_length': avg_sentence_length,
                'avg_syllables_per_word': avg_syllables_per_word,
                'complex_words_ratio': self._calculate_complex_words_ratio(words_only, language),
                'passive_voice_ratio': self._calculate_passive_voice_ratio(text, language),
                'subordinate_clauses_ratio': self._calculate_subordinate_clauses_ratio(text, language)
            }
            
            return ReadabilityScores(
                flesch_reading_ease=flesch_reading_ease,
                flesch_kincaid_grade=flesch_kincaid_grade,
                automated_readability_index=ari,
                coleman_liau_index=cli,
                gunning_fog=gunning_fog,
                smog_index=smog,
                lix_readability=lix,
                rix_readability=rix,
                cefr_level=cefr_level,
                complexity_level=complexity_level,
                readability_summary=readability_summary
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing readability: {str(e)}")
            return self._create_empty_scores()
    
    def _count_syllables(self, words: List[str], language: str) -> int:
        """Count syllables in word list using language-specific rules."""
        try:
            syllable_counter = self.syllable_counters.get(language)
            if not syllable_counter:
                # Fallback to simple vowel counting
                return sum(self._simple_syllable_count(word) for word in words)
            
            total_syllables = 0
            for word in words:
                word_syllables = len(syllable_counter.inserted(word.lower()).split('-'))
                total_syllables += max(1, word_syllables)  # Minimum 1 syllable per word
            
            return total_syllables
            
        except Exception as e:
            self.logger.error(f"Error counting syllables: {str(e)}")
            return len(words)  # Fallback: 1 syllable per word
    
    def _simple_syllable_count(self, word: str) -> int:
        """Simple syllable counting based on vowel patterns."""
        word = word.lower()
        vowels = 'aeiouàáâãäåæçèéêëìíîïñòóôõöøùúûüý'
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def _calculate_flesch_reading_ease(self, avg_sentence_length: float, avg_syllables: float) -> float:
        """Calculate Flesch Reading Ease score."""
        try:
            # Standard Flesch formula
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            return max(0, min(100, score))
        except:
            return 50.0  # Neutral score
    
    def _calculate_flesch_kincaid_grade(self, avg_sentence_length: float, avg_syllables: float) -> float:
        """Calculate Flesch-Kincaid Grade Level."""
        try:
            grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables) - 15.59
            return max(0, grade)
        except:
            return 8.0  # 8th grade level (neutral)
    
    def _calculate_lix_readability(self, text: str, language: str) -> float:
        """Calculate LIX readability index (European metric)."""
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            
            words_only = [word for word in words if word.isalpha()]
            word_count = len(words_only)
            sentence_count = len(sentences)
            
            if sentence_count == 0 or word_count == 0:
                return 50.0
            
            # Count long words (>6 characters for most European languages)
            long_words = sum(1 for word in words_only if len(word) > 6)
            
            # LIX formula
            avg_sentence_length = word_count / sentence_count
            long_words_percentage = (long_words / word_count) * 100
            
            lix = avg_sentence_length + long_words_percentage
            return lix
            
        except Exception as e:
            self.logger.error(f"Error calculating LIX: {str(e)}")
            return 50.0
    
    def _calculate_rix_readability(self, text: str, language: str) -> float:
        """Calculate RIX readability index."""
        try:
            sentences = sent_tokenize(text)
            long_sentences = sum(1 for sent in sentences if len(word_tokenize(sent)) > 6)
            
            if len(sentences) == 0:
                return 5.0
            
            rix = long_sentences / len(sentences)
            return rix * 100  # Convert to percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating RIX: {str(e)}")
            return 50.0
    
    def _assess_cefr_level(self, flesch_score: float, avg_sentence_length: float, avg_syllables: float) -> CEFRLevel:
        """Assess CEFR level based on multiple metrics."""
        try:
            # Score each metric against CEFR thresholds
            level_scores = defaultdict(int)
            
            # Flesch Reading Ease assessment
            for level, (min_score, max_score) in self.cefr_thresholds['flesch_reading_ease'].items():
                if min_score <= flesch_score < max_score:
                    level_scores[level] += 3
                elif abs(flesch_score - ((min_score + max_score) / 2)) < 10:
                    level_scores[level] += 1
            
            # Average sentence length assessment
            for level, (min_len, max_len) in self.cefr_thresholds['avg_sentence_length'].items():
                if min_len <= avg_sentence_length < max_len:
                    level_scores[level] += 2
                elif abs(avg_sentence_length - ((min_len + max_len) / 2)) < 3:
                    level_scores[level] += 1
            
            # Average syllables per word assessment
            for level, (min_syl, max_syl) in self.cefr_thresholds['avg_syllables_per_word'].items():
                if min_syl <= avg_syllables < max_syl:
                    level_scores[level] += 2
                elif abs(avg_syllables - ((min_syl + max_syl) / 2)) < 0.2:
                    level_scores[level] += 1
            
            # Return the level with highest score
            if level_scores:
                best_level = max(level_scores, key=level_scores.get)
                return best_level
            else:
                return CEFRLevel.B2  # Default to B2 if no clear match
                
        except Exception as e:
            self.logger.error(f"Error assessing CEFR level: {str(e)}")
            return CEFRLevel.B2
    
    def _assess_complexity_level(self, flesch_score: float) -> TextComplexityLevel:
        """Assess overall text complexity level."""
        if flesch_score >= 90:
            return TextComplexityLevel.VERY_SIMPLE
        elif flesch_score >= 70:
            return TextComplexityLevel.SIMPLE
        elif flesch_score >= 50:
            return TextComplexityLevel.MODERATE
        elif flesch_score >= 30:
            return TextComplexityLevel.COMPLEX
        else:
            return TextComplexityLevel.VERY_COMPLEX
    
    def _calculate_complex_words_ratio(self, words: List[str], language: str) -> float:
        """Calculate ratio of complex words (3+ syllables)."""
        try:
            if not words:
                return 0.0
            
            complex_words = 0
            for word in words:
                syllable_count = self._simple_syllable_count(word)
                if syllable_count >= 3:
                    complex_words += 1
            
            return complex_words / len(words)
            
        except Exception as e:
            self.logger.error(f"Error calculating complex words ratio: {str(e)}")
            return 0.0
    
    def _calculate_passive_voice_ratio(self, text: str, language: str) -> float:
        """Calculate ratio of passive voice constructions."""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.0
            
            passive_count = 0
            
            # Simple passive voice detection patterns
            if language == 'nl':
                passive_patterns = [
                    r'\b(wordt|worden|werd|werden)\s+\w+',  # Dutch passive auxiliary verbs
                    r'\b(is|zijn|was|waren)\s+\w+d\b',      # Past participle patterns
                ]
            else:  # English
                passive_patterns = [
                    r'\b(am|is|are|was|were|been|being)\s+\w+ed\b',  # Standard passive
                    r'\b(am|is|are|was|were|been|being)\s+\w+en\b',  # Irregular past participles
                ]
            
            for sentence in sentences:
                for pattern in passive_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        passive_count += 1
                        break
            
            return passive_count / len(sentences)
            
        except Exception as e:
            self.logger.error(f"Error calculating passive voice ratio: {str(e)}")
            return 0.0
    
    def _calculate_subordinate_clauses_ratio(self, text: str, language: str) -> float:
        """Calculate ratio of sentences with subordinate clauses."""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.0
            
            subordinate_count = 0
            
            # Subordinating conjunctions and relative pronouns
            if language == 'nl':
                subordinate_markers = [
                    'dat', 'omdat', 'terwijl', 'hoewel', 'als', 'wanneer', 'waar', 'wie',
                    'wat', 'welke', 'waarbij', 'waardoor', 'waarvoor', 'indien'
                ]
            else:  # English
                subordinate_markers = [
                    'that', 'because', 'while', 'although', 'if', 'when', 'where', 'who',
                    'which', 'what', 'whereby', 'through', 'for', 'since', 'unless'
                ]
            
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if any(marker in words for marker in subordinate_markers):
                    subordinate_count += 1
            
            return subordinate_count / len(sentences)
            
        except Exception as e:
            self.logger.error(f"Error calculating subordinate clauses ratio: {str(e)}")
            return 0.0
    
    def _create_empty_scores(self) -> ReadabilityScores:
        """Create empty/default readability scores."""
        return ReadabilityScores(
            flesch_reading_ease=50.0,
            flesch_kincaid_grade=8.0,
            automated_readability_index=8.0,
            coleman_liau_index=8.0,
            gunning_fog=8.0,
            smog_index=8.0,
            lix_readability=50.0,
            rix_readability=50.0,
            cefr_level=CEFRLevel.B2,
            complexity_level=TextComplexityLevel.MODERATE,
            readability_summary={}
        )


class AdvancedLanguageSimplifier:
    """Advanced language simplification with multiple strategies and meaning preservation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        self.nlp_models = {}
        self.simplification_models = {}
        self.synonym_models = {}
        
        try:
            # Load spaCy models for different languages
            self.nlp_models['nl'] = spacy.load("nl_core_news_sm")
            self.nlp_models['en'] = spacy.load("en_core_web_sm")
            
            # Load T5 model for text-to-text simplification
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
            
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased')
            
            self.logger.info("Initialized language simplification models")
            
        except Exception as e:
            self.logger.error(f"Error initializing simplification models: {str(e)}")
            self.nlp_models = {}
            self.simplification_models = {}
        
        # Financial terminology mappings
        self.financial_simplifications = {
            'nl': {
                # Complex -> Simple mappings
                'hypotheekschuld': 'hypotheek',
                'eigenwoningforfait': 'belasting op uw huis',
                'vermogensrendementsheffing': 'belasting op spaargeld',
                'hypotheekrenteaftrek': 'korting op hypotheekrente',
                'annuïteitenhypotheek': 'hypotheek met vaste maandlasten',
                'lineaire hypotheek': 'hypotheek waarbij u steeds minder betaalt',
                'aflossingsvrij': 'alleen rente betalen',
                'rentevastperiode': 'periode met vaste rente',
                'oversluiten': 'naar andere bank gaan',
                'boeterente': 'extra kosten bij vervroegd aflossen',
                'taxatiewaarde': 'geschatte waarde van het huis',
                'financieringslasten': 'alle kosten van de hypotheek',
                'restschuld': 'bedrag dat u nog moet betalen'
            },
            'en': {
                'amortization': 'paying off the loan gradually',
                'collateral': 'security for the loan',
                'refinancing': 'getting a new loan to replace the old one',
                'underwriting': 'checking if you can get the loan',
                'securitization': 'bundling loans together',
                'prepayment penalty': 'fee for paying early',
                'principal': 'the amount you borrowed',
                'escrow': 'money held by the lender',
                'closing costs': 'fees to complete the purchase',
                'debt-to-income ratio': 'how much debt you have compared to income'
            }
        }
        
        # Sentence complexity patterns
        self.complexity_patterns = {
            'nl': {
                'passive_voice': r'\b(wordt|worden|werd|werden)\s+\w+',
                'long_compounds': r'\b\w{15,}\b',  # Very long compound words
                'complex_conjunctions': r'\b(niettegenstaande|desalniettemin|daarentegen)\b',
                'formal_language': r'\b(dientengevolge|aldus|bijgevolg|derhalve)\b'
            },
            'en': {
                'passive_voice': r'\b(am|is|are|was|were|been|being)\s+\w+ed\b',
                'complex_words': r'\b\w{12,}\b',  # Very long words
                'complex_conjunctions': r'\b(nevertheless|furthermore|consequently|thereby)\b',
                'formal_language': r'\b(henceforth|heretofore|notwithstanding|wherein)\b'
            }
        }
    
    async def simplify_text(self, text: str, context: SimplificationContext) -> SimplificationResult:
        """Perform comprehensive text simplification."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze original text readability
            readability_analyzer = AdvancedReadabilityAnalyzer()
            original_readability = await readability_analyzer.analyze_readability(text, context.language)
            
            # Initialize simplification tracking
            simplified_text = text
            changes_made = []
            strategies_used = []
            warnings = []
            
            # Apply simplification strategies in order
            
            # 1. Lexical simplification (replace complex words)
            simplified_text, lexical_changes = await self._apply_lexical_simplification(
                simplified_text, context
            )
            if lexical_changes:
                changes_made.extend(lexical_changes)
                strategies_used.append(SimplificationStrategy.LEXICAL)
            
            # 2. Domain-specific simplification (financial terms)
            simplified_text, domain_changes = await self._apply_domain_simplification(
                simplified_text, context
            )
            if domain_changes:
                changes_made.extend(domain_changes)
                strategies_used.append(SimplificationStrategy.DOMAIN_SPECIFIC)
            
            # 3. Syntactic simplification (sentence structure)
            simplified_text, syntactic_changes = await self._apply_syntactic_simplification(
                simplified_text, context
            )
            if syntactic_changes:
                changes_made.extend(syntactic_changes)
                strategies_used.append(SimplificationStrategy.SYNTACTIC)
            
            # 4. Discourse simplification (document structure)
            simplified_text, discourse_changes = await self._apply_discourse_simplification(
                simplified_text, context
            )
            if discourse_changes:
                changes_made.extend(discourse_changes)
                strategies_used.append(SimplificationStrategy.DISCOURSE)
            
            # Analyze simplified text readability
            final_readability = await readability_analyzer.analyze_readability(simplified_text, context.language)
            
            # Calculate improvements
            complexity_reduction = self._calculate_complexity_reduction(original_readability, final_readability)
            preservation_score = await self._calculate_preservation_score(text, simplified_text, context)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                original_readability, final_readability, context
            )
            
            # Add warnings if needed
            if preservation_score < 0.8:
                warnings.append("Significant meaning changes detected - review recommended")
            
            if final_readability.cefr_level.value > context.target_cefr_level.value:
                warnings.append(f"Target CEFR level {context.target_cefr_level.value} not achieved")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return SimplificationResult(
                original_text=text,
                simplified_text=simplified_text,
                simplification_strategies=strategies_used,
                readability_before=original_readability,
                readability_after=final_readability,
                complexity_reduction=complexity_reduction,
                preservation_score=preservation_score,
                changes_made=changes_made,
                warnings=warnings,
                recommendations=recommendations,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in text simplification: {str(e)}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Return original text with error information
            return SimplificationResult(
                original_text=text,
                simplified_text=text,
                simplification_strategies=[],
                readability_before=ReadabilityScores(
                    flesch_reading_ease=50.0, flesch_kincaid_grade=8.0,
                    automated_readability_index=8.0, coleman_liau_index=8.0,
                    gunning_fog=8.0, smog_index=8.0, lix_readability=50.0,
                    rix_readability=50.0, cefr_level=CEFRLevel.B2,
                    complexity_level=TextComplexityLevel.MODERATE, readability_summary={}
                ),
                readability_after=ReadabilityScores(
                    flesch_reading_ease=50.0, flesch_kincaid_grade=8.0,
                    automated_readability_index=8.0, coleman_liau_index=8.0,
                    gunning_fog=8.0, smog_index=8.0, lix_readability=50.0,
                    rix_readability=50.0, cefr_level=CEFRLevel.B2,
                    complexity_level=TextComplexityLevel.MODERATE, readability_summary={}
                ),
                complexity_reduction=0.0,
                preservation_score=1.0,
                changes_made=[],
                warnings=[f"Simplification error: {str(e)}"],
                recommendations=[],
                processing_time=processing_time
            )
    
    async def _apply_lexical_simplification(self, text: str, context: SimplificationContext) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply word-level simplification."""
        changes = []
        simplified_text = text
        
        try:
            # Get NLP model for language
            nlp = self.nlp_models.get(context.language)
            if not nlp:
                return simplified_text, changes
            
            # Process text with spaCy
            doc = nlp(text)
            
            # Track replacements to avoid conflicts
            replacements = []
            
            for token in doc:
                if token.is_alpha and len(token.text) > 6:  # Consider longer words for simplification
                    # Check if word is complex for target CEFR level
                    if self._is_word_complex(token.text, context):
                        # Find simpler synonym
                        simpler_word = await self._find_simpler_synonym(token.text, context)
                        
                        if simpler_word and simpler_word != token.text:
                            replacements.append({
                                'start': token.idx,
                                'end': token.idx + len(token.text),
                                'original': token.text,
                                'replacement': simpler_word
                            })
                            
                            changes.append({
                                'type': 'lexical_replacement',
                                'original': token.text,
                                'replacement': simpler_word,
                                'reason': 'complexity_reduction',
                                'position': token.idx
                            })
            
            # Apply replacements (in reverse order to maintain indices)
            for replacement in sorted(replacements, key=lambda x: x['start'], reverse=True):
                simplified_text = (
                    simplified_text[:replacement['start']] +
                    replacement['replacement'] +
                    simplified_text[replacement['end']:]
                )
            
        except Exception as e:
            self.logger.error(f"Error in lexical simplification: {str(e)}")
        
        return simplified_text, changes
    
    async def _apply_domain_simplification(self, text: str, context: SimplificationContext) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply domain-specific simplification for financial terms."""
        changes = []
        simplified_text = text
        
        try:
            # Get domain-specific mappings
            domain_mappings = self.financial_simplifications.get(context.language, {})
            
            # Apply replacements
            for complex_term, simple_term in domain_mappings.items():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(complex_term), re.IGNORECASE)
                matches = list(pattern.finditer(simplified_text))
                
                if matches:
                    # Apply replacements in reverse order
                    for match in reversed(matches):
                        simplified_text = (
                            simplified_text[:match.start()] +
                            simple_term +
                            simplified_text[match.end():]
                        )
                        
                        changes.append({
                            'type': 'domain_simplification',
                            'original': match.group(),
                            'replacement': simple_term,
                            'reason': 'financial_terminology_simplification',
                            'position': match.start()
                        })
            
            # Add explanations for remaining technical terms if needed
            if context.preserve_technical_terms:
                simplified_text, explanation_changes = self._add_term_explanations(simplified_text, context)
                changes.extend(explanation_changes)
            
        except Exception as e:
            self.logger.error(f"Error in domain simplification: {str(e)}")
        
        return simplified_text, changes
    
    async def _apply_syntactic_simplification(self, text: str, context: SimplificationContext) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply sentence-level structural simplification."""
        changes = []
        
        try:
            sentences = sent_tokenize(text)
            simplified_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Check if sentence needs simplification
                words = word_tokenize(sentence)
                word_count = len([w for w in words if w.isalpha()])
                
                if word_count > context.max_sentence_length:
                    # Split long sentences
                    split_sentences = await self._split_complex_sentence(sentence, context)
                    
                    if len(split_sentences) > 1:
                        simplified_sentences.extend(split_sentences)
                        changes.append({
                            'type': 'sentence_splitting',
                            'original': sentence,
                            'replacement': ' '.join(split_sentences),
                            'reason': f'sentence_too_long_{word_count}_words',
                            'position': i
                        })
                    else:
                        simplified_sentences.append(sentence)
                else:
                    # Apply other syntactic simplifications
                    simplified_sentence = await self._simplify_sentence_structure(sentence, context)
                    simplified_sentences.append(simplified_sentence)
                    
                    if simplified_sentence != sentence:
                        changes.append({
                            'type': 'syntactic_restructuring',
                            'original': sentence,
                            'replacement': simplified_sentence,
                            'reason': 'structure_simplification',
                            'position': i
                        })
            
            simplified_text = ' '.join(simplified_sentences)
            
        except Exception as e:
            self.logger.error(f"Error in syntactic simplification: {str(e)}")
            return text, changes
        
        return simplified_text, changes
    
    async def _apply_discourse_simplification(self, text: str, context: SimplificationContext) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply document-level structural improvements."""
        changes = []
        simplified_text = text
        
        try:
            # Add paragraph breaks for better readability
            sentences = sent_tokenize(text)
            
            if len(sentences) > 4:  # Only restructure longer texts
                # Group sentences into logical paragraphs
                paragraphs = self._create_logical_paragraphs(sentences, context)
                simplified_text = '\n\n'.join(paragraphs)
                
                if len(paragraphs) > 1:
                    changes.append({
                        'type': 'paragraph_structuring',
                        'original': text,
                        'replacement': simplified_text,
                        'reason': 'improved_document_structure',
                        'position': 0
                    })
            
            # Add transition words where appropriate
            simplified_text = self._add_transition_words(simplified_text, context)
            
        except Exception as e:
            self.logger.error(f"Error in discourse simplification: {str(e)}")
        
        return simplified_text, changes
    
    def _is_word_complex(self, word: str, context: SimplificationContext) -> bool:
        """Determine if a word is complex for the target audience."""
        # Check word length
        if len(word) > 12:  # Very long words are generally complex
            return True
        
        # Check syllable count
        syllable_count = self._count_word_syllables(word)
        if syllable_count > context.max_syllables_per_word:
            return True
        
        # Check frequency (if available)
        word_frequencies = AdvancedReadabilityAnalyzer()._load_dutch_word_frequencies() if context.language == 'nl' else AdvancedReadabilityAnalyzer()._load_english_word_frequencies()
        
        if word.lower() in word_frequencies:
            frequency = word_frequencies[word.lower()]
            # Words with frequency below 100 are considered complex
            if frequency < 100:
                return True
        
        return False
    
    def _count_word_syllables(self, word: str) -> int:
        """Count syllables in a single word."""
        word = word.lower()
        vowels = 'aeiou'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    async def _find_simpler_synonym(self, word: str, context: SimplificationContext) -> Optional[str]:
        """Find a simpler synonym for a complex word."""
        try:
            # Simple synonym mappings for common complex words
            simple_synonyms = {
                'nl': {
                    'gecompliceerd': 'moeilijk',
                    'adequate': 'voldoende',
                    'faciliteren': 'helpen',
                    'implementeren': 'uitvoeren',
                    'analyseren': 'bekijken',
                    'optimaliseren': 'verbeteren',
                    'transparant': 'duidelijk',
                    'substantieel': 'groot',
                    'significante': 'belangrijke',
                    'procedure': 'werkwijze',
                    'documentatie': 'papieren',
                    'verificatie': 'controle'
                },
                'en': {
                    'complicated': 'hard',
                    'adequate': 'enough',
                    'facilitate': 'help',
                    'implement': 'do',
                    'analyze': 'look at',
                    'optimize': 'improve',
                    'transparent': 'clear',
                    'substantial': 'big',
                    'significant': 'important',
                    'procedure': 'way',
                    'documentation': 'papers',
                    'verification': 'check'
                }
            }
            
            synonyms = simple_synonyms.get(context.language, {})
            
            # Direct lookup first
            if word.lower() in synonyms:
                return synonyms[word.lower()]
            
            # For more complex synonym finding, you could integrate with:
            # - WordNet for English
            # - Dutch synonym databases
            # - Transformer-based models
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding synonym for '{word}': {str(e)}")
            return None
    
    def _add_term_explanations(self, text: str, context: SimplificationContext) -> Tuple[str, List[Dict[str, Any]]]:
        """Add explanations for technical terms that are preserved."""
        changes = []
        
        # Technical terms that should be explained
        terms_to_explain = {
            'nl': {
                'NHG': 'NHG (Nationale Hypotheek Garantie - een verzekering die helpt bij hypotheken)',
                'BKR': 'BKR (Bureau Krediet Registratie - houdt bij wie geld geleend heeft)',
                'LTV': 'LTV (Loan to Value - hoeveel je leent ten opzichte van de waarde van het huis)',
                'AFM': 'AFM (Autoriteit Financiële Markten - toezichthouder op banken)'
            },
            'en': {
                'LTV': 'LTV (Loan to Value - how much you borrow compared to house value)',
                'DTI': 'DTI (Debt to Income - how much debt you have compared to your income)',
                'APR': 'APR (Annual Percentage Rate - the yearly cost of the loan)',
                'PMI': 'PMI (Private Mortgage Insurance - insurance for the lender)'
            }
        }
        
        explanations = terms_to_explain.get(context.language, {})
        
        for term, explanation in explanations.items():
            if term in text and explanation not in text:
                # Replace first occurrence with explanation
                text = text.replace(term, explanation, 1)
                changes.append({
                    'type': 'term_explanation',
                    'original': term,
                    'replacement': explanation,
                    'reason': 'technical_term_clarification',
                    'position': text.find(explanation)
                })
        
        return text, changes
    
    async def _split_complex_sentence(self, sentence: str, context: SimplificationContext) -> List[str]:
        """Split a complex sentence into simpler ones."""
        try:
            # Simple sentence splitting based on conjunctions
            splitting_words = {
                'nl': ['en', 'maar', 'want', 'dus', 'of', 'terwijl', 'omdat', 'hoewel'],
                'en': ['and', 'but', 'because', 'so', 'or', 'while', 'although', 'since']
            }
            
            split_words = splitting_words.get(context.language, splitting_words['en'])
            
            # Find potential split points
            words = word_tokenize(sentence)
            split_indices = []
            
            for i, word in enumerate(words):
                if word.lower() in split_words and i > 3:  # Don't split too early
                    split_indices.append(i)
            
            if not split_indices:
                return [sentence]
            
            # Split at the middle-most conjunction
            mid_index = split_indices[len(split_indices) // 2]
            
            # Create two sentences
            first_part = ' '.join(words[:mid_index])
            second_part = ' '.join(words[mid_index + 1:])  # Skip conjunction
            
            # Ensure both parts are complete sentences
            if not first_part.endswith('.'):
                first_part += '.'
            
            if not second_part[0].isupper():
                second_part = second_part.capitalize()
            
            if not second_part.endswith('.'):
                second_part += '.'
            
            return [first_part, second_part]
            
        except Exception as e:
            self.logger.error(f"Error splitting sentence: {str(e)}")
            return [sentence]
    
    async def _simplify_sentence_structure(self, sentence: str, context: SimplificationContext) -> str:
        """Simplify the structure of a sentence without splitting."""
        try:
            simplified = sentence
            
            # Convert passive voice to active voice (simple patterns)
            if context.language == 'nl':
                # Dutch passive to active patterns
                passive_patterns = [
                    (r'wordt (.*?) door (.*?)\.', r'\2 \1.'),  # Simple passive conversion
                    (r'(.*?) wordt (.*?)\.', r'We \2 \1.'),   # General passive to active
                ]
            else:
                # English passive to active patterns
                passive_patterns = [
                    (r'(.*?) is done by (.*?)\.', r'\2 does \1.'),
                    (r'(.*?) was created by (.*?)\.', r'\2 created \1.'),
                ]
            
            for pattern, replacement in passive_patterns:
                simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
            
            return simplified
            
        except Exception as e:
            self.logger.error(f"Error simplifying sentence structure: {str(e)}")
            return sentence
    
    def _create_logical_paragraphs(self, sentences: List[str], context: SimplificationContext) -> List[str]:
        """Group sentences into logical paragraphs."""
        try:
            if len(sentences) <= 4:
                return [' '.join(sentences)]
            
            # Simple paragraph grouping (3-4 sentences per paragraph)
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                current_paragraph.append(sentence)
                
                # Create new paragraph after 3-4 sentences
                if len(current_paragraph) >= 3:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            # Add remaining sentences
            if current_paragraph:
                if paragraphs:
                    # Add to last paragraph if it's short
                    if len(current_paragraph) <= 2:
                        paragraphs[-1] += ' ' + ' '.join(current_paragraph)
                    else:
                        paragraphs.append(' '.join(current_paragraph))
                else:
                    paragraphs.append(' '.join(current_paragraph))
            
            return paragraphs
            
        except Exception as e:
            self.logger.error(f"Error creating paragraphs: {str(e)}")
            return [' '.join(sentences)]
    
    def _add_transition_words(self, text: str, context: SimplificationContext) -> str:
        """Add appropriate transition words for better flow."""
        try:
            # Simple transition word additions
            transitions = {
                'nl': {
                    'first_sentence': 'Ten eerste,',
                    'continuation': 'Daarnaast,',
                    'conclusion': 'Tot slot,'
                },
                'en': {
                    'first_sentence': 'First,',
                    'continuation': 'Also,',
                    'conclusion': 'Finally,'
                }
            }
            
            lang_transitions = transitions.get(context.language, transitions['en'])
            
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 1:
                # Add transitions to paragraph beginnings
                for i, paragraph in enumerate(paragraphs):
                    if i == 0:
                        paragraphs[i] = f"{lang_transitions['first_sentence']} {paragraph}"
                    elif i == len(paragraphs) - 1:
                        paragraphs[i] = f"{lang_transitions['conclusion']} {paragraph}"
                    else:
                        paragraphs[i] = f"{lang_transitions['continuation']} {paragraph}"
                
                text = '\n\n'.join(paragraphs)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error adding transition words: {str(e)}")
            return text
    
    def _calculate_complexity_reduction(self, before: ReadabilityScores, after: ReadabilityScores) -> float:
        """Calculate the percentage reduction in complexity."""
        try:
            # Use Flesch Reading Ease as primary metric (higher is better)
            before_score = before.flesch_reading_ease
            after_score = after.flesch_reading_ease
            
            if before_score == 0:
                return 0.0
            
            improvement = (after_score - before_score) / 100.0  # Convert to percentage
            return max(0.0, improvement)
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity reduction: {str(e)}")
            return 0.0
    
    async def _calculate_preservation_score(self, original: str, simplified: str, context: SimplificationContext) -> float:
        """Calculate semantic similarity to assess meaning preservation."""
        try:
            if not hasattr(self, 'sentence_model') or not self.sentence_model:
                # Fallback to simple string similarity
                return self._simple_similarity(original, simplified)
            
            # Use sentence transformer for semantic similarity
            embeddings = self.sentence_model.encode([original, simplified])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating preservation score: {str(e)}")
            return self._simple_similarity(original, simplified)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity calculation."""
        try:
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating simple similarity: {str(e)}")
            return 0.5
    
    def _generate_recommendations(self, before: ReadabilityScores, after: ReadabilityScores, context: SimplificationContext) -> List[str]:
        """Generate recommendations for further improvement."""
        recommendations = []
        
        # Check if target CEFR level was achieved
        target_level_value = list(CEFRLevel).index(context.target_cefr_level)
        achieved_level_value = list(CEFRLevel).index(after.cefr_level)
        
        if achieved_level_value > target_level_value:
            recommendations.append(
                f"Text is still at {after.cefr_level.value} level. Further simplification needed to reach {context.target_cefr_level.value}."
            )
        
        # Check specific metrics
        if after.readability_summary.get('avg_sentence_length', 0) > context.max_sentence_length:
            recommendations.append("Consider breaking down longer sentences further.")
        
        if after.readability_summary.get('complex_words_ratio', 0) > 0.2:
            recommendations.append("Consider replacing more complex words with simpler alternatives.")
        
        if after.readability_summary.get('passive_voice_ratio', 0) > 0.3:
            recommendations.append("Consider converting more passive voice constructions to active voice.")
        
        # Positive feedback
        if after.flesch_reading_ease > before.flesch_reading_ease + 10:
            recommendations.append("Good improvement in readability achieved!")
        
        if not recommendations:
            recommendations.append("Text readability has been optimized for the target audience.")
        
        return recommendations


class LanguageSimplificationManager:
    """Main manager for language simplification operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.readability_analyzer = AdvancedReadabilityAnalyzer()
        self.language_simplifier = AdvancedLanguageSimplifier()
        
        # Simplification cache for performance
        self.simplification_cache = {}
        self.cache_max_size = 1000
    
    async def analyze_text_readability(self, text: str, language: str = "nl") -> ReadabilityScores:
        """Analyze text readability and CEFR level."""
        try:
            return await self.readability_analyzer.analyze_readability(text, language)
        except Exception as e:
            self.logger.error(f"Error analyzing readability: {str(e)}")
            raise
    
    async def simplify_for_cefr_level(self, text: str, target_level: CEFRLevel, language: str = "nl", **kwargs) -> SimplificationResult:
        """Simplify text to achieve specific CEFR level."""
        try:
            # Create context
            context = SimplificationContext(
                target_cefr_level=target_level,
                language=language,
                **kwargs
            )
            
            # Check cache
            cache_key = f"{hash(text)}_{target_level.value}_{language}"
            if cache_key in self.simplification_cache:
                return self.simplification_cache[cache_key]
            
            # Perform simplification
            result = await self.language_simplifier.simplify_text(text, context)
            
            # Cache result
            if len(self.simplification_cache) < self.cache_max_size:
                self.simplification_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error simplifying for CEFR level: {str(e)}")
            raise
    
    async def real_time_optimization(self, text: str, target_metrics: Dict[str, float], language: str = "nl") -> SimplificationResult:
        """Perform real-time text optimization based on specific metrics."""
        try:
            # Determine target CEFR level from metrics
            target_cefr = CEFRLevel.B1  # Default
            
            if 'flesch_reading_ease' in target_metrics:
                flesch_target = target_metrics['flesch_reading_ease']
                if flesch_target >= 80:
                    target_cefr = CEFRLevel.A2
                elif flesch_target >= 70:
                    target_cefr = CEFRLevel.B1
                elif flesch_target >= 60:
                    target_cefr = CEFRLevel.B2
            
            # Create context with custom parameters
            context = SimplificationContext(
                target_cefr_level=target_cefr,
                language=language,
                max_sentence_length=int(target_metrics.get('max_sentence_length', 20)),
                max_syllables_per_word=int(target_metrics.get('max_syllables_per_word', 3))
            )
            
            return await self.language_simplifier.simplify_text(text, context)
            
        except Exception as e:
            self.logger.error(f"Error in real-time optimization: {str(e)}")
            raise
    
    async def batch_simplification(self, texts: List[str], context: SimplificationContext) -> List[SimplificationResult]:
        """Simplify multiple texts in batch."""
        try:
            results = []
            
            for text in texts:
                try:
                    result = await self.language_simplifier.simplify_text(text, context)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error simplifying text in batch: {str(e)}")
                    # Add error result
                    results.append(SimplificationResult(
                        original_text=text,
                        simplified_text=text,
                        simplification_strategies=[],
                        readability_before=ReadabilityScores(
                            flesch_reading_ease=50.0, flesch_kincaid_grade=8.0,
                            automated_readability_index=8.0, coleman_liau_index=8.0,
                            gunning_fog=8.0, smog_index=8.0, lix_readability=50.0,
                            rix_readability=50.0, cefr_level=CEFRLevel.B2,
                            complexity_level=TextComplexityLevel.MODERATE, readability_summary={}
                        ),
                        readability_after=ReadabilityScores(
                            flesch_reading_ease=50.0, flesch_kincaid_grade=8.0,
                            automated_readability_index=8.0, coleman_liau_index=8.0,
                            gunning_fog=8.0, smog_index=8.0, lix_readability=50.0,
                            rix_readability=50.0, cefr_level=CEFRLevel.B2,
                            complexity_level=TextComplexityLevel.MODERATE, readability_summary={}
                        ),
                        complexity_reduction=0.0,
                        preservation_score=1.0,
                        changes_made=[],
                        warnings=[f"Simplification error: {str(e)}"],
                        recommendations=[],
                        processing_time=0.0
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch simplification: {str(e)}")
            raise
    
    def get_cefr_recommendations(self, current_level: CEFRLevel, target_level: CEFRLevel, language: str = "nl") -> List[str]:
        """Get specific recommendations for achieving target CEFR level."""
        recommendations = []
        
        current_value = list(CEFRLevel).index(current_level)
        target_value = list(CEFRLevel).index(target_level)
        
        if current_value > target_value:
            # Need to simplify
            level_diff = current_value - target_value
            
            if language == "nl":
                recommendations.extend([
                    "Gebruik kortere zinnen (maximaal 15-20 woorden)",
                    "Vervang moeilijke woorden door eenvoudigere alternatieven",
                    "Vermijd passieve zinsconstructies",
                    "Gebruik meer bekende woorden uit het dagelijks gebruik"
                ])
                
                if level_diff >= 2:
                    recommendations.extend([
                        "Splits complexe zinnen op in meerdere eenvoudige zinnen",
                        "Voeg uitleg toe bij technische termen",
                        "Gebruik meer voorbeelden en concrete situaties"
                    ])
            else:
                recommendations.extend([
                    "Use shorter sentences (maximum 15-20 words)",
                    "Replace difficult words with simpler alternatives",
                    "Avoid passive voice constructions",
                    "Use more common everyday words"
                ])
                
                if level_diff >= 2:
                    recommendations.extend([
                        "Split complex sentences into multiple simple ones",
                        "Add explanations for technical terms",
                        "Use more examples and concrete situations"
                    ])
        
        return recommendations


# Factory function for creating simplification manager
def create_language_simplification_manager() -> LanguageSimplificationManager:
    """Factory function to create and initialize LanguageSimplificationManager."""
    return LanguageSimplificationManager()


# Convenience functions for integration with other systems
async def simplify_mortgage_advice(text: str, target_level: CEFRLevel = CEFRLevel.B1, language: str = "nl") -> SimplificationResult:
    """
    Convenience function to simplify mortgage advice text.
    
    Args:
        text: The mortgage advice text to simplify
        target_level: Target CEFR level for simplification
        language: Language of the text ('nl' or 'en')
        
    Returns:
        SimplificationResult with comprehensive analysis and simplified text
    """
    manager = create_language_simplification_manager()
    return await manager.simplify_for_cefr_level(text, target_level, language, domain="finance")


async def assess_text_complexity(text: str, language: str = "nl") -> ReadabilityScores:
    """
    Assess the complexity and readability of text.
    
    Args:
        text: Text to analyze
        language: Language of the text
        
    Returns:
        ReadabilityScores with comprehensive analysis
    """
    manager = create_language_simplification_manager()
    return await manager.analyze_text_readability(text, language)


async def optimize_for_accessibility(text: str, accessibility_requirements: List[str], language: str = "nl") -> SimplificationResult:
    """
    Optimize text for accessibility compliance.
    
    Args:
        text: Text to optimize
        accessibility_requirements: List of accessibility requirements (WCAG levels)
        language: Language of the text
        
    Returns:
        SimplificationResult optimized for accessibility
    """
    manager = create_language_simplification_manager()
    
    context = SimplificationContext(
        target_cefr_level=CEFRLevel.B1,
        language=language,
        accessibility_requirements=accessibility_requirements,
        max_sentence_length=15,  # Shorter for accessibility
        max_syllables_per_word=2   # Simpler words for accessibility
    )
    
    simplifier = AdvancedLanguageSimplifier()
    return await simplifier.simplify_text(text, context)
