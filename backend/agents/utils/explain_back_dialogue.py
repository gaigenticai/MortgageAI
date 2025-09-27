"""
Interactive Explain-Back Dialogue System for MortgageAI

This module provides advanced dialogue management capabilities for validating client comprehension
through interactive explain-back conversations, adaptive questioning, and personalized learning feedback.

Features:
- Comprehension validation through interactive dialogues
- Adaptive questioning based on client responses and understanding levels
- Learning feedback mechanisms with personalized improvement paths
- Multi-modal dialogue support (text, voice, visual aids)
- Real-time comprehension scoring and progress tracking
- AFM compliance for financial advice explanation requirements
- Multilingual support (Dutch, English) with cultural context awareness
- Advanced NLP for intent recognition and response generation
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json
import re
import uuid
import numpy as np
from collections import defaultdict, deque
import statistics

# Advanced NLP and ML libraries
import spacy
import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, GPT2LMHeadModel, GPT2Tokenizer
)
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat

# Database and configuration
from ..config import settings


class DialogueState(Enum):
    """States in the explain-back dialogue process."""
    INITIAL = "initial"
    EXPLANATION_PROVIDED = "explanation_provided"
    COMPREHENSION_TESTING = "comprehension_testing"
    REMEDIATION = "remediation"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(Enum):
    """Types of comprehension validation questions."""
    FACTUAL_RECALL = "factual_recall"
    CONCEPTUAL_UNDERSTANDING = "conceptual_understanding"
    APPLICATION = "application"
    SCENARIO_BASED = "scenario_based"
    COMPARISON = "comparison"
    CONSEQUENCE_ANALYSIS = "consequence_analysis"
    DECISION_MAKING = "decision_making"


class ComprehensionLevel(Enum):
    """Client comprehension levels."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 75-89%
    ADEQUATE = "adequate"   # 60-74%
    POOR = "poor"          # 40-59%
    INADEQUATE = "inadequate"  # 0-39%


class LearningStyle(Enum):
    """Learning style preferences."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"


@dataclass
class DialogueContext:
    """Context information for the dialogue session."""
    session_id: str
    client_id: str
    advisor_id: str
    topic: str
    language: str = "nl"
    complexity_level: str = "intermediate"
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    accessibility_needs: List[str] = field(default_factory=list)
    time_constraints: Optional[int] = None  # minutes
    session_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueTurn:
    """Represents a single turn in the dialogue."""
    turn_id: str
    timestamp: datetime
    speaker: str  # 'system', 'client', 'advisor'
    content: str
    intent: Optional[str] = None
    sentiment: Optional[float] = None
    confidence: Optional[float] = None
    comprehension_indicators: Dict[str, Any] = field(default_factory=dict)
    response_time: Optional[float] = None  # seconds
    multimodal_elements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensionAssessment:
    """Assessment of client comprehension at a point in time."""
    assessment_id: str
    timestamp: datetime
    topic: str
    question_type: QuestionType
    question_content: str
    client_response: str
    expected_concepts: List[str]
    identified_concepts: List[str]
    comprehension_score: float  # 0-1
    confidence_score: float  # 0-1
    response_quality: str  # excellent, good, adequate, poor, inadequate
    knowledge_gaps: List[str]
    remediation_suggestions: List[str]


@dataclass
class LearningProfile:
    """Client learning profile and progress tracking."""
    client_id: str
    learning_style: LearningStyle
    preferred_explanation_length: str  # short, medium, long
    concept_mastery: Dict[str, float]  # concept -> mastery score (0-1)
    learning_pace: str  # fast, medium, slow
    attention_span: int  # estimated minutes
    previous_sessions: List[str]  # session IDs
    strengths: List[str]
    challenges: List[str]
    progress_over_time: Dict[str, List[float]]  # concept -> [scores over time]
    adaptive_parameters: Dict[str, Any]


class AdvancedNLPProcessor:
    """Advanced NLP processing for dialogue analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        try:
            # Multi-language models
            self.nlp_nl = spacy.load("nl_core_news_sm")
            self.nlp_en = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy models for Dutch and English")
        except OSError as e:
            self.logger.error(f"Failed to load spaCy models: {e}")
            self.nlp_nl = None
            self.nlp_en = None
        
        # Initialize transformers
        try:
            # Multilingual BERT for embeddings
            self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased')
            
            # Intent classification model
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium"
            )
            
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
            
            # Question generation model
            self.question_generator = pipeline(
                "text2text-generation",
                model="valhalla/t5-base-qg-hl"
            )
            
            self.logger.info("Initialized transformer models")
        except Exception as e:
            self.logger.error(f"Failed to initialize transformer models: {e}")
            self.sentence_model = None
            self.intent_classifier = None
            self.sentiment_analyzer = None
            self.question_generator = None
        
        # Initialize NLTK components
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            self.logger.error(f"Failed to initialize NLTK: {e}")
            self.vader = None
        
        # Financial domain concepts
        self.financial_concepts = {
            'nl': {
                'hypotheek': ['lening', 'financiering', 'krediet'],
                'rente': ['percentage', 'kosten', 'tarief'],
                'aflossing': ['terugbetaling', 'maandlast', 'schuld'],
                'nhg': ['nationale hypotheek garantie', 'garantie', 'waarborg'],
                'bkr': ['bureau krediet registratie', 'kredietcheck', 'screening'],
                'ltv': ['loan to value', 'financieringsratio', 'verhouding'],
                'afm': ['autoriteit financiele markten', 'toezichthouder', 'regulator']
            },
            'en': {
                'mortgage': ['loan', 'financing', 'credit'],
                'interest': ['rate', 'percentage', 'cost'],
                'repayment': ['payment', 'installment', 'amortization'],
                'guarantee': ['nhg', 'insurance', 'protection'],
                'credit_check': ['bkr', 'screening', 'assessment'],
                'ltv': ['loan to value', 'ratio', 'financing'],
                'regulator': ['afm', 'authority', 'supervisor']
            }
        }
    
    async def analyze_text(self, text: str, language: str = "nl") -> Dict[str, Any]:
        """Comprehensive text analysis for dialogue understanding."""
        try:
            analysis = {
                'text': text,
                'language': language,
                'length': len(text.split()),
                'complexity': self._calculate_complexity(text, language),
                'sentiment': await self._analyze_sentiment(text),
                'entities': self._extract_entities(text, language),
                'concepts': self._identify_concepts(text, language),
                'intent': await self._classify_intent(text),
                'readability': self._calculate_readability(text, language),
                'discourse_markers': self._identify_discourse_markers(text, language)
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error in text analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_complexity(self, text: str, language: str) -> Dict[str, float]:
        """Calculate text complexity metrics."""
        try:
            # Basic complexity metrics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            avg_word_length = statistics.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = statistics.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
            
            # Advanced complexity using textstat
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            
            return {
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'flesch_reading_ease': flesch_reading_ease,
                'flesch_kincaid_grade': flesch_kincaid_grade,
                'overall_complexity': max(0, min(1, (100 - flesch_reading_ease) / 100))
            }
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple approaches."""
        try:
            sentiment = {}
            
            # VADER sentiment (good for social media text)
            if self.vader:
                vader_scores = self.vader.polarity_scores(text)
                sentiment['vader'] = vader_scores
            
            # Transformer-based sentiment
            if self.sentiment_analyzer:
                transformer_sentiment = self.sentiment_analyzer(text)
                sentiment['transformer'] = transformer_sentiment[0] if transformer_sentiment else None
            
            # Overall sentiment score
            if 'vader' in sentiment:
                compound_score = sentiment['vader']['compound']
                if compound_score >= 0.05:
                    overall_sentiment = 'positive'
                elif compound_score <= -0.05:
                    overall_sentiment = 'negative'
                else:
                    overall_sentiment = 'neutral'
                
                sentiment['overall'] = {
                    'label': overall_sentiment,
                    'score': abs(compound_score)
                }
            
            return sentiment
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'error': str(e)}
    
    def _extract_entities(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            nlp = self.nlp_nl if language == 'nl' else self.nlp_en
            if not nlp:
                return []
            
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'confidence', 0.5)
                })
            
            return entities
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _identify_concepts(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Identify financial concepts in text."""
        try:
            concepts = []
            text_lower = text.lower()
            
            concept_dict = self.financial_concepts.get(language, {})
            
            for concept, synonyms in concept_dict.items():
                # Check for main concept
                if concept in text_lower:
                    concepts.append({
                        'concept': concept,
                        'matched_text': concept,
                        'confidence': 1.0,
                        'category': 'financial'
                    })
                
                # Check for synonyms
                for synonym in synonyms:
                    if synonym in text_lower:
                        concepts.append({
                            'concept': concept,
                            'matched_text': synonym,
                            'confidence': 0.8,
                            'category': 'financial'
                        })
            
            return concepts
        except Exception as e:
            self.logger.error(f"Error identifying concepts: {str(e)}")
            return []
    
    async def _classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify the intent of the text."""
        try:
            # Simple rule-based intent classification for financial domain
            text_lower = text.lower()
            
            # Question indicators
            question_words = ['wat', 'waar', 'wanneer', 'wie', 'waarom', 'hoe',
                             'what', 'where', 'when', 'who', 'why', 'how']
            
            # Confirmation indicators
            confirmation_words = ['ja', 'nee', 'klopt', 'correct', 'juist',
                                 'yes', 'no', 'right', 'correct', 'true', 'false']
            
            # Uncertainty indicators
            uncertainty_words = ['misschien', 'denk', 'geloof', 'weet niet',
                               'maybe', 'think', 'believe', 'don\'t know', 'unsure']
            
            intent_scores = {}
            
            # Calculate intent probabilities
            if any(word in text_lower for word in question_words):
                intent_scores['question'] = 0.8
            
            if any(word in text_lower for word in confirmation_words):
                intent_scores['confirmation'] = 0.7
            
            if any(word in text_lower for word in uncertainty_words):
                intent_scores['uncertainty'] = 0.6
            
            if '?' in text:
                intent_scores['question'] = intent_scores.get('question', 0) + 0.3
            
            # Default to statement if no clear intent
            if not intent_scores:
                intent_scores['statement'] = 0.5
            
            # Get most likely intent
            primary_intent = max(intent_scores, key=intent_scores.get)
            
            return {
                'primary_intent': primary_intent,
                'confidence': intent_scores[primary_intent],
                'all_intents': intent_scores
            }
        except Exception as e:
            self.logger.error(f"Error classifying intent: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_readability(self, text: str, language: str) -> Dict[str, float]:
        """Calculate readability scores."""
        try:
            readability = {}
            
            # Use textstat for readability metrics
            readability['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            readability['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            readability['automated_readability_index'] = textstat.automated_readability_index(text)
            readability['coleman_liau_index'] = textstat.coleman_liau_index(text)
            
            # CEFR level estimation (for European framework)
            flesch_score = readability['flesch_reading_ease']
            if flesch_score >= 90:
                cefr_level = 'A1'
            elif flesch_score >= 80:
                cefr_level = 'A2'
            elif flesch_score >= 70:
                cefr_level = 'B1'
            elif flesch_score >= 60:
                cefr_level = 'B2'
            elif flesch_score >= 50:
                cefr_level = 'C1'
            else:
                cefr_level = 'C2'
            
            readability['cefr_level'] = cefr_level
            
            return readability
        except Exception as e:
            self.logger.error(f"Error calculating readability: {str(e)}")
            return {'error': str(e)}
    
    def _identify_discourse_markers(self, text: str, language: str) -> List[str]:
        """Identify discourse markers that indicate comprehension level."""
        try:
            markers = []
            text_lower = text.lower()
            
            # Discourse markers by language
            if language == 'nl':
                markers_dict = {
                    'understanding': ['begrijp', 'snap', 'duidelijk', 'helder'],
                    'confusion': ['verwarrend', 'onduidelijk', 'snap niet', 'begrijp niet'],
                    'agreement': ['eens', 'akkoord', 'klopt', 'juist'],
                    'disagreement': ['oneens', 'niet akkoord', 'klopt niet'],
                    'certainty': ['zeker', 'absoluut', 'definitief'],
                    'uncertainty': ['onzeker', 'misschien', 'weet niet']
                }
            else:  # English
                markers_dict = {
                    'understanding': ['understand', 'clear', 'obvious', 'makes sense'],
                    'confusion': ['confusing', 'unclear', 'don\'t understand', 'don\'t get'],
                    'agreement': ['agree', 'correct', 'right', 'exactly'],
                    'disagreement': ['disagree', 'wrong', 'incorrect'],
                    'certainty': ['certain', 'sure', 'definitely', 'absolutely'],
                    'uncertainty': ['uncertain', 'maybe', 'don\'t know', 'unsure']
                }
            
            for category, marker_list in markers_dict.items():
                for marker in marker_list:
                    if marker in text_lower:
                        markers.append(category)
                        break
            
            return list(set(markers))  # Remove duplicates
        except Exception as e:
            self.logger.error(f"Error identifying discourse markers: {str(e)}")
            return []


class AdaptiveQuestionGenerator:
    """Generates adaptive questions based on comprehension level and learning profile."""
    
    def __init__(self, nlp_processor: AdvancedNLPProcessor):
        self.logger = logging.getLogger(__name__)
        self.nlp = nlp_processor
        
        # Question templates by type and language
        self.question_templates = self._initialize_question_templates()
        
        # Difficulty progression rules
        self.difficulty_rules = {
            ComprehensionLevel.EXCELLENT: {'next_level': 'increase', 'complexity_factor': 1.2},
            ComprehensionLevel.GOOD: {'next_level': 'maintain', 'complexity_factor': 1.0},
            ComprehensionLevel.ADEQUATE: {'next_level': 'simplify', 'complexity_factor': 0.8},
            ComprehensionLevel.POOR: {'next_level': 'remediate', 'complexity_factor': 0.6},
            ComprehensionLevel.INADEQUATE: {'next_level': 'restart', 'complexity_factor': 0.4}
        }
    
    def _initialize_question_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize question templates for different types and languages."""
        return {
            'nl': {
                'factual_recall': [
                    "Wat is {concept}?",
                    "Kunt u uitleggen wat {concept} betekent?",
                    "Wat zijn de belangrijkste kenmerken van {concept}?",
                    "Hoe zou u {concept} definiëren?"
                ],
                'conceptual_understanding': [
                    "Waarom is {concept} belangrijk voor uw hypotheek?",
                    "Hoe beïnvloedt {concept} uw maandlasten?",
                    "Wat is het verband tussen {concept1} en {concept2}?",
                    "Hoe werkt {concept} in de praktijk?"
                ],
                'application': [
                    "Hoe zou u {concept} toepassen in uw situatie?",
                    "Wat zou er gebeuren als {scenario}?",
                    "Welke keuze zou u maken tussen {option1} en {option2}?",
                    "Hoe zou u {concept} gebruiken om {goal} te bereiken?"
                ],
                'scenario_based': [
                    "Stel dat {scenario}, wat zou u dan doen?",
                    "In welke situatie zou {concept} het meest relevant zijn?",
                    "Hoe zou uw beslissing veranderen als {condition}?",
                    "Wat zijn de gevolgen van {action} in uw situatie?"
                ]
            },
            'en': {
                'factual_recall': [
                    "What is {concept}?",
                    "Can you explain what {concept} means?",
                    "What are the key features of {concept}?",
                    "How would you define {concept}?"
                ],
                'conceptual_understanding': [
                    "Why is {concept} important for your mortgage?",
                    "How does {concept} affect your monthly payments?",
                    "What is the relationship between {concept1} and {concept2}?",
                    "How does {concept} work in practice?"
                ],
                'application': [
                    "How would you apply {concept} in your situation?",
                    "What would happen if {scenario}?",
                    "Which choice would you make between {option1} and {option2}?",
                    "How would you use {concept} to achieve {goal}?"
                ],
                'scenario_based': [
                    "Suppose {scenario}, what would you do?",
                    "In which situation would {concept} be most relevant?",
                    "How would your decision change if {condition}?",
                    "What are the consequences of {action} in your situation?"
                ]
            }
        }
    
    async def generate_question(self, 
                              topic: str, 
                              question_type: QuestionType,
                              comprehension_level: ComprehensionLevel,
                              learning_profile: LearningProfile,
                              context: DialogueContext,
                              previous_responses: List[DialogueTurn] = None) -> str:
        """Generate an adaptive question based on multiple factors."""
        try:
            # Select appropriate template
            templates = self.question_templates[context.language][question_type.value]
            
            # Adapt based on comprehension level
            difficulty_rule = self.difficulty_rules[comprehension_level]
            complexity_factor = difficulty_rule['complexity_factor']
            
            # Select template based on complexity
            if complexity_factor >= 1.0:
                # More complex questions
                template_idx = min(len(templates) - 1, int(complexity_factor * len(templates)) - 1)
            else:
                # Simpler questions
                template_idx = max(0, int(complexity_factor * len(templates)))
            
            selected_template = templates[template_idx]
            
            # Extract relevant concepts from topic
            topic_analysis = await self.nlp.analyze_text(topic, context.language)
            concepts = [c['concept'] for c in topic_analysis.get('concepts', [])]
            
            # Fill in template with concepts
            if concepts:
                question = selected_template.format(
                    concept=concepts[0] if '{concept}' in selected_template else '',
                    concept1=concepts[0] if len(concepts) > 0 and '{concept1}' in selected_template else '',
                    concept2=concepts[1] if len(concepts) > 1 and '{concept2}' in selected_template else '',
                    scenario=self._generate_scenario(topic, context),
                    option1=self._generate_option(concepts[0] if concepts else topic, 1),
                    option2=self._generate_option(concepts[0] if concepts else topic, 2),
                    goal=self._generate_goal(topic, context),
                    condition=self._generate_condition(topic, context),
                    action=self._generate_action(topic, context)
                )
            else:
                question = selected_template
            
            # Adapt for learning style
            question = self._adapt_for_learning_style(question, learning_profile.learning_style, context)
            
            return question
            
        except Exception as e:
            self.logger.error(f"Error generating question: {str(e)}")
            return "Kunt u meer vertellen over uw begrip van dit onderwerp?" if context.language == 'nl' else "Can you tell me more about your understanding of this topic?"
    
    def _generate_scenario(self, topic: str, context: DialogueContext) -> str:
        """Generate a relevant scenario for the topic."""
        # Simplified scenario generation - in production, this would be more sophisticated
        scenarios = {
            'nl': {
                'hypotheek': 'u een huis wilt kopen van €400.000',
                'rente': 'de rente stijgt met 1%',
                'aflossing': 'u extra wilt aflossen',
                'nhg': 'u gebruik maakt van de NHG'
            },
            'en': {
                'mortgage': 'you want to buy a house for €400,000',
                'interest': 'interest rates increase by 1%',
                'repayment': 'you want to make extra payments',
                'guarantee': 'you use the NHG guarantee'
            }
        }
        
        topic_lower = topic.lower()
        scenario_dict = scenarios.get(context.language, scenarios['en'])
        
        for key, scenario in scenario_dict.items():
            if key in topic_lower:
                return scenario
        
        return 'uw situatie verandert' if context.language == 'nl' else 'your situation changes'
    
    def _generate_option(self, concept: str, option_num: int) -> str:
        """Generate options for comparison questions."""
        options = {
            'nl': {
                'hypotheek': ['annuïteitenhypotheek', 'lineaire hypotheek'],
                'rente': ['variabele rente', 'vaste rente'],
                'aflossing': ['aflossingsvrij', 'annuïtair']
            },
            'en': {
                'mortgage': ['annuity mortgage', 'linear mortgage'],
                'interest': ['variable rate', 'fixed rate'],
                'repayment': ['interest-only', 'repayment']
            }
        }
        
        # Default options if concept not found
        default_options = {
            'nl': ['optie A', 'optie B'],
            'en': ['option A', 'option B']
        }
        
        concept_lower = concept.lower()
        for lang, lang_options in options.items():
            for key, option_list in lang_options.items():
                if key in concept_lower:
                    return option_list[option_num - 1] if option_num <= len(option_list) else option_list[0]
        
        return default_options.get('nl' if 'nl' in str(options) else 'en', default_options['en'])[option_num - 1]
    
    def _generate_goal(self, topic: str, context: DialogueContext) -> str:
        """Generate a relevant goal for the topic."""
        goals = {
            'nl': {
                'hypotheek': 'de laagste maandlasten',
                'rente': 'rentezekerheid',
                'aflossing': 'sneller schuldenvrij'
            },
            'en': {
                'mortgage': 'the lowest monthly payments',
                'interest': 'interest rate certainty',
                'repayment': 'faster debt freedom'
            }
        }
        
        topic_lower = topic.lower()
        goal_dict = goals.get(context.language, goals['en'])
        
        for key, goal in goal_dict.items():
            if key in topic_lower:
                return goal
        
        return 'uw doel' if context.language == 'nl' else 'your goal'
    
    def _generate_condition(self, topic: str, context: DialogueContext) -> str:
        """Generate a condition for scenario questions."""
        conditions = {
            'nl': {
                'hypotheek': 'uw inkomen daalt',
                'rente': 'de rente verder stijgt',
                'aflossing': 'u een erfenis ontvangt'
            },
            'en': {
                'mortgage': 'your income decreases',
                'interest': 'interest rates rise further',
                'repayment': 'you receive an inheritance'
            }
        }
        
        topic_lower = topic.lower()
        condition_dict = conditions.get(context.language, conditions['en'])
        
        for key, condition in condition_dict.items():
            if key in topic_lower:
                return condition
        
        return 'de omstandigheden veranderen' if context.language == 'nl' else 'circumstances change'
    
    def _generate_action(self, topic: str, context: DialogueContext) -> str:
        """Generate an action for consequence questions."""
        actions = {
            'nl': {
                'hypotheek': 'oversluiten naar een andere bank',
                'rente': 'de rente vast zetten',
                'aflossing': 'extra aflossen'
            },
            'en': {
                'mortgage': 'switching to another bank',
                'interest': 'fixing the interest rate',
                'repayment': 'making extra payments'
            }
        }
        
        topic_lower = topic.lower()
        action_dict = actions.get(context.language, actions['en'])
        
        for key, action in action_dict.items():
            if key in topic_lower:
                return action
        
        return 'deze actie ondernemen' if context.language == 'nl' else 'taking this action'
    
    def _adapt_for_learning_style(self, question: str, learning_style: LearningStyle, context: DialogueContext) -> str:
        """Adapt question presentation for different learning styles."""
        adaptations = {
            LearningStyle.VISUAL: {
                'nl': " (Stel je voor hoe dit eruit zou zien in een grafiek of diagram.)",
                'en': " (Imagine how this would look in a chart or diagram.)"
            },
            LearningStyle.AUDITORY: {
                'nl': " (Beschrijf dit alsof je het aan iemand uitlegt.)",
                'en': " (Describe this as if you're explaining it to someone.)"
            },
            LearningStyle.KINESTHETIC: {
                'nl': " (Denk aan de praktische stappen die je zou nemen.)",
                'en': " (Think about the practical steps you would take.)"
            },
            LearningStyle.READING_WRITING: {
                'nl': " (Je mag dit opschrijven om je gedachten te ordenen.)",
                'en': " (You may write this down to organize your thoughts.)"
            }
        }
        
        if learning_style in adaptations:
            adaptation = adaptations[learning_style].get(context.language, "")
            return question + adaptation
        
        return question


class ComprehensionValidator:
    """Validates client responses and assesses comprehension levels."""
    
    def __init__(self, nlp_processor: AdvancedNLPProcessor):
        self.logger = logging.getLogger(__name__)
        self.nlp = nlp_processor
        
        # Comprehension scoring weights
        self.scoring_weights = {
            'concept_coverage': 0.4,      # How many key concepts are mentioned
            'accuracy': 0.3,              # Correctness of information
            'depth': 0.2,                 # Level of understanding shown
            'coherence': 0.1              # Logical flow of response
        }
    
    async def assess_comprehension(self,
                                 question: str,
                                 response: str,
                                 expected_concepts: List[str],
                                 context: DialogueContext) -> ComprehensionAssessment:
        """Assess client comprehension based on their response."""
        try:
            # Analyze the response
            response_analysis = await self.nlp.analyze_text(response, context.language)
            
            # Extract concepts from response
            identified_concepts = [c['concept'] for c in response_analysis.get('concepts', [])]
            
            # Calculate concept coverage
            concept_coverage = self._calculate_concept_coverage(expected_concepts, identified_concepts)
            
            # Assess accuracy
            accuracy = await self._assess_accuracy(response, expected_concepts, context)
            
            # Assess depth of understanding
            depth = self._assess_depth(response_analysis, context)
            
            # Assess coherence
            coherence = self._assess_coherence(response_analysis)
            
            # Calculate overall comprehension score
            comprehension_score = (
                concept_coverage * self.scoring_weights['concept_coverage'] +
                accuracy * self.scoring_weights['accuracy'] +
                depth * self.scoring_weights['depth'] +
                coherence * self.scoring_weights['coherence']
            )
            
            # Determine response quality
            quality = self._determine_quality_level(comprehension_score)
            
            # Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(expected_concepts, identified_concepts)
            
            # Generate remediation suggestions
            remediation_suggestions = self._generate_remediation_suggestions(
                knowledge_gaps, comprehension_score, context
            )
            
            # Estimate confidence
            confidence = self._estimate_confidence(response_analysis, comprehension_score)
            
            # Create assessment
            assessment = ComprehensionAssessment(
                assessment_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                topic=context.topic,
                question_type=QuestionType.FACTUAL_RECALL,  # This should be passed as parameter
                question_content=question,
                client_response=response,
                expected_concepts=expected_concepts,
                identified_concepts=identified_concepts,
                comprehension_score=comprehension_score,
                confidence_score=confidence,
                response_quality=quality,
                knowledge_gaps=knowledge_gaps,
                remediation_suggestions=remediation_suggestions
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing comprehension: {str(e)}")
            # Return a default assessment
            return ComprehensionAssessment(
                assessment_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                topic=context.topic,
                question_type=QuestionType.FACTUAL_RECALL,
                question_content=question,
                client_response=response,
                expected_concepts=expected_concepts,
                identified_concepts=[],
                comprehension_score=0.0,
                confidence_score=0.0,
                response_quality="error",
                knowledge_gaps=expected_concepts,
                remediation_suggestions=["Er is een fout opgetreden bij de beoordeling"]
            )
    
    def _calculate_concept_coverage(self, expected: List[str], identified: List[str]) -> float:
        """Calculate how well the response covers expected concepts."""
        if not expected:
            return 1.0
        
        # Count matches (case-insensitive)
        expected_lower = [c.lower() for c in expected]
        identified_lower = [c.lower() for c in identified]
        
        matches = sum(1 for concept in expected_lower if concept in identified_lower)
        
        return matches / len(expected)
    
    async def _assess_accuracy(self, response: str, expected_concepts: List[str], context: DialogueContext) -> float:
        """Assess the accuracy of the information in the response."""
        try:
            # This is a simplified accuracy assessment
            # In production, this would use more sophisticated fact-checking
            
            # Check for common misconceptions or errors
            error_indicators = {
                'nl': ['verkeerd', 'fout', 'niet waar', 'onjuist'],
                'en': ['wrong', 'incorrect', 'false', 'not true']
            }
            
            response_lower = response.lower()
            errors = error_indicators.get(context.language, [])
            
            # Basic accuracy heuristics
            accuracy = 1.0
            
            # Deduct for obvious error indicators
            for error in errors:
                if error in response_lower:
                    accuracy -= 0.2
            
            # Check for presence of key concepts (indicates some accuracy)
            concept_presence = sum(1 for concept in expected_concepts 
                                 if concept.lower() in response_lower)
            
            if expected_concepts:
                concept_accuracy = concept_presence / len(expected_concepts)
                accuracy = (accuracy + concept_accuracy) / 2
            
            return max(0.0, min(1.0, accuracy))
            
        except Exception as e:
            self.logger.error(f"Error assessing accuracy: {str(e)}")
            return 0.5  # Neutral accuracy if assessment fails
    
    def _assess_depth(self, response_analysis: Dict[str, Any], context: DialogueContext) -> float:
        """Assess the depth of understanding shown in the response."""
        try:
            # Depth indicators
            depth_score = 0.0
            
            # Length indicates some depth (but not everything)
            length_score = min(1.0, response_analysis.get('length', 0) / 50)  # Normalize to 50 words
            depth_score += length_score * 0.3
            
            # Complexity indicates deeper understanding
            complexity = response_analysis.get('complexity', {})
            complexity_score = complexity.get('overall_complexity', 0)
            depth_score += complexity_score * 0.4
            
            # Presence of multiple concepts indicates depth
            concepts = response_analysis.get('concepts', [])
            concept_diversity = min(1.0, len(concepts) / 5)  # Normalize to 5 concepts
            depth_score += concept_diversity * 0.3
            
            return min(1.0, depth_score)
            
        except Exception as e:
            self.logger.error(f"Error assessing depth: {str(e)}")
            return 0.0
    
    def _assess_coherence(self, response_analysis: Dict[str, Any]) -> float:
        """Assess the logical coherence of the response."""
        try:
            # Simplified coherence assessment
            coherence_score = 0.5  # Base coherence
            
            # Check for discourse markers that indicate good structure
            discourse_markers = response_analysis.get('discourse_markers', [])
            
            # Positive indicators
            positive_markers = ['understanding', 'agreement', 'certainty']
            negative_markers = ['confusion', 'uncertainty']
            
            positive_count = sum(1 for marker in discourse_markers if marker in positive_markers)
            negative_count = sum(1 for marker in discourse_markers if marker in negative_markers)
            
            # Adjust coherence based on markers
            coherence_score += positive_count * 0.1
            coherence_score -= negative_count * 0.1
            
            # Length too short or too long may indicate poor coherence
            length = response_analysis.get('length', 0)
            if length < 5:  # Too short
                coherence_score -= 0.2
            elif length > 200:  # Too long might be rambling
                coherence_score -= 0.1
            
            return max(0.0, min(1.0, coherence_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing coherence: {str(e)}")
            return 0.5
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine response quality level from comprehension score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "adequate"
        elif score >= 0.4:
            return "poor"
        else:
            return "inadequate"
    
    def _identify_knowledge_gaps(self, expected: List[str], identified: List[str]) -> List[str]:
        """Identify concepts that were expected but not demonstrated."""
        expected_lower = [c.lower() for c in expected]
        identified_lower = [c.lower() for c in identified]
        
        gaps = [concept for concept in expected 
                if concept.lower() not in identified_lower]
        
        return gaps
    
    def _generate_remediation_suggestions(self, gaps: List[str], score: float, context: DialogueContext) -> List[str]:
        """Generate suggestions for improving understanding."""
        suggestions = []
        lang = context.language
        
        if score < 0.4:
            # Major remediation needed
            suggestions.extend([
                "Laten we opnieuw beginnen met de basisprincipes" if lang == 'nl' 
                else "Let's start again with the basic principles",
                "Ik zal het concept stap voor stap uitleggen" if lang == 'nl'
                else "I'll explain the concept step by step"
            ])
        elif score < 0.6:
            # Moderate remediation
            suggestions.extend([
                "Laten we de belangrijkste punten nogmaals doornemen" if lang == 'nl'
                else "Let's review the main points again",
                "Ik geef u een voorbeeld om het te verduidelijken" if lang == 'nl'
                else "I'll give you an example to clarify"
            ])
        elif score < 0.75:
            # Minor remediation
            suggestions.extend([
                "U bent op de goede weg, laten we dit detail uitdiepen" if lang == 'nl'
                else "You're on the right track, let's explore this detail further"
            ])
        
        # Specific gap remediation
        for gap in gaps[:3]:  # Limit to top 3 gaps
            suggestions.append(
                f"We moeten het concept '{gap}' nog behandelen" if lang == 'nl'
                else f"We need to cover the concept '{gap}'"
            )
        
        return suggestions
    
    def _estimate_confidence(self, response_analysis: Dict[str, Any], comprehension_score: float) -> float:
        """Estimate confidence in the comprehension assessment."""
        confidence = comprehension_score
        
        # Adjust based on response length (very short or very long responses are less reliable)
        length = response_analysis.get('length', 0)
        if length < 5 or length > 200:
            confidence *= 0.8
        
        # Adjust based on sentiment (neutral/positive responses are more reliable)
        sentiment = response_analysis.get('sentiment', {})
        if sentiment.get('overall', {}).get('label') == 'negative':
            confidence *= 0.9
        
        return max(0.1, min(1.0, confidence))


class ExplainBackDialogueManager:
    """Main manager for explain-back dialogue sessions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.nlp_processor = AdvancedNLPProcessor()
        self.question_generator = AdaptiveQuestionGenerator(self.nlp_processor)
        self.comprehension_validator = ComprehensionValidator(self.nlp_processor)
        
        # Active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Learning profiles cache
        self.learning_profiles: Dict[str, LearningProfile] = {}
        
        # Session history
        self.session_history: Dict[str, List[DialogueTurn]] = defaultdict(list)
    
    async def start_dialogue_session(self, context: DialogueContext, initial_explanation: str) -> Dict[str, Any]:
        """Start a new explain-back dialogue session."""
        try:
            session_id = context.session_id
            
            # Initialize session state
            self.active_sessions[session_id] = {
                'context': context,
                'state': DialogueState.INITIAL,
                'start_time': datetime.now(timezone.utc),
                'explanation': initial_explanation,
                'comprehension_history': [],
                'current_question': None,
                'question_count': 0,
                'max_questions': 10,  # Configurable
                'overall_comprehension': 0.0,
                'remediation_attempts': 0,
                'max_remediation_attempts': 3
            }
            
            # Load or create learning profile
            learning_profile = await self._get_or_create_learning_profile(context.client_id)
            self.learning_profiles[context.client_id] = learning_profile
            
            # Provide initial explanation
            initial_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=initial_explanation,
                intent='explanation',
                multimodal_elements=await self._add_multimodal_elements(initial_explanation, learning_profile, context)
            )
            
            self.session_history[session_id].append(initial_turn)
            
            # Generate first comprehension question
            expected_concepts = await self._extract_key_concepts(initial_explanation, context)
            
            first_question = await self.question_generator.generate_question(
                topic=context.topic,
                question_type=QuestionType.FACTUAL_RECALL,
                comprehension_level=ComprehensionLevel.ADEQUATE,  # Start at adequate level
                learning_profile=learning_profile,
                context=context
            )
            
            self.active_sessions[session_id]['current_question'] = first_question
            self.active_sessions[session_id]['expected_concepts'] = expected_concepts
            self.active_sessions[session_id]['state'] = DialogueState.COMPREHENSION_TESTING
            
            # Create question turn
            question_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=first_question,
                intent='comprehension_question',
                multimodal_elements=await self._add_multimodal_elements(first_question, learning_profile, context)
            )
            
            self.session_history[session_id].append(question_turn)
            
            return {
                'success': True,
                'session_id': session_id,
                'state': DialogueState.COMPREHENSION_TESTING.value,
                'question': first_question,
                'multimodal_elements': question_turn.multimodal_elements,
                'expected_response_time': self._estimate_response_time(learning_profile),
                'progress': {'current_step': 1, 'total_steps': self.active_sessions[session_id]['max_questions']}
            }
            
        except Exception as e:
            self.logger.error(f"Error starting dialogue session: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': context.session_id
            }
    
    async def process_client_response(self, session_id: str, response: str, response_time: Optional[float] = None) -> Dict[str, Any]:
        """Process client response and continue the dialogue."""
        try:
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': 'Session not found',
                    'session_id': session_id
                }
            
            session = self.active_sessions[session_id]
            context = session['context']
            learning_profile = self.learning_profiles[context.client_id]
            
            # Record client response
            response_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='client',
                content=response,
                response_time=response_time,
                comprehension_indicators=await self._analyze_response_indicators(response, context)
            )
            
            self.session_history[session_id].append(response_turn)
            
            # Assess comprehension
            assessment = await self.comprehension_validator.assess_comprehension(
                question=session['current_question'],
                response=response,
                expected_concepts=session.get('expected_concepts', []),
                context=context
            )
            
            session['comprehension_history'].append(assessment)
            session['question_count'] += 1
            
            # Update overall comprehension
            session['overall_comprehension'] = await self._calculate_overall_comprehension(session['comprehension_history'])
            
            # Determine next action based on comprehension level
            next_action = await self._determine_next_action(assessment, session, learning_profile)
            
            if next_action['action'] == 'continue_questioning':
                return await self._continue_questioning(session_id, assessment, next_action)
            elif next_action['action'] == 'remediation':
                return await self._provide_remediation(session_id, assessment, next_action)
            elif next_action['action'] == 'complete':
                return await self._complete_session(session_id, assessment)
            elif next_action['action'] == 'escalate':
                return await self._escalate_to_human(session_id, assessment)
            else:
                return {
                    'success': False,
                    'error': 'Unknown next action',
                    'session_id': session_id
                }
                
        except Exception as e:
            self.logger.error(f"Error processing client response: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a dialogue session."""
        try:
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': 'Session not found'
                }
            
            session = self.active_sessions[session_id]
            
            return {
                'success': True,
                'session_id': session_id,
                'state': session['state'].value,
                'progress': {
                    'questions_asked': session['question_count'],
                    'max_questions': session['max_questions'],
                    'overall_comprehension': session['overall_comprehension'],
                    'remediation_attempts': session['remediation_attempts']
                },
                'duration': (datetime.now(timezone.utc) - session['start_time']).total_seconds(),
                'comprehension_trend': [a.comprehension_score for a in session['comprehension_history']],
                'current_state': session['state'].value
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session status: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _get_or_create_learning_profile(self, client_id: str) -> LearningProfile:
        """Get existing learning profile or create a new one."""
        # In production, this would load from database
        # For now, create a default profile
        return LearningProfile(
            client_id=client_id,
            learning_style=LearningStyle.MULTIMODAL,
            preferred_explanation_length="medium",
            concept_mastery={},
            learning_pace="medium",
            attention_span=15,  # minutes
            previous_sessions=[],
            strengths=[],
            challenges=[],
            progress_over_time={},
            adaptive_parameters={}
        )
    
    async def _extract_key_concepts(self, text: str, context: DialogueContext) -> List[str]:
        """Extract key concepts from the explanation text."""
        try:
            analysis = await self.nlp_processor.analyze_text(text, context.language)
            concepts = [c['concept'] for c in analysis.get('concepts', [])]
            
            # Add domain-specific concepts
            financial_concepts = self.nlp_processor.financial_concepts.get(context.language, {})
            for concept in financial_concepts.keys():
                if concept.lower() in text.lower():
                    concepts.append(concept)
            
            return list(set(concepts))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error extracting concepts: {str(e)}")
            return []
    
    async def _add_multimodal_elements(self, content: str, learning_profile: LearningProfile, context: DialogueContext) -> Dict[str, Any]:
        """Add multimodal elements based on learning style."""
        elements = {}
        
        if learning_profile.learning_style == LearningStyle.VISUAL:
            # Add visual aids
            elements['visual_aids'] = {
                'charts': await self._suggest_charts(content, context),
                'diagrams': await self._suggest_diagrams(content, context),
                'infographics': await self._suggest_infographics(content, context)
            }
        
        elif learning_profile.learning_style == LearningStyle.AUDITORY:
            # Add audio elements
            elements['audio'] = {
                'text_to_speech': True,
                'pronunciation_guide': await self._create_pronunciation_guide(content, context),
                'audio_examples': await self._suggest_audio_examples(content, context)
            }
        
        elif learning_profile.learning_style == LearningStyle.KINESTHETIC:
            # Add interactive elements
            elements['interactive'] = {
                'simulations': await self._suggest_simulations(content, context),
                'calculators': await self._suggest_calculators(content, context),
                'scenarios': await self._create_interactive_scenarios(content, context)
            }
        
        return elements
    
    async def _suggest_charts(self, content: str, context: DialogueContext) -> List[str]:
        """Suggest relevant charts for visual learners."""
        # Simplified chart suggestions
        chart_keywords = {
            'rente': ['interest_rate_chart', 'payment_schedule'],
            'aflossing': ['amortization_schedule', 'payment_breakdown'],
            'hypotheek': ['mortgage_comparison', 'cost_breakdown']
        }
        
        suggestions = []
        content_lower = content.lower()
        
        for keyword, charts in chart_keywords.items():
            if keyword in content_lower:
                suggestions.extend(charts)
        
        return suggestions
    
    async def _suggest_diagrams(self, content: str, context: DialogueContext) -> List[str]:
        """Suggest relevant diagrams."""
        return ['mortgage_process_flow', 'decision_tree', 'concept_map']
    
    async def _suggest_infographics(self, content: str, context: DialogueContext) -> List[str]:
        """Suggest relevant infographics."""
        return ['mortgage_types_comparison', 'cost_factors', 'timeline']
    
    async def _create_pronunciation_guide(self, content: str, context: DialogueContext) -> Dict[str, str]:
        """Create pronunciation guide for difficult terms."""
        # Simplified pronunciation guide
        dutch_terms = {
            'hypotheek': 'HI-po-teek',
            'aflossing': 'AF-los-sing',
            'eigenwoningforfait': 'EI-gen-wo-ning-for-FAIT'
        }
        
        guide = {}
        content_lower = content.lower()
        
        for term, pronunciation in dutch_terms.items():
            if term in content_lower:
                guide[term] = pronunciation
        
        return guide
    
    async def _suggest_audio_examples(self, content: str, context: DialogueContext) -> List[str]:
        """Suggest relevant audio examples."""
        return ['expert_explanation', 'client_testimonial', 'case_study_narration']
    
    async def _suggest_simulations(self, content: str, context: DialogueContext) -> List[str]:
        """Suggest interactive simulations."""
        return ['mortgage_calculator', 'interest_rate_simulator', 'payment_scheduler']
    
    async def _suggest_calculators(self, content: str, context: DialogueContext) -> List[str]:
        """Suggest relevant calculators."""
        return ['affordability_calculator', 'interest_calculator', 'ltv_calculator']
    
    async def _create_interactive_scenarios(self, content: str, context: DialogueContext) -> List[Dict[str, Any]]:
        """Create interactive scenarios."""
        return [
            {
                'name': 'mortgage_decision',
                'description': 'Interactive mortgage decision scenario',
                'parameters': {'house_price': 400000, 'income': 60000}
            }
        ]
    
    def _estimate_response_time(self, learning_profile: LearningProfile) -> int:
        """Estimate expected response time based on learning profile."""
        base_time = 60  # 1 minute base
        
        if learning_profile.learning_pace == 'fast':
            return int(base_time * 0.7)
        elif learning_profile.learning_pace == 'slow':
            return int(base_time * 1.5)
        else:
            return base_time
    
    async def _analyze_response_indicators(self, response: str, context: DialogueContext) -> Dict[str, Any]:
        """Analyze response for comprehension indicators."""
        try:
            analysis = await self.nlp_processor.analyze_text(response, context.language)
            
            return {
                'confidence_level': self._assess_response_confidence(analysis),
                'uncertainty_markers': self._count_uncertainty_markers(analysis),
                'comprehension_signals': self._identify_comprehension_signals(analysis),
                'confusion_indicators': self._identify_confusion_indicators(analysis)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing response indicators: {str(e)}")
            return {}
    
    def _assess_response_confidence(self, analysis: Dict[str, Any]) -> float:
        """Assess confidence level from response analysis."""
        # Simple confidence assessment
        confidence = 0.5  # Base confidence
        
        discourse_markers = analysis.get('discourse_markers', [])
        
        if 'certainty' in discourse_markers:
            confidence += 0.3
        if 'uncertainty' in discourse_markers:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _count_uncertainty_markers(self, analysis: Dict[str, Any]) -> int:
        """Count uncertainty markers in response."""
        discourse_markers = analysis.get('discourse_markers', [])
        return discourse_markers.count('uncertainty')
    
    def _identify_comprehension_signals(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify signals of good comprehension."""
        signals = []
        discourse_markers = analysis.get('discourse_markers', [])
        
        if 'understanding' in discourse_markers:
            signals.append('explicit_understanding')
        if 'agreement' in discourse_markers:
            signals.append('agreement')
        if len(analysis.get('concepts', [])) > 2:
            signals.append('concept_integration')
        
        return signals
    
    def _identify_confusion_indicators(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify indicators of confusion."""
        indicators = []
        discourse_markers = analysis.get('discourse_markers', [])
        
        if 'confusion' in discourse_markers:
            indicators.append('explicit_confusion')
        if 'uncertainty' in discourse_markers:
            indicators.append('uncertainty')
        
        return indicators
    
    async def _calculate_overall_comprehension(self, assessments: List[ComprehensionAssessment]) -> float:
        """Calculate overall comprehension from all assessments."""
        if not assessments:
            return 0.0
        
        # Weight recent assessments more heavily
        weights = [1.0 + 0.1 * i for i in range(len(assessments))]
        weighted_scores = [a.comprehension_score * w for a, w in zip(assessments, weights)]
        
        return sum(weighted_scores) / sum(weights)
    
    async def _determine_next_action(self, assessment: ComprehensionAssessment, session: Dict[str, Any], learning_profile: LearningProfile) -> Dict[str, Any]:
        """Determine the next action based on comprehension assessment."""
        comprehension_score = assessment.comprehension_score
        question_count = session['question_count']
        max_questions = session['max_questions']
        remediation_attempts = session['remediation_attempts']
        max_remediation = session['max_remediation_attempts']
        
        # Decision logic
        if comprehension_score >= 0.8 and question_count >= 3:
            return {'action': 'complete', 'reason': 'high_comprehension_achieved'}
        
        elif comprehension_score < 0.4 and remediation_attempts < max_remediation:
            return {'action': 'remediation', 'reason': 'low_comprehension_needs_help'}
        
        elif comprehension_score < 0.4 and remediation_attempts >= max_remediation:
            return {'action': 'escalate', 'reason': 'persistent_comprehension_issues'}
        
        elif question_count >= max_questions:
            if session['overall_comprehension'] >= 0.6:
                return {'action': 'complete', 'reason': 'max_questions_adequate_comprehension'}
            else:
                return {'action': 'escalate', 'reason': 'max_questions_inadequate_comprehension'}
        
        else:
            return {'action': 'continue_questioning', 'reason': 'continue_assessment'}
    
    async def _continue_questioning(self, session_id: str, assessment: ComprehensionAssessment, next_action: Dict[str, Any]) -> Dict[str, Any]:
        """Continue with the next question."""
        try:
            session = self.active_sessions[session_id]
            context = session['context']
            learning_profile = self.learning_profiles[context.client_id]
            
            # Determine comprehension level for adaptive questioning
            comprehension_level = self._score_to_comprehension_level(assessment.comprehension_score)
            
            # Generate next question with increased difficulty or remediation
            next_question = await self.question_generator.generate_question(
                topic=context.topic,
                question_type=self._select_question_type(assessment, session),
                comprehension_level=comprehension_level,
                learning_profile=learning_profile,
                context=context,
                previous_responses=self.session_history[session_id]
            )
            
            session['current_question'] = next_question
            session['state'] = DialogueState.COMPREHENSION_TESTING
            
            # Create question turn
            question_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=next_question,
                intent='comprehension_question',
                multimodal_elements=await self._add_multimodal_elements(next_question, learning_profile, context)
            )
            
            self.session_history[session_id].append(question_turn)
            
            return {
                'success': True,
                'session_id': session_id,
                'state': DialogueState.COMPREHENSION_TESTING.value,
                'question': next_question,
                'multimodal_elements': question_turn.multimodal_elements,
                'assessment_feedback': self._create_assessment_feedback(assessment, context),
                'progress': {
                    'current_step': session['question_count'] + 1,
                    'total_steps': session['max_questions'],
                    'comprehension_score': assessment.comprehension_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error continuing questioning: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def _provide_remediation(self, session_id: str, assessment: ComprehensionAssessment, next_action: Dict[str, Any]) -> Dict[str, Any]:
        """Provide remediation for comprehension gaps."""
        try:
            session = self.active_sessions[session_id]
            context = session['context']
            learning_profile = self.learning_profiles[context.client_id]
            
            session['remediation_attempts'] += 1
            session['state'] = DialogueState.REMEDIATION
            
            # Create remediation explanation
            remediation_content = await self._create_remediation_content(assessment, context, learning_profile)
            
            remediation_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=remediation_content,
                intent='remediation',
                multimodal_elements=await self._add_multimodal_elements(remediation_content, learning_profile, context)
            )
            
            self.session_history[session_id].append(remediation_turn)
            
            # Generate follow-up question to test improved understanding
            follow_up_question = await self.question_generator.generate_question(
                topic=context.topic,
                question_type=QuestionType.FACTUAL_RECALL,  # Start simple after remediation
                comprehension_level=ComprehensionLevel.ADEQUATE,
                learning_profile=learning_profile,
                context=context
            )
            
            session['current_question'] = follow_up_question
            
            question_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=follow_up_question,
                intent='follow_up_question'
            )
            
            self.session_history[session_id].append(question_turn)
            
            return {
                'success': True,
                'session_id': session_id,
                'state': DialogueState.REMEDIATION.value,
                'remediation': remediation_content,
                'follow_up_question': follow_up_question,
                'multimodal_elements': remediation_turn.multimodal_elements,
                'remediation_attempt': session['remediation_attempts'],
                'max_attempts': session['max_remediation_attempts']
            }
            
        except Exception as e:
            self.logger.error(f"Error providing remediation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def _complete_session(self, session_id: str, assessment: ComprehensionAssessment) -> Dict[str, Any]:
        """Complete the dialogue session."""
        try:
            session = self.active_sessions[session_id]
            context = session['context']
            
            session['state'] = DialogueState.COMPLETED
            session['end_time'] = datetime.now(timezone.utc)
            
            # Generate session summary
            summary = await self._generate_session_summary(session_id)
            
            # Update learning profile
            await self._update_learning_profile(context.client_id, session, session['comprehension_history'])
            
            completion_message = await self._create_completion_message(summary, context)
            
            completion_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=completion_message,
                intent='session_completion'
            )
            
            self.session_history[session_id].append(completion_turn)
            
            return {
                'success': True,
                'session_id': session_id,
                'state': DialogueState.COMPLETED.value,
                'completion_message': completion_message,
                'session_summary': summary,
                'final_comprehension_score': session['overall_comprehension'],
                'recommendations': summary.get('recommendations', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error completing session: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def _escalate_to_human(self, session_id: str, assessment: ComprehensionAssessment) -> Dict[str, Any]:
        """Escalate session to human advisor."""
        try:
            session = self.active_sessions[session_id]
            context = session['context']
            
            session['state'] = DialogueState.FAILED
            
            escalation_summary = await self._create_escalation_summary(session_id, assessment)
            
            escalation_message = (
                "Ik zie dat u moeite heeft met dit onderwerp. Laat me een van onze adviseurs erbij halen "
                "die u persoonlijk kan helpen." if context.language == 'nl' else
                "I can see you're having difficulty with this topic. Let me bring in one of our advisors "
                "who can help you personally."
            )
            
            escalation_turn = DialogueTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                speaker='system',
                content=escalation_message,
                intent='escalation'
            )
            
            self.session_history[session_id].append(escalation_turn)
            
            return {
                'success': True,
                'session_id': session_id,
                'state': DialogueState.FAILED.value,
                'escalation_message': escalation_message,
                'escalation_summary': escalation_summary,
                'requires_human_intervention': True,
                'advisor_briefing': escalation_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error escalating to human: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def _score_to_comprehension_level(self, score: float) -> ComprehensionLevel:
        """Convert numeric score to comprehension level enum."""
        if score >= 0.9:
            return ComprehensionLevel.EXCELLENT
        elif score >= 0.75:
            return ComprehensionLevel.GOOD
        elif score >= 0.6:
            return ComprehensionLevel.ADEQUATE
        elif score >= 0.4:
            return ComprehensionLevel.POOR
        else:
            return ComprehensionLevel.INADEQUATE
    
    def _select_question_type(self, assessment: ComprehensionAssessment, session: Dict[str, Any]) -> QuestionType:
        """Select appropriate question type based on assessment and session state."""
        question_count = session['question_count']
        comprehension_score = assessment.comprehension_score
        
        # Progressive question types
        if question_count <= 2:
            return QuestionType.FACTUAL_RECALL
        elif question_count <= 4:
            return QuestionType.CONCEPTUAL_UNDERSTANDING
        elif comprehension_score >= 0.7:
            return QuestionType.APPLICATION
        elif comprehension_score >= 0.5:
            return QuestionType.SCENARIO_BASED
        else:
            return QuestionType.FACTUAL_RECALL  # Go back to basics
    
    def _create_assessment_feedback(self, assessment: ComprehensionAssessment, context: DialogueContext) -> str:
        """Create feedback message based on assessment."""
        lang = context.language
        score = assessment.comprehension_score
        
        if score >= 0.8:
            return "Uitstekend! U toont een goed begrip van dit onderwerp." if lang == 'nl' else "Excellent! You show a good understanding of this topic."
        elif score >= 0.6:
            return "Goed gedaan. Laten we nog wat dieper ingaan op enkele aspecten." if lang == 'nl' else "Well done. Let's explore some aspects in more depth."
        elif score >= 0.4:
            return "U bent op de goede weg. Laten we samen nog wat punten verhelderen." if lang == 'nl' else "You're on the right track. Let's clarify some points together."
        else:
            return "Laten we dit stap voor stap doornemen om het duidelijker te maken." if lang == 'nl' else "Let's go through this step by step to make it clearer."
    
    async def _create_remediation_content(self, assessment: ComprehensionAssessment, context: DialogueContext, learning_profile: LearningProfile) -> str:
        """Create personalized remediation content."""
        lang = context.language
        gaps = assessment.knowledge_gaps
        
        if not gaps:
            return "Laten we het concept nog eens bekijken vanuit een andere invalshoek." if lang == 'nl' else "Let's look at the concept from another angle."
        
        # Create targeted remediation for knowledge gaps
        remediation = []
        
        if lang == 'nl':
            remediation.append("Ik merk dat er nog wat onduidelijkheden zijn. Laat me het anders uitleggen:")
            for gap in gaps[:2]:  # Focus on top 2 gaps
                remediation.append(f"• {gap}: {self._get_simple_explanation(gap, lang)}")
        else:
            remediation.append("I notice there are still some unclear points. Let me explain it differently:")
            for gap in gaps[:2]:
                remediation.append(f"• {gap}: {self._get_simple_explanation(gap, lang)}")
        
        return "\n".join(remediation)
    
    def _get_simple_explanation(self, concept: str, language: str) -> str:
        """Get simple explanation for a concept."""
        # Simplified explanations database
        explanations = {
            'nl': {
                'hypotheek': 'Een hypotheek is geld dat u leent van de bank om een huis te kopen',
                'rente': 'Rente zijn de kosten die u betaalt voor het lenen van geld',
                'aflossing': 'Aflossing is het terugbetalen van de geleende hypotheek',
                'nhg': 'NHG is een verzekering die de bank beschermt als u niet kunt betalen'
            },
            'en': {
                'mortgage': 'A mortgage is money you borrow from the bank to buy a house',
                'interest': 'Interest is the cost you pay for borrowing money',
                'repayment': 'Repayment is paying back the borrowed mortgage',
                'guarantee': 'NHG is insurance that protects the bank if you cannot pay'
            }
        }
        
        concept_explanations = explanations.get(language, explanations['en'])
        return concept_explanations.get(concept, f"Dit is een belangrijk concept in hypotheken" if language == 'nl' else "This is an important concept in mortgages")
    
    async def _generate_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive session summary."""
        session = self.active_sessions[session_id]
        history = self.session_history[session_id]
        
        duration = (session.get('end_time', datetime.now(timezone.utc)) - session['start_time']).total_seconds()
        
        summary = {
            'session_id': session_id,
            'duration_seconds': duration,
            'questions_asked': session['question_count'],
            'remediation_attempts': session['remediation_attempts'],
            'final_comprehension_score': session['overall_comprehension'],
            'comprehension_trend': [a.comprehension_score for a in session['comprehension_history']],
            'topics_covered': [session['context'].topic],
            'completion_status': session['state'].value,
            'total_turns': len(history),
            'recommendations': await self._generate_recommendations(session)
        }
        
        return summary
    
    async def _generate_recommendations(self, session: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on session performance."""
        recommendations = []
        context = session['context']
        lang = context.language
        
        overall_comprehension = session['overall_comprehension']
        
        if overall_comprehension >= 0.8:
            recommendations.extend([
                "U toont uitstekend begrip. U kunt doorgaan naar geavanceerdere onderwerpen." if lang == 'nl'
                else "You show excellent understanding. You can proceed to more advanced topics.",
                "Overweeg aanvullende financiële educatie voor nog betere besluitvorming." if lang == 'nl'
                else "Consider additional financial education for even better decision-making."
            ])
        elif overall_comprehension >= 0.6:
            recommendations.extend([
                "Goed gedaan! Herhaal de belangrijkste concepten nog eens." if lang == 'nl'
                else "Well done! Review the key concepts once more.",
                "Vraag om voorbeelden uit de praktijk voor beter begrip." if lang == 'nl'
                else "Ask for practical examples for better understanding."
            ])
        else:
            recommendations.extend([
                "Meer oefening met de basisconcepten is aan te raden." if lang == 'nl'
                else "More practice with basic concepts is recommended.",
                "Overweeg een persoonlijke sessie met een adviseur." if lang == 'nl'
                else "Consider a personal session with an advisor."
            ])
        
        return recommendations
    
    async def _create_completion_message(self, summary: Dict[str, Any], context: DialogueContext) -> str:
        """Create completion message for the session."""
        lang = context.language
        score = summary['final_comprehension_score']
        
        if lang == 'nl':
            if score >= 0.8:
                message = f"Gefeliciteerd! U heeft een uitstekend begrip getoond (score: {score:.1%}). "
                message += "U bent goed voorbereid om verder te gaan met uw hypotheekaanvraag."
            elif score >= 0.6:
                message = f"Goed gedaan! U heeft een adequate begrip getoond (score: {score:.1%}). "
                message += "Met wat extra aandacht voor enkele punten bent u klaar om door te gaan."
            else:
                message = f"We hebben vooruitgang geboekt (score: {score:.1%}), maar "
                message += "ik raad aan om nog eens te praten met een van onze adviseurs."
        else:
            if score >= 0.8:
                message = f"Congratulations! You've shown excellent understanding (score: {score:.1%}). "
                message += "You're well prepared to proceed with your mortgage application."
            elif score >= 0.6:
                message = f"Well done! You've shown adequate understanding (score: {score:.1%}). "
                message += "With some extra attention to a few points, you're ready to proceed."
            else:
                message = f"We've made progress (score: {score:.1%}), but "
                message += "I recommend speaking with one of our advisors again."
        
        return message
    
    async def _create_escalation_summary(self, session_id: str, assessment: ComprehensionAssessment) -> Dict[str, Any]:
        """Create summary for human advisor escalation."""
        session = self.active_sessions[session_id]
        context = session['context']
        
        return {
            'client_id': context.client_id,
            'session_id': session_id,
            'topic': context.topic,
            'language': context.language,
            'comprehension_scores': [a.comprehension_score for a in session['comprehension_history']],
            'knowledge_gaps': assessment.knowledge_gaps,
            'remediation_attempts': session['remediation_attempts'],
            'persistent_difficulties': self._identify_persistent_difficulties(session['comprehension_history']),
            'recommended_approach': self._recommend_human_approach(assessment, context),
            'session_duration': (datetime.now(timezone.utc) - session['start_time']).total_seconds(),
            'client_responses': [turn.content for turn in self.session_history[session_id] if turn.speaker == 'client']
        }
    
    def _identify_persistent_difficulties(self, assessments: List[ComprehensionAssessment]) -> List[str]:
        """Identify concepts that consistently cause difficulty."""
        gap_counts = defaultdict(int)
        
        for assessment in assessments:
            for gap in assessment.knowledge_gaps:
                gap_counts[gap] += 1
        
        # Return gaps that appear in more than half the assessments
        threshold = len(assessments) / 2
        return [gap for gap, count in gap_counts.items() if count > threshold]
    
    def _recommend_human_approach(self, assessment: ComprehensionAssessment, context: DialogueContext) -> List[str]:
        """Recommend approach for human advisor."""
        recommendations = []
        lang = context.language
        
        if assessment.comprehension_score < 0.3:
            recommendations.extend([
                "Start met zeer eenvoudige uitleg en veel voorbeelden" if lang == 'nl'
                else "Start with very simple explanations and many examples",
                "Gebruik visuele hulpmiddelen" if lang == 'nl'
                else "Use visual aids"
            ])
        
        if assessment.knowledge_gaps:
            gap_text = ', '.join(assessment.knowledge_gaps[:3])
            recommendations.append(
                f"Focus op deze concepten: {gap_text}" if lang == 'nl'
                else f"Focus on these concepts: {gap_text}"
            )
        
        return recommendations
    
    async def _update_learning_profile(self, client_id: str, session: Dict[str, Any], assessments: List[ComprehensionAssessment]):
        """Update client learning profile based on session results."""
        try:
            profile = self.learning_profiles[client_id]
            
            # Update concept mastery
            for assessment in assessments:
                for concept in assessment.identified_concepts:
                    if concept not in profile.concept_mastery:
                        profile.concept_mastery[concept] = assessment.comprehension_score
                    else:
                        # Running average
                        current = profile.concept_mastery[concept]
                        profile.concept_mastery[concept] = (current + assessment.comprehension_score) / 2
            
            # Update learning pace based on response times
            avg_response_time = statistics.mean([
                turn.response_time for turn in self.session_history[session['context'].session_id]
                if turn.speaker == 'client' and turn.response_time is not None
            ])
            
            if avg_response_time:
                if avg_response_time < 30:  # Quick responses
                    profile.learning_pace = 'fast'
                elif avg_response_time > 120:  # Slow responses
                    profile.learning_pace = 'slow'
                else:
                    profile.learning_pace = 'medium'
            
            # Add session to history
            profile.previous_sessions.append(session['context'].session_id)
            
            # Update progress tracking
            topic = session['context'].topic
            if topic not in profile.progress_over_time:
                profile.progress_over_time[topic] = []
            
            profile.progress_over_time[topic].append(session['overall_comprehension'])
            
            # Update adaptive parameters
            profile.adaptive_parameters.update({
                'last_session_performance': session['overall_comprehension'],
                'preferred_question_types': self._identify_preferred_question_types(assessments),
                'effective_remediation_strategies': self._identify_effective_strategies(session)
            })
            
        except Exception as e:
            self.logger.error(f"Error updating learning profile: {str(e)}")
    
    def _identify_preferred_question_types(self, assessments: List[ComprehensionAssessment]) -> List[str]:
        """Identify question types that work best for this client."""
        type_performance = defaultdict(list)
        
        for assessment in assessments:
            type_performance[assessment.question_type.value].append(assessment.comprehension_score)
        
        # Calculate average performance per type
        avg_performance = {
            q_type: statistics.mean(scores) for q_type, scores in type_performance.items()
        }
        
        # Return types with above-average performance
        overall_avg = statistics.mean([score for scores in type_performance.values() for score in scores])
        return [q_type for q_type, avg in avg_performance.items() if avg > overall_avg]
    
    def _identify_effective_strategies(self, session: Dict[str, Any]) -> List[str]:
        """Identify effective remediation strategies from the session."""
        strategies = []
        
        if session['remediation_attempts'] > 0 and session['overall_comprehension'] > 0.6:
            strategies.append('step_by_step_explanation')
            strategies.append('multiple_examples')
        
        # This would be more sophisticated in production
        return strategies


# Factory function for creating dialogue manager
def create_explain_back_dialogue_manager() -> ExplainBackDialogueManager:
    """Factory function to create and initialize ExplainBackDialogueManager."""
    return ExplainBackDialogueManager()


# Convenience functions for integration with other systems
async def start_explanation_validation(topic: str, explanation: str, context: DialogueContext) -> Dict[str, Any]:
    """
    Convenience function to start an explain-back dialogue session.
    
    Args:
        topic: The topic being explained
        explanation: The initial explanation to validate understanding of
        context: Dialogue context with client and session information
        
    Returns:
        Dictionary with session initialization results
    """
    manager = create_explain_back_dialogue_manager()
    context.topic = topic
    return await manager.start_dialogue_session(context, explanation)


async def process_client_explanation(session_id: str, client_response: str, response_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Convenience function to process client responses in an active session.
    
    Args:
        session_id: Active session identifier
        client_response: Client's response to comprehension question
        response_time: Time taken to respond (seconds)
        
    Returns:
        Dictionary with response processing results and next steps
    """
    manager = create_explain_back_dialogue_manager()
    return await manager.process_client_response(session_id, client_response, response_time)


async def get_dialogue_session_status(session_id: str) -> Dict[str, Any]:
    """
    Get the current status of a dialogue session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary with current session status and progress
    """
    manager = create_explain_back_dialogue_manager()
    return await manager.get_session_status(session_id)
