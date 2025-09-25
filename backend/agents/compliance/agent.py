"""
Compliance & Plain-Language Advisor Agent Implementation

This module implements the core compliance agent that ensures mortgage advice
meets AFM regulatory requirements and uses plain language understandable to consumers.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
import hashlib
from functools import lru_cache

from anthropic import Anthropic
from openai import OpenAI
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import cachetools

from ..config import settings
from ..database import get_db_connection
from ..utils.text_processor import TextProcessor
from ..utils.regulation_store import RegulationStore


class AdvancedTextSimplifier:
    """
    Advanced NLP-based text simplification using Hugging Face transformers.
    Provides production-grade text simplification with caching, async processing,
    and comprehensive readability assessment.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = 0 if torch.cuda.is_available() else -1
        self.cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL

        # Initialize T5 model for text simplification
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.model.to(self.device)
            self.logger.info("T5-small model loaded successfully for text simplification")
        except Exception as e:
            self.logger.error(f"Failed to load T5 model: {str(e)}")
            raise

    async def simplify_text_async(self, text: str, max_length: int = 512) -> str:
        """
        Asynchronously simplify text using T5 model.

        Args:
            text: Input text to simplify
            max_length: Maximum length for model input

        Returns:
            Simplified text
        """
        try:
            # Check cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.cache:
                self.logger.info("Returning cached simplified text")
                return self.cache[cache_key]

            # Run model inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            simplified = await loop.run_in_executor(None, self._simplify_text_sync, text, max_length)

            # Cache result
            self.cache[cache_key] = simplified
            self.logger.info("Text simplified and cached successfully")
            return simplified

        except Exception as e:
            self.logger.error(f"Error in async text simplification: {str(e)}")
            raise

    def _simplify_text_sync(self, text: str, max_length: int) -> str:
        """Synchronous text simplification using T5."""
        try:
            # Prepare input for T5
            input_text = f"simplify: {text[:max_length]}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate simplified text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )

            simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return simplified

        except Exception as e:
            self.logger.error(f"Error in synchronous text simplification: {str(e)}")
            raise

    def assess_readability_advanced(self, text: str) -> Dict[str, Any]:
        """
        Advanced readability assessment using multiple metrics for CEFR B1 compliance.

        Returns:
            Dictionary with various readability scores and CEFR B1 assessment
        """
        try:
            metrics = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'difficult_words': textstat.difficult_words(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'gunning_fog': textstat.gunning_fog(text),
                'text_standard': textstat.text_standard(text, float_output=True)
            }

            # CEFR B1 assessment (roughly grade 6-8 level)
            cefr_b1_score = self._calculate_cefr_b1_score(metrics)
            is_cefr_b1_compliant = cefr_b1_score >= 0.7  # Threshold for B1

            result = {
                'metrics': metrics,
                'cefr_b1_score': cefr_b1_score,
                'is_cefr_b1_compliant': is_cefr_b1_compliant,
                'assessment_timestamp': datetime.utcnow().isoformat()
            }

            self.logger.info(f"Advanced readability assessment completed. CEFR B1 compliant: {is_cefr_b1_compliant}")
            return result

        except Exception as e:
            self.logger.error(f"Error in advanced readability assessment: {str(e)}")
            raise

    def _calculate_cefr_b1_score(self, metrics: Dict[str, float]) -> float:
        """Calculate CEFR B1 compliance score based on multiple metrics."""
        # CEFR B1 typically corresponds to:
        # Flesch Reading Ease: 60-70
        # Flesch-Kincaid Grade: 6-8
        # SMOG: 8-10

        score = 0.0
        total_weight = 0.0

        # Flesch Reading Ease (weight: 0.3)
        if 60 <= metrics['flesch_reading_ease'] <= 70:
            score += 0.3
        elif 50 <= metrics['flesch_reading_ease'] <= 80:
            score += 0.2
        total_weight += 0.3

        # Flesch-Kincaid Grade (weight: 0.3)
        if 6 <= metrics['flesch_kincaid_grade'] <= 8:
            score += 0.3
        elif 5 <= metrics['flesch_kincaid_grade'] <= 9:
            score += 0.2
        total_weight += 0.3

        # SMOG Index (weight: 0.2)
        if 8 <= metrics['smog_index'] <= 10:
            score += 0.2
        elif 7 <= metrics['smog_index'] <= 11:
            score += 0.15
        total_weight += 0.2

        # Gunning Fog (weight: 0.2)
        if 7 <= metrics['gunning_fog'] <= 9:
            score += 0.2
        elif 6 <= metrics['gunning_fog'] <= 10:
            score += 0.15
        total_weight += 0.2

        return score / total_weight if total_weight > 0 else 0.0


class ComplianceAgent:
    """
    AFM-certified Mortgage Advisor Agent that ensures advice quality and plain-language compliance.

    The agent performs four main functions:
    1. Regulation verification against latest AFM Wft rules
    2. Advanced language simplification to CEFR B1 level (short sentences, active voice, optimized vocabulary)
    3. Required disclosure highlighting (fees, risks, adviser remuneration)
    4. User comprehension validation through explain-back dialogue
    """

    def __init__(self):
        """Initialize the Compliance Agent with required components."""
        # Determine which LLM provider to use
        if settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != 'your_anthropic_api_key_here':
            self.provider = 'anthropic'
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.logger.info("Using Anthropic Claude for LLM operations")
        else:
            self.provider = 'openai'
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.logger.info("Using OpenAI for LLM operations")

        self.text_processor = TextProcessor()
        self.regulation_store = RegulationStore()
        self.advanced_simplifier = AdvancedTextSimplifier()
        self.logger = logging.getLogger(__name__)

        # CEFR B1 readability targets
        self.max_sentence_length = 12  # words per sentence
        self.target_readability_score = 60  # Flesch Reading Ease (B1 level)

        # Required AFM disclosures
        self.required_disclosures = {
            'fees': ['advisory fee', 'arrangement fee', 'valuation fee', 'legal fee'],
            'risks': ['interest rate risk', 'early repayment charges', 'negative equity', 'payment difficulties'],
            'remuneration': ['commission structure', 'fee basis', 'conflict of interest']
        }

    async def generate_advice_draft(self, user_profile: Dict[str, Any], product_features: List[str]) -> Dict[str, Any]:
        """
        Generate initial mortgage advice draft based on user profile and product features.

        Args:
            user_profile: Dictionary containing user information (first-time buyer, amount, preferences)
            product_features: List of mortgage product features to explain

        Returns:
            Dictionary containing advice draft and metadata
        """
        try:
            # Build user context
            user_context = self._build_user_context(user_profile)

            # Generate advice using Claude
            prompt = self._build_advice_prompt(user_context, product_features)
            response = await self._call_claude(prompt)

            advice_draft = {
                'content': response,
                'user_profile': user_profile,
                'product_features': product_features,
                'generated_at': datetime.utcnow().isoformat(),
                'readability_score': textstat.flesch_reading_ease(response),
                'compliance_check_required': True
            }

            self.logger.info(f"Generated advice draft for user profile: {user_profile.get('buyer_type', 'unknown')}")
            return advice_draft

        except Exception as e:
            self.logger.error(f"Error generating advice draft: {str(e)}")
            raise

    async def check_compliance(self, advice_text: str) -> Dict[str, Any]:
        """
        Check advice text for AFM Wft compliance and regulatory adherence.

        Args:
            advice_text: The advice text to check for compliance

        Returns:
            Dictionary containing compliance results and any issues found
        """
        try:
            compliance_issues = []
            compliance_score = 100

            # Check for required disclosures
            disclosure_check = self._check_required_disclosures(advice_text)
            if not disclosure_check['passed']:
                compliance_issues.extend(disclosure_check['missing'])
                compliance_score -= len(disclosure_check['missing']) * 10

            # Check language complexity
            readability_check = self._check_readability(advice_text)
            if not readability_check['passed']:
                compliance_issues.extend(readability_check['issues'])
                compliance_score -= readability_check['penalty']

            # Check against specific regulations
            regulation_check = await self._check_regulation_compliance(advice_text)
            if not regulation_check['passed']:
                compliance_issues.extend(regulation_check['violations'])
                compliance_score = max(0, compliance_score - regulation_check['penalty'])

            # Ensure positive score
            compliance_score = max(0, compliance_score)

            result = {
                'passed': len(compliance_issues) == 0,
                'score': compliance_score,
                'issues': compliance_issues,
                'disclosure_check': disclosure_check,
                'readability_check': readability_check,
                'regulation_check': regulation_check,
                'checked_at': datetime.utcnow().isoformat()
            }

            self.logger.info(f"Compliance check completed with score: {compliance_score}")
            return result

        except Exception as e:
            self.logger.error(f"Error checking compliance: {str(e)}")
            raise

    async def simplify_language(self, advice_text: str, compliance_issues: List[str]) -> Dict[str, Any]:
        """
        Advanced language simplification using NLP models to meet CEFR B1 readability requirements.

        Args:
            advice_text: Original advice text
            compliance_issues: List of compliance issues to address

        Returns:
            Dictionary containing simplified text and comprehensive improvement metrics
        """
        try:
            # Get original advanced readability assessment
            original_assessment = self.advanced_simplifier.assess_readability_advanced(advice_text)
            original_sentences = len(sent_tokenize(advice_text))

            # Use advanced NLP model for simplification
            simplified_text = await self.advanced_simplifier.simplify_text_async(advice_text)

            # Get new advanced readability assessment
            new_assessment = self.advanced_simplifier.assess_readability_advanced(simplified_text)
            new_sentences = len(sent_tokenize(simplified_text))

            # Validate simplification maintained meaning using Claude
            meaning_preserved = await self._validate_meaning_preservation(advice_text, simplified_text)

            # Calculate comprehensive improvements
            improvements = self._calculate_comprehensive_improvements(original_assessment, new_assessment)

            result = {
                'original_text': advice_text,
                'advanced_simplified_text': simplified_text,
                'original_readability_assessment': original_assessment,
                'new_readability_assessment': new_assessment,
                'comprehensive_improvements': improvements,
                'sentence_count_change': new_sentences - original_sentences,
                'meaning_preserved': meaning_preserved,
                'advanced_simplified_at': datetime.utcnow().isoformat()
            }

            self.logger.info(f"Advanced language simplification completed. CEFR B1 compliance improved: {new_assessment['is_cefr_b1_compliant']}")
            return result

        except Exception as e:
            self.logger.error(f"Error in advanced language simplification: {str(e)}")
            raise

    async def embed_explain_back(self, advice_text: str) -> Dict[str, Any]:
        """
        Embed explain-back questions to validate user understanding.

        Args:
            advice_text: The advice text to add explain-back validation to

        Returns:
            Dictionary containing advice with embedded questions
        """
        try:
            # Identify key concepts that need comprehension validation
            key_concepts = self._extract_key_concepts(advice_text)

            # Generate explain-back questions
            questions = []
            for concept in key_concepts:
                question = await self._generate_explain_back_question(concept, advice_text)
                questions.append({
                    'concept': concept,
                    'question': question,
                    'required': True
                })

            # Embed questions in the advice text
            enhanced_advice = self._embed_questions_in_text(advice_text, questions)

            result = {
                'original_advice': advice_text,
                'enhanced_advice': enhanced_advice,
                'explain_back_questions': questions,
                'key_concepts_covered': key_concepts,
                'embedded_at': datetime.utcnow().isoformat()
            }

            self.logger.info(f"Embedded {len(questions)} explain-back questions in advice")
            return result

        except Exception as e:
            self.logger.error(f"Error embedding explain-back questions: {str(e)}")
            raise

    def _build_user_context(self, user_profile: Dict[str, Any]) -> str:
        """Build user context string from profile data."""
        context_parts = []

        if 'buyer_type' in user_profile:
            context_parts.append(f"Buyer type: {user_profile['buyer_type']}")

        if 'mortgage_amount' in user_profile:
            context_parts.append(f"Mortgage amount: €{user_profile['mortgage_amount']:,}")

        if 'interest_preference' in user_profile:
            context_parts.append(f"Interest rate preference: {user_profile['interest_preference']}")

        if 'property_value' in user_profile:
            context_parts.append(f"Property value: €{user_profile['property_value']:,}")

        if 'income' in user_profile:
            context_parts.append(f"Annual income: €{user_profile['income']:,}")

        return " | ".join(context_parts)

    def _build_advice_prompt(self, user_context: str, product_features: List[str]) -> str:
        """Build the system prompt for advice generation."""
        features_text = "\n".join(f"- {feature}" for feature in product_features)

        return f"""You are an AFM-certified Mortgage Advisor Agent. Before providing advice:

1. Verify all statements against the latest AFM Wft rules.
2. Ensure language meets CEFR B1 readability: short sentences, active voice, optimized vocabulary.
3. Highlight required disclosures: fees, risks, adviser remuneration.
4. Confirm user comprehension by asking them to summarize key points.

User Profile: {user_context}

Provide mortgage advice that explains these product features clearly:
{features_text}

Requirements:
- Use plain language (max 12 words/sentence)
- Include all required AFM disclosures
- Be transparent about costs and risks
- Use advanced clear explanations
"""

    def _check_required_disclosures(self, text: str) -> Dict[str, Any]:
        """Check if required AFM disclosures are present in the text."""
        missing_disclosures = []
        text_lower = text.lower()

        for category, disclosures in self.required_disclosures.items():
            category_covered = False
            for disclosure in disclosures:
                if disclosure.lower() in text_lower:
                    category_covered = True
                    break
            if not category_covered:
                missing_disclosures.append(f"Missing {category} disclosure: {disclosures[0]}")

        return {
            'passed': len(missing_disclosures) == 0,
            'missing': missing_disclosures
        }

    def _check_readability(self, text: str) -> Dict[str, Any]:
        """Advanced readability check using multiple metrics for CEFR B1 compliance."""
        try:
            assessment = self.advanced_simplifier.assess_readability_advanced(text)
            issues = []
            penalty = 0

            # Check sentence length
            sentences = sent_tokenize(text)
            long_sentences = [s for s in sentences if len(word_tokenize(s)) > self.max_sentence_length]

            if long_sentences:
                issues.append(f"{len(long_sentences)} sentences exceed {self.max_sentence_length} words")
                penalty += len(long_sentences) * 5

            # Check CEFR B1 compliance
            if not assessment['is_cefr_b1_compliant']:
                issues.append("Text does not meet CEFR B1 readability standards")
                penalty += 20

            # Check individual metrics for additional issues
            metrics = assessment['metrics']
            if metrics['flesch_reading_ease'] < self.target_readability_score:
                issues.append(".1f")
                penalty += int((self.target_readability_score - metrics['flesch_reading_ease']) / 2)

            if metrics['flesch_kincaid_grade'] > 8:
                issues.append(".1f")
                penalty += int(metrics['flesch_kincaid_grade'] - 8) * 5

            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'penalty': penalty,
                'score': metrics['flesch_reading_ease'],
                'advanced_assessment': assessment
            }

        except Exception as e:
            self.logger.error(f"Error in advanced readability check: {str(e)}")
            # Fallback to basic check
            return self._fallback_readability_check(text)

    async def _check_regulation_compliance(self, text: str) -> Dict[str, Any]:
        """Check text against specific AFM regulations."""
        violations = []
        penalty = 0

        # Get relevant regulations from store
        relevant_regs = await self.regulation_store.get_relevant_regulations("mortgage_advice")

        for reg in relevant_regs:
            # Simple keyword matching (in production, use more sophisticated NLP)
            if reg['category'] == 'disclosure' and not self._contains_disclosure(text, reg['keywords']):
                violations.append(f"Regulation {reg['code']}: Missing required disclosure")
                penalty += 15

        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'penalty': penalty
        }

    def _contains_disclosure(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains required disclosure keywords."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def _build_simplification_prompt(self, text: str, issues: List[str]) -> str:
        """Build prompt for language simplification."""
        issues_text = "\n".join(f"- {issue}" for issue in issues)

        return f"""Simplify this mortgage advice to meet CEFR B1 readability requirements:

Original text:
{text}

Issues to address:
{issues_text}

Requirements:
- Use short sentences (max 12 words each)
- Replace complex words with optimized alternatives
- Use active voice
- Maintain all important information and disclosures
- Keep the same meaning and advice

Simplified version:"""

    async def _validate_meaning_preservation(self, original: str, simplified: str) -> bool:
        """Validate that simplification preserved the original meaning."""
        prompt = f"""Compare these two texts and determine if they convey the same mortgage advice and disclosures:

Original: {original}

Simplified: {simplified}

Do they convey the same meaning and advice? Answer only YES or NO:"""

        response = await self._call_claude(prompt)
        return response.strip().upper() == "YES"

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts that need explain-back validation."""
        # Simple extraction based on mortgage-specific keywords
        key_terms = [
            'interest rate', 'fixed rate', 'variable rate', 'mortgage term',
            'deposit', 'loan-to-value', 'debt-to-income', 'early repayment',
            'arrangement fee', 'valuation fee', 'legal fee'
        ]

        found_concepts = []
        text_lower = text.lower()

        for term in key_terms:
            if term in text_lower:
                found_concepts.append(term)

        return found_concepts[:5]  # Limit to 5 key concepts

    async def _generate_explain_back_question(self, concept: str, context: str) -> str:
        """Generate an explain-back question for a key concept."""
        prompt = f"""Generate an advanced comprehension question to check if the user understands this mortgage concept.

Concept: {concept}
Context: {context[:200]}...

Question should:
- Be advanced and clear
- Ask the user to explain the concept in their own words
- Help validate understanding

Question:"""

        return await self._call_claude(prompt)

    def _embed_questions_in_text(self, text: str, questions: List[Dict[str, Any]]) -> str:
        """Embed explain-back questions in the advice text."""
        enhanced_text = text + "\n\n--- Understanding Check ---\n\n"

        for i, q in enumerate(questions, 1):
            enhanced_text += f"{i}. {q['question']}\n\n"

        enhanced_text += "Please answer these questions to help us ensure you understand the advice."
        return enhanced_text

    def _calculate_comprehensive_improvements(self, original: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive improvements across all readability metrics."""
        improvements = {}
        original_metrics = original['metrics']
        new_metrics = new['metrics']

        for metric in original_metrics:
            if metric in new_metrics:
                improvement = new_metrics[metric] - original_metrics[metric]
                # For some metrics, lower is better (grades), for others higher (ease scores)
                if metric in ['flesch_reading_ease', 'dale_chall_readability_score']:
                    improvements[metric] = improvement  # Higher is better
                else:
                    improvements[metric] = -improvement  # Lower is better, so negate

        improvements['cefr_b1_score'] = new['cefr_b1_score'] - original['cefr_b1_score']
        improvements['overall_cefr_compliance'] = new['is_cefr_b1_compliant'] and not original['is_cefr_b1_compliant']

        return improvements

    def _fallback_readability_check(self, text: str) -> Dict[str, Any]:
        """Fallback basic readability check in case advanced assessment fails."""
        issues = []
        penalty = 0

        sentences = sent_tokenize(text)
        long_sentences = [s for s in sentences if len(word_tokenize(s)) > self.max_sentence_length]

        if long_sentences:
            issues.append(f"{len(long_sentences)} sentences exceed {self.max_sentence_length} words")
            penalty += len(long_sentences) * 5

        score = textstat.flesch_reading_ease(text)
        if score < self.target_readability_score:
            issues.append(".1f")
            penalty += int((self.target_readability_score - score) / 2)

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'penalty': penalty,
            'score': score,
            'fallback_used': True
        }

    async def _call_claude(self, prompt: str) -> str:
        """Make a call to LLM API (Claude or OpenAI) with error handling."""
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=settings.AI_MODEL,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-4",  # Use GPT-4 as equivalent to Claude
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            raise
