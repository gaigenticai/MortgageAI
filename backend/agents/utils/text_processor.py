"""
Text Processing Utilities for MortgageAI

This module provides advanced text analysis and processing utilities for:
- Linguistic analysis with spaCy
- Contextual text simplification with transformers
- Dependency parsing for sentence optimization
- Multiple readability assessment metrics
- Contextual embeddings for vocabulary simplification
- Async processing and caching
"""

import asyncio
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from functools import lru_cache
import time

import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
from cachetools import TTLCache


class TextProcessor:
    """
    Advanced text processing utilities for mortgage advice analysis using NLP techniques.

    Features:
    - CEFR level assessment with contextual embeddings
    - Multiple readability scoring metrics
    - Jargon detection using semantic similarity
    - Sentence structure optimization with dependency parsing
    - Mortgage terminology validation
    - Async processing for model inference
    - Caching mechanisms for performance
    - Comprehensive error handling and logging
    """

    def __init__(self, cache_ttl: int = 3600):
        self.logger = logging.getLogger(__name__)

        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found, downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize transformer models
        try:
            self.simplification_model = pipeline(
                "text2text-generation",
                model="facebook/bart-large-cnn",  # Can be replaced with simplification-specific model
                device=0 if torch.cuda.is_available() else -1
            )
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.logger.info("Loaded simplification model: facebook/bart-large-cnn")
        except Exception as e:
            self.logger.error(f"Failed to load simplification model: {e}")
            self.simplification_model = None

        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

        # Mortgage-specific terminology with embeddings
        self.mortgage_terms = [
            'amortization', 'capitalization', 'collateral', 'covenant', 'equity',
            'lien', 'principal', 'subordination', 'underwriting', 'variable rate',
            'mortgage', 'deposit', 'repayment', 'valuation', 'interest rate'
        ]

        # Simple word replacements for fallback
        self.simple_replacements = {
            'amortization': 'paying down the loan',
            'capitalization': 'adding to the loan amount',
            'collateral': 'property used as security',
            'covenant': 'promise in the loan agreement',
            'equity': 'ownership value in your home',
            'lien': 'legal claim on your property',
            'principal': 'the amount you borrowed',
            'subordination': 'changing priority of claims',
            'underwriting': 'loan approval process',
            'variable rate': 'changing interest rate'
        }

        # CEFR level word lists
        self.cefr_levels = {
            'A1': {'house', 'money', 'pay', 'year', 'buy', 'sell', 'bank'},
            'A2': {'interest', 'rate', 'loan', 'property', 'income', 'credit'},
            'B1': {'mortgage', 'deposit', 'repayment', 'equity', 'valuation'},
            'B2': {'amortization', 'capitalization', 'underwriting', 'covenant'},
            'C1': {'subordination', 'collateralization', 'securitization'}
        }

        # Caching
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.logger.info(f"Initialized cache with TTL: {cache_ttl}s")

    async def analyze_readability(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive readability analysis using multiple metrics and linguistic features.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with various readability metrics and linguistic features
        """
        cache_key = f"readability:{hash(text)}"
        if cache_key in self.cache:
            self.logger.debug("Returning cached readability analysis")
            return self.cache[cache_key]

        try:
            start_time = time.time()

            # Basic readability scores
            flesch_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
            smog_index = textstat.smog_index(text)
            coleman_liau = textstat.coleman_liau_index(text)
            automated_readability = textstat.automated_readability_index(text)
            dale_chall = textstat.dale_chall_readability_score(text)
            linsear_write = textstat.linsear_write_formula(text)
            gunning_fog = textstat.gunning_fog(text)

            # Linguistic analysis with spaCy
            doc = self.nlp(text)
            sentences = list(doc.sents)
            sentence_lengths = [len(sent) for sent in sentences]

            # Dependency parsing features
            avg_dependency_depth = self._calculate_avg_dependency_depth(doc)
            passive_constructions = self._count_passive_constructions(doc)
            complex_clauses = self._count_complex_clauses(doc)

            # Word analysis
            words = [token.text.lower() for token in doc if token.is_alpha]
            word_count = len(words)
            unique_words = len(set(words))

            # Stopword ratio
            stop_words = set(stopwords.words('english'))
            stopword_count = len([w for w in words if w in stop_words])
            stopword_ratio = stopword_count / word_count if word_count > 0 else 0

            # Lexical diversity
            lexical_diversity = unique_words / word_count if word_count > 0 else 0

            # CEFR level assessment
            cefr_level = await self._assess_cefr_level_async(text)

            # Syntactic complexity
            avg_words_per_sentence = word_count / len(sentences) if sentences else 0
            avg_syllables_per_word = textstat.avg_syllables_per_word(text)

            result = {
                'flesch_reading_ease': flesch_ease,
                'flesch_kincaid_grade': flesch_kincaid,
                'smog_index': smog_index,
                'coleman_liau_index': coleman_liau,
                'automated_readability_index': automated_readability,
                'dale_chall_score': dale_chall,
                'linsear_write_score': linsear_write,
                'gunning_fog_index': gunning_fog,
                'sentence_count': len(sentences),
                'avg_sentence_length': avg_words_per_sentence,
                'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
                'word_count': word_count,
                'unique_words': unique_words,
                'lexical_diversity': lexical_diversity,
                'stopword_ratio': stopword_ratio,
                'cefr_level': cefr_level,
                'avg_dependency_depth': avg_dependency_depth,
                'passive_constructions': passive_constructions,
                'complex_clauses': complex_clauses,
                'avg_syllables_per_word': avg_syllables_per_word,
                'target_met': flesch_ease >= 60 and cefr_level in ['A2', 'B1'],
                'processing_time': time.time() - start_time
            }

            self.cache[cache_key] = result
            self.logger.info(f"Completed readability analysis in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing readability: {str(e)}", exc_info=True)
            return {'error': str(e), 'processing_time': time.time() - time.time()}

    async def detect_jargon(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect mortgage-specific jargon using semantic similarity and contextual analysis.

        Args:
            text: Text to analyze for jargon

        Returns:
            List of detected jargon with suggestions and complexity scores
        """
        cache_key = f"jargon:{hash(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            detected_jargon = []
            doc = self.nlp(text)

            # Get embeddings for mortgage terms
            if self.embedding_model:
                term_embeddings = self.embedding_model.encode(self.mortgage_terms, convert_to_tensor=True)

                for token in doc:
                    if token.is_alpha and not token.is_stop:
                        word_embedding = self.embedding_model.encode([token.text], convert_to_tensor=True)
                        similarities = util.cos_sim(word_embedding, term_embeddings)[0]

                        # Find most similar term
                        max_sim_idx = torch.argmax(similarities).item()
                        max_similarity = similarities[max_sim_idx].item()

                        if max_similarity > 0.7:  # Similarity threshold
                            term = self.mortgage_terms[max_sim_idx]
                            replacement = self.simple_replacements.get(term, f"simpler term for {term}")

                            detected_jargon.append({
                                'term': token.text,
                                'matched_term': term,
                                'replacement': replacement,
                                'position': (token.idx, token.idx + len(token.text)),
                                'context': text[max(0, token.idx-20):token.idx + len(token.text)+20],
                                'similarity_score': max_similarity,
                                'complexity': self._assess_word_complexity(token)
                            })

            # Fallback to simple matching
            else:
                text_lower = text.lower()
                for term, replacement in self.simple_replacements.items():
                    if term in text_lower:
                        start = text_lower.find(term)
                        end = start + len(term)
                        detected_jargon.append({
                            'term': term,
                            'replacement': replacement,
                            'position': (start, end),
                            'context': text[max(0, start-20):end+20],
                            'similarity_score': 1.0,
                            'complexity': 'high'
                        })

            self.cache[cache_key] = detected_jargon
            return detected_jargon

        except Exception as e:
            self.logger.error(f"Error detecting jargon: {str(e)}", exc_info=True)
            return []

    async def simplify_text(self, text: str) -> Dict[str, Any]:
        """
        Simplify text using contextual transformers and linguistic analysis.

        Args:
            text: Text to simplify

        Returns:
            Dictionary with original and simplified text, metrics
        """
        cache_key = f"simplify:{hash(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            start_time = time.time()
            original_score = textstat.flesch_reading_ease(text)

            # Step 1: Replace jargon with contextual alternatives
            jargon_replaced = await self._replace_jargon_contextually(text)

            # Step 2: Optimize sentence structure
            structure_optimized = await self._optimize_sentence_structure(jargon_replaced)

            # Step 3: Apply transformer-based simplification if available
            if self.simplification_model:
                simplified = await self._apply_transformer_simplification(structure_optimized)
            else:
                simplified = structure_optimized

            # Calculate metrics
            simplified_score = textstat.flesch_reading_ease(simplified)
            jargon_count = len(await self.detect_jargon(text))

            result = {
                'original_text': text,
                'simplified_text': simplified,
                'original_score': original_score,
                'simplified_score': simplified_score,
                'improvement': simplified_score - original_score,
                'jargon_replaced': jargon_count,
                'processing_time': time.time() - start_time
            }

            self.cache[cache_key] = result
            self.logger.info(f"Completed text simplification in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error simplifying text: {str(e)}", exc_info=True)
            return {'error': str(e), 'original_text': text, 'processing_time': time.time() - time.time()}

    async def _replace_jargon_contextually(self, text: str) -> str:
        """Replace jargon using contextual embeddings."""
        if not self.embedding_model:
            # Fallback to simple replacement
            for term, replacement in self.simple_replacements.items():
                pattern = r'\b' + re.escape(term) + r'\b'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return text

        doc = self.nlp(text)
        replacements = []

        for token in doc:
            if token.text.lower() in self.simple_replacements:
                # Get context window
                start = max(0, token.i - 5)
                end = min(len(doc), token.i + 6)
                context = doc[start:end]

                # Find best replacement based on context similarity
                context_text = context.text
                context_embedding = self.embedding_model.encode([context_text], convert_to_tensor=True)

                candidates = [self.simple_replacements[token.text.lower()]]
                candidate_embeddings = self.embedding_model.encode(candidates, convert_to_tensor=True)

                similarities = util.cos_sim(context_embedding, candidate_embeddings)[0]
                best_idx = torch.argmax(similarities).item()

                replacements.append((token.idx, token.idx + len(token.text), candidates[best_idx]))

        # Apply replacements in reverse order
        for start, end, replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]

        return text

    async def _optimize_sentence_structure(self, text: str) -> str:
        """Optimize sentence structure using dependency parsing."""
        doc = self.nlp(text)
        optimized_sentences = []

        for sent in doc.sents:
            # Analyze sentence complexity
            if len(sent) > 15:
                # Try to split complex sentences
                split_sentences = self._split_complex_sentence(sent)
                optimized_sentences.extend(split_sentences)
            else:
                optimized_sentences.append(sent.text)

        return ' '.join(optimized_sentences)

    def _split_complex_sentence(self, sent) -> List[str]:
        """Split complex sentences based on dependency parsing."""
        # Find conjunctions or relative clauses
        splits = []
        current_start = 0

        for token in sent:
            if token.dep_ in ['conj', 'cc'] and token.i - current_start > 8:
                # Split before conjunction if sentence part is long enough
                split_text = sent[current_start:token.i].text.strip()
                if split_text:
                    splits.append(split_text)
                current_start = token.i

        # Add remaining part
        if current_start < len(sent):
            remaining = sent[current_start:].text.strip()
            if remaining:
                splits.append(remaining)

        return splits if splits else [sent.text]

    async def _apply_transformer_simplification(self, text: str) -> str:
        """Apply transformer-based text simplification."""
        try:
            # Split into manageable chunks
            sentences = sent_tokenize(text)
            simplified_sentences = []

            for sentence in sentences:
                if len(sentence.split()) > 50:  # Skip very long sentences
                    simplified_sentences.append(sentence)
                    continue

                # Use transformer for simplification
                inputs = self.tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.simplification_model.model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=4,
                        early_stopping=True
                    )

                simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                simplified_sentences.append(simplified)

            return ' '.join(simplified_sentences)

        except Exception as e:
            self.logger.warning(f"Transformer simplification failed: {e}, using original text")
            return text

    def _calculate_avg_dependency_depth(self, doc) -> float:
        """Calculate average dependency tree depth."""
        depths = []
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 20:  # Prevent infinite loops
                    break
            depths.append(depth)
        return sum(depths) / len(depths) if depths else 0

    def _count_passive_constructions(self, doc) -> int:
        """Count passive voice constructions."""
        count = 0
        for token in doc:
            if token.dep_ == 'auxpass':
                count += 1
        return count

    def _count_complex_clauses(self, doc) -> int:
        """Count complex clauses (subordinate clauses)."""
        count = 0
        for token in doc:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'relcl']:
                count += 1
        return count

    def _assess_word_complexity(self, token) -> str:
        """Assess word complexity based on linguistic features."""
        # Simple heuristic based on length and frequency
        if len(token.text) > 8:
            return 'high'
        elif token.is_stop:
            return 'low'
        else:
            return 'medium'

    async def _assess_cefr_level_async(self, text: str) -> str:
        """Assess CEFR level asynchronously."""
        # This can be made async if needed, but for now it's synchronous
        return self._assess_cefr_level(text)

    def _assess_cefr_level(self, text: str) -> str:
        """Assess the CEFR level of the text based on vocabulary."""
        words = set(word_tokenize(text.lower()))
        level_counts = {}

        for level, vocab in self.cefr_levels.items():
            level_counts[level] = len(words.intersection(vocab))

        # Return the highest level with significant vocabulary usage
        for level in ['C1', 'B2', 'B1', 'A2', 'A1']:
            if level_counts.get(level, 0) > 0:
                return level

        return 'A1'

    def check_compliance_language(self, text: str) -> Dict[str, Any]:
        """Check compliance language (synchronous for compatibility)."""
        # Implementation similar to original, but can be enhanced
        issues = []
        score = 100

        required_phrases = [
            'independent advice',
            'not regulated advice',
            'seek independent financial advice',
            'terms and conditions apply'
        ]

        text_lower = text.lower()
        missing_phrases = []

        for phrase in required_phrases:
            if phrase not in text_lower:
                missing_phrases.append(phrase)

        if missing_phrases:
            issues.append(f"Missing required disclosures: {', '.join(missing_phrases)}")
            score -= len(missing_phrases) * 15

        prohibited_phrases = [
            'guaranteed returns',
            'risk-free investment',
            '100% safe',
            'no risk of loss'
        ]

        found_prohibited = []
        for phrase in prohibited_phrases:
            if phrase in text_lower:
                found_prohibited.append(phrase)

        if found_prohibited:
            issues.append(f"Prohibited language found: {', '.join(found_prohibited)}")
            score -= len(found_prohibited) * 20

        # Enhanced sentence complexity check with spaCy
        doc = self.nlp(text)
        complex_sentences = sum(1 for sent in doc.sents if len(sent) > 15)

        if complex_sentences > 0:
            issues.append(f"{complex_sentences} sentences exceed 15-word limit")
            score -= complex_sentences * 5

        return {
            'compliant': len(issues) == 0,
            'score': max(0, score),
            'issues': issues,
            'missing_required': missing_phrases,
            'prohibited_found': found_prohibited,
            'complex_sentences': complex_sentences
        }

    def extract_mortgage_terms(self, text: str) -> List[str]:
        """Extract mortgage terms (synchronous)."""
        mortgage_terms = set()
        all_vocab = set()
        for vocab in self.cefr_levels.values():
            all_vocab.update(vocab)

        words = word_tokenize(text.lower())
        for word in words:
            if word in all_vocab:
                mortgage_terms.add(word)

        return list(mortgage_terms)

    async def calculate_comprehension_score(self, text: str) -> float:
        """Calculate comprehension score asynchronously."""
        try:
            readability = await self.analyze_readability(text)
            compliance = self.check_compliance_language(text)
            jargon = await self.detect_jargon(text)

            readability_weight = 0.4
            compliance_weight = 0.4
            jargon_weight = 0.2

            readability_score = min(100, max(0, readability.get('flesch_reading_ease', 0)))
            compliance_score = compliance.get('score', 0)
            jargon_penalty = len(jargon) * 5
            jargon_score = max(0, 100 - jargon_penalty)

            total_score = (
                readability_score * readability_weight +
                compliance_score * compliance_weight +
                jargon_score * jargon_weight
            )

            return round(total_score, 1)

        except Exception as e:
            self.logger.error(f"Error calculating comprehension score: {str(e)}")
            return 0.0
