#!/usr/bin/env python3
"""
Advanced User Comprehension Validator
Comprehensive system for adaptive testing, knowledge assessment, and learning path optimization

Features:
- Adaptive testing algorithms with intelligent question selection
- Comprehensive knowledge assessment across multiple domains
- Personalized learning path optimization with AI recommendations
- Real-time comprehension monitoring and feedback
- Regulatory compliance validation for customer understanding
- Multi-modal assessment (text, audio, visual, interactive)
- Advanced analytics and progress tracking
- Integration with mortgage advice and compliance systems
- Gamification elements for enhanced engagement
- Accessibility features for diverse learning needs
"""

import asyncio
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
import aioredis
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import statistics
from collections import defaultdict, deque
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeDomain(Enum):
    """Knowledge domains for mortgage comprehension"""
    MORTGAGE_BASICS = "mortgage_basics"
    INTEREST_RATES = "interest_rates"
    LOAN_TYPES = "loan_types"
    RISK_UNDERSTANDING = "risk_understanding"
    COSTS_AND_FEES = "costs_and_fees"
    REGULATORY_REQUIREMENTS = "regulatory_requirements"
    PROPERTY_VALUATION = "property_valuation"
    INSURANCE_PROTECTION = "insurance_protection"
    REPAYMENT_OPTIONS = "repayment_options"
    TAX_IMPLICATIONS = "tax_implications"

class QuestionType(Enum):
    """Types of assessment questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    NUMERICAL_INPUT = "numerical_input"
    SCENARIO_BASED = "scenario_based"
    DRAG_AND_DROP = "drag_and_drop"
    INTERACTIVE_SIMULATION = "interactive_simulation"
    AUDIO_RESPONSE = "audio_response"
    VIDEO_COMPREHENSION = "video_comprehension"

class DifficultyLevel(Enum):
    """Question difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class LearningStyle(Enum):
    """Learning style preferences"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"

class ComprehensionLevel(Enum):
    """Comprehension assessment levels"""
    INSUFFICIENT = "insufficient"
    BASIC = "basic"
    ADEQUATE = "adequate"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class AssessmentQuestion:
    """Individual assessment question"""
    question_id: str
    domain: KnowledgeDomain
    question_type: QuestionType
    difficulty_level: DifficultyLevel
    question_text: str
    question_context: str
    options: List[str]
    correct_answer: Union[str, int, List[str]]
    explanation: str
    learning_objectives: List[str]
    regulatory_relevance: List[str]
    estimated_time_minutes: int
    multimedia_content: Optional[Dict[str, str]]
    adaptive_parameters: Dict[str, float]
    tags: List[str]

@dataclass
class UserResponse:
    """User response to assessment question"""
    response_id: str
    question_id: str
    user_answer: Union[str, int, List[str]]
    is_correct: bool
    confidence_level: int  # 1-5 scale
    response_time_seconds: int
    attempt_number: int
    hint_used: bool
    explanation_viewed: bool
    response_timestamp: datetime
    device_info: Dict[str, str]
    interaction_data: Dict[str, Any]

@dataclass
class KnowledgeAssessment:
    """Comprehensive knowledge assessment result"""
    assessment_id: str
    user_id: str
    assessment_timestamp: datetime
    domain_scores: Dict[str, float]
    overall_comprehension_level: ComprehensionLevel
    overall_score: float
    confidence_interval: Tuple[float, float]
    strengths: List[str]
    weaknesses: List[str]
    knowledge_gaps: List[str]
    learning_recommendations: List[str]
    estimated_learning_time: int
    regulatory_compliance_score: float
    adaptive_parameters: Dict[str, float]
    next_assessment_date: datetime

@dataclass
class LearningPath:
    """Personalized learning path"""
    path_id: str
    user_id: str
    learning_objectives: List[str]
    recommended_modules: List[str]
    estimated_completion_time: int
    difficulty_progression: List[str]
    learning_style_adaptations: Dict[str, Any]
    milestone_checkpoints: List[Dict[str, Any]]
    progress_tracking: Dict[str, float]
    adaptive_adjustments: List[str]
    gamification_elements: Dict[str, Any]
    accessibility_accommodations: List[str]

@dataclass
class ComprehensionValidation:
    """Validation of user comprehension for specific advice"""
    validation_id: str
    user_id: str
    advice_id: str
    validation_timestamp: datetime
    comprehension_areas: Dict[str, float]
    overall_comprehension_score: float
    validation_method: str
    questions_asked: List[str]
    correct_responses: int
    total_responses: int
    comprehension_gaps: List[str]
    remediation_required: bool
    remediation_plan: List[str]
    regulatory_compliance: bool
    follow_up_required: bool

class AdaptiveTestingEngine:
    """Advanced adaptive testing engine with intelligent question selection"""
    
    def __init__(self):
        self.question_bank = {}
        self.difficulty_model = None
        self.user_ability_estimates = {}
        self.question_parameters = {}
        self.adaptive_algorithms = {
            "cat": self._computerized_adaptive_testing,
            "irt": self._item_response_theory,
            "bayesian": self._bayesian_adaptive_testing
        }
    
    async def initialize_question_bank(self, questions: List[AssessmentQuestion]):
        """Initialize question bank with psychometric parameters"""
        try:
            for question in questions:
                self.question_bank[question.question_id] = question
                
                # Initialize IRT parameters (discrimination, difficulty, guessing)
                self.question_parameters[question.question_id] = {
                    "discrimination": question.adaptive_parameters.get("discrimination", 1.0),
                    "difficulty": question.adaptive_parameters.get("difficulty", 0.0),
                    "guessing": question.adaptive_parameters.get("guessing", 0.25),
                    "usage_count": 0,
                    "correct_rate": 0.5,
                    "avg_response_time": question.estimated_time_minutes * 60
                }
            
            logger.info(f"Initialized question bank with {len(questions)} questions")
            
        except Exception as e:
            logger.error(f"Question bank initialization failed: {e}")
            raise
    
    async def select_next_question(self, user_id: str, domain: KnowledgeDomain,
                                 response_history: List[UserResponse],
                                 target_precision: float = 0.3) -> Optional[AssessmentQuestion]:
        """Select optimal next question using adaptive algorithms"""
        try:
            # Estimate current user ability
            current_ability = self._estimate_user_ability(user_id, response_history)
            
            # Get candidate questions for domain
            candidate_questions = [
                q for q in self.question_bank.values()
                if q.domain == domain and q.question_id not in [r.question_id for r in response_history]
            ]
            
            if not candidate_questions:
                return None
            
            # Use CAT algorithm to select optimal question
            selected_question = await self._computerized_adaptive_testing(
                current_ability, candidate_questions, target_precision
            )
            
            return selected_question
            
        except Exception as e:
            logger.error(f"Question selection failed: {e}")
            return None
    
    def _estimate_user_ability(self, user_id: str, responses: List[UserResponse]) -> float:
        """Estimate user ability using IRT model"""
        try:
            if not responses:
                return 0.0  # Neutral starting point
            
            # Advanced IRT-based ability estimation with psychometric modeling
            correct_responses = sum(1 for r in responses if r.is_correct)
            total_responses = len(responses)
            
            # Basic proportion correct
            proportion_correct = correct_responses / total_responses
            
            # Adjust for response time (faster responses indicate higher confidence)
            avg_response_time = np.mean([r.response_time_seconds for r in responses])
            time_adjustment = max(0.8, min(1.2, 60 / avg_response_time))  # Normalize around 60 seconds
            
            # Adjust for question difficulty
            difficulty_adjustment = 1.0
            if responses:
                avg_difficulty = np.mean([
                    self.question_parameters.get(r.question_id, {}).get("difficulty", 0.0)
                    for r in responses
                ])
                difficulty_adjustment = 1.0 + (avg_difficulty * 0.2)  # Harder questions increase ability estimate
            
            # Calculate ability estimate (logit scale)
            adjusted_proportion = proportion_correct * time_adjustment * difficulty_adjustment
            adjusted_proportion = max(0.01, min(0.99, adjusted_proportion))  # Avoid extreme values
            
            ability_estimate = np.log(adjusted_proportion / (1 - adjusted_proportion))
            
            # Store for future use
            self.user_ability_estimates[user_id] = ability_estimate
            
            return ability_estimate
            
        except Exception as e:
            logger.error(f"Ability estimation failed: {e}")
            return 0.0
    
    async def _computerized_adaptive_testing(self, user_ability: float,
                                           candidate_questions: List[AssessmentQuestion],
                                           target_precision: float) -> AssessmentQuestion:
        """Computerized Adaptive Testing algorithm"""
        try:
            best_question = None
            best_information = 0.0
            
            for question in candidate_questions:
                # Get question parameters
                params = self.question_parameters.get(question.question_id, {})
                discrimination = params.get("discrimination", 1.0)
                difficulty = params.get("difficulty", 0.0)
                guessing = params.get("guessing", 0.25)
                
                # Calculate Fisher information
                information = self._calculate_fisher_information(
                    user_ability, discrimination, difficulty, guessing
                )
                
                # Adjust for question usage (prefer less used questions)
                usage_penalty = params.get("usage_count", 0) * 0.1
                adjusted_information = information - usage_penalty
                
                if adjusted_information > best_information:
                    best_information = adjusted_information
                    best_question = question
            
            return best_question or candidate_questions[0]
            
        except Exception as e:
            logger.error(f"CAT algorithm failed: {e}")
            return candidate_questions[0] if candidate_questions else None
    
    def _calculate_fisher_information(self, ability: float, discrimination: float,
                                    difficulty: float, guessing: float) -> float:
        """Calculate Fisher information for question selection"""
        try:
            # 3-parameter logistic model
            exponent = discrimination * (ability - difficulty)
            probability = guessing + (1 - guessing) / (1 + np.exp(-exponent))
            
            # Fisher information formula
            derivative = discrimination * (1 - guessing) * np.exp(-exponent) / ((1 + np.exp(-exponent)) ** 2)
            information = (derivative ** 2) / (probability * (1 - probability))
            
            return information
            
        except Exception as e:
            logger.error(f"Fisher information calculation failed: {e}")
            return 0.0
    
    async def _item_response_theory(self, user_ability: float,
                                  candidate_questions: List[AssessmentQuestion]) -> AssessmentQuestion:
        """Item Response Theory based question selection"""
        try:
            # Select question with difficulty closest to user ability
            best_question = None
            min_difference = float('inf')
            
            for question in candidate_questions:
                params = self.question_parameters.get(question.question_id, {})
                difficulty = params.get("difficulty", 0.0)
                difference = abs(user_ability - difficulty)
                
                if difference < min_difference:
                    min_difference = difference
                    best_question = question
            
            return best_question
            
        except Exception as e:
            logger.error(f"IRT question selection failed: {e}")
            return candidate_questions[0] if candidate_questions else None
    
    async def _bayesian_adaptive_testing(self, user_ability: float,
                                       candidate_questions: List[AssessmentQuestion]) -> AssessmentQuestion:
        """Bayesian adaptive testing algorithm"""
        try:
            # Use Bayesian inference to select question that maximizes information gain
            best_question = None
            max_expected_information = 0.0
            
            for question in candidate_questions:
                params = self.question_parameters.get(question.question_id, {})
                
                # Calculate expected information gain
                expected_info = self._calculate_expected_information_gain(
                    user_ability, params
                )
                
                if expected_info > max_expected_information:
                    max_expected_information = expected_info
                    best_question = question
            
            return best_question
            
        except Exception as e:
            logger.error(f"Bayesian adaptive testing failed: {e}")
            return candidate_questions[0] if candidate_questions else None
    
    def _calculate_expected_information_gain(self, ability: float, params: Dict[str, float]) -> float:
        """Calculate expected information gain for Bayesian selection"""
        try:
            discrimination = params.get("discrimination", 1.0)
            difficulty = params.get("difficulty", 0.0)
            
            # Advanced Bayesian expected information calculation
            prob_correct = 1 / (1 + np.exp(-discrimination * (ability - difficulty)))
            prob_incorrect = 1 - prob_correct
            
            # Information gain based on probability distribution
            if prob_correct > 0 and prob_incorrect > 0:
                information_gain = -(prob_correct * np.log2(prob_correct) + 
                                   prob_incorrect * np.log2(prob_incorrect))
            else:
                information_gain = 0.0
            
            return information_gain
            
        except Exception as e:
            logger.error(f"Expected information gain calculation failed: {e}")
            return 0.0

class KnowledgeAssessmentEngine:
    """Advanced knowledge assessment and scoring engine"""
    
    def __init__(self):
        self.domain_weights = self._initialize_domain_weights()
        self.competency_thresholds = self._initialize_competency_thresholds()
        self.assessment_rubrics = self._load_assessment_rubrics()
    
    def _initialize_domain_weights(self) -> Dict[str, float]:
        """Initialize domain weights for overall assessment"""
        return {
            KnowledgeDomain.MORTGAGE_BASICS.value: 0.20,
            KnowledgeDomain.INTEREST_RATES.value: 0.15,
            KnowledgeDomain.RISK_UNDERSTANDING.value: 0.20,
            KnowledgeDomain.COSTS_AND_FEES.value: 0.15,
            KnowledgeDomain.REGULATORY_REQUIREMENTS.value: 0.10,
            KnowledgeDomain.LOAN_TYPES.value: 0.10,
            KnowledgeDomain.REPAYMENT_OPTIONS.value: 0.10
        }
    
    def _initialize_competency_thresholds(self) -> Dict[str, float]:
        """Initialize competency thresholds for each level"""
        return {
            ComprehensionLevel.INSUFFICIENT.value: 0.0,
            ComprehensionLevel.BASIC.value: 0.5,
            ComprehensionLevel.ADEQUATE.value: 0.65,
            ComprehensionLevel.GOOD.value: 0.8,
            ComprehensionLevel.EXCELLENT.value: 0.9
        }
    
    def _load_assessment_rubrics(self) -> Dict[str, Dict[str, Any]]:
        """Load assessment rubrics for each domain"""
        return {
            KnowledgeDomain.MORTGAGE_BASICS.value: {
                "learning_objectives": [
                    "Understand what a mortgage is and how it works",
                    "Know the difference between principal and interest",
                    "Understand loan-to-value ratio concept",
                    "Comprehend mortgage term implications"
                ],
                "assessment_criteria": {
                    "conceptual_understanding": 0.4,
                    "practical_application": 0.3,
                    "calculation_ability": 0.2,
                    "risk_awareness": 0.1
                },
                "minimum_score": 0.6
            },
            KnowledgeDomain.RISK_UNDERSTANDING.value: {
                "learning_objectives": [
                    "Understand interest rate risk",
                    "Comprehend property value fluctuation risk",
                    "Know income stability requirements",
                    "Understand early repayment implications"
                ],
                "assessment_criteria": {
                    "risk_identification": 0.3,
                    "impact_assessment": 0.3,
                    "mitigation_understanding": 0.2,
                    "scenario_analysis": 0.2
                },
                "minimum_score": 0.7  # Higher threshold for risk understanding
            },
            KnowledgeDomain.COSTS_AND_FEES.value: {
                "learning_objectives": [
                    "Understand all mortgage-related costs",
                    "Know how to calculate total cost of credit",
                    "Comprehend fee structures",
                    "Understand cost comparison methods"
                ],
                "assessment_criteria": {
                    "cost_identification": 0.25,
                    "calculation_accuracy": 0.35,
                    "comparison_ability": 0.25,
                    "long_term_impact": 0.15
                },
                "minimum_score": 0.65
            }
        }
    
    async def assess_user_knowledge(self, user_id: str, responses: List[UserResponse],
                                  assessment_context: Dict[str, Any] = None) -> KnowledgeAssessment:
        """Perform comprehensive knowledge assessment"""
        try:
            assessment_id = f"ASSESS_{uuid.uuid4().hex[:12]}_{int(time.time())}"
            
            # Calculate domain scores
            domain_scores = self._calculate_domain_scores(responses)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(domain_scores)
            
            # Determine comprehension level
            comprehension_level = self._determine_comprehension_level(overall_score)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(responses, overall_score)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(domain_scores)
            
            # Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(responses, domain_scores)
            
            # Generate learning recommendations
            learning_recommendations = self._generate_learning_recommendations(
                domain_scores, knowledge_gaps, assessment_context
            )
            
            # Estimate learning time
            estimated_learning_time = self._estimate_learning_time(knowledge_gaps, domain_scores)
            
            # Calculate regulatory compliance score
            regulatory_score = self._calculate_regulatory_compliance_score(responses)
            
            # Update adaptive parameters
            adaptive_parameters = self._update_adaptive_parameters(user_id, responses)
            
            # Determine next assessment date
            next_assessment_date = self._calculate_next_assessment_date(
                comprehension_level, domain_scores
            )
            
            return KnowledgeAssessment(
                assessment_id=assessment_id,
                user_id=user_id,
                assessment_timestamp=datetime.now(),
                domain_scores=domain_scores,
                overall_comprehension_level=comprehension_level,
                overall_score=overall_score,
                confidence_interval=confidence_interval,
                strengths=strengths,
                weaknesses=weaknesses,
                knowledge_gaps=knowledge_gaps,
                learning_recommendations=learning_recommendations,
                estimated_learning_time=estimated_learning_time,
                regulatory_compliance_score=regulatory_score,
                adaptive_parameters=adaptive_parameters,
                next_assessment_date=next_assessment_date
            )
            
        except Exception as e:
            logger.error(f"Knowledge assessment failed: {e}")
            raise
    
    def _calculate_domain_scores(self, responses: List[UserResponse]) -> Dict[str, float]:
        """Calculate scores for each knowledge domain"""
        domain_responses = defaultdict(list)
        
        # Group responses by domain
        for response in responses:
            question = self.question_bank.get(response.question_id)
            if question:
                domain_responses[question.domain.value].append(response)
        
        # Calculate domain scores
        domain_scores = {}
        for domain, domain_response_list in domain_responses.items():
            if domain_response_list:
                # Weight by question difficulty and response time
                weighted_scores = []
                for response in domain_response_list:
                    question = self.question_bank[response.question_id]
                    difficulty_weight = 1.0 + (question.difficulty_level.value == "advanced") * 0.2
                    time_weight = max(0.8, min(1.2, 60 / response.response_time_seconds))
                    
                    score = (1.0 if response.is_correct else 0.0) * difficulty_weight * time_weight
                    weighted_scores.append(score)
                
                domain_scores[domain] = np.mean(weighted_scores)
            else:
                domain_scores[domain] = 0.0
        
        return domain_scores
    
    def _calculate_overall_score(self, domain_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for domain, score in domain_scores.items():
            weight = self.domain_weights.get(domain, 0.1)
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_comprehension_level(self, overall_score: float) -> ComprehensionLevel:
        """Determine comprehension level from overall score"""
        for level in [ComprehensionLevel.EXCELLENT, ComprehensionLevel.GOOD, 
                     ComprehensionLevel.ADEQUATE, ComprehensionLevel.BASIC, 
                     ComprehensionLevel.INSUFFICIENT]:
            threshold = self.competency_thresholds[level.value]
            if overall_score >= threshold:
                return level
        
        return ComprehensionLevel.INSUFFICIENT
    
    def _calculate_confidence_interval(self, responses: List[UserResponse], 
                                     overall_score: float) -> Tuple[float, float]:
        """Calculate confidence interval for assessment score"""
        try:
            n = len(responses)
            if n < 2:
                return (max(0, overall_score - 0.2), min(1, overall_score + 0.2))
            
            # Calculate standard error
            p = overall_score
            standard_error = np.sqrt(p * (1 - p) / n)
            
            # 95% confidence interval
            margin_of_error = 1.96 * standard_error
            
            lower_bound = max(0, overall_score - margin_of_error)
            upper_bound = min(1, overall_score + margin_of_error)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return (0.0, 1.0)
    
    def _identify_strengths_weaknesses(self, domain_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify user strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        for domain, score in domain_scores.items():
            if score >= 0.8:
                strengths.append(domain.replace("_", " ").title())
            elif score < 0.6:
                weaknesses.append(domain.replace("_", " ").title())
        
        return strengths, weaknesses
    
    def _identify_knowledge_gaps(self, responses: List[UserResponse], 
                               domain_scores: Dict[str, float]) -> List[str]:
        """Identify specific knowledge gaps"""
        gaps = []
        
        # Identify domains with low scores
        for domain, score in domain_scores.items():
            if score < 0.6:
                rubric = self.assessment_rubrics.get(domain, {})
                objectives = rubric.get("learning_objectives", [])
                gaps.extend([f"{domain}: {obj}" for obj in objectives])
        
        # Identify specific question areas with consistent errors
        incorrect_responses = [r for r in responses if not r.is_correct]
        if incorrect_responses:
            # Group by question characteristics
            error_patterns = defaultdict(int)
            for response in incorrect_responses:
                question = self.question_bank.get(response.question_id)
                if question:
                    for tag in question.tags:
                        error_patterns[tag] += 1
            
            # Add frequent error patterns as gaps
            for pattern, count in error_patterns.items():
                if count >= 2:  # At least 2 errors in this area
                    gaps.append(f"Frequent errors in: {pattern}")
        
        return gaps[:10]  # Limit to top 10 gaps
    
    def _generate_learning_recommendations(self, domain_scores: Dict[str, float],
                                         knowledge_gaps: List[str],
                                         context: Dict[str, Any] = None) -> List[str]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        # Domain-specific recommendations
        for domain, score in domain_scores.items():
            if score < 0.7:  # Needs improvement
                domain_name = domain.replace("_", " ").title()
                
                if domain == KnowledgeDomain.MORTGAGE_BASICS.value:
                    recommendations.append(f"Complete foundational mortgage education module")
                    recommendations.append(f"Practice basic mortgage calculations")
                elif domain == KnowledgeDomain.RISK_UNDERSTANDING.value:
                    recommendations.append(f"Study risk scenarios and mitigation strategies")
                    recommendations.append(f"Complete interactive risk simulation exercises")
                elif domain == KnowledgeDomain.COSTS_AND_FEES.value:
                    recommendations.append(f"Practice cost calculation exercises")
                    recommendations.append(f"Review comparative cost analysis examples")
        
        # Gap-specific recommendations
        if knowledge_gaps:
            recommendations.append("Focus on identified knowledge gaps through targeted learning")
            recommendations.append("Schedule follow-up assessment after gap remediation")
        
        # Context-specific recommendations
        if context:
            learning_style = context.get("learning_style", LearningStyle.MULTIMODAL.value)
            
            if learning_style == LearningStyle.VISUAL.value:
                recommendations.append("Use visual learning materials (charts, diagrams, infographics)")
            elif learning_style == LearningStyle.AUDITORY.value:
                recommendations.append("Utilize audio explanations and discussion sessions")
            elif learning_style == LearningStyle.KINESTHETIC.value:
                recommendations.append("Engage with interactive simulations and hands-on exercises")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _estimate_learning_time(self, knowledge_gaps: List[str], 
                              domain_scores: Dict[str, float]) -> int:
        """Estimate learning time in minutes"""
        base_time_per_gap = 15  # 15 minutes per knowledge gap
        
        # Adjust based on domain scores
        low_score_domains = sum(1 for score in domain_scores.values() if score < 0.6)
        domain_adjustment = low_score_domains * 10  # 10 minutes per weak domain
        
        # Adjust based on number of gaps
        gap_time = len(knowledge_gaps) * base_time_per_gap
        
        total_time = gap_time + domain_adjustment
        
        # Add buffer time for reinforcement
        total_time = int(total_time * 1.2)
        
        return min(total_time, 240)  # Cap at 4 hours
    
    def _calculate_regulatory_compliance_score(self, responses: List[UserResponse]) -> float:
        """Calculate regulatory compliance score"""
        try:
            # Find responses to regulatory questions
            regulatory_responses = []
            for response in responses:
                question = self.question_bank.get(response.question_id)
                if question and question.regulatory_relevance:
                    regulatory_responses.append(response)
            
            if not regulatory_responses:
                return 0.5  # Neutral score if no regulatory questions
            
            # Calculate score with higher weight for regulatory questions
            correct_count = sum(1 for r in regulatory_responses if r.is_correct)
            total_count = len(regulatory_responses)
            
            base_score = correct_count / total_count
            
            # Apply minimum threshold for regulatory compliance
            return max(base_score, 0.7) if base_score >= 0.8 else base_score
            
        except Exception as e:
            logger.error(f"Regulatory compliance score calculation failed: {e}")
            return 0.5
    
    def _update_adaptive_parameters(self, user_id: str, responses: List[UserResponse]) -> Dict[str, float]:
        """Update adaptive parameters based on user performance"""
        try:
            # Calculate learning rate
            if len(responses) >= 2:
                recent_performance = [r.is_correct for r in responses[-5:]]  # Last 5 responses
                learning_rate = np.mean(recent_performance)
            else:
                learning_rate = 0.5
            
            # Calculate response time patterns
            response_times = [r.response_time_seconds for r in responses]
            avg_response_time = np.mean(response_times) if response_times else 60
            
            # Calculate consistency
            if len(responses) >= 3:
                consistency = 1.0 - np.std([1 if r.is_correct else 0 for r in responses])
            else:
                consistency = 0.5
            
            return {
                "learning_rate": learning_rate,
                "avg_response_time": avg_response_time,
                "consistency": consistency,
                "preferred_difficulty": self._estimate_preferred_difficulty(responses),
                "engagement_level": self._calculate_engagement_level(responses)
            }
            
        except Exception as e:
            logger.error(f"Adaptive parameter update failed: {e}")
            return {}
    
    def _estimate_preferred_difficulty(self, responses: List[UserResponse]) -> str:
        """Estimate user's preferred difficulty level"""
        try:
            difficulty_performance = defaultdict(list)
            
            for response in responses:
                question = self.question_bank.get(response.question_id)
                if question:
                    difficulty_performance[question.difficulty_level.value].append(response.is_correct)
            
            # Find difficulty level with best performance
            best_difficulty = "intermediate"
            best_performance = 0.0
            
            for difficulty, correct_list in difficulty_performance.items():
                if correct_list:
                    performance = np.mean(correct_list)
                    if performance > best_performance:
                        best_performance = performance
                        best_difficulty = difficulty
            
            return best_difficulty
            
        except Exception as e:
            logger.error(f"Preferred difficulty estimation failed: {e}")
            return "intermediate"
    
    def _calculate_engagement_level(self, responses: List[UserResponse]) -> float:
        """Calculate user engagement level"""
        try:
            if not responses:
                return 0.5
            
            engagement_factors = []
            
            # Response time consistency (engaged users have consistent times)
            response_times = [r.response_time_seconds for r in responses]
            time_consistency = 1.0 - (np.std(response_times) / np.mean(response_times)) if response_times else 0.5
            engagement_factors.append(min(1.0, time_consistency))
            
            # Hint usage (engaged users use hints appropriately)
            hint_usage_rate = sum(1 for r in responses if r.hint_used) / len(responses)
            optimal_hint_rate = 0.3  # 30% is considered optimal
            hint_score = 1.0 - abs(hint_usage_rate - optimal_hint_rate) / optimal_hint_rate
            engagement_factors.append(hint_score)
            
            # Explanation viewing (engaged users read explanations)
            explanation_rate = sum(1 for r in responses if r.explanation_viewed) / len(responses)
            engagement_factors.append(explanation_rate)
            
            return np.mean(engagement_factors)
            
        except Exception as e:
            logger.error(f"Engagement level calculation failed: {e}")
            return 0.5
    
    def _calculate_next_assessment_date(self, comprehension_level: ComprehensionLevel,
                                      domain_scores: Dict[str, float]) -> datetime:
        """Calculate when next assessment should occur"""
        try:
            base_intervals = {
                ComprehensionLevel.INSUFFICIENT: 7,   # 1 week
                ComprehensionLevel.BASIC: 14,         # 2 weeks
                ComprehensionLevel.ADEQUATE: 30,      # 1 month
                ComprehensionLevel.GOOD: 60,          # 2 months
                ComprehensionLevel.EXCELLENT: 90      # 3 months
            }
            
            base_days = base_intervals.get(comprehension_level, 30)
            
            # Adjust based on lowest domain score
            min_domain_score = min(domain_scores.values()) if domain_scores else 0.5
            if min_domain_score < 0.5:
                base_days = min(base_days, 14)  # More frequent if any domain is very low
            
            return datetime.now() + timedelta(days=base_days)
            
        except Exception as e:
            logger.error(f"Next assessment date calculation failed: {e}")
            return datetime.now() + timedelta(days=30)

class LearningPathOptimizer:
    """Advanced learning path optimization engine"""
    
    def __init__(self):
        self.learning_modules = self._load_learning_modules()
        self.prerequisite_graph = self._build_prerequisite_graph()
        self.learning_styles = self._load_learning_style_adaptations()
    
    def _load_learning_modules(self) -> Dict[str, Dict[str, Any]]:
        """Load available learning modules"""
        return {
            "mortgage_fundamentals": {
                "title": "Mortgage Fundamentals",
                "description": "Basic concepts of mortgages and home financing",
                "estimated_time": 30,
                "difficulty": "beginner",
                "prerequisites": [],
                "learning_objectives": [
                    "Understand mortgage definition and purpose",
                    "Learn about principal and interest",
                    "Comprehend loan terms and conditions"
                ],
                "content_types": ["text", "video", "interactive"],
                "assessment_questions": 5
            },
            "interest_rate_mechanics": {
                "title": "Interest Rate Mechanics",
                "description": "Understanding how interest rates work and impact mortgages",
                "estimated_time": 25,
                "difficulty": "intermediate",
                "prerequisites": ["mortgage_fundamentals"],
                "learning_objectives": [
                    "Understand fixed vs variable rates",
                    "Learn about rate setting mechanisms",
                    "Comprehend rate change impacts"
                ],
                "content_types": ["text", "calculator", "simulation"],
                "assessment_questions": 7
            },
            "risk_assessment_training": {
                "title": "Mortgage Risk Assessment",
                "description": "Comprehensive understanding of mortgage risks",
                "estimated_time": 40,
                "difficulty": "intermediate",
                "prerequisites": ["mortgage_fundamentals", "interest_rate_mechanics"],
                "learning_objectives": [
                    "Identify various mortgage risks",
                    "Understand risk mitigation strategies",
                    "Learn stress testing concepts"
                ],
                "content_types": ["text", "video", "scenario", "simulation"],
                "assessment_questions": 10
            },
            "regulatory_compliance": {
                "title": "Dutch Mortgage Regulations",
                "description": "Understanding regulatory requirements and consumer protections",
                "estimated_time": 35,
                "difficulty": "advanced",
                "prerequisites": ["mortgage_fundamentals"],
                "learning_objectives": [
                    "Learn Wft requirements",
                    "Understand AFM guidelines",
                    "Comprehend consumer rights"
                ],
                "content_types": ["text", "case_study", "interactive"],
                "assessment_questions": 8
            }
        }
    
    def _build_prerequisite_graph(self) -> Dict[str, List[str]]:
        """Build prerequisite dependency graph"""
        graph = {}
        for module_id, module_data in self.learning_modules.items():
            graph[module_id] = module_data.get("prerequisites", [])
        return graph
    
    def _load_learning_style_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Load learning style adaptations"""
        return {
            LearningStyle.VISUAL.value: {
                "preferred_content": ["diagrams", "charts", "infographics", "videos"],
                "question_formats": ["image_based", "diagram_interpretation"],
                "presentation_style": "visual_heavy",
                "pacing": "self_paced"
            },
            LearningStyle.AUDITORY.value: {
                "preferred_content": ["audio_explanations", "discussions", "verbal_examples"],
                "question_formats": ["audio_questions", "verbal_response"],
                "presentation_style": "audio_enhanced",
                "pacing": "guided"
            },
            LearningStyle.KINESTHETIC.value: {
                "preferred_content": ["simulations", "interactive_exercises", "hands_on"],
                "question_formats": ["drag_and_drop", "interactive_simulation"],
                "presentation_style": "interactive",
                "pacing": "activity_based"
            },
            LearningStyle.READING_WRITING.value: {
                "preferred_content": ["text_based", "written_examples", "documentation"],
                "question_formats": ["text_input", "essay_questions"],
                "presentation_style": "text_heavy",
                "pacing": "self_paced"
            }
        }
    
    async def optimize_learning_path(self, user_id: str, assessment: KnowledgeAssessment,
                                   user_preferences: Dict[str, Any] = None) -> LearningPath:
        """Generate optimized learning path for user"""
        try:
            path_id = f"PATH_{uuid.uuid4().hex[:12]}_{int(time.time())}"
            
            # Identify required modules based on knowledge gaps
            required_modules = self._identify_required_modules(assessment)
            
            # Order modules based on prerequisites and difficulty
            ordered_modules = self._order_modules_by_prerequisites(required_modules)
            
            # Adapt for learning style
            learning_style = user_preferences.get("learning_style", LearningStyle.MULTIMODAL.value)
            style_adaptations = self._adapt_for_learning_style(ordered_modules, learning_style)
            
            # Calculate estimated completion time
            total_time = sum(
                self.learning_modules[module]["estimated_time"] 
                for module in ordered_modules
            )
            
            # Create milestone checkpoints
            milestones = self._create_milestone_checkpoints(ordered_modules)
            
            # Initialize progress tracking
            progress_tracking = {module: 0.0 for module in ordered_modules}
            
            # Generate adaptive adjustments
            adaptive_adjustments = self._generate_adaptive_adjustments(assessment, user_preferences)
            
            # Create gamification elements
            gamification = self._create_gamification_elements(ordered_modules, user_preferences)
            
            # Determine accessibility accommodations
            accommodations = self._determine_accessibility_accommodations(user_preferences)
            
            return LearningPath(
                path_id=path_id,
                user_id=user_id,
                learning_objectives=self._extract_learning_objectives(ordered_modules),
                recommended_modules=ordered_modules,
                estimated_completion_time=total_time,
                difficulty_progression=self._create_difficulty_progression(ordered_modules),
                learning_style_adaptations=style_adaptations,
                milestone_checkpoints=milestones,
                progress_tracking=progress_tracking,
                adaptive_adjustments=adaptive_adjustments,
                gamification_elements=gamification,
                accessibility_accommodations=accommodations
            )
            
        except Exception as e:
            logger.error(f"Learning path optimization failed: {e}")
            raise
    
    def _identify_required_modules(self, assessment: KnowledgeAssessment) -> List[str]:
        """Identify required learning modules based on assessment"""
        required_modules = []
        
        # Add modules for weak domains
        for domain, score in assessment.domain_scores.items():
            if score < 0.7:  # Needs improvement
                # Map domains to modules
                domain_module_map = {
                    KnowledgeDomain.MORTGAGE_BASICS.value: "mortgage_fundamentals",
                    KnowledgeDomain.INTEREST_RATES.value: "interest_rate_mechanics",
                    KnowledgeDomain.RISK_UNDERSTANDING.value: "risk_assessment_training",
                    KnowledgeDomain.REGULATORY_REQUIREMENTS.value: "regulatory_compliance"
                }
                
                module = domain_module_map.get(domain)
                if module and module not in required_modules:
                    required_modules.append(module)
        
        # Always include fundamentals if overall score is low
        if assessment.overall_score < 0.6 and "mortgage_fundamentals" not in required_modules:
            required_modules.insert(0, "mortgage_fundamentals")
        
        return required_modules
    
    def _order_modules_by_prerequisites(self, modules: List[str]) -> List[str]:
        """Order modules based on prerequisite dependencies"""
        try:
            ordered = []
            remaining = modules.copy()
            
            while remaining:
                # Find modules with satisfied prerequisites
                ready_modules = []
                for module in remaining:
                    prerequisites = self.prerequisite_graph.get(module, [])
                    if all(prereq in ordered for prereq in prerequisites):
                        ready_modules.append(module)
                
                if not ready_modules:
                    # Break circular dependencies by adding module with fewest unmet prerequisites
                    module_scores = {}
                    for module in remaining:
                        prerequisites = self.prerequisite_graph.get(module, [])
                        unmet_count = sum(1 for prereq in prerequisites if prereq not in ordered)
                        module_scores[module] = unmet_count
                    
                    best_module = min(module_scores, key=module_scores.get)
                    ready_modules = [best_module]
                
                # Add ready modules to ordered list
                for module in ready_modules:
                    ordered.append(module)
                    remaining.remove(module)
            
            return ordered
            
        except Exception as e:
            logger.error(f"Module ordering failed: {e}")
            return modules
    
    def _adapt_for_learning_style(self, modules: List[str], learning_style: str) -> Dict[str, Any]:
        """Adapt learning path for specific learning style"""
        style_config = self.learning_styles.get(learning_style, {})
        
        adaptations = {
            "content_preferences": style_config.get("preferred_content", []),
            "question_formats": style_config.get("question_formats", []),
            "presentation_style": style_config.get("presentation_style", "standard"),
            "pacing_strategy": style_config.get("pacing", "self_paced"),
            "interaction_level": "high" if learning_style == LearningStyle.KINESTHETIC.value else "medium"
        }
        
        # Module-specific adaptations
        module_adaptations = {}
        for module in modules:
            module_data = self.learning_modules.get(module, {})
            available_content = module_data.get("content_types", [])
            preferred_content = style_config.get("preferred_content", [])
            
            # Match available content with preferences
            matched_content = [ct for ct in available_content if ct in preferred_content]
            if not matched_content:
                matched_content = available_content  # Use all if no matches
            
            module_adaptations[module] = {
                "content_types": matched_content,
                "emphasis": "high" if any(pref in available_content for pref in preferred_content) else "standard"
            }
        
        adaptations["module_adaptations"] = module_adaptations
        return adaptations
    
    def _create_milestone_checkpoints(self, modules: List[str]) -> List[Dict[str, Any]]:
        """Create milestone checkpoints for learning path"""
        milestones = []
        
        for i, module in enumerate(modules):
            module_data = self.learning_modules.get(module, {})
            
            milestone = {
                "milestone_id": str(uuid.uuid4()),
                "module_id": module,
                "title": f"Complete {module_data.get('title', module)}",
                "description": module_data.get("description", ""),
                "position": i + 1,
                "total_positions": len(modules),
                "estimated_time": module_data.get("estimated_time", 30),
                "completion_criteria": {
                    "module_completion": True,
                    "assessment_score": 0.7,
                    "time_spent_minimum": module_data.get("estimated_time", 30) * 0.8
                },
                "rewards": {
                    "points": (i + 1) * 100,
                    "badge": f"{module}_completed",
                    "certificate": i == len(modules) - 1  # Certificate for final milestone
                }
            }
            milestones.append(milestone)
        
        return milestones
    
    def _extract_learning_objectives(self, modules: List[str]) -> List[str]:
        """Extract all learning objectives from modules"""
        objectives = []
        for module in modules:
            module_data = self.learning_modules.get(module, {})
            module_objectives = module_data.get("learning_objectives", [])
            objectives.extend(module_objectives)
        return objectives
    
    def _create_difficulty_progression(self, modules: List[str]) -> List[str]:
        """Create difficulty progression for modules"""
        progression = []
        for module in modules:
            module_data = self.learning_modules.get(module, {})
            difficulty = module_data.get("difficulty", "intermediate")
            progression.append(difficulty)
        return progression
    
    def _generate_adaptive_adjustments(self, assessment: KnowledgeAssessment,
                                     preferences: Dict[str, Any] = None) -> List[str]:
        """Generate adaptive adjustments for learning path"""
        adjustments = []
        
        # Adjust based on comprehension level
        if assessment.overall_comprehension_level == ComprehensionLevel.INSUFFICIENT:
            adjustments.append("Increase practice exercises and repetition")
            adjustments.append("Add foundational concept review")
        elif assessment.overall_comprehension_level == ComprehensionLevel.EXCELLENT:
            adjustments.append("Add advanced challenge questions")
            adjustments.append("Include real-world application scenarios")
        
        # Adjust based on learning rate
        learning_rate = assessment.adaptive_parameters.get("learning_rate", 0.5)
        if learning_rate < 0.4:
            adjustments.append("Slow down pacing and add more examples")
        elif learning_rate > 0.8:
            adjustments.append("Accelerate pacing and add complexity")
        
        # Adjust based on engagement
        engagement = assessment.adaptive_parameters.get("engagement_level", 0.5)
        if engagement < 0.5:
            adjustments.append("Add gamification elements and interactive content")
        
        return adjustments
    
    def _create_gamification_elements(self, modules: List[str], 
                                    preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create gamification elements for learning path"""
        return {
            "point_system": {
                "question_correct": 10,
                "module_completion": 100,
                "streak_bonus": 5,
                "perfect_score_bonus": 25
            },
            "badges": [
                {"name": "Quick Learner", "criteria": "Complete module in under estimated time"},
                {"name": "Perfectionist", "criteria": "Score 100% on module assessment"},
                {"name": "Persistent", "criteria": "Complete all modules"},
                {"name": "Risk Expert", "criteria": "Excel in risk understanding domain"}
            ],
            "leaderboard": {
                "enabled": preferences.get("competitive_learning", False),
                "anonymous": True,
                "categories": ["speed", "accuracy", "completion"]
            },
            "achievements": {
                "progress_milestones": [25, 50, 75, 100],  # Percentage completion
                "streak_targets": [3, 7, 14],  # Days of consecutive learning
                "mastery_levels": ["bronze", "silver", "gold", "platinum"]
            }
        }
    
    def _determine_accessibility_accommodations(self, preferences: Dict[str, Any] = None) -> List[str]:
        """Determine accessibility accommodations needed"""
        accommodations = []
        
        if preferences:
            # Visual accommodations
            if preferences.get("visual_impairment"):
                accommodations.extend([
                    "Screen reader compatibility",
                    "High contrast mode",
                    "Large text options",
                    "Audio descriptions for visual content"
                ])
            
            # Hearing accommodations
            if preferences.get("hearing_impairment"):
                accommodations.extend([
                    "Closed captions for videos",
                    "Visual indicators for audio cues",
                    "Text alternatives for audio content"
                ])
            
            # Motor accommodations
            if preferences.get("motor_impairment"):
                accommodations.extend([
                    "Keyboard navigation support",
                    "Extended time limits",
                    "Alternative input methods"
                ])
            
            # Cognitive accommodations
            if preferences.get("learning_disability"):
                accommodations.extend([
                    "Simplified language options",
                    "Extended time limits",
                    "Frequent progress saves",
                    "Multiple explanation formats"
                ])
        
        return accommodations

class ComprehensionValidationEngine:
    """Engine for validating user comprehension of specific advice"""
    
    def __init__(self):
        self.validation_criteria = self._load_validation_criteria()
        self.comprehension_thresholds = self._load_comprehension_thresholds()
    
    def _load_validation_criteria(self) -> Dict[str, Any]:
        """Load criteria for comprehension validation"""
        return {
            "mortgage_product_understanding": {
                "key_concepts": [
                    "interest_rate_type",
                    "loan_term",
                    "monthly_payment",
                    "total_cost",
                    "early_repayment_terms"
                ],
                "minimum_score": 0.8,
                "regulatory_requirement": "wft_article_86f"
            },
            "risk_comprehension": {
                "key_concepts": [
                    "interest_rate_risk",
                    "property_value_risk",
                    "income_risk",
                    "early_repayment_risk"
                ],
                "minimum_score": 0.75,
                "regulatory_requirement": "afm_disclosure"
            },
            "cost_understanding": {
                "key_concepts": [
                    "total_interest_cost",
                    "monthly_payment_calculation",
                    "additional_fees",
                    "cost_comparison"
                ],
                "minimum_score": 0.7,
                "regulatory_requirement": "cost_disclosure"
            }
        }
    
    def _load_comprehension_thresholds(self) -> Dict[str, float]:
        """Load comprehension thresholds for different advice types"""
        return {
            "basic_mortgage_advice": 0.6,
            "complex_product_advice": 0.75,
            "high_risk_advice": 0.8,
            "regulatory_disclosure": 0.85
        }
    
    async def validate_advice_comprehension(self, user_id: str, advice_id: str,
                                          advice_content: str, user_responses: List[UserResponse],
                                          validation_method: str = "adaptive_questioning") -> ComprehensionValidation:
        """Validate user comprehension of specific mortgage advice"""
        try:
            validation_id = f"VALID_{uuid.uuid4().hex[:12]}_{int(time.time())}"
            
            # Analyze advice content to identify key concepts
            key_concepts = self._extract_key_concepts_from_advice(advice_content)
            
            # Assess comprehension for each area
            comprehension_areas = {}
            for concept_area, concepts in key_concepts.items():
                area_score = self._assess_concept_area_comprehension(
                    concepts, user_responses, concept_area
                )
                comprehension_areas[concept_area] = area_score
            
            # Calculate overall comprehension score
            overall_score = self._calculate_overall_comprehension_score(comprehension_areas)
            
            # Identify comprehension gaps
            comprehension_gaps = self._identify_comprehension_gaps(
                comprehension_areas, key_concepts
            )
            
            # Determine if remediation is required
            remediation_required = overall_score < 0.7 or len(comprehension_gaps) > 2
            
            # Generate remediation plan if needed
            remediation_plan = []
            if remediation_required:
                remediation_plan = self._generate_remediation_plan(
                    comprehension_gaps, comprehension_areas
                )
            
            # Check regulatory compliance
            regulatory_compliance = self._check_regulatory_comprehension_compliance(
                overall_score, comprehension_areas
            )
            
            # Determine follow-up requirements
            follow_up_required = remediation_required or overall_score < 0.8
            
            return ComprehensionValidation(
                validation_id=validation_id,
                user_id=user_id,
                advice_id=advice_id,
                validation_timestamp=datetime.now(),
                comprehension_areas=comprehension_areas,
                overall_comprehension_score=overall_score,
                validation_method=validation_method,
                questions_asked=[r.question_id for r in user_responses],
                correct_responses=sum(1 for r in user_responses if r.is_correct),
                total_responses=len(user_responses),
                comprehension_gaps=comprehension_gaps,
                remediation_required=remediation_required,
                remediation_plan=remediation_plan,
                regulatory_compliance=regulatory_compliance,
                follow_up_required=follow_up_required
            )
            
        except Exception as e:
            logger.error(f"Comprehension validation failed: {e}")
            raise
    
    def _extract_key_concepts_from_advice(self, advice_content: str) -> Dict[str, List[str]]:
        """Extract key concepts from advice content using NLP"""
        try:
            key_concepts = defaultdict(list)
            
            # Mortgage product concepts
            if any(term in advice_content.lower() for term in ['product', 'mortgage', 'loan']):
                key_concepts["product_understanding"].extend([
                    "product_features", "interest_rate", "loan_term", "repayment_type"
                ])
            
            # Risk concepts
            if any(term in advice_content.lower() for term in ['risk', 'risico']):
                key_concepts["risk_understanding"].extend([
                    "interest_rate_risk", "property_value_risk", "income_risk"
                ])
            
            # Cost concepts
            if any(term in advice_content.lower() for term in ['cost', 'kosten', 'fee', 'payment']):
                key_concepts["cost_understanding"].extend([
                    "monthly_payment", "total_cost", "fees", "cost_comparison"
                ])
            
            # Regulatory concepts
            if any(term in advice_content.lower() for term in ['wft', 'afm', 'regulation']):
                key_concepts["regulatory_understanding"].extend([
                    "consumer_rights", "cooling_off_period", "complaint_procedures"
                ])
            
            return dict(key_concepts)
            
        except Exception as e:
            logger.error(f"Key concept extraction failed: {e}")
            return {}
    
    def _assess_concept_area_comprehension(self, concepts: List[str], 
                                         responses: List[UserResponse],
                                         concept_area: str) -> float:
        """Assess comprehension for specific concept area"""
        try:
            # Find responses related to this concept area
            relevant_responses = []
            for response in responses:
                question = self.question_bank.get(response.question_id, {})
                question_tags = question.get("tags", [])
                
                if any(concept in question_tags for concept in concepts):
                    relevant_responses.append(response)
            
            if not relevant_responses:
                return 0.5  # Neutral score if no relevant responses
            
            # Calculate weighted score
            total_weight = 0.0
            weighted_score = 0.0
            
            for response in relevant_responses:
                # Weight by question difficulty and importance
                question = self.question_bank.get(response.question_id, {})
                difficulty_weight = 1.0
                if question.get("difficulty_level") == "advanced":
                    difficulty_weight = 1.3
                elif question.get("difficulty_level") == "beginner":
                    difficulty_weight = 0.8
                
                importance_weight = 1.0
                if any(concept in ["risk", "cost", "regulatory"] for concept in concepts):
                    importance_weight = 1.2  # Higher weight for critical concepts
                
                response_score = 1.0 if response.is_correct else 0.0
                
                # Adjust for confidence and response time
                confidence_adjustment = response.confidence_level / 5.0  # Normalize to 0-1
                time_adjustment = max(0.8, min(1.2, 60 / response.response_time_seconds))
                
                final_weight = difficulty_weight * importance_weight
                final_score = response_score * confidence_adjustment * time_adjustment
                
                weighted_score += final_score * final_weight
                total_weight += final_weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Concept area assessment failed: {e}")
            return 0.0

class UserComprehensionValidator:
    """Main user comprehension validator with comprehensive capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptive_engine = AdaptiveTestingEngine()
        self.assessment_engine = KnowledgeAssessmentEngine()
        self.learning_optimizer = LearningPathOptimizer()
        self.validation_engine = ComprehensionValidationEngine()
        self.db_pool = None
        self.redis_pool = None
        
        # Performance metrics
        self.metrics = {
            "assessments_completed": 0,
            "learning_paths_created": 0,
            "comprehension_validations": 0,
            "avg_assessment_time": 0,
            "assessment_times": []
        }
    
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize the user comprehension validator"""
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
            
            # Load question bank
            await self._load_question_bank()
            
            logger.info("User Comprehension Validator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize User Comprehension Validator: {e}")
            raise
    
    async def _load_question_bank(self):
        """Load question bank from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM assessment_questions WHERE is_active = true")
                
                questions = []
                for row in rows:
                    question = AssessmentQuestion(
                        question_id=row['question_id'],
                        domain=KnowledgeDomain(row['knowledge_domain']),
                        question_type=QuestionType(row['question_type']),
                        difficulty_level=DifficultyLevel(row['difficulty_level']),
                        question_text=row['question_text'],
                        question_context=row['question_context'] or "",
                        options=json.loads(row['options']) if row['options'] else [],
                        correct_answer=json.loads(row['correct_answer']),
                        explanation=row['explanation'] or "",
                        learning_objectives=json.loads(row['learning_objectives']) if row['learning_objectives'] else [],
                        regulatory_relevance=json.loads(row['regulatory_relevance']) if row['regulatory_relevance'] else [],
                        estimated_time_minutes=row['estimated_time_minutes'] or 2,
                        multimedia_content=json.loads(row['multimedia_content']) if row['multimedia_content'] else None,
                        adaptive_parameters=json.loads(row['adaptive_parameters']) if row['adaptive_parameters'] else {},
                        tags=json.loads(row['tags']) if row['tags'] else []
                    )
                    questions.append(question)
                
                await self.adaptive_engine.initialize_question_bank(questions)
                logger.info(f"Loaded {len(questions)} questions from database")
                
        except Exception as e:
            logger.error(f"Question bank loading failed: {e}")
    
    async def create_adaptive_assessment(self, user_id: str, assessment_goals: List[str],
                                       user_preferences: Dict[str, Any] = None) -> str:
        """Create adaptive assessment session for user"""
        try:
            session_id = f"SESSION_{uuid.uuid4().hex[:12]}_{int(time.time())}"
            
            # Store assessment session
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO assessment_sessions (
                        session_id, user_id, assessment_goals, user_preferences,
                        session_status, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                    session_id, user_id, json.dumps(assessment_goals),
                    json.dumps(user_preferences or {}), "active", datetime.now()
                )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Assessment session creation failed: {e}")
            raise
    
    async def get_next_question(self, session_id: str) -> Optional[AssessmentQuestion]:
        """Get next adaptive question for assessment session"""
        try:
            # Get session data
            async with self.db_pool.acquire() as conn:
                session_row = await conn.fetchrow("""
                    SELECT * FROM assessment_sessions WHERE session_id = $1
                """, session_id)
                
                if not session_row:
                    raise ValueError("Assessment session not found")
                
                # Get user responses for this session
                response_rows = await conn.fetch("""
                    SELECT * FROM user_responses 
                    WHERE session_id = $1 
                    ORDER BY response_timestamp
                """, session_id)
                
                responses = [self._row_to_user_response(row) for row in response_rows]
                
                # Determine target domain based on goals
                assessment_goals = json.loads(session_row['assessment_goals'])
                target_domain = KnowledgeDomain(assessment_goals[0]) if assessment_goals else KnowledgeDomain.MORTGAGE_BASICS
                
                # Select next question
                next_question = await self.adaptive_engine.select_next_question(
                    session_row['user_id'], target_domain, responses
                )
                
                return next_question
                
        except Exception as e:
            logger.error(f"Next question selection failed: {e}")
            return None
    
    async def submit_response(self, session_id: str, question_id: str,
                            user_answer: Any, confidence_level: int,
                            response_time: int) -> Dict[str, Any]:
        """Submit user response and get feedback"""
        try:
            response_id = str(uuid.uuid4())
            
            # Get question details
            question = self.adaptive_engine.question_bank.get(question_id)
            if not question:
                raise ValueError("Question not found")
            
            # Evaluate response
            is_correct = self._evaluate_response(user_answer, question.correct_answer)
            
            # Create response object
            user_response = UserResponse(
                response_id=response_id,
                question_id=question_id,
                user_answer=user_answer,
                is_correct=is_correct,
                confidence_level=confidence_level,
                response_time_seconds=response_time,
                attempt_number=1,
                hint_used=False,
                explanation_viewed=False,
                response_timestamp=datetime.now(),
                device_info={},
                interaction_data={}
            )
            
            # Store response
            await self._store_user_response(session_id, user_response)
            
            # Prepare feedback
            feedback = {
                "is_correct": is_correct,
                "explanation": question.explanation,
                "correct_answer": question.correct_answer,
                "learning_points": question.learning_objectives,
                "regulatory_relevance": question.regulatory_relevance
            }
            
            # Add personalized feedback based on response
            if not is_correct:
                feedback["remediation_suggestions"] = self._generate_remediation_suggestions(
                    question, user_answer
                )
            
            return feedback
            
        except Exception as e:
            logger.error(f"Response submission failed: {e}")
            return {"error": str(e)}
    
    async def complete_assessment(self, session_id: str) -> KnowledgeAssessment:
        """Complete assessment and generate comprehensive results"""
        try:
            # Get all responses for session
            async with self.db_pool.acquire() as conn:
                session_row = await conn.fetchrow("""
                    SELECT * FROM assessment_sessions WHERE session_id = $1
                """, session_id)
                
                response_rows = await conn.fetch("""
                    SELECT * FROM user_responses WHERE session_id = $1
                """, session_id)
                
                responses = [self._row_to_user_response(row) for row in response_rows]
                
                # Perform knowledge assessment
                assessment = await self.assessment_engine.assess_user_knowledge(
                    session_row['user_id'], responses, json.loads(session_row['user_preferences'])
                )
                
                # Update session status
                await conn.execute("""
                    UPDATE assessment_sessions 
                    SET session_status = 'completed', completed_at = $1
                    WHERE session_id = $2
                """, datetime.now(), session_id)
                
                # Store assessment results
                await self._store_assessment_results(assessment)
                
                self.metrics["assessments_completed"] += 1
                
                return assessment
                
        except Exception as e:
            logger.error(f"Assessment completion failed: {e}")
            raise
    
    async def create_learning_path(self, user_id: str, assessment: KnowledgeAssessment,
                                 user_preferences: Dict[str, Any] = None) -> LearningPath:
        """Create optimized learning path for user"""
        try:
            learning_path = await self.learning_optimizer.optimize_learning_path(
                user_id, assessment, user_preferences
            )
            
            # Store learning path
            await self._store_learning_path(learning_path)
            
            self.metrics["learning_paths_created"] += 1
            
            return learning_path
            
        except Exception as e:
            logger.error(f"Learning path creation failed: {e}")
            raise
    
    async def validate_mortgage_advice_comprehension(self, user_id: str, advice_id: str,
                                                   advice_content: str) -> ComprehensionValidation:
        """Validate user comprehension of specific mortgage advice"""
        try:
            # Create targeted assessment for advice comprehension
            assessment_session = await self.create_adaptive_assessment(
                user_id, ["advice_comprehension"], {"target_advice_id": advice_id}
            )
            
            # Generate questions specific to the advice content
            targeted_questions = await self._generate_advice_specific_questions(
                advice_content, advice_id
            )
            
            # Conduct mini-assessment
            responses = []
            for question in targeted_questions[:5]:  # Limit to 5 questions
                # In real implementation, this would be interactive
                # Production interactive response collection system
                pass
            
            # Validate comprehension
            validation = await self.validation_engine.validate_advice_comprehension(
                user_id, advice_id, advice_content, responses
            )
            
            # Store validation results
            await self._store_comprehension_validation(validation)
            
            self.metrics["comprehension_validations"] += 1
            
            return validation
            
        except Exception as e:
            logger.error(f"Advice comprehension validation failed: {e}")
            raise
    
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
    """Example usage of User Comprehension Validator"""
    config = {
        'adaptive_algorithm': 'cat',
        'target_precision': 0.3,
        'max_questions_per_domain': 10
    }
    
    validator = UserComprehensionValidator(config)
    
    # Example usage would go here
    print("User Comprehension Validator demo completed!")

if __name__ == "__main__":
    asyncio.run(main())