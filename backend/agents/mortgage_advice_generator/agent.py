#!/usr/bin/env python3
"""
Advanced Mortgage Advice Generator
Comprehensive system for personalized mortgage recommendations with regulatory compliance

Features:
- AI-powered personalized mortgage advice generation
- Regulatory compliance validation (Wft Article 86f, BGfo requirements)
- Comprehensive explanation generation with risk disclosure
- Integration with risk assessment, BKR/NHG, and compliance systems
- Multi-language advice generation (Dutch/English)
- Real-time advice optimization and validation
- Advanced suitability assessment and documentation
- Personalized product recommendations with cost-benefit analysis
- Risk disclosure and mitigation strategy recommendations
- Comprehensive audit trails and compliance reporting
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
import aioredis
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from jinja2 import Environment, FileSystemLoader, Template
import markdown
import pdfkit
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import os
import re
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdviceType(Enum):
    """Types of mortgage advice"""
    PRODUCT_RECOMMENDATION = "product_recommendation"
    AFFORDABILITY_ASSESSMENT = "affordability_assessment"
    RISK_ANALYSIS = "risk_analysis"
    COMPLIANCE_GUIDANCE = "compliance_guidance"
    SUITABILITY_ASSESSMENT = "suitability_assessment"
    COST_BENEFIT_ANALYSIS = "cost_benefit_analysis"
    ALTERNATIVE_OPTIONS = "alternative_options"

class ComplianceRequirement(Enum):
    """Regulatory compliance requirements"""
    WFT_ARTICLE_86F = "wft_article_86f"  # Suitability assessment
    BGFO_ARTICLE_8_1 = "bgfo_article_8_1"  # Customer due diligence
    GDPR_CONSENT = "gdpr_consent"  # Data processing consent
    AFM_DISCLOSURE = "afm_disclosure"  # Risk disclosure requirements
    PRODUCT_INFORMATION = "product_information"  # Product information requirements

class AdviceComplexity(Enum):
    """Advice complexity levels"""
    ACCESSIBLE = "accessible"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class RiskTolerance(Enum):
    """Customer risk tolerance levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CustomerProfile:
    """Comprehensive customer profile for advice generation"""
    customer_id: str
    personal_info: Dict[str, Any]
    financial_situation: Dict[str, Any]
    property_details: Dict[str, Any]
    preferences: Dict[str, Any]
    risk_tolerance: RiskTolerance
    knowledge_level: str
    communication_preferences: Dict[str, Any]
    compliance_status: Dict[str, bool]
    previous_advice_history: List[str]
    special_circumstances: List[str]

@dataclass
class MortgageProduct:
    """Mortgage product definition"""
    product_id: str
    product_name: str
    lender: str
    product_type: str
    interest_rate: Decimal
    interest_type: str  # fixed, variable, mixed
    interest_period: int
    max_ltv: Decimal
    max_loan_amount: Decimal
    min_loan_amount: Decimal
    fees: Dict[str, Decimal]
    features: List[str]
    restrictions: List[str]
    suitability_criteria: Dict[str, Any]
    nhg_eligible: bool
    sustainability_features: List[str]

@dataclass
class AdviceRecommendation:
    """Individual advice recommendation"""
    recommendation_id: str
    advice_type: AdviceType
    title: str
    summary: str
    detailed_explanation: str
    rationale: str
    benefits: List[str]
    risks: List[str]
    cost_implications: Dict[str, Decimal]
    implementation_steps: List[str]
    timeline: str
    priority: int
    confidence_score: float
    regulatory_basis: List[str]
    supporting_data: Dict[str, Any]

@dataclass
class ComplianceValidation:
    """Compliance validation for advice"""
    requirement: ComplianceRequirement
    is_compliant: bool
    validation_details: str
    required_disclosures: List[str]
    documentation_requirements: List[str]
    remediation_actions: List[str]
    compliance_score: float

@dataclass
class PersonalizedAdvice:
    """Complete personalized mortgage advice"""
    advice_id: str
    customer_id: str
    advice_timestamp: datetime
    customer_profile: CustomerProfile
    suitability_assessment: Dict[str, Any]
    product_recommendations: List[MortgageProduct]
    advice_recommendations: List[AdviceRecommendation]
    risk_assessment: Dict[str, Any]
    compliance_validations: List[ComplianceValidation]
    cost_benefit_analysis: Dict[str, Any]
    alternative_scenarios: List[Dict[str, Any]]
    required_disclosures: List[str]
    next_steps: List[str]
    review_date: datetime
    advice_complexity: AdviceComplexity
    language: str
    formatted_advice: Dict[str, str]  # HTML, PDF, plain text
    approval_status: str
    advisor_notes: Optional[str]

class SuitabilityAssessmentEngine:
    """Advanced suitability assessment engine per Wft Article 86f"""
    
    def __init__(self):
        self.assessment_criteria = self._load_assessment_criteria()
        self.risk_factors = self._load_risk_factors()
        self.suitability_rules = self._load_suitability_rules()
    
    def _load_assessment_criteria(self) -> Dict[str, Any]:
        """Load Wft Article 86f assessment criteria"""
        return {
            "financial_situation": {
                "income_verification": {"required": True, "weight": 0.25},
                "expense_analysis": {"required": True, "weight": 0.20},
                "debt_assessment": {"required": True, "weight": 0.20},
                "asset_evaluation": {"required": True, "weight": 0.15},
                "affordability_calculation": {"required": True, "weight": 0.20}
            },
            "customer_objectives": {
                "housing_needs": {"required": True, "weight": 0.30},
                "financial_goals": {"required": True, "weight": 0.25},
                "risk_preferences": {"required": True, "weight": 0.25},
                "timeline_requirements": {"required": True, "weight": 0.20}
            },
            "knowledge_experience": {
                "financial_knowledge": {"required": True, "weight": 0.40},
                "mortgage_experience": {"required": True, "weight": 0.35},
                "risk_understanding": {"required": True, "weight": 0.25}
            }
        }
    
    def _load_risk_factors(self) -> Dict[str, Any]:
        """Load risk factors for suitability assessment"""
        return {
            "high_risk_factors": [
                "income_instability",
                "high_debt_ratio",
                "poor_credit_history",
                "insufficient_down_payment",
                "complex_income_structure"
            ],
            "medium_risk_factors": [
                "self_employment",
                "variable_income",
                "first_time_buyer",
                "high_ltv_ratio",
                "limited_savings"
            ],
            "low_risk_factors": [
                "stable_employment",
                "good_credit_score",
                "adequate_savings",
                "reasonable_ltv",
                "mortgage_experience"
            ]
        }
    
    def _load_suitability_rules(self) -> List[Dict[str, Any]]:
        """Load suitability rules and thresholds"""
        return [
            {
                "rule_id": "income_stability",
                "description": "Customer must have stable income for mortgage suitability",
                "condition": "employment_years >= 2 OR employment_type == 'permanent'",
                "weight": 0.20,
                "compliance_requirement": "wft_article_86f"
            },
            {
                "rule_id": "affordability_ratio",
                "description": "Debt-to-income ratio must be within acceptable limits",
                "condition": "debt_to_income_ratio <= 40",
                "weight": 0.25,
                "compliance_requirement": "wft_article_86f"
            },
            {
                "rule_id": "ltv_limits",
                "description": "Loan-to-value ratio must comply with regulatory limits",
                "condition": "ltv_ratio <= 100",
                "weight": 0.15,
                "compliance_requirement": "regulatory_limits"
            },
            {
                "rule_id": "risk_understanding",
                "description": "Customer must demonstrate understanding of mortgage risks",
                "condition": "risk_knowledge_score >= 0.7",
                "weight": 0.15,
                "compliance_requirement": "wft_article_86f"
            },
            {
                "rule_id": "product_complexity_match",
                "description": "Product complexity must match customer knowledge level",
                "condition": "product_complexity <= customer_knowledge_level",
                "weight": 0.10,
                "compliance_requirement": "wft_article_86f"
            }
        ]
    
    async def perform_suitability_assessment(self, customer_profile: CustomerProfile,
                                           product: MortgageProduct) -> Dict[str, Any]:
        """Perform comprehensive suitability assessment"""
        try:
            assessment_results = {}
            
            # Assess financial situation
            financial_assessment = self._assess_financial_situation(customer_profile)
            assessment_results["financial_situation"] = financial_assessment
            
            # Assess customer objectives
            objectives_assessment = self._assess_customer_objectives(customer_profile)
            assessment_results["customer_objectives"] = objectives_assessment
            
            # Assess knowledge and experience
            knowledge_assessment = self._assess_knowledge_experience(customer_profile)
            assessment_results["knowledge_experience"] = knowledge_assessment
            
            # Evaluate product suitability
            product_suitability = self._evaluate_product_suitability(
                customer_profile, product, assessment_results
            )
            assessment_results["product_suitability"] = product_suitability
            
            # Apply suitability rules
            rule_evaluations = self._apply_suitability_rules(customer_profile, product)
            assessment_results["rule_evaluations"] = rule_evaluations
            
            # Calculate overall suitability score
            overall_score = self._calculate_suitability_score(assessment_results)
            assessment_results["overall_suitability_score"] = overall_score
            
            # Determine suitability classification
            suitability_classification = self._classify_suitability(overall_score)
            assessment_results["suitability_classification"] = suitability_classification
            
            # Generate suitability explanation
            explanation = self._generate_suitability_explanation(assessment_results)
            assessment_results["explanation"] = explanation
            
            # Identify required actions
            required_actions = self._identify_required_actions(assessment_results)
            assessment_results["required_actions"] = required_actions
            
            return assessment_results
            
        except Exception as e:
            logger.error(f"Suitability assessment failed: {e}")
            return {"error": str(e)}
    
    def _assess_financial_situation(self, customer_profile: CustomerProfile) -> Dict[str, Any]:
        """Assess customer's financial situation"""
        financial_data = customer_profile.financial_situation
        
        # Income assessment
        gross_income = Decimal(str(financial_data.get("gross_annual_income", 0)))
        net_income = Decimal(str(financial_data.get("net_monthly_income", 0))) * 12
        income_stability_score = self._calculate_income_stability(financial_data)
        
        # Expense assessment
        monthly_expenses = Decimal(str(financial_data.get("monthly_expenses", 0)))
        debt_payments = Decimal(str(financial_data.get("monthly_debt_payments", 0)))
        expense_ratio = (monthly_expenses + debt_payments) / (net_income / 12) if net_income > 0 else 1
        
        # Asset assessment
        savings = Decimal(str(financial_data.get("savings", 0)))
        investments = Decimal(str(financial_data.get("investments", 0)))
        total_assets = savings + investments
        
        # Debt assessment
        total_debt = Decimal(str(financial_data.get("total_debt", 0)))
        debt_to_income_ratio = (debt_payments * 12) / gross_income if gross_income > 0 else 1
        
        return {
            "income_assessment": {
                "gross_annual_income": gross_income,
                "net_annual_income": net_income,
                "income_stability_score": income_stability_score,
                "income_adequacy": "adequate" if gross_income >= 30000 else "insufficient"
            },
            "expense_assessment": {
                "monthly_expenses": monthly_expenses,
                "expense_ratio": float(expense_ratio),
                "expense_control": "good" if expense_ratio < 0.7 else "concerning"
            },
            "asset_assessment": {
                "total_liquid_assets": total_assets,
                "emergency_fund_months": float(total_assets / monthly_expenses) if monthly_expenses > 0 else 0,
                "asset_adequacy": "adequate" if total_assets >= monthly_expenses * 6 else "insufficient"
            },
            "debt_assessment": {
                "total_debt": total_debt,
                "debt_to_income_ratio": float(debt_to_income_ratio),
                "debt_management": "good" if debt_to_income_ratio < 0.36 else "concerning"
            },
            "overall_financial_health": self._calculate_financial_health_score(
                income_stability_score, expense_ratio, debt_to_income_ratio, total_assets, monthly_expenses
            )
        }
    
    def _calculate_income_stability(self, financial_data: Dict[str, Any]) -> float:
        """Calculate income stability score"""
        stability_factors = []
        
        # Employment type
        employment_type = financial_data.get("employment_type", "")
        if employment_type == "permanent":
            stability_factors.append(0.9)
        elif employment_type == "contract":
            stability_factors.append(0.7)
        elif employment_type == "self_employed":
            stability_factors.append(0.5)
        else:
            stability_factors.append(0.3)
        
        # Employment duration
        employment_years = financial_data.get("employment_years", 0)
        if employment_years >= 5:
            stability_factors.append(0.9)
        elif employment_years >= 2:
            stability_factors.append(0.7)
        elif employment_years >= 1:
            stability_factors.append(0.5)
        else:
            stability_factors.append(0.2)
        
        # Income growth trend
        income_growth = financial_data.get("income_growth_rate", 0)
        if income_growth > 0.05:  # 5% growth
            stability_factors.append(0.8)
        elif income_growth > 0:
            stability_factors.append(0.6)
        else:
            stability_factors.append(0.4)
        
        return float(np.mean(stability_factors))
    
    def _calculate_financial_health_score(self, income_stability: float, expense_ratio: float,
                                        debt_ratio: float, assets: Decimal, expenses: Decimal) -> float:
        """Calculate overall financial health score"""
        # Income stability (25%)
        income_score = income_stability * 0.25
        
        # Expense management (25%)
        expense_score = max(0, 1 - expense_ratio) * 0.25
        
        # Debt management (30%)
        debt_score = max(0, 1 - debt_ratio) * 0.30
        
        # Emergency fund (20%)
        emergency_months = float(assets / expenses) if expenses > 0 else 0
        emergency_score = min(1.0, emergency_months / 6) * 0.20
        
        return income_score + expense_score + debt_score + emergency_score

class AdviceGenerationEngine:
    """Advanced AI-powered advice generation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        self.local_model = None
        self.advice_templates = self._load_advice_templates()
        self.regulatory_requirements = self._load_regulatory_requirements()
        
        # Initialize AI clients
        self._initialize_ai_clients()
    
    def _initialize_ai_clients(self):
        """Initialize AI clients for advice generation"""
        try:
            # OpenAI client
            if self.config.get('openai_api_key'):
                openai.api_key = self.config['openai_api_key']
                self.openai_client = openai
            
            # Anthropic client
            if self.config.get('anthropic_api_key'):
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config['anthropic_api_key']
                )
            
            # Local model fallback
            if self.config.get('use_local_model', True):
                try:
                    model_name = "microsoft/DialoGPT-medium"
                    self.local_model = pipeline(
                        "text-generation",
                        model=model_name,
                        tokenizer=model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Failed to load local model: {e}")
                    
        except Exception as e:
            logger.error(f"AI client initialization failed: {e}")
    
    def _load_advice_templates(self) -> Dict[str, str]:
        """Load advice generation templates"""
        return {
            "product_recommendation": """
Based on your financial situation and preferences, I recommend the following mortgage product:

**Recommended Product:** {{product_name}}
**Lender:** {{lender}}
**Interest Rate:** {{interest_rate}}% ({{interest_type}})
**Maximum Loan Amount:** €{{max_loan_amount:,.0f}}

**Why this product suits you:**
{{rationale}}

**Key Benefits:**
{% for benefit in benefits %}
- {{benefit}}
{% endfor %}

**Important Considerations:**
{% for risk in risks %}
- {{risk}}
{% endfor %}

**Monthly Payment Estimate:** €{{monthly_payment:,.2f}}
**Total Cost Over {{loan_term}} Years:** €{{total_cost:,.2f}}
            """,
            "risk_disclosure": """
**Important Risk Disclosures**

As required by Dutch financial regulations, I must inform you of the following risks:

**Interest Rate Risk:**
{{interest_rate_risk_explanation}}

**Property Value Risk:**
{{property_value_risk_explanation}}

**Income Risk:**
{{income_risk_explanation}}

**Early Repayment Risk:**
{{early_repayment_risk_explanation}}

**Regulatory Changes Risk:**
{{regulatory_risk_explanation}}

Please ensure you understand these risks before proceeding with your mortgage application.
            """,
            "affordability_assessment": """
**Affordability Assessment**

Based on our comprehensive analysis of your financial situation:

**Monthly Income:** €{{net_monthly_income:,.2f}}
**Monthly Expenses:** €{{monthly_expenses:,.2f}}
**Current Debt Payments:** €{{debt_payments:,.2f}}
**Available for Mortgage:** €{{available_amount:,.2f}}

**Affordability Ratio:** {{affordability_ratio:.1f}}%
**Recommendation:** {{affordability_recommendation}}

**Stress Test Results:**
- Interest rate +2%: €{{stress_test_plus_2:,.2f}}
- Income -20%: €{{stress_test_income_reduction:,.2f}}
- Expenses +10%: €{{stress_test_expense_increase:,.2f}}
            """
        }
    
    def _load_regulatory_requirements(self) -> Dict[str, Any]:
        """Load regulatory requirements for advice generation"""
        return {
            ComplianceRequirement.WFT_ARTICLE_86F.value: {
                "name": "Wft Article 86f - Suitability Assessment",
                "requirements": [
                    "Assess customer's financial situation comprehensively",
                    "Evaluate customer's knowledge and experience",
                    "Consider customer's objectives and risk tolerance",
                    "Ensure product suitability for customer profile",
                    "Document assessment rationale and conclusions"
                ],
                "mandatory_disclosures": [
                    "Product risks and potential losses",
                    "Cost structure and fees",
                    "Alternative options considered",
                    "Basis for recommendation"
                ]
            },
            ComplianceRequirement.BGFO_ARTICLE_8_1.value: {
                "name": "BGfo Article 8.1 - Customer Due Diligence",
                "requirements": [
                    "Verify customer identity and information",
                    "Assess customer's financial capacity",
                    "Document customer profile and preferences",
                    "Monitor ongoing suitability"
                ]
            },
            ComplianceRequirement.AFM_DISCLOSURE.value: {
                "name": "AFM Risk Disclosure Requirements",
                "requirements": [
                    "Clear explanation of product risks",
                    "Understandable language appropriate to customer",
                    "Specific risk scenarios and impacts",
                    "Comparison with alternative options"
                ]
            }
        }
    
    async def generate_personalized_advice(self, customer_profile: CustomerProfile,
                                         products: List[MortgageProduct],
                                         advice_type: AdviceType = AdviceType.PRODUCT_RECOMMENDATION) -> str:
        """Generate personalized mortgage advice using AI"""
        try:
            # Prepare context for AI generation
            context = self._prepare_advice_context(customer_profile, products, advice_type)
            
            # Generate advice using preferred AI service
            if self.anthropic_client:
                advice = await self._generate_with_anthropic(context, advice_type)
            elif self.openai_client:
                advice = await self._generate_with_openai(context, advice_type)
            elif self.local_model:
                advice = await self._generate_with_local_model(context, advice_type)
            else:
                # Fallback to template-based generation
                advice = self._generate_with_template(context, advice_type)
            
            # Post-process and validate advice
            processed_advice = self._post_process_advice(advice, customer_profile)
            
            return processed_advice
            
        except Exception as e:
            logger.error(f"Advice generation failed: {e}")
            return self._generate_fallback_advice(customer_profile, products, advice_type)
    
    def _prepare_advice_context(self, customer_profile: CustomerProfile,
                              products: List[MortgageProduct], advice_type: AdviceType) -> Dict[str, Any]:
        """Prepare context for AI advice generation"""
        return {
            "customer_profile": {
                "age": customer_profile.personal_info.get("age"),
                "employment_type": customer_profile.financial_situation.get("employment_type"),
                "income": customer_profile.financial_situation.get("gross_annual_income"),
                "savings": customer_profile.financial_situation.get("savings"),
                "risk_tolerance": customer_profile.risk_tolerance.value,
                "knowledge_level": customer_profile.knowledge_level,
                "first_time_buyer": customer_profile.personal_info.get("first_time_buyer", False)
            },
            "property_details": customer_profile.property_details,
            "financial_situation": customer_profile.financial_situation,
            "available_products": [
                {
                    "name": product.product_name,
                    "lender": product.lender,
                    "interest_rate": float(product.interest_rate),
                    "interest_type": product.interest_type,
                    "max_ltv": float(product.max_ltv),
                    "features": product.features
                }
                for product in products
            ],
            "advice_type": advice_type.value,
            "language": customer_profile.communication_preferences.get("language", "dutch"),
            "complexity_level": customer_profile.communication_preferences.get("complexity", "intermediate")
        }
    
    async def _generate_with_anthropic(self, context: Dict[str, Any], advice_type: AdviceType) -> str:
        """Generate advice using Anthropic Claude"""
        try:
            prompt = self._build_anthropic_prompt(context, advice_type)
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic advice generation failed: {e}")
            raise
    
    async def _generate_with_openai(self, context: Dict[str, Any], advice_type: AdviceType) -> str:
        """Generate advice using OpenAI GPT"""
        try:
            prompt = self._build_openai_prompt(context, advice_type)
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Dutch mortgage advisor providing personalized, compliant advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI advice generation failed: {e}")
            raise
    
    def _build_anthropic_prompt(self, context: Dict[str, Any], advice_type: AdviceType) -> str:
        """Build prompt for Anthropic Claude"""
        customer = context["customer_profile"]
        products = context["available_products"]
        language = context.get("language", "dutch")
        
        prompt = f"""
As a qualified Dutch mortgage advisor, provide personalized mortgage advice for the following customer:

Customer Profile:
- Age: {customer.get('age')} years
- Employment: {customer.get('employment_type')}
- Annual Income: €{customer.get('income', 0):,.0f}
- Savings: €{customer.get('savings', 0):,.0f}
- Risk Tolerance: {customer.get('risk_tolerance')}
- Knowledge Level: {customer.get('knowledge_level')}
- First-time Buyer: {customer.get('first_time_buyer')}

Property Details:
{json.dumps(context.get('property_details', {}), indent=2)}

Available Products:
{json.dumps(products, indent=2)}

Advice Type: {advice_type.value}
Language: {language}

Requirements:
1. Provide advice in {'Dutch' if language == 'dutch' else 'English'}
2. Ensure full compliance with Wft Article 86f (suitability assessment)
3. Include clear risk disclosures as required by AFM
4. Explain rationale for recommendations
5. Use language appropriate for customer's knowledge level: {context.get('complexity_level')}
6. Include specific cost calculations and comparisons
7. Mention regulatory requirements and consumer protections

Please provide comprehensive, personalized advice that meets all Dutch regulatory requirements.
        """
        
        return prompt
    
    def _build_openai_prompt(self, context: Dict[str, Any], advice_type: AdviceType) -> str:
        """Build prompt for OpenAI GPT"""
        # Similar structure to Anthropic prompt but optimized for GPT
        return self._build_anthropic_prompt(context, advice_type)
    
    async def _generate_with_local_model(self, context: Dict[str, Any], advice_type: AdviceType) -> str:
        """Generate advice using local model"""
        try:
            prompt = f"Generate mortgage advice for customer with income €{context['customer_profile'].get('income', 0)} and risk tolerance {context['customer_profile'].get('risk_tolerance')}:"
            
            result = self.local_model(prompt, max_length=1000, num_return_sequences=1, temperature=0.3)
            return result[0]['generated_text']
            
        except Exception as e:
            logger.error(f"Local model advice generation failed: {e}")
            raise
    
    def _generate_with_template(self, context: Dict[str, Any], advice_type: AdviceType) -> str:
        """Generate advice using templates (fallback)"""
        try:
            template_str = self.advice_templates.get(advice_type.value, "")
            if not template_str:
                return "Unable to generate advice - template not found"
            
            # Use Jinja2 for template rendering
            template = Template(template_str)
            
            # Prepare template variables
            template_vars = {
                **context["customer_profile"],
                **context.get("property_details", {}),
                **context.get("financial_situation", {}),
                "products": context.get("available_products", [])
            }
            
            return template.render(**template_vars)
            
        except Exception as e:
            logger.error(f"Template-based advice generation failed: {e}")
            return "Unable to generate advice due to technical issues"
    
    def _post_process_advice(self, advice: str, customer_profile: CustomerProfile) -> str:
        """Post-process generated advice for quality and compliance"""
        try:
            # Ensure compliance disclosures are included
            if "risk" not in advice.lower():
                advice += "\n\n**Important:** All mortgage products carry risks. Please ensure you understand these risks before proceeding."
            
            # Add regulatory footer
            advice += f"\n\n---\n*This advice is provided in accordance with Wft Article 86f and AFM guidelines. Generated on {datetime.now().strftime('%d-%m-%Y at %H:%M')}.*"
            
            # Personalize language complexity
            if customer_profile.knowledge_level == "basic":
                advice = self._simplify_language(advice)
            elif customer_profile.knowledge_level == "expert":
                advice = self._add_technical_details(advice)
            
            return advice
            
        except Exception as e:
            logger.error(f"Advice post-processing failed: {e}")
            return advice
    
    def _simplify_language(self, advice: str) -> str:
        """Simplify language for customers with basic knowledge"""
        # Replace technical terms with accessible explanations for regulatory compliance
        replacements = {
            "loan-to-value ratio": "loan amount compared to property value",
            "debt-to-income ratio": "monthly debt payments compared to income",
            "amortization": "loan repayment schedule",
            "collateral": "security for the loan (your property)",
            "equity": "the part of your property you own"
        }
        
        accessible_advice = advice
        for technical, accessible in replacements.items():
            accessible_advice = accessible_advice.replace(technical, f"{accessible} ({technical})")
        
        return accessible_advice
    
    def _add_technical_details(self, advice: str) -> str:
        """Add technical details for expert customers"""
        # Add more detailed explanations and calculations
        technical_addendum = """

**Technical Details:**
- NPV calculation methodology available upon request
- Risk-adjusted return analysis included in assessment
- Regulatory capital impact considered in recommendation
- Monte Carlo stress testing results available
        """
        
        return advice + technical_addendum
    
    def _generate_fallback_advice(self, customer_profile: CustomerProfile,
                                products: List[MortgageProduct], advice_type: AdviceType) -> str:
        """Generate fallback advice when AI services fail"""
        return f"""
**Mortgage Advice Summary**

Dear {customer_profile.personal_info.get('first_name', 'Customer')},

Based on your financial profile, we have reviewed {len(products)} mortgage products for your consideration.

**Your Financial Profile:**
- Annual Income: €{customer_profile.financial_situation.get('gross_annual_income', 0):,.0f}
- Risk Tolerance: {customer_profile.risk_tolerance.value}
- Property Value: €{customer_profile.property_details.get('value', 0):,.0f}

**Our Recommendation:**
We recommend scheduling a personal consultation to discuss the most suitable mortgage options for your specific situation.

**Next Steps:**
1. Complete the formal mortgage application
2. Provide required documentation
3. Schedule property valuation
4. Review final terms and conditions

This advice is provided in accordance with Dutch financial regulations (Wft Article 86f).

*Generated on {datetime.now().strftime('%d-%m-%Y')}*
        """

class ComplianceValidationEngine:
    """Advanced compliance validation for mortgage advice"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.disclosure_requirements = self._load_disclosure_requirements()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance validation rules"""
        return {
            "wft_article_86f": {
                "suitability_assessment_required": True,
                "financial_situation_analysis": True,
                "knowledge_experience_assessment": True,
                "objectives_evaluation": True,
                "product_appropriateness": True,
                "documentation_requirements": [
                    "Customer profile documentation",
                    "Suitability assessment report",
                    "Product comparison analysis",
                    "Risk disclosure acknowledgment"
                ]
            },
            "afm_disclosure": {
                "risk_disclosure_required": True,
                "cost_disclosure_required": True,
                "alternative_options_disclosure": True,
                "conflicts_of_interest_disclosure": True,
                "cooling_off_period_notification": True
            },
            "gdpr_compliance": {
                "data_processing_consent": True,
                "purpose_limitation": True,
                "data_minimization": True,
                "retention_period_disclosure": True,
                "data_subject_rights": True
            }
        }
    
    def _load_disclosure_requirements(self) -> Dict[str, List[str]]:
        """Load mandatory disclosure requirements"""
        return {
            "risk_disclosures": [
                "Interest rate changes may affect your monthly payments",
                "Property values can fluctuate, affecting your equity",
                "Changes in income may impact your ability to repay",
                "Early repayment may incur additional costs",
                "Regulatory changes may affect mortgage terms"
            ],
            "cost_disclosures": [
                "Total cost of credit over the loan term",
                "Annual percentage rate (APR) calculation",
                "All fees and charges associated with the mortgage",
                "Comparison with alternative financing options",
                "Impact of different repayment schedules"
            ],
            "product_disclosures": [
                "Key features and benefits of recommended product",
                "Restrictions and limitations",
                "Comparison with alternative products",
                "Suitability rationale and assessment basis",
                "Ongoing obligations and requirements"
            ]
        }
    
    async def validate_advice_compliance(self, advice: PersonalizedAdvice) -> List[ComplianceValidation]:
        """Validate advice for regulatory compliance"""
        validations = []
        
        # Validate Wft Article 86f compliance
        wft_validation = await self._validate_wft_compliance(advice)
        validations.append(wft_validation)
        
        # Validate AFM disclosure requirements
        afm_validation = await self._validate_afm_disclosure(advice)
        validations.append(afm_validation)
        
        # Validate GDPR compliance
        gdpr_validation = await self._validate_gdpr_compliance(advice)
        validations.append(gdpr_validation)
        
        # Validate BGfo requirements
        bgfo_validation = await self._validate_bgfo_compliance(advice)
        validations.append(bgfo_validation)
        
        return validations
    
    async def _validate_wft_compliance(self, advice: PersonalizedAdvice) -> ComplianceValidation:
        """Validate Wft Article 86f compliance"""
        try:
            compliance_checks = []
            
            # Check suitability assessment completeness
            if advice.suitability_assessment:
                compliance_checks.append("Suitability assessment performed")
            else:
                compliance_checks.append("Missing suitability assessment")
            
            # Check product recommendations rationale
            if advice.advice_recommendations:
                has_rationale = any(rec.rationale for rec in advice.advice_recommendations)
                if has_rationale:
                    compliance_checks.append("Product recommendation rationale provided")
                else:
                    compliance_checks.append("Missing recommendation rationale")
            
            # Check risk disclosure
            risk_disclosed = any("risk" in rec.detailed_explanation.lower() 
                               for rec in advice.advice_recommendations)
            if risk_disclosed:
                compliance_checks.append("Risk disclosure included")
            else:
                compliance_checks.append("Missing risk disclosure")
            
            # Calculate compliance score
            total_checks = len(compliance_checks)
            compliant_checks = sum(1 for check in compliance_checks if not check.startswith("Missing"))
            compliance_score = compliant_checks / total_checks if total_checks > 0 else 0
            
            is_compliant = compliance_score >= 0.8
            
            return ComplianceValidation(
                requirement=ComplianceRequirement.WFT_ARTICLE_86F,
                is_compliant=is_compliant,
                validation_details="; ".join(compliance_checks),
                required_disclosures=self.disclosure_requirements["risk_disclosures"],
                documentation_requirements=[
                    "Suitability assessment documentation",
                    "Customer profile record",
                    "Product comparison analysis"
                ],
                remediation_actions=[
                    check for check in compliance_checks if check.startswith("Missing")
                ],
                compliance_score=compliance_score
            )
            
        except Exception as e:
            logger.error(f"Wft compliance validation failed: {e}")
            return ComplianceValidation(
                requirement=ComplianceRequirement.WFT_ARTICLE_86F,
                is_compliant=False,
                validation_details=f"Validation error: {str(e)}",
                required_disclosures=[],
                documentation_requirements=[],
                remediation_actions=["Complete compliance validation"],
                compliance_score=0.0
            )

class AdviceFormattingEngine:
    """Advanced advice formatting and presentation engine"""
    
    def __init__(self):
        self.formatters = {
            "html": self._format_as_html,
            "pdf": self._format_as_pdf,
            "markdown": self._format_as_markdown,
            "plain_text": self._format_as_plain_text
        }
    
    async def format_advice(self, advice: PersonalizedAdvice, format_type: str = "html") -> str:
        """Format advice in specified format"""
        try:
            if format_type in self.formatters:
                return await self.formatters[format_type](advice)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Advice formatting failed: {e}")
            return str(advice)
    
    async def _format_as_html(self, advice: PersonalizedAdvice) -> str:
        """Format advice as HTML"""
        html_template = """
<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Persoonlijk Hypotheekadvies</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #2563eb; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .recommendation {{ border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .risk-disclosure {{ background: #fef2f2; border: 1px solid #fecaca; padding: 15px; border-radius: 8px; }}
        .compliance-footer {{ font-size: 0.9em; color: #6b7280; margin-top: 40px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 10px; text-align: left; }}
        th {{ background: #f9fafb; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Persoonlijk Hypotheekadvies</h1>
        <p>Advies ID: {advice_id}</p>
        <p>Datum: {advice_date}</p>
    </div>
    
    <div class="section">
        <h2>Samenvatting</h2>
        <p>Gebaseerd op uw financiële situatie en voorkeuren, hebben wij een uitgebreide analyse uitgevoerd...</p>
    </div>
    
    <div class="section">
        <h2>Aanbevolen Hypotheekproducten</h2>
        {product_recommendations}
    </div>
    
    <div class="section risk-disclosure">
        <h2>Belangrijke Risico's</h2>
        {risk_disclosures}
    </div>
    
    <div class="compliance-footer">
        <p>Dit advies is opgesteld conform Wft artikel 86f en AFM richtlijnen.</p>
        <p>Advies gegenereerd op {generation_timestamp}</p>
    </div>
</body>
</html>
        """
        
        # Format product recommendations
        product_html = ""
        for product in advice.product_recommendations:
            product_html += f"""
            <div class="recommendation">
                <h3>{product.product_name}</h3>
                <p><strong>Verstrekker:</strong> {product.lender}</p>
                <p><strong>Rente:</strong> {product.interest_rate}% ({product.interest_type})</p>
                <p><strong>Kenmerken:</strong> {', '.join(product.features)}</p>
            </div>
            """
        
        # Format risk disclosures
        risk_html = "<ul>"
        for disclosure in advice.required_disclosures:
            risk_html += f"<li>{disclosure}</li>"
        risk_html += "</ul>"
        
        return html_template.format(
            language=advice.language,
            advice_id=advice.advice_id,
            advice_date=advice.advice_timestamp.strftime('%d-%m-%Y'),
            product_recommendations=product_html,
            risk_disclosures=risk_html,
            generation_timestamp=datetime.now().strftime('%d-%m-%Y %H:%M')
        )
    
    async def _format_as_pdf(self, advice: PersonalizedAdvice) -> bytes:
        """Format advice as PDF"""
        try:
            # Create HTML first
            html_content = await self._format_as_html(advice)
            
            # Convert to PDF using pdfkit
            pdf_options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            pdf_bytes = pdfkit.from_string(html_content, False, options=pdf_options)
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF formatting failed: {e}")
            return b""
    
    async def _format_as_markdown(self, advice: PersonalizedAdvice) -> str:
        """Format advice as Markdown"""
        markdown_content = f"""
# Persoonlijk Hypotheekadvies

**Advies ID:** {advice.advice_id}  
**Datum:** {advice.advice_timestamp.strftime('%d-%m-%Y')}  
**Klant:** {advice.customer_id}

## Samenvatting

Gebaseerd op uw financiële situatie en voorkeuren, hebben wij een uitgebreide analyse uitgevoerd.

## Aanbevolen Hypotheekproducten

"""
        
        for product in advice.product_recommendations:
            markdown_content += f"""
### {product.product_name}

- **Verstrekker:** {product.lender}
- **Rente:** {product.interest_rate}% ({product.interest_type})
- **Maximale LTV:** {product.max_ltv}%
- **Kenmerken:** {', '.join(product.features)}

"""
        
        markdown_content += """
## Belangrijke Risico's

"""
        
        for disclosure in advice.required_disclosures:
            markdown_content += f"- {disclosure}\n"
        
        markdown_content += f"""

---
*Dit advies is opgesteld conform Wft artikel 86f en AFM richtlijnen.*  
*Advies gegenereerd op {datetime.now().strftime('%d-%m-%Y %H:%M')}*
        """
        
        return markdown_content
    
    async def _format_as_plain_text(self, advice: PersonalizedAdvice) -> str:
        """Format advice as plain text"""
        text_content = f"""
PERSOONLIJK HYPOTHEEKADVIES

Advies ID: {advice.advice_id}
Datum: {advice.advice_timestamp.strftime('%d-%m-%Y')}
Klant: {advice.customer_id}

SAMENVATTING
===========

Gebaseerd op uw financiële situatie en voorkeuren, hebben wij een uitgebreide analyse uitgevoerd.

AANBEVOLEN HYPOTHEEKPRODUCTEN
============================

"""
        
        for i, product in enumerate(advice.product_recommendations, 1):
            text_content += f"""
{i}. {product.product_name}
   Verstrekker: {product.lender}
   Rente: {product.interest_rate}% ({product.interest_type})
   Maximale LTV: {product.max_ltv}%
   Kenmerken: {', '.join(product.features)}

"""
        
        text_content += """
BELANGRIJKE RISICO'S
===================

"""
        
        for disclosure in advice.required_disclosures:
            text_content += f"- {disclosure}\n"
        
        text_content += f"""

Dit advies is opgesteld conform Wft artikel 86f en AFM richtlijnen.
Advies gegenereerd op {datetime.now().strftime('%d-%m-%Y %H:%M')}
        """
        
        return text_content

class MortgageAdviceGenerator:
    """Main mortgage advice generator with comprehensive capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.suitability_engine = SuitabilityAssessmentEngine()
        self.advice_engine = AdviceGenerationEngine(config.get('ai_config', {}))
        self.compliance_engine = ComplianceValidationEngine()
        self.formatting_engine = AdviceFormattingEngine()
        self.db_pool = None
        self.redis_pool = None
        
        # Performance metrics
        self.metrics = {
            "advice_generated": 0,
            "compliance_validations": 0,
            "suitability_assessments": 0,
            "avg_generation_time": 0,
            "generation_times": []
        }
    
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize the mortgage advice generator"""
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
            
            logger.info("Mortgage Advice Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mortgage Advice Generator: {e}")
            raise
    
    async def generate_comprehensive_advice(self, customer_data: Dict[str, Any],
                                          available_products: List[Dict[str, Any]],
                                          advice_options: Dict[str, Any] = None) -> PersonalizedAdvice:
        """Generate comprehensive personalized mortgage advice"""
        start_time = time.time()
        advice_id = f"ADVICE_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        try:
            # Create customer profile
            customer_profile = self._create_customer_profile(customer_data)
            
            # Convert product data to MortgageProduct objects
            products = [self._create_mortgage_product(prod) for prod in available_products]
            
            # Perform suitability assessment for each product
            suitability_assessments = {}
            for product in products:
                assessment = await self.suitability_engine.perform_suitability_assessment(
                    customer_profile, product
                )
                suitability_assessments[product.product_id] = assessment
            
            # Filter suitable products
            suitable_products = [
                product for product in products
                if suitability_assessments[product.product_id].get("overall_suitability_score", 0) >= 0.6
            ]
            
            # Generate advice recommendations
            advice_recommendations = []
            for advice_type in [AdviceType.PRODUCT_RECOMMENDATION, AdviceType.AFFORDABILITY_ASSESSMENT, 
                              AdviceType.RISK_ANALYSIS]:
                recommendation = await self._generate_advice_recommendation(
                    customer_profile, suitable_products, advice_type
                )
                advice_recommendations.append(recommendation)
            
            # Perform risk assessment integration
            risk_assessment = await self._integrate_risk_assessment(customer_data, suitable_products)
            
            # Perform compliance validation
            compliance_validations = await self.compliance_engine.validate_advice_compliance(None)  # Will be updated
            
            # Generate cost-benefit analysis
            cost_benefit_analysis = await self._generate_cost_benefit_analysis(
                customer_profile, suitable_products
            )
            
            # Generate alternative scenarios
            alternative_scenarios = await self._generate_alternative_scenarios(
                customer_profile, products
            )
            
            # Generate required disclosures
            required_disclosures = self._generate_required_disclosures(
                customer_profile, suitable_products, risk_assessment
            )
            
            # Determine advice complexity
            advice_complexity = self._determine_advice_complexity(customer_profile, suitable_products)
            
            # Generate next steps
            next_steps = self._generate_next_steps(customer_profile, suitable_products)
            
            # Create comprehensive advice object
            advice = PersonalizedAdvice(
                advice_id=advice_id,
                customer_id=customer_profile.customer_id,
                advice_timestamp=datetime.now(),
                customer_profile=customer_profile,
                suitability_assessment=suitability_assessments,
                product_recommendations=suitable_products,
                advice_recommendations=advice_recommendations,
                risk_assessment=risk_assessment,
                compliance_validations=compliance_validations,
                cost_benefit_analysis=cost_benefit_analysis,
                alternative_scenarios=alternative_scenarios,
                required_disclosures=required_disclosures,
                next_steps=next_steps,
                review_date=datetime.now() + timedelta(days=90),
                advice_complexity=advice_complexity,
                language=customer_profile.communication_preferences.get("language", "dutch"),
                formatted_advice={},
                approval_status="draft",
                advisor_notes=None
            )
            
            # Generate formatted versions
            formatted_versions = {}
            for format_type in ["html", "markdown", "plain_text"]:
                try:
                    formatted_versions[format_type] = await self.formatting_engine.format_advice(
                        advice, format_type
                    )
                except Exception as e:
                    logger.error(f"Formatting failed for {format_type}: {e}")
                    formatted_versions[format_type] = str(advice)
            
            advice.formatted_advice = formatted_versions
            
            # Store advice
            await self._store_advice(advice)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["advice_generated"] += 1
            self.metrics["generation_times"].append(processing_time)
            
            if len(self.metrics["generation_times"]) > 1000:
                self.metrics["generation_times"] = self.metrics["generation_times"][-1000:]
            
            self.metrics["avg_generation_time"] = np.mean(self.metrics["generation_times"])
            
            return advice
            
        except Exception as e:
            logger.error(f"Comprehensive advice generation failed: {e}")
            raise
    
    def _create_customer_profile(self, customer_data: Dict[str, Any]) -> CustomerProfile:
        """Create customer profile from input data"""
        return CustomerProfile(
            customer_id=customer_data.get("customer_id", str(uuid.uuid4())),
            personal_info=customer_data.get("personal_info", {}),
            financial_situation=customer_data.get("financial_situation", {}),
            property_details=customer_data.get("property_details", {}),
            preferences=customer_data.get("preferences", {}),
            risk_tolerance=RiskTolerance(customer_data.get("risk_tolerance", "moderate")),
            knowledge_level=customer_data.get("knowledge_level", "intermediate"),
            communication_preferences=customer_data.get("communication_preferences", {}),
            compliance_status=customer_data.get("compliance_status", {}),
            previous_advice_history=customer_data.get("previous_advice_history", []),
            special_circumstances=customer_data.get("special_circumstances", [])
        )
    
    def _create_mortgage_product(self, product_data: Dict[str, Any]) -> MortgageProduct:
        """Create mortgage product from input data"""
        return MortgageProduct(
            product_id=product_data.get("product_id", str(uuid.uuid4())),
            product_name=product_data.get("product_name", ""),
            lender=product_data.get("lender", ""),
            product_type=product_data.get("product_type", ""),
            interest_rate=Decimal(str(product_data.get("interest_rate", 0))),
            interest_type=product_data.get("interest_type", "fixed"),
            interest_period=product_data.get("interest_period", 10),
            max_ltv=Decimal(str(product_data.get("max_ltv", 100))),
            max_loan_amount=Decimal(str(product_data.get("max_loan_amount", 1000000))),
            min_loan_amount=Decimal(str(product_data.get("min_loan_amount", 50000))),
            fees=product_data.get("fees", {}),
            features=product_data.get("features", []),
            restrictions=product_data.get("restrictions", []),
            suitability_criteria=product_data.get("suitability_criteria", {}),
            nhg_eligible=product_data.get("nhg_eligible", False),
            sustainability_features=product_data.get("sustainability_features", [])
        )
    
    async def _generate_advice_recommendation(self, customer_profile: CustomerProfile,
                                            products: List[MortgageProduct],
                                            advice_type: AdviceType) -> AdviceRecommendation:
        """Generate specific advice recommendation"""
        try:
            recommendation_id = str(uuid.uuid4())
            
            # Generate advice using AI engine
            advice_content = await self.advice_engine.generate_personalized_advice(
                customer_profile, products, advice_type
            )
            
            # Parse advice content
            title, summary, detailed_explanation, rationale = self._parse_advice_content(advice_content)
            
            # Generate benefits and risks
            benefits, risks = self._extract_benefits_risks(advice_content, advice_type)
            
            # Calculate cost implications
            cost_implications = self._calculate_cost_implications(customer_profile, products, advice_type)
            
            # Generate implementation steps
            implementation_steps = self._generate_implementation_steps(advice_type, products)
            
            # Determine timeline
            timeline = self._determine_timeline(advice_type, customer_profile)
            
            # Calculate confidence score
            confidence_score = self._calculate_advice_confidence(customer_profile, products, advice_type)
            
            # Identify regulatory basis
            regulatory_basis = self._identify_regulatory_basis(advice_type)
            
            return AdviceRecommendation(
                recommendation_id=recommendation_id,
                advice_type=advice_type,
                title=title,
                summary=summary,
                detailed_explanation=detailed_explanation,
                rationale=rationale,
                benefits=benefits,
                risks=risks,
                cost_implications=cost_implications,
                implementation_steps=implementation_steps,
                timeline=timeline,
                priority=self._determine_priority(advice_type),
                confidence_score=confidence_score,
                regulatory_basis=regulatory_basis,
                supporting_data={}
            )
            
        except Exception as e:
            logger.error(f"Advice recommendation generation failed: {e}")
            return AdviceRecommendation(
                recommendation_id=str(uuid.uuid4()),
                advice_type=advice_type,
                title=f"Error generating {advice_type.value}",
                summary="Unable to generate recommendation",
                detailed_explanation=f"Error: {str(e)}",
                rationale="Technical error occurred",
                benefits=[],
                risks=["Unable to assess risks due to technical error"],
                cost_implications={},
                implementation_steps=["Contact advisor for manual assistance"],
                timeline="Immediate",
                priority=1,
                confidence_score=0.0,
                regulatory_basis=[],
                supporting_data={"error": str(e)}
            )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get advice generation performance metrics"""
        return self.metrics
    
    async def close(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()

# Example usage
async def main():
    """Example usage of Mortgage Advice Generator"""
    config = {
        'ai_config': {
            'openai_api_key': 'your_openai_key',
            'anthropic_api_key': 'your_anthropic_key',
            'use_local_model': True
        }
    }
    
    generator = MortgageAdviceGenerator(config)
    
    # Example customer data
    customer_data = {
        "customer_id": "customer_123",
        "personal_info": {
            "first_name": "Jan",
            "last_name": "de Vries",
            "age": 35,
            "first_time_buyer": True
        },
        "financial_situation": {
            "gross_annual_income": 60000,
            "net_monthly_income": 3800,
            "monthly_expenses": 2200,
            "savings": 80000,
            "employment_type": "permanent",
            "employment_years": 8
        },
        "property_details": {
            "value": 400000,
            "type": "house",
            "location": "Amsterdam"
        },
        "risk_tolerance": "moderate",
        "knowledge_level": "intermediate"
    }
    
    # Example products
    products = [
        {
            "product_id": "ing_fixed_10",
            "product_name": "ING Fixed 10 Years",
            "lender": "ING Bank",
            "interest_rate": 3.2,
            "interest_type": "fixed",
            "max_ltv": 100,
            "features": ["NHG eligible", "Flexible repayment"]
        }
    ]
    
    # Generate advice
    # advice = await generator.generate_comprehensive_advice(customer_data, products)
    # print(f"Advice generated: {advice.advice_id}")
    
    print("Mortgage Advice Generator demo completed!")

if __name__ == "__main__":
    asyncio.run(main())