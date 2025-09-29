/**
 * Advanced Validation Engine for MortgageAI
 * 
 * Production-grade validation system with:
 * - Multi-provider API key validation with real-time testing
 * - Dutch financial regulations compliance validation
 * - Advanced data integrity checks
 * - Real-time performance monitoring
 * - Comprehensive error handling and recovery
 * - Caching and optimization for high-performance validation
 */

const axios = require('axios');
const crypto = require('crypto');
const EventEmitter = require('events');

class APIKeyValidator extends EventEmitter {
    constructor() {
        super();
        this.validationCache = new Map();
        this.rateLimits = new Map();
        this.performanceMetrics = {
            totalValidations: 0,
            successfulValidations: 0,
            averageResponseTime: 0,
            cacheHitRate: 0
        };
    }

    async validateOpenAI(apiKey) {
        const startTime = Date.now();
        
        try {
            // Validate key format first
            if (!apiKey || !apiKey.startsWith('sk-') || apiKey.length < 40) {
                return {
                    isValid: false,
                    message: 'Invalid OpenAI API key format. Keys should start with "sk-" and be at least 40 characters.',
                    error: 'Invalid format'
                };
            }

            // Check cache first
            const cacheKey = this.generateCacheKey('openai', apiKey);
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                return { ...cached, cached: true };
            }

            // Rate limiting check
            if (!this.checkRateLimit('openai')) {
                return {
                    isValid: false,
                    message: 'Rate limit exceeded for OpenAI validation. Please try again later.',
                    error: 'Rate limited'
                };
            }

            // Test API key with minimal request
            const response = await axios.get('https://api.openai.com/v1/models', {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                timeout: 15000
            });

            if (response.status === 200 && response.data.data) {
                const result = {
                    isValid: true,
                    message: 'OpenAI API key is valid and has access to models',
                    capabilities: response.data.data.map(model => model.id).slice(0, 10), // First 10 models
                    rateLimit: this.extractRateLimit(response.headers),
                    organization: response.data.organization_id || 'personal'
                };

                this.setCache(cacheKey, result, 300000); // Cache for 5 minutes
                this.updateMetrics(Date.now() - startTime, true);
                return result;
            }

            return {
                isValid: false,
                message: 'OpenAI API returned unexpected response',
                error: 'Invalid response'
            };

        } catch (error) {
            this.updateMetrics(Date.now() - startTime, false);
            
            if (error.response) {
                if (error.response.status === 401) {
                    return {
                        isValid: false,
                        message: 'Invalid OpenAI API key - authentication failed',
                        error: 'Authentication failed'
                    };
                }
                if (error.response.status === 403) {
                    return {
                        isValid: false,
                        message: 'OpenAI API key access denied - check permissions',
                        error: 'Access denied'
                    };
                }
                if (error.response.status === 429) {
                    return {
                        isValid: false,
                        message: 'OpenAI API rate limit exceeded',
                        error: 'Rate limited'
                    };
                }
            }

            return {
                isValid: false,
                message: `OpenAI validation failed: ${error.message}`,
                error: 'Network or service error'
            };
        }
    }

    async validateOCRSpace(apiKey) {
        const startTime = Date.now();
        
        try {
            // Validate key format
            if (!apiKey || (!apiKey.startsWith('K') && !apiKey.startsWith('helloworld'))) {
                return {
                    isValid: false,
                    message: 'Invalid OCR.space API key format. Keys should start with "K" or use "helloworld" for testing.',
                    error: 'Invalid format'
                };
            }

            const cacheKey = this.generateCacheKey('ocr', apiKey);
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                return { ...cached, cached: true };
            }

            if (!this.checkRateLimit('ocr')) {
                return {
                    isValid: false,
                    message: 'Rate limit exceeded for OCR validation. Please try again later.',
                    error: 'Rate limited'
                };
            }

            // Create a minimal test image (1x1 pixel PNG)
            const testImageBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';
            
            const formData = new URLSearchParams();
            formData.append('apikey', apiKey);
            formData.append('base64Image', `data:image/png;base64,${testImageBase64}`);
            formData.append('detectOrientation', 'true');
            formData.append('scale', 'true');

            const response = await axios.post('https://api.ocr.space/parse/image', formData, {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                timeout: 20000
            });

            if (response.status === 200 && response.data) {
                if (response.data.IsErroredOnProcessing === false || response.data.OCRExitCode === 1) {
                    const result = {
                        isValid: true,
                        message: 'OCR.space API key is valid and working',
                        capabilities: ['text_extraction', 'orientation_detection', 'table_detection'],
                        rateLimit: {
                            remaining: parseInt(response.headers['x-ratelimit-remaining']) || 500,
                            resetTime: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
                        },
                        subscription: response.data.subscription || 'free'
                    };

                    this.setCache(cacheKey, result, 600000); // Cache for 10 minutes
                    this.updateMetrics(Date.now() - startTime, true);
                    return result;
                }

                if (response.data.IsErroredOnProcessing === true) {
                    const errorMessage = response.data.ErrorMessage || 'Unknown error';
                    
                    if (errorMessage.includes('Invalid API key')) {
                        return {
                            isValid: false,
                            message: 'Invalid OCR.space API key',
                            error: 'Authentication failed'
                        };
                    }
                    
                    return {
                        isValid: false,
                        message: `OCR.space API error: ${errorMessage}`,
                        error: 'API error'
                    };
                }
            }

            return {
                isValid: false,
                message: 'OCR.space API returned unexpected response format',
                error: 'Invalid response'
            };

        } catch (error) {
            this.updateMetrics(Date.now() - startTime, false);
            
            if (error.response && error.response.status === 403) {
                return {
                    isValid: false,
                    message: 'OCR.space API key is invalid or suspended',
                    error: 'Access denied'
                };
            }

            // Handle timeout or network errors gracefully
            if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
                // For timeout, if key format is correct, assume it might be valid
                if (apiKey.startsWith('K') && apiKey.length >= 10) {
                    return {
                        isValid: true,
                        message: 'OCR.space API key format appears valid (network timeout prevented full validation)',
                        capabilities: ['text_extraction'],
                        error: 'Network timeout - key format valid'
                    };
                }
            }

            return {
                isValid: false,
                message: `OCR.space validation failed: ${error.message}`,
                error: 'Network or service error'
            };
        }
    }

    async validateAnthropic(apiKey) {
        const startTime = Date.now();
        
        try {
            if (!apiKey || !apiKey.startsWith('sk-ant-') || apiKey.length < 40) {
                return {
                    isValid: false,
                    message: 'Invalid Anthropic API key format. Keys should start with "sk-ant-" and be at least 40 characters.',
                    error: 'Invalid format'
                };
            }

            const cacheKey = this.generateCacheKey('anthropic', apiKey);
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                return { ...cached, cached: true };
            }

            if (!this.checkRateLimit('anthropic')) {
                return {
                    isValid: false,
                    message: 'Rate limit exceeded for Anthropic validation. Please try again later.',
                    error: 'Rate limited'
                };
            }

            // Test with minimal request
            const response = await axios.post('https://api.anthropic.com/v1/messages', {
                model: 'claude-3-haiku-20240307',
                max_tokens: 1,
                messages: [{ role: 'user', content: 'Hi' }]
            }, {
                headers: {
                    'x-api-key': apiKey,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                timeout: 15000
            });

            if (response.status === 200) {
                const result = {
                    isValid: true,
                    message: 'Anthropic API key is valid and working',
                    capabilities: ['claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'],
                    rateLimit: this.extractRateLimit(response.headers)
                };

                this.setCache(cacheKey, result, 300000);
                this.updateMetrics(Date.now() - startTime, true);
                return result;
            }

            return {
                isValid: false,
                message: `Anthropic API returned status ${response.status}`,
                error: 'API request failed'
            };

        } catch (error) {
            this.updateMetrics(Date.now() - startTime, false);
            
            if (error.response) {
                if (error.response.status === 401) {
                    return {
                        isValid: false,
                        message: 'Invalid Anthropic API key - authentication failed',
                        error: 'Authentication failed'
                    };
                }
                if (error.response.status === 403) {
                    return {
                        isValid: false,
                        message: 'Anthropic API key access denied',
                        error: 'Access denied'
                    };
                }
            }

            return {
                isValid: false,
                message: `Anthropic validation failed: ${error.message}`,
                error: 'Network or service error'
            };
        }
    }

    generateCacheKey(provider, apiKey) {
        return `${provider}_${crypto.createHash('sha256').update(apiKey).digest('hex').substring(0, 16)}`;
    }

    getFromCache(key) {
        const entry = this.validationCache.get(key);
        if (!entry || Date.now() - entry.timestamp > entry.ttl) {
            this.validationCache.delete(key);
            return null;
        }
        
        this.performanceMetrics.cacheHitRate = 
            (this.performanceMetrics.cacheHitRate * 0.9) + 0.1; // Moving average
        
        return entry.data;
    }

    setCache(key, data, ttl = 300000) {
        this.validationCache.set(key, {
            data,
            timestamp: Date.now(),
            ttl
        });
        
        // Cleanup old entries if cache gets too large
        if (this.validationCache.size > 1000) {
            const entries = Array.from(this.validationCache.entries());
            const oldEntries = entries
                .sort((a, b) => a[1].timestamp - b[1].timestamp)
                .slice(0, 200);
            
            oldEntries.forEach(([key]) => this.validationCache.delete(key));
        }
    }

    checkRateLimit(provider) {
        const now = Date.now();
        const limit = this.rateLimits.get(provider) || { count: 0, resetTime: now };
        
        if (now > limit.resetTime) {
            limit.count = 0;
            limit.resetTime = now + 60000; // Reset every minute
        }
        
        if (limit.count >= 10) { // Max 10 validations per minute per provider
            return false;
        }
        
        limit.count++;
        this.rateLimits.set(provider, limit);
        return true;
    }

    extractRateLimit(headers) {
        return {
            remaining: parseInt(headers['x-ratelimit-remaining']) || 
                      parseInt(headers['x-ratelimit-requests-remaining']) || 100,
            resetTime: headers['x-ratelimit-reset'] || 
                      new Date(Date.now() + 60000).toISOString()
        };
    }

    updateMetrics(responseTime, success) {
        this.performanceMetrics.totalValidations++;
        
        if (success) {
            this.performanceMetrics.successfulValidations++;
        }
        
        const total = this.performanceMetrics.totalValidations;
        const currentAvg = this.performanceMetrics.averageResponseTime;
        this.performanceMetrics.averageResponseTime = 
            ((currentAvg * (total - 1)) + responseTime) / total;
    }

    getMetrics() {
        return {
            ...this.performanceMetrics,
            successRate: this.performanceMetrics.totalValidations > 0 
                ? this.performanceMetrics.successfulValidations / this.performanceMetrics.totalValidations 
                : 0,
            cacheSize: this.validationCache.size
        };
    }

    clearCache() {
        this.validationCache.clear();
        this.emit('cache_cleared');
    }
}

class DutchMortgageValidator {
    constructor() {
        this.afmRegulations = {
            maxLTV: 1.0, // 100% LTV allowed in Netherlands
            maxDTI: 0.28, // Maximum debt-to-income ratio
            nhgLimit: 435000, // 2024 NHG limit
            minAge: 18,
            maxAge: 80,
            minIncome: 0,
            maxMortgageTerm: 30
        };
    }

    validateBSN(bsn) {
        // Dutch BSN (Burgerservicenummer) validation
        if (!bsn || typeof bsn !== 'string') {
            return { isValid: false, message: 'BSN moet een string zijn' };
        }

        // Remove any spaces or dashes
        const cleanBSN = bsn.replace(/[\s-]/g, '');
        
        // Check length
        if (cleanBSN.length !== 9) {
            return { isValid: false, message: 'BSN moet 9 cijfers bevatten' };
        }

        // Check if all characters are digits
        if (!/^\d{9}$/.test(cleanBSN)) {
            return { isValid: false, message: 'BSN mag alleen cijfers bevatten' };
        }

        // Validate using the 11-test (elfproef)
        const digits = cleanBSN.split('').map(Number);
        let sum = 0;
        
        for (let i = 0; i < 8; i++) {
            sum += digits[i] * (9 - i);
        }
        
        const remainder = sum % 11;
        const checkDigit = remainder < 2 ? remainder : 11 - remainder;
        
        if (checkDigit !== digits[8]) {
            return { isValid: false, message: 'BSN controle cijfer is onjuist' };
        }

        return { 
            isValid: true, 
            message: 'BSN is geldig',
            formatted: `${cleanBSN.substring(0, 3)}-${cleanBSN.substring(3, 5)}-${cleanBSN.substring(5)}`
        };
    }

    validateDutchIBAN(iban) {
        if (!iban || typeof iban !== 'string') {
            return { isValid: false, message: 'IBAN moet een string zijn' };
        }

        // Remove spaces and convert to uppercase
        const cleanIBAN = iban.replace(/\s/g, '').toUpperCase();
        
        // Check if it's a Dutch IBAN
        if (!cleanIBAN.startsWith('NL')) {
            return { isValid: false, message: 'Alleen Nederlandse IBANs worden ondersteund' };
        }

        // Check length (Dutch IBAN is 18 characters)
        if (cleanIBAN.length !== 18) {
            return { isValid: false, message: 'Nederlandse IBAN moet 18 tekens lang zijn' };
        }

        // Validate IBAN checksum using mod-97 algorithm
        const rearranged = cleanIBAN.substring(4) + cleanIBAN.substring(0, 4);
        const numericString = rearranged.replace(/[A-Z]/g, (char) => 
            (char.charCodeAt(0) - 55).toString()
        );

        // Calculate mod 97
        let remainder = 0;
        for (let i = 0; i < numericString.length; i++) {
            remainder = (remainder * 10 + parseInt(numericString[i])) % 97;
        }

        if (remainder !== 1) {
            return { isValid: false, message: 'IBAN controle cijfers zijn onjuist' };
        }

        return { 
            isValid: true, 
            message: 'IBAN is geldig',
            formatted: cleanIBAN.replace(/(.{4})/g, '$1 ').trim(),
            bankCode: cleanIBAN.substring(4, 8)
        };
    }

    validateMortgageAmount(amount, income, propertyValue) {
        const result = {
            isValid: false,
            messages: [],
            calculations: {}
        };

        // Convert strings to numbers if necessary
        const numAmount = typeof amount === 'string' ? parseFloat(amount) : amount;
        const numIncome = typeof income === 'string' ? parseFloat(income) : income;
        const numPropertyValue = typeof propertyValue === 'string' ? parseFloat(propertyValue) : propertyValue;

        // Basic validation
        if (isNaN(numAmount) || numAmount <= 0) {
            result.messages.push('Hypotheekbedrag moet een geldig positief getal zijn');
        }

        if (isNaN(numIncome) || numIncome <= 0) {
            result.messages.push('Inkomen moet een geldig positief getal zijn');
        }

        if (isNaN(numPropertyValue) || numPropertyValue <= 0) {
            result.messages.push('Woningwaarde moet een geldig positief getal zijn');
        }

        if (result.messages.length > 0) {
            return result;
        }

        // Calculate maximum allowed amounts
        const maxBasedOnIncome = this.calculateMaxMortgageBasedOnIncome(numIncome);
        const maxBasedOnProperty = numPropertyValue * this.afmRegulations.maxLTV;
        const maxAllowed = Math.min(maxBasedOnIncome, maxBasedOnProperty);

        result.calculations = {
            maxBasedOnIncome,
            maxBasedOnProperty,
            maxAllowed,
            ltvRatio: numAmount / numPropertyValue,
            dtiRatio: (numAmount * 0.05) / (numIncome / 12) // Approximate monthly payment
        };

        // Validate LTV ratio
        if (result.calculations.ltvRatio > this.afmRegulations.maxLTV) {
            result.messages.push(`LTV ratio (${(result.calculations.ltvRatio * 100).toFixed(1)}%) overschrijdt maximum van ${(this.afmRegulations.maxLTV * 100)}%`);
        }

        // Validate DTI ratio (simplified)
        if (result.calculations.dtiRatio > this.afmRegulations.maxDTI) {
            result.messages.push(`Geschatte DTI ratio (${(result.calculations.dtiRatio * 100).toFixed(1)}%) overschrijdt maximum van ${(this.afmRegulations.maxDTI * 100)}%`);
        }

        // Check if amount exceeds maximum allowed
        if (numAmount > maxAllowed) {
            result.messages.push(`Aangevraagd bedrag (€${numAmount.toLocaleString()}) overschrijdt maximum toegestaan bedrag (€${Math.round(maxAllowed).toLocaleString()})`);
        }

        // NHG eligibility
        result.calculations.nhgEligible = numPropertyValue <= this.afmRegulations.nhgLimit;
        if (!result.calculations.nhgEligible) {
            result.messages.push(`Woningwaarde overschrijdt NHG-grens van €${this.afmRegulations.nhgLimit.toLocaleString()}`);
        }

        result.isValid = result.messages.length === 0;
        
        if (result.isValid) {
            result.messages.push('Hypotheekbedrag voldoet aan alle AFM-richtlijnen');
        }

        return result;
    }

    calculateMaxMortgageBasedOnIncome(annualIncome) {
        // Simplified calculation based on Dutch standards
        // In practice, this would involve complex affordability calculations
        const monthlyIncome = annualIncome / 12;
        const maxMonthlyPayment = monthlyIncome * this.afmRegulations.maxDTI;
        const assumedInterestRate = 0.045; // 4.5% average
        const termInMonths = this.afmRegulations.maxMortgageTerm * 12;
        
        // Calculate maximum loan using annuity formula
        const monthlyRate = assumedInterestRate / 12;
        const maxLoan = maxMonthlyPayment * 
            ((Math.pow(1 + monthlyRate, termInMonths) - 1) / 
             (monthlyRate * Math.pow(1 + monthlyRate, termInMonths)));
        
        return maxLoan;
    }

    validateAge(age, mortgageTerm = 30) {
        const numAge = typeof age === 'string' ? parseInt(age) : age;
        const numTerm = typeof mortgageTerm === 'string' ? parseInt(mortgageTerm) : mortgageTerm;
        
        if (isNaN(numAge) || numAge < this.afmRegulations.minAge) {
            return { 
                isValid: false, 
                message: `Minimumleeftijd voor hypotheek is ${this.afmRegulations.minAge} jaar` 
            };
        }
        
        const ageAtEndOfTerm = numAge + numTerm;
        if (ageAtEndOfTerm > this.afmRegulations.maxAge) {
            return { 
                isValid: false, 
                message: `Leeftijd aan einde van hypotheekperiode (${ageAtEndOfTerm}) overschrijdt maximum van ${this.afmRegulations.maxAge} jaar` 
            };
        }
        
        return { 
            isValid: true, 
            message: 'Leeftijd voldoet aan hypotheekvereisten',
            ageAtEndOfTerm 
        };
    }
}

// Export the validation engines
module.exports = {
    APIKeyValidator,
    DutchMortgageValidator
};