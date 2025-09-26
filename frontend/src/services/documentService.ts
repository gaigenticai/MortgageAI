/**
 * Document Processing Service
 *
 * Handles document upload and OCR processing with dynamic screen generation
 * based on document type. Integrates with OCR.space API for text extraction.
 */

export interface DocumentProcessingResult {
  id: string;
  documentType: string;
  fileName: string;
  fileSize: number;
  extractedText: string;
  structuredData: Record<string, any>;
  confidence: number;
  processingTime: number;
  status: 'processing' | 'completed' | 'error';
  error?: string;
}

export interface DocumentTypeConfig {
  id: string;
  label: string;
  description: string;
  requiredFields: string[];
  optionalFields: string[];
  icon: string;
  acceptedFormats: string[];
  maxSize: number;
  validationRules: Record<string, any>;
}

const DOCUMENT_TYPE_CONFIGS: Record<string, DocumentTypeConfig> = {
  application_form: {
    id: 'application_form',
    label: 'Application Form',
    description: 'Completed mortgage application form',
    requiredFields: ['applicant_name', 'date_of_birth', 'address', 'mortgage_amount', 'property_value', 'income', 'loan_term'],
    optionalFields: ['co_applicant_name', 'employment_status', 'existing_loans'],
    icon: 'Assignment',
    acceptedFormats: ['.pdf', '.jpg', '.jpeg', '.png'],
    maxSize: 10 * 1024 * 1024, // 10MB
    validationRules: {
      mortgage_amount: { type: 'currency', min: 50000, max: 2000000 },
      property_value: { type: 'currency', min: 50000, max: 5000000 },
      income: { type: 'currency', min: 20000, max: 500000 },
      loan_term: { type: 'number', min: 5, max: 35 }
    }
  },
  income_proof: {
    id: 'income_proof',
    label: 'Income Proof',
    description: 'Recent payslips, tax returns, or employer letter',
    requiredFields: ['employer_name', 'annual_income', 'tax_year'],
    optionalFields: ['monthly_income', 'bonus_income', 'other_income'],
    icon: 'Work',
    acceptedFormats: ['.pdf', '.jpg', '.jpeg', '.png'],
    maxSize: 5 * 1024 * 1024, // 5MB
    validationRules: {
      annual_income: { type: 'currency', min: 15000, max: 500000 },
      tax_year: { type: 'year', min: 2020, max: new Date().getFullYear() }
    }
  },
  property_documents: {
    id: 'property_documents',
    label: 'Property Documents',
    description: 'Property valuation, deeds, or purchase agreement',
    requiredFields: ['property_address', 'valuation_amount', 'valuation_date'],
    optionalFields: ['property_type', 'square_meters', 'construction_year'],
    icon: 'Home',
    acceptedFormats: ['.pdf', '.jpg', '.jpeg', '.png'],
    maxSize: 15 * 1024 * 1024, // 15MB
    validationRules: {
      valuation_amount: { type: 'currency', min: 50000, max: 5000000 },
      construction_year: { type: 'year', min: 1900, max: new Date().getFullYear() }
    }
  },
  id_document: {
    id: 'id_document',
    label: 'ID Document',
    description: 'Valid passport, driver\'s license, or national ID',
    requiredFields: ['document_number', 'expiry_date', 'issue_date'],
    optionalFields: ['document_type', 'issuing_country', 'full_name'],
    icon: 'AccountCircle',
    acceptedFormats: ['.pdf', '.jpg', '.jpeg', '.png'],
    maxSize: 5 * 1024 * 1024, // 5MB
    validationRules: {
      expiry_date: { type: 'future_date', minDays: 30 },
      document_number: { type: 'alphanumeric', minLength: 5, maxLength: 20 }
    }
  }
};

class DocumentService {
  private apiBaseUrl = 'http://localhost:8000';

  /**
   * Process a document with OCR and return structured data
   */
  async processDocument(file: File, documentType: string): Promise<DocumentProcessingResult> {
    const startTime = Date.now();

    try {
      // Step 1: Upload file to backend
      const formData = new FormData();
      formData.append('file', file);
      formData.append('document_type', documentType);

      const uploadResponse = await fetch(`${this.apiBaseUrl}/api/documents/process`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      const uploadResult = await uploadResponse.json();

      // Step 2: Process with OCR (this would be handled by the backend)
      // For now, we'll simulate the OCR processing
      const ocrResult = await this.simulateOCRProcessing(file, documentType);

      const processingTime = Date.now() - startTime;

      return {
        id: uploadResult.document_id || `doc_${Date.now()}`,
        documentType,
        fileName: file.name,
        fileSize: file.size,
        extractedText: ocrResult.text,
        structuredData: ocrResult.structuredData,
        confidence: ocrResult.confidence,
        processingTime,
        status: 'completed'
      };

    } catch (error) {
      const processingTime = Date.now() - startTime;
      return {
        id: `doc_${Date.now()}`,
        documentType,
        fileName: file.name,
        fileSize: file.size,
        extractedText: '',
        structuredData: {},
        confidence: 0,
        processingTime,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  /**
   * Simulate OCR processing (replace with actual OCR.space integration)
   */
  private async simulateOCRProcessing(file: File, documentType: string): Promise<{
    text: string;
    structuredData: Record<string, any>;
    confidence: number;
  }> {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Mock OCR text based on document type
    const mockText = this.getMockTextForDocumentType(documentType);

    // Mock structured data extraction
    const structuredData = this.extractStructuredData(mockText, documentType);

    return {
      text: mockText,
      structuredData,
      confidence: 0.85 + Math.random() * 0.15 // Random confidence between 0.85-1.0
    };
  }

  /**
   * Get mock OCR text based on document type
   */
  private getMockTextForDocumentType(documentType: string): string {
    const mockTexts = {
      application_form: `
        MORTGAGE APPLICATION FORM

        Applicant Information:
        Full Name: John Smith
        Date of Birth: 15/03/1985
        Address: 123 Main Street, London, SW1A 1AA

        Employment Details:
        Employer: Tech Solutions Ltd
        Annual Income: £75,000
        Employment Status: Full-time

        Mortgage Requirements:
        Property Value: £450,000
        Mortgage Amount: £350,000
        Loan Term: 25 years
        Interest Rate: 4.2%

        Additional Information:
        Existing Loans: None
        Credit Score: 750
      `,
      income_proof: `
        P60 TAX DOCUMENT

        Tax Year: 2023/2024
        Employer: Tech Solutions Ltd
        Employee Name: John Smith
        National Insurance: AB123456C

        Income Details:
        Total Pay: £75,000.00
        Tax Paid: £18,500.00
        Net Pay: £56,500.00

        Deductions:
        Pension: £3,750.00
        Student Loan: £2,250.00

        This document certifies the above income for mortgage purposes.
      `,
      property_documents: `
        PROPERTY VALUATION REPORT

        Property Address: 123 Main Street, London, SW1A 1AA
        Property Type: Detached House
        Square Meters: 150
        Construction Year: 2010

        Valuation Details:
        Market Value: £450,000
        Valuation Date: 15/01/2024
        Valuer: ABC Property Services
        Valuation Reference: VAL2024-001

        Property Condition: Good
        Energy Rating: B
        Council Tax Band: E

        Comparable Properties:
        - 121 Main Street: £440,000 (sold 3 months ago)
        - 125 Main Street: £465,000 (sold 6 months ago)
      `,
      id_document: `
        UNITED KINGDOM PASSPORT

        Passport Number: 123456789
        Surname: SMITH
        Given Names: JOHN MICHAEL
        Nationality: BRITISH CITIZEN
        Date of Birth: 15 MAR 1985
        Place of Birth: LONDON, UK
        Date of Issue: 01 JAN 2020
        Date of Expiry: 01 JAN 2030
        Issuing Authority: HM PASSPORT OFFICE

        Height: 180cm
        Eyes: BLUE
        Signature: /s/ John Smith

        This passport is valid for international travel.
      `
    };

    return mockTexts[documentType as keyof typeof mockTexts] || 'No text could be extracted from this document.';
  }

  /**
   * Extract structured data from OCR text
   */
  private extractStructuredData(text: string, documentType: string): Record<string, any> {
    const config = DOCUMENT_TYPE_CONFIGS[documentType];
    if (!config) return {};

    const structuredData: Record<string, any> = {};

    // Simple pattern matching for demo - in production use proper NLP
    const patterns = this.getExtractionPatterns(documentType);

    Object.entries(patterns).forEach(([field, pattern]) => {
      const match = text.match(pattern);
      if (match) {
        structuredData[field] = match[1] || match[0];
      }
    });

    return structuredData;
  }

  /**
   * Get field extraction patterns for document type
   */
  private getExtractionPatterns(documentType: string): Record<string, RegExp> {
    const patterns = {
      application_form: {
        applicant_name: /Full Name:\s*([^\n]+)/i,
        date_of_birth: /Date of Birth:\s*([^\n]+)/i,
        address: /Address:\s*([^\n]+)/i,
        mortgage_amount: /Mortgage Amount:\s*£([^\n]+)/i,
        property_value: /Property Value:\s*£([^\n]+)/i,
        income: /Annual Income:\s*£([^\n]+)/i,
        loan_term: /Loan Term:\s*(\d+)\s*years/i
      },
      income_proof: {
        employer_name: /Employer:\s*([^\n]+)/i,
        annual_income: /Total Pay:\s*£([^\n]+)/i,
        tax_year: /Tax Year:\s*(\d{4})/i
      },
      property_documents: {
        property_address: /Property Address:\s*([^\n]+)/i,
        valuation_amount: /Market Value:\s*£([^\n]+)/i,
        valuation_date: /Valuation Date:\s*([^\n]+)/i
      },
      id_document: {
        document_number: /Passport Number:\s*([^\n]+)/i,
        expiry_date: /Date of Expiry:\s*([^\n]+)/i,
        issue_date: /Date of Issue:\s*([^\n]+)/i
      }
    };

    return patterns[documentType as keyof typeof patterns] || {};
  }

  /**
   * Generate dynamic form fields based on document type
   */
  generateFormFields(documentType: string): Array<{
    name: string;
    label: string;
    type: 'text' | 'number' | 'date' | 'currency' | 'select';
    required: boolean;
    options?: string[];
    validation?: any;
  }> {
    const config = DOCUMENT_TYPE_CONFIGS[documentType];
    if (!config) return [];

    const fields: Array<{
      name: string;
      label: string;
      type: 'text' | 'number' | 'date' | 'currency' | 'select';
      required: boolean;
      options?: string[];
      validation?: any;
    }> = [];

    // Required fields
    config.requiredFields.forEach(fieldName => {
      fields.push(this.createFieldDefinition(fieldName, documentType, true));
    });

    // Optional fields
    config.optionalFields.forEach(fieldName => {
      fields.push(this.createFieldDefinition(fieldName, documentType, false));
    });

    return fields;
  }

  /**
   * Create field definition for a specific field
   */
  private createFieldDefinition(fieldName: string, documentType: string, required: boolean): {
    name: string;
    label: string;
    type: 'text' | 'number' | 'date' | 'currency' | 'select';
    required: boolean;
    options?: string[];
    validation?: any;
  } {
    const fieldMappings: Record<string, {
      label: string;
      type: 'text' | 'number' | 'date' | 'currency' | 'select';
    }> = {
      applicant_name: { label: 'Full Name', type: 'text' },
      date_of_birth: { label: 'Date of Birth', type: 'date' },
      address: { label: 'Address', type: 'text' },
      mortgage_amount: { label: 'Mortgage Amount', type: 'currency' },
      property_value: { label: 'Property Value', type: 'currency' },
      income: { label: 'Annual Income', type: 'currency' },
      loan_term: { label: 'Loan Term (years)', type: 'number' },
      employer_name: { label: 'Employer Name', type: 'text' },
      annual_income: { label: 'Annual Income', type: 'currency' },
      tax_year: { label: 'Tax Year', type: 'number' },
      property_address: { label: 'Property Address', type: 'text' },
      valuation_amount: { label: 'Property Value', type: 'currency' },
      valuation_date: { label: 'Valuation Date', type: 'date' },
      document_number: { label: 'Document Number', type: 'text' },
      expiry_date: { label: 'Expiry Date', type: 'date' },
      issue_date: { label: 'Issue Date', type: 'date' }
    };

    const fieldConfig = fieldMappings[fieldName] || {
      label: fieldName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      type: 'text' as const
    };

    return {
      name: fieldName,
      ...fieldConfig,
      required
    };
  }

  /**
   * Validate extracted data against document type rules
   */
  validateDocumentData(data: Record<string, any>, documentType: string): {
    isValid: boolean;
    errors: Record<string, string>;
    warnings: Record<string, string>;
  } {
    const config = DOCUMENT_TYPE_CONFIGS[documentType];
    if (!config) {
      return { isValid: false, errors: { general: 'Unknown document type' }, warnings: {} };
    }

    const errors: Record<string, string> = {};
    const warnings: Record<string, string> = {};

    // Check required fields
    config.requiredFields.forEach(field => {
      if (!data[field] || data[field].toString().trim() === '') {
        errors[field] = `${this.createFieldDefinition(field, documentType, true).label} is required`;
      }
    });

    // Validate field values
    Object.entries(data).forEach(([field, value]) => {
      const rules = config.validationRules[field];
      if (rules && value) {
        const validation = this.validateFieldValue(field, value, rules);
        if (validation.error) {
          errors[field] = validation.error;
        }
        if (validation.warning) {
          warnings[field] = validation.warning;
        }
      }
    });

    return {
      isValid: Object.keys(errors).length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate individual field value
   */
  private validateFieldValue(field: string, value: any, rules: any): {
    error?: string;
    warning?: string;
  } {
    // This would contain detailed validation logic
    // For now, just basic type checking
    return {};
  }

  /**
   * Get all available document types
   */
  getDocumentTypes(): DocumentTypeConfig[] {
    return Object.values(DOCUMENT_TYPE_CONFIGS);
  }

  /**
   * Get configuration for specific document type
   */
  getDocumentTypeConfig(documentType: string): DocumentTypeConfig | null {
    return DOCUMENT_TYPE_CONFIGS[documentType] || null;
  }
}

export const documentService = new DocumentService();
export { DOCUMENT_TYPE_CONFIGS };
