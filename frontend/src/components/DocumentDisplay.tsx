/**
 * Dynamic Document Display Component
 *
 * Renders extracted document content in a structured, professional format
 * based on the document type. Supports validation, editing, and preview modes.
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Grid,
  TextField,
  Button,
  Chip,
  Alert,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  ExpandMore,
  CheckCircle,
  Warning,
  Error,
  Edit,
  Save,
  Cancel,
  Refresh
} from '@mui/icons-material';
import { DocumentProcessingResult, DocumentTypeConfig, documentService } from '../services/documentService';

interface DocumentDisplayProps {
  document: DocumentProcessingResult;
  onUpdate?: (updatedData: Record<string, any>) => void;
  onValidate?: (document: DocumentProcessingResult) => void;
  editable?: boolean;
  showValidation?: boolean;
}

const DocumentDisplay: React.FC<DocumentDisplayProps> = ({
  document,
  onUpdate,
  onValidate,
  editable = true,
  showValidation = true
}) => {
  const [editMode, setEditMode] = useState(false);
  const [editedData, setEditedData] = useState<Record<string, any>>({});
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean;
    errors: Record<string, string>;
    warnings: Record<string, string>;
  } | null>(null);

  const config = documentService.getDocumentTypeConfig(document.documentType);

  useEffect(() => {
    if (document.structuredData) {
      setEditedData({ ...document.structuredData });
    }

    if (showValidation && document.status === 'completed') {
      const result = documentService.validateDocumentData(document.structuredData, document.documentType);
      setValidationResult(result);
    }
  }, [document, showValidation]);

  const handleFieldChange = (field: string, value: any) => {
    setEditedData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSave = () => {
    if (onUpdate) {
      onUpdate(editedData);
    }
    setEditMode(false);
  };

  const handleCancel = () => {
    setEditedData({ ...document.structuredData });
    setEditMode(false);
  };

  const handleValidate = () => {
    if (onValidate) {
      onValidate(document);
    } else {
      const result = documentService.validateDocumentData(editedData, document.documentType);
      setValidationResult(result);
    }
  };

  const renderFieldValue = (field: string, value: any, fieldDef: any) => {
    if (fieldDef.type === 'currency') {
      return new Intl.NumberFormat('en-GB', {
        style: 'currency',
        currency: 'GBP'
      }).format(value);
    }

    if (fieldDef.type === 'date') {
      try {
        return new Date(value).toLocaleDateString('en-GB');
      } catch {
        return value;
      }
    }

    return value;
  };

  const getFieldIcon = (field: string) => {
    if (validationResult) {
      if (validationResult.errors[field]) return <Error color="error" />;
      if (validationResult.warnings[field]) return <Warning color="warning" />;
    }
    return <CheckCircle color="success" />;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.9) return 'High';
    if (confidence >= 0.7) return 'Medium';
    return 'Low';
  };

  if (!config) {
    return (
      <Alert severity="error">
        Unknown document type: {document.documentType}
      </Alert>
    );
  }

  return (
    <Card sx={{ mb: 3 }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6">{config.label}</Typography>
            <Chip
              label={`${getConfidenceLabel(document.confidence)} Confidence`}
              color={getConfidenceColor(document.confidence)}
              size="small"
            />
            {document.status === 'processing' && (
              <Chip label="Processing..." color="info" size="small" />
            )}
          </Box>
        }
        subheader={
          <Box>
            <Typography variant="body2" color="text.secondary">
              {config.description}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {document.fileName} • {(document.fileSize / 1024 / 1024).toFixed(2)} MB
            </Typography>
          </Box>
        }
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            {editable && document.status === 'completed' && (
              <>
                {editMode ? (
                  <>
                    <Button
                      size="small"
                      startIcon={<Save />}
                      onClick={handleSave}
                      variant="contained"
                    >
                      Save
                    </Button>
                    <Button
                      size="small"
                      startIcon={<Cancel />}
                      onClick={handleCancel}
                    >
                      Cancel
                    </Button>
                  </>
                ) : (
                  <Button
                    size="small"
                    startIcon={<Edit />}
                    onClick={() => setEditMode(true)}
                  >
                    Edit
                  </Button>
                )}
              </>
            )}
            {showValidation && (
              <Button
                size="small"
                startIcon={<Refresh />}
                onClick={handleValidate}
              >
                Validate
              </Button>
            )}
          </Box>
        }
      />

      <CardContent>
        {/* Processing Progress */}
        {document.status === 'processing' && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" gutterBottom>
              Processing document...
            </Typography>
            <LinearProgress />
          </Box>
        )}

        {/* Validation Results */}
        {validationResult && (
          <Box sx={{ mb: 3 }}>
            {Object.keys(validationResult.errors).length > 0 && (
              <Alert severity="error" sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Validation Errors:
                </Typography>
                {Object.entries(validationResult.errors).map(([field, error]) => (
                  <Typography key={field} variant="body2">
                    • {error}
                  </Typography>
                ))}
              </Alert>
            )}

            {Object.keys(validationResult.warnings).length > 0 && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Validation Warnings:
                </Typography>
                {Object.entries(validationResult.warnings).map(([field, warning]) => (
                  <Typography key={field} variant="body2">
                    • {warning}
                  </Typography>
                ))}
              </Alert>
            )}

            {validationResult.isValid && (
              <Alert severity="success">
                All validation checks passed successfully!
              </Alert>
            )}
          </Box>
        )}

        {/* Document Content */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle1">
              Extracted Information ({Object.keys(editedData).length} fields)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {config.requiredFields.map(field => {
                const fieldDef = documentService.generateFormFields(document.documentType)
                  .find(f => f.name === field);

                if (!fieldDef) return null;

                return (
                  <Grid item xs={12} md={6} key={field}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                      {getFieldIcon(field)}
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {fieldDef.label}
                          {fieldDef.required && <span style={{ color: 'red' }}> *</span>}
                        </Typography>

                        {editMode ? (
                          <TextField
                            fullWidth
                            size="small"
                            value={editedData[field] || ''}
                            onChange={(e) => handleFieldChange(field, e.target.value)}
                            error={!!validationResult?.errors[field]}
                            helperText={validationResult?.errors[field] || validationResult?.warnings[field]}
                            type={fieldDef.type === 'number' ? 'number' : 'text'}
                          />
                        ) : (
                          <Typography variant="body1" sx={{
                            color: validationResult?.errors[field] ? 'error.main' :
                                   validationResult?.warnings[field] ? 'warning.main' : 'text.primary'
                          }}>
                            {renderFieldValue(field, editedData[field], fieldDef)}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  </Grid>
                );
              })}

              {config.optionalFields.length > 0 && (
                <>
                  <Grid item xs={12}>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Optional Fields
                    </Typography>
                  </Grid>

                  {config.optionalFields.map(field => {
                    const fieldDef = documentService.generateFormFields(document.documentType)
                      .find(f => f.name === field);

                    if (!fieldDef) return null;

                    return (
                      <Grid item xs={12} md={6} key={field}>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                          {getFieldIcon(field)}
                          <Box sx={{ flex: 1 }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              {fieldDef.label}
                              {!fieldDef.required && <span style={{ fontStyle: 'italic' }}> (Optional)</span>}
                            </Typography>

                            {editMode ? (
                              <TextField
                                fullWidth
                                size="small"
                                value={editedData[field] || ''}
                                onChange={(e) => handleFieldChange(field, e.target.value)}
                                error={!!validationResult?.errors[field]}
                                helperText={validationResult?.errors[field] || validationResult?.warnings[field]}
                                type={fieldDef.type === 'number' ? 'number' : 'text'}
                              />
                            ) : (
                              <Typography variant="body1" sx={{
                                color: validationResult?.errors[field] ? 'error.main' :
                                       validationResult?.warnings[field] ? 'warning.main' : 'text.primary'
                              }}>
                                {editedData[field] ? renderFieldValue(field, editedData[field], fieldDef) : 'Not provided'}
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      </Grid>
                    );
                  })}
                </>
              )}
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Raw OCR Text */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle1">
              Raw OCR Text ({document.extractedText.length} characters)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{
              backgroundColor: 'grey.50',
              p: 2,
              borderRadius: 1,
              maxHeight: 300,
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              whiteSpace: 'pre-wrap'
            }}>
              {document.extractedText}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Processing Metadata */}
        <Box sx={{ mt: 3, pt: 3, borderTop: '1px solid', borderColor: 'divider' }}>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">
                Processing Time
              </Typography>
              <Typography variant="body1">
                {document.processingTime}ms
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">
                Confidence Score
              </Typography>
              <Typography variant="body1">
                {Math.round(document.confidence * 100)}%
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">
                Fields Extracted
              </Typography>
              <Typography variant="body1">
                {Object.keys(document.structuredData).length}
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">
                Status
              </Typography>
              <Chip
                size="small"
                label={document.status}
                color={document.status === 'completed' ? 'success' : 'error'}
              />
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default DocumentDisplay;

