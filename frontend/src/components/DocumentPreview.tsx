/**
 * Document Preview Component
 *
 * Shows a preview of the document processing results with dynamic form generation
 * based on document type. Allows users to review and edit extracted data before
 * proceeding to the next step.
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Grid,
  TextField,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  LinearProgress,
  Card,
  CardContent
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Edit,
  Save,
  Close,
  Refresh
} from '@mui/icons-material';
import { DocumentProcessingResult, documentService } from '../services/documentService';

interface DocumentPreviewProps {
  open: boolean;
  onClose: () => void;
  document: DocumentProcessingResult | null;
  onSave: (updatedDocument: DocumentProcessingResult) => void;
}

const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  open,
  onClose,
  document,
  onSave
}) => {
  const [editMode, setEditMode] = useState(false);
  const [editedData, setEditedData] = useState<Record<string, any>>({});
  const [formFields, setFormFields] = useState<any[]>([]);
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean;
    errors: Record<string, string>;
    warnings: Record<string, string>;
  } | null>(null);

  useEffect(() => {
    if (document && open) {
      setEditedData({ ...document.structuredData });
      const fields = documentService.generateFormFields(document.documentType);
      setFormFields(fields);

      // Validate the document
      const result = documentService.validateDocumentData(document.structuredData, document.documentType);
      setValidationResult(result);
    }
  }, [document, open]);

  const handleFieldChange = (fieldName: string, value: any) => {
    setEditedData(prev => ({
      ...prev,
      [fieldName]: value
    }));
  };

  const handleSave = () => {
    if (document && onSave) {
      const updatedDocument: DocumentProcessingResult = {
        ...document,
        structuredData: editedData
      };
      onSave(updatedDocument);
      setEditMode(false);
    }
  };

  const handleValidate = () => {
    if (document) {
      const result = documentService.validateDocumentData(editedData, document.documentType);
      setValidationResult(result);
    }
  };

  const renderFieldInput = (field: any) => {
    const value = editedData[field.name] || '';
    const hasError = validationResult?.errors[field.name];
    const hasWarning = validationResult?.warnings[field.name];

    const commonProps = {
      fullWidth: true,
      size: 'small' as const,
      value,
      onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
        handleFieldChange(field.name, e.target.value),
      error: !!hasError,
      helperText: hasError || hasWarning || '',
      label: field.label + (field.required ? ' *' : ' (Optional)'),
      disabled: !editMode
    };

    switch (field.type) {
      case 'number':
        return <TextField {...commonProps} type="number" />;
      case 'date':
        return <TextField {...commonProps} type="date" />;
      case 'currency':
        return <TextField {...commonProps} type="number" InputProps={{
          startAdornment: <span>£</span>
        }} />;
      default:
        return <TextField {...commonProps} />;
    }
  };

  const getValidationSummary = () => {
    if (!validationResult) return null;

    const errorCount = Object.keys(validationResult.errors).length;
    const warningCount = Object.keys(validationResult.warnings).length;

    if (errorCount === 0 && warningCount === 0) {
      return (
        <Alert severity="success" sx={{ mb: 3 }}>
          <Typography variant="body2">
            All validation checks passed successfully!
          </Typography>
        </Alert>
      );
    }

    return (
      <Alert severity={errorCount > 0 ? "error" : "warning"} sx={{ mb: 3 }}>
        <Typography variant="body2">
          {errorCount > 0 && `${errorCount} validation error(s) found. `}
          {warningCount > 0 && `${warningCount} warning(s) found. `}
          Please review and correct the issues below.
        </Typography>
      </Alert>
    );
  };

  if (!document) {
    return null;
  }

  const config = documentService.getDocumentTypeConfig(document.documentType);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: '80vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="h6" component="div">
              Document Preview - {config?.label || document.documentType}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {document.fileName} • {(document.fileSize / 1024 / 1024).toFixed(2)} MB
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`${Math.round(document.confidence * 100)}% confidence`}
              color={document.confidence >= 0.8 ? 'success' : document.confidence >= 0.6 ? 'warning' : 'error'}
              size="small"
            />
            <Chip
              label={document.status}
              color={document.status === 'completed' ? 'success' : 'error'}
              size="small"
            />
          </Box>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {/* Processing Progress */}
        {document.status === 'processing' && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" gutterBottom>
              Processing document with OCR...
            </Typography>
            <LinearProgress />
          </Box>
        )}

        {/* Validation Summary */}
        {getValidationSummary()}

        {/* Validation Errors and Warnings */}
        {validationResult && (Object.keys(validationResult.errors).length > 0 || Object.keys(validationResult.warnings).length > 0) && (
          <Card variant="outlined" sx={{ mb: 3, borderColor: 'error.main' }}>
            <CardContent>
              <Typography variant="h6" color="error.main" gutterBottom>
                Validation Issues
              </Typography>

              {Object.keys(validationResult.errors).length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="error.main" gutterBottom>
                    Errors ({Object.keys(validationResult.errors).length}):
                  </Typography>
                  {Object.entries(validationResult.errors).map(([field, error]) => (
                    <Typography key={field} variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Error color="error" fontSize="small" />
                      <strong>{field}:</strong> {error}
                    </Typography>
                  ))}
                </Box>
              )}

              {Object.keys(validationResult.warnings).length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="warning.main" gutterBottom>
                    Warnings ({Object.keys(validationResult.warnings).length}):
                  </Typography>
                  {Object.entries(validationResult.warnings).map(([field, warning]) => (
                    <Typography key={field} variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Error color="warning" fontSize="small" />
                      <strong>{field}:</strong> {warning}
                    </Typography>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        )}

        {/* Form Fields */}
        <Typography variant="h6" gutterBottom>
          Extracted Information
        </Typography>

        <Grid container spacing={3}>
          {formFields.map((field) => (
            <Grid item xs={12} md={6} key={field.name}>
              <Box sx={{ mb: 2 }}>
                {renderFieldInput(field)}
              </Box>
            </Grid>
          ))}
        </Grid>

        {/* Raw Text Preview */}
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Raw OCR Text
          </Typography>
          <Card variant="outlined">
            <CardContent sx={{
              maxHeight: 200,
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              backgroundColor: 'grey.50'
            }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {document.extractedText}
              </pre>
            </CardContent>
          </Card>
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} startIcon={<Close />}>
          Close
        </Button>

        {document.status === 'completed' && (
          <>
            <Button
              onClick={handleValidate}
              startIcon={<Refresh />}
              disabled={editMode}
            >
              Validate
            </Button>

            {editMode ? (
              <>
                <Button
                  onClick={() => setEditMode(false)}
                  startIcon={<Close />}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSave}
                  variant="contained"
                  startIcon={<Save />}
                >
                  Save Changes
                </Button>
              </>
            ) : (
              <Button
                onClick={() => setEditMode(true)}
                variant="contained"
                startIcon={<Edit />}
              >
                Edit
              </Button>
            )}
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default DocumentPreview;

