/**
 * Document Upload Interface
 *
 * Professional document upload interface for mortgage applications with:
 * - Drag-and-drop file upload
 * - Multiple document type support (application forms, income proofs, property docs, IDs)
 * - File validation and preview
 * - Progress tracking and error handling
 * - Integration with backend document processing
 */

import React, { useState, useCallback } from 'react';
import {
  Container,
  Paper,
  Typography,
  Grid,
  Box,
  Button,
  Alert,
  LinearProgress,
  Chip,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Tooltip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import {
  CloudUpload,
  Description,
  PictureAsPdf,
  Image,
  Delete,
  CheckCircle,
  Error,
  UploadFile,
  Assignment,
  Work,
  Home,
  AccountCircle
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { useNavigate } from 'react-router-dom';
import { documentService, DocumentProcessingResult, DOCUMENT_TYPE_CONFIGS } from '../services/documentService';
import DocumentPreview from '../components/DocumentPreview';
import DocumentDisplay from '../components/DocumentDisplay';

// Get document types from the service
const DOCUMENT_TYPES = DOCUMENT_TYPE_CONFIGS;

interface UploadedFile {
  id: string;
  file: File;
  documentType: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  processedResult?: DocumentProcessingResult;
}

const DocumentUpload: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [previewDocument, setPreviewDocument] = useState<DocumentProcessingResult | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      errors.forEach((error: { message: string }) => {
        enqueueSnackbar(`File ${file.name}: ${error.message}`, { variant: 'error' });
      });
    });

    // Add accepted files
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      id: `${Date.now()}-${Math.random()}`,
      file,
      documentType: '',
      status: 'uploading',
      progress: 0
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);

    // Process each file
    newFiles.forEach(uploadedFile => {
      processDocument(uploadedFile.id, uploadedFile.file);
    });
  }, [enqueueSnackbar]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    maxSize: 15 * 1024 * 1024, // 15MB
    multiple: true
  });

  const processDocument = useCallback(async (fileId: string, file: File) => {
    // Update status to processing
    setUploadedFiles(prev =>
      prev.map(f => f.id === fileId ? { ...f, status: 'processing', progress: 50 } : f)
    );

    try {
      // Get the document type from the uploaded file
      const uploadedFile = uploadedFiles.find(f => f.id === fileId);
      const documentType = uploadedFile?.documentType || 'application_form';

      // Process the document using the OCR service
      const result = await documentService.processDocument(file, documentType);

      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId
            ? { ...f, status: 'completed', progress: 100, processedResult: result }
            : f
        )
      );

      enqueueSnackbar(`Document "${file.name}" processed successfully!`, { variant: 'success' });

    } catch (error) {
      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId
            ? { ...f, status: 'error', error: 'Processing failed' }
            : f
        )
      );

      enqueueSnackbar(`Failed to process "${file.name}": Unknown error occurred`, {
        variant: 'error'
      });
    }
  }, [uploadedFiles, enqueueSnackbar]);

  const handleDocumentTypeChange = (fileId: string, documentType: string) => {
    setUploadedFiles(prev =>
      prev.map(file =>
        file.id === fileId ? { ...file, documentType } : file
      )
    );
  };

  const handleDeleteFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    if (extension === 'pdf') return <PictureAsPdf color="error" />;
    return <Image color="primary" />;
  };

  const getDocumentTypeIcon = (type: string) => {
    const docType = DOCUMENT_TYPES[type as keyof typeof DOCUMENT_TYPES] as any;
    if (docType && docType.icon) {
      const IconComponent = docType.icon;
      return <IconComponent />;
    }
    return <Description />;
  };

  const handlePreviewDocument = (document: DocumentProcessingResult) => {
    setPreviewDocument(document);
    setPreviewOpen(true);
  };

  const handleClosePreview = () => {
    setPreviewOpen(false);
    setPreviewDocument(null);
  };

  const handleSaveDocument = (updatedData: Record<string, any>) => {
    if (previewDocument) {
      const updatedDocument: DocumentProcessingResult = {
        ...previewDocument,
        structuredData: updatedData
      };
      setUploadedFiles(prev =>
        prev.map(file =>
          file.processedResult?.id === updatedDocument.id
            ? { ...file, processedResult: updatedDocument }
            : file
        )
      );
      enqueueSnackbar('Document updated successfully!', { variant: 'success' });
    }
  };

  const handleContinue = async () => {
    const completedFiles = uploadedFiles.filter(f => f.status === 'completed');

    if (completedFiles.length === 0) {
      enqueueSnackbar('Please upload and process at least one document', { variant: 'warning' });
      return;
    }

    // Check if all required document types are present
    const requiredTypes = Object.keys(DOCUMENT_TYPES).filter(
      type => (DOCUMENT_TYPES as any)[type].required === true
    );

    const uploadedTypes = [...new Set(
      completedFiles
        .filter(f => f.processedResult)
        .map(f => f.processedResult!.documentType)
    )];

    const missingRequired = requiredTypes.filter(type => !uploadedTypes.includes(type));

    if (missingRequired.length > 0) {
      enqueueSnackbar(
        `Please upload required documents: ${missingRequired.map(type =>
          (DOCUMENT_TYPES as any)[type].label
        ).join(', ')}`,
        { variant: 'error' }
      );
      return;
    }

    // All validations pass, navigate to next step
    enqueueSnackbar('All documents processed successfully!', { variant: 'success' });
    navigate('/quality-control');
  };

  const completedFiles = uploadedFiles.filter(f => f.status === 'completed');
  const totalRequired = Object.values(DOCUMENT_TYPES).filter((doc: any) => doc.required === true).length;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold', mb: 1 }}>
          Document Upload
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Upload your mortgage application documents for automated processing and validation
        </Typography>

        {/* Progress Overview */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Document Processing Progress
          </Typography>
          <Alert severity={completedFiles.length >= totalRequired ? "success" : "info"} sx={{ mb: 2 }}>
            {completedFiles.length} of {totalRequired} required document types processed
          </Alert>
          <LinearProgress
            variant="determinate"
            value={(completedFiles.length / totalRequired) * 100}
            sx={{ height: 8, borderRadius: 4 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Upload documents and they will be automatically processed using OCR technology
          </Typography>
        </Box>

        {/* Required Documents List */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Required Documents
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(DOCUMENT_TYPES).map(([type, config]: [string, any]) => {
              const processedForType = completedFiles.filter(f =>
                f.processedResult?.documentType === type
              ).length;
              const uploadedForType = uploadedFiles.filter(f => f.documentType === type).length;

              const IconComponent = config.icon;
              const isRequired = config.required === true;

              return (
                <Grid item xs={12} sm={6} md={3} key={type}>
                  <Card variant="outlined" sx={{
                    height: '100%',
                    borderColor: processedForType > 0 ? 'success.main' :
                               uploadedForType > 0 ? 'warning.main' : 'divider'
                  }}>
                    <CardContent sx={{ pb: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <IconComponent />
                        <Typography variant="subtitle2" sx={{ ml: 1, fontWeight: 'bold' }}>
                          {config.label}
                          {isRequired && <span style={{ color: 'red' }}> *</span>}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {config.description}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {config.acceptedFormats?.join(', ')} • Max {Math.round(config.maxSize / (1024 * 1024))}MB
                      </Typography>
                    </CardContent>
                    <CardActions sx={{ pt: 0 }}>
                      <Chip
                        size="small"
                        label={
                          processedForType > 0 ? `${processedForType} processed` :
                          uploadedForType > 0 ? 'Processing...' : 'Not uploaded'
                        }
                        color={
                          processedForType > 0 ? "success" :
                          uploadedForType > 0 ? "warning" : "default"
                        }
                      />
                    </CardActions>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Box>

        <Divider sx={{ my: 4 }} />

        {/* File Upload Area */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Upload Files
          </Typography>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: isDragActive ? 'primary.50' : 'grey.50',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'primary.50'
              }
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop files here...' : 'Drag & drop files here, or click to select'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supports PDF, JPG, PNG files • Maximum 15MB per file
            </Typography>
          </Box>
        </Box>

        {/* Processed Documents */}
        {completedFiles.length > 0 && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Processed Documents ({completedFiles.length})
            </Typography>
            {completedFiles.map((uploadedFile) => (
              uploadedFile.processedResult && (
                <DocumentDisplay
                  key={uploadedFile.id}
                  document={uploadedFile.processedResult}
                  onUpdate={handleSaveDocument}
                  editable={true}
                  showValidation={true}
                />
              )
            ))}
          </Box>
        )}

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Upload Queue ({uploadedFiles.length})
            </Typography>
            <List>
              {uploadedFiles.map((uploadedFile) => (
                <Card key={uploadedFile.id} variant="outlined" sx={{ mb: 2 }}>
                  <CardContent sx={{ pb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      {getFileIcon(uploadedFile.file.name)}
                      <Typography variant="subtitle1" sx={{ ml: 1, flexGrow: 1 }}>
                        {uploadedFile.file.name}
                      </Typography>
                      <Chip
                        size="small"
                        label={uploadedFile.status}
                        color={
                          uploadedFile.status === 'completed' ? 'success' :
                          uploadedFile.status === 'error' ? 'error' :
                          uploadedFile.status === 'processing' ? 'info' : 'warning'
                        }
                        icon={
                          uploadedFile.status === 'completed' ? <CheckCircle /> :
                          uploadedFile.status === 'error' ? <Error /> : undefined
                        }
                      />
                      <IconButton
                        onClick={() => handleDeleteFile(uploadedFile.id)}
                        color="error"
                        size="small"
                      >
                        <Delete />
                      </IconButton>
                    </Box>

                    {/* Progress Bar */}
                    {(uploadedFile.status === 'uploading' || uploadedFile.status === 'processing') && (
                      <Box sx={{ mb: 2 }}>
                        <LinearProgress
                          variant="determinate"
                          value={uploadedFile.progress}
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          {uploadedFile.status === 'processing' ? 'Processing with OCR...' : 'Uploading...'}
                        </Typography>
                      </Box>
                    )}

                    {/* Error Message */}
                    {uploadedFile.status === 'error' && uploadedFile.error && (
                      <Alert severity="error" sx={{ mb: 2 }}>
                        {uploadedFile.error}
                      </Alert>
                    )}

                    {/* Document Type Selection */}
                    {uploadedFile.status !== 'completed' && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" gutterBottom>
                          Document Type:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                          {Object.entries(DOCUMENT_TYPES).map(([type, config]: [string, any]) => {
                            const IconComponent = config.icon;
                            return (
                              <Chip
                                key={type}
                                label={config.label}
                                variant={uploadedFile.documentType === type ? "filled" : "outlined"}
                                color={uploadedFile.documentType === type ? "primary" : "default"}
                                icon={<IconComponent />}
                                onClick={() => handleDocumentTypeChange(uploadedFile.id, type)}
                                sx={{ cursor: 'pointer' }}
                              />
                            );
                          })}
                        </Box>
                      </Box>
                    )}

                    {/* Preview Button for completed files */}
                    {uploadedFile.status === 'completed' && uploadedFile.processedResult && (
                      <Box sx={{ mb: 2 }}>
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={() => handlePreviewDocument(uploadedFile.processedResult!)}
                          startIcon={<Description />}
                        >
                          Preview & Edit
                        </Button>
                      </Box>
                    )}

                    {/* File Size */}
                    <Typography variant="caption" color="text.secondary">
                      {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </List>
          </Box>
        )}

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            variant="outlined"
            onClick={() => navigate('/')}
            sx={{ borderRadius: 2 }}
          >
            Back to Application
          </Button>
          <Button
            variant="contained"
            onClick={handleContinue}
            disabled={completedFiles.length === 0 || uploading}
            startIcon={<UploadFile />}
            sx={{ borderRadius: 2, px: 4 }}
          >
            Continue to Quality Control
          </Button>
        </Box>
      </Paper>

      {/* Document Preview Dialog */}
      <DocumentPreview
        open={previewOpen}
        onClose={handleClosePreview}
        document={previewDocument}
        onSave={handleSaveDocument}
      />
    </Container>
  );
};

export default DocumentUpload;
