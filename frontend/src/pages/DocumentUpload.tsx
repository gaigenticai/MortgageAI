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
    <Container maxWidth="lg" sx={{ mt: 8, mb: 8 }}>
      <Paper elevation={0} sx={{
        p: 6,
        borderRadius: 4,
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(226, 232, 240, 0.8)',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1), 0 4px 10px rgba(0, 0, 0, 0.05)',
      }}>
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Box sx={{
            width: 80,
            height: 80,
            borderRadius: 4,
            background: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mx: 'auto',
            mb: 4,
            boxShadow: '0 8px 24px rgba(59, 130, 246, 0.25), 0 4px 12px rgba(0, 0, 0, 0.1)',
          }}>
            <UploadFile sx={{ color: 'white', fontSize: 40 }} />
          </Box>
          <Typography
            variant="h2"
            component="h1"
            gutterBottom
            align="center"
            sx={{
              fontWeight: 700,
              mb: 3,
              background: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Document Upload
          </Typography>
          <Typography
            variant="h6"
            align="center"
            color="text.secondary"
            sx={{ fontWeight: 400, maxWidth: 600, mx: 'auto', lineHeight: 1.6 }}
          >
            Upload your mortgage application documents for automated AI-powered processing and validation
          </Typography>
        </Box>

        {/* Progress Overview */}
        <Box sx={{ mb: 6 }}>
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 3,
            mb: 4,
            p: 4,
            borderRadius: 3,
            background: 'rgba(99, 102, 241, 0.05)',
            border: '1px solid rgba(99, 102, 241, 0.1)',
          }}>
            <Box sx={{
              width: 56,
              height: 56,
              borderRadius: 3,
              background: 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(16, 185, 129, 0.25)',
            }}>
              <CheckCircle sx={{ color: 'white', fontSize: 28 }} />
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'text.primary', mb: 1 }}>
                Document Processing Progress
              </Typography>
              <Typography variant="body1" color="text.secondary">
                {completedFiles.length} of {totalRequired} required document types processed
              </Typography>
              <Box sx={{
                mt: 2,
                p: 2,
                borderRadius: 2,
                background: completedFiles.length >= totalRequired
                  ? 'rgba(16, 185, 129, 0.1)'
                  : 'rgba(59, 130, 246, 0.1)',
                border: `1px solid ${completedFiles.length >= totalRequired ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)'}`,
              }}>
                <LinearProgress
                  variant="determinate"
                  value={(completedFiles.length / totalRequired) * 100}
                  sx={{
                    height: 10,
                    borderRadius: 2,
                    background: 'rgba(226, 232, 240, 0.8)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 2,
                      background: completedFiles.length >= totalRequired
                        ? 'linear-gradient(135deg, #10B981 0%, #34D399 100%)'
                        : 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                    }
                  }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1, fontSize: '0.875rem' }}>
                  Upload documents and they will be automatically processed using advanced AI OCR technology
                </Typography>
              </Box>
            </Box>
          </Box>
        </Box>

        {/* Required Documents List */}
        <Box sx={{ mb: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, color: 'text.primary', mb: 4 }}>
            Required Documents
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(DOCUMENT_TYPES).map(([type, config]: [string, any]) => {
              const processedForType = completedFiles.filter(f =>
                f.processedResult?.documentType === type
              ).length;
              const uploadedForType = uploadedFiles.filter(f => f.documentType === type).length;

              const IconComponent = config.icon;
              const isRequired = config.required === true;

              return (
                <Grid item xs={12} sm={6} md={4} key={type}>
                  <Card sx={{
                    height: '100%',
                    background: 'rgba(255, 255, 255, 0.9)',
                    backdropFilter: 'blur(10px)',
                    border: processedForType > 0
                      ? '1px solid rgba(16, 185, 129, 0.2)'
                      : uploadedForType > 0
                      ? '1px solid rgba(245, 158, 11, 0.2)'
                      : '1px solid rgba(226, 232, 240, 0.8)',
                    borderRadius: 3,
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15), 0 4px 10px rgba(0, 0, 0, 0.1)',
                    }
                  }}>
                    <CardContent sx={{ p: 4 }}>
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 3 }}>
                        <Box sx={{
                          width: 56,
                          height: 56,
                          borderRadius: 3,
                          background: processedForType > 0
                            ? 'linear-gradient(135deg, #10B981 0%, #34D399 100%)'
                            : uploadedForType > 0
                            ? 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)'
                            : 'linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 100%)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          boxShadow: processedForType > 0
                            ? '0 4px 12px rgba(16, 185, 129, 0.25)'
                            : uploadedForType > 0
                            ? '0 4px 12px rgba(245, 158, 11, 0.25)'
                            : '0 4px 12px rgba(0, 0, 0, 0.1)',
                        }}>
                          <IconComponent sx={{
                            color: processedForType > 0 || uploadedForType > 0 ? 'white' : 'text.secondary',
                            fontSize: 24
                          }} />
                        </Box>
                        {isRequired && (
                          <Chip
                            label="Required"
                            size="small"
                            sx={{
                              background: 'rgba(239, 68, 68, 0.1)',
                              color: 'error.main',
                              border: '1px solid rgba(239, 68, 68, 0.2)',
                              fontWeight: 600,
                              fontSize: '0.75rem',
                            }}
                          />
                        )}
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary', mb: 2 }}>
                        {config.label}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.5 }}>
                        {config.description}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.8125rem' }}>
                        {config.acceptedFormats?.join(', ')} • Max {Math.round(config.maxSize / (1024 * 1024))}MB
                      </Typography>
                    </CardContent>
                    <CardActions sx={{ px: 4, pb: 4, pt: 0 }}>
                      <Chip
                        size="small"
                        label={
                          processedForType > 0 ? `${processedForType} processed` :
                          uploadedForType > 0 ? 'Processing...' : 'Not uploaded'
                        }
                        sx={{
                          fontWeight: 600,
                          fontSize: '0.8125rem',
                          background: processedForType > 0
                            ? 'rgba(16, 185, 129, 0.1)'
                            : uploadedForType > 0
                            ? 'rgba(245, 158, 11, 0.1)'
                            : 'rgba(226, 232, 240, 0.8)',
                          color: processedForType > 0
                            ? 'success.main'
                            : uploadedForType > 0
                            ? 'warning.main'
                            : 'text.secondary',
                          border: processedForType > 0
                            ? '1px solid rgba(16, 185, 129, 0.2)'
                            : uploadedForType > 0
                            ? '1px solid rgba(245, 158, 11, 0.2)'
                            : '1px solid rgba(226, 232, 240, 0.8)',
                        }}
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
        <Box sx={{ mb: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, color: 'text.primary', mb: 4 }}>
            Upload Files
          </Typography>
          <Box
            {...getRootProps()}
            sx={{
              border: '3px dashed',
              borderColor: isDragActive ? '#6366F1' : '#E2E8F0',
              borderRadius: 4,
              p: 8,
              textAlign: 'center',
              cursor: 'pointer',
              background: isDragActive
                ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%)'
                : 'linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(248, 250, 252, 0.8) 100%)',
              backdropFilter: 'blur(10px)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: isDragActive
                  ? 'radial-gradient(circle at center, rgba(99, 102, 241, 0.03) 0%, transparent 70%)'
                  : 'transparent',
                transition: 'all 0.3s ease',
              },
              '&:hover': {
                borderColor: '#6366F1',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.03) 0%, rgba(139, 92, 246, 0.03) 100%)',
                transform: 'translateY(-2px)',
                boxShadow: '0 8px 24px rgba(99, 102, 241, 0.15), 0 4px 12px rgba(0, 0, 0, 0.1)',
                '& .upload-icon': {
                  transform: 'scale(1.1) rotate(5deg)',
                },
              }
            }}
          >
            <input {...getInputProps()} />
            <Box sx={{
              width: 80,
              height: 80,
              borderRadius: 4,
              background: isDragActive
                ? 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)'
                : 'linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mx: 'auto',
              mb: 4,
              boxShadow: isDragActive
                ? '0 8px 24px rgba(99, 102, 241, 0.25), 0 4px 12px rgba(0, 0, 0, 0.1)'
                : '0 4px 12px rgba(0, 0, 0, 0.1)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            }}>
              <CloudUpload
                className="upload-icon"
                sx={{
                  fontSize: 40,
                  color: isDragActive ? 'white' : 'text.secondary',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
              />
            </Box>
            <Typography
              variant="h5"
              gutterBottom
              sx={{
                fontWeight: 600,
                color: 'text.primary',
                mb: 2
              }}
            >
              {isDragActive ? 'Drop files here...' : 'Drag & drop files here, or click to select'}
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{
                fontSize: '1rem',
                maxWidth: 400,
                mx: 'auto',
                lineHeight: 1.6
              }}
            >
              Supports PDF, JPG, PNG files • Maximum 15MB per file • AI-powered OCR processing
            </Typography>
            <Box sx={{
              mt: 3,
              display: 'flex',
              justifyContent: 'center',
              gap: 2,
              flexWrap: 'wrap'
            }}>
              {['PDF', 'JPG', 'PNG'].map((format) => (
                <Chip
                  key={format}
                  label={format}
                  size="small"
                  sx={{
                    background: 'rgba(99, 102, 241, 0.1)',
                    color: 'primary.main',
                    border: '1px solid rgba(99, 102, 241, 0.2)',
                    fontWeight: 500,
                    fontSize: '0.8125rem',
                  }}
                />
              ))}
            </Box>
          </Box>
        </Box>

        {/* Processed Documents */}
        {completedFiles.length > 0 && (
          <Box sx={{ mb: 6 }}>
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 3,
              mb: 4,
              p: 3,
              borderRadius: 3,
              background: 'rgba(16, 185, 129, 0.05)',
              border: '1px solid rgba(16, 185, 129, 0.1)',
            }}>
              <Box sx={{
                width: 48,
                height: 48,
                borderRadius: 3,
                background: 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(16, 185, 129, 0.25)',
              }}>
                <CheckCircle sx={{ color: 'white', fontSize: 24 }} />
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 600, color: 'text.primary' }}>
                  Processed Documents ({completedFiles.length})
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Documents have been successfully processed with AI-powered OCR
                </Typography>
              </Box>
            </Box>
            <Grid container spacing={3}>
              {completedFiles.map((uploadedFile) => (
                uploadedFile.processedResult && (
                  <Grid item xs={12} key={uploadedFile.id}>
                    <DocumentDisplay
                      document={uploadedFile.processedResult}
                      onUpdate={handleSaveDocument}
                      editable={true}
                      showValidation={true}
                    />
                  </Grid>
                )
              ))}
            </Grid>
          </Box>
        )}

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <Box sx={{ mb: 6 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, color: 'text.primary', mb: 4 }}>
              Upload Queue ({uploadedFiles.length})
            </Typography>
            <Grid container spacing={3}>
              {uploadedFiles.map((uploadedFile) => (
                <Grid item xs={12} md={6} key={uploadedFile.id}>
                  <Card sx={{
                    background: 'rgba(255, 255, 255, 0.9)',
                    backdropFilter: 'blur(10px)',
                    border: uploadedFile.status === 'completed'
                      ? '1px solid rgba(16, 185, 129, 0.2)'
                      : uploadedFile.status === 'error'
                      ? '1px solid rgba(239, 68, 68, 0.2)'
                      : uploadedFile.status === 'processing'
                      ? '1px solid rgba(245, 158, 11, 0.2)'
                      : '1px solid rgba(226, 232, 240, 0.8)',
                    borderRadius: 3,
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15), 0 4px 12px rgba(0, 0, 0, 0.1)',
                    }
                  }}>
                    <CardContent sx={{ p: 4 }}>
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, flex: 1 }}>
                          <Box sx={{
                            width: 56,
                            height: 56,
                            borderRadius: 3,
                            background: uploadedFile.status === 'completed'
                              ? 'linear-gradient(135deg, #10B981 0%, #34D399 100%)'
                              : uploadedFile.status === 'error'
                              ? 'linear-gradient(135deg, #EF4444 0%, #F87171 100%)'
                              : uploadedFile.status === 'processing'
                              ? 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)'
                              : 'linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 100%)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                          }}>
                            {getFileIcon(uploadedFile.file.name)}
                          </Box>
                          <Box sx={{ flex: 1, minWidth: 0 }}>
                            <Typography
                              variant="h6"
                              sx={{
                                fontWeight: 600,
                                color: 'text.primary',
                                mb: 1,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}
                            >
                              {uploadedFile.file.name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                            </Typography>
                            <Chip
                              size="small"
                              label={uploadedFile.status.charAt(0).toUpperCase() + uploadedFile.status.slice(1)}
                              sx={{
                                fontWeight: 600,
                                fontSize: '0.8125rem',
                                background: uploadedFile.status === 'completed'
                                  ? 'rgba(16, 185, 129, 0.1)'
                                  : uploadedFile.status === 'error'
                                  ? 'rgba(239, 68, 68, 0.1)'
                                  : uploadedFile.status === 'processing'
                                  ? 'rgba(245, 158, 11, 0.1)'
                                  : 'rgba(59, 130, 246, 0.1)',
                                color: uploadedFile.status === 'completed'
                                  ? 'success.main'
                                  : uploadedFile.status === 'error'
                                  ? 'error.main'
                                  : uploadedFile.status === 'processing'
                                  ? 'warning.main'
                                  : 'info.main',
                                border: uploadedFile.status === 'completed'
                                  ? '1px solid rgba(16, 185, 129, 0.2)'
                                  : uploadedFile.status === 'error'
                                  ? '1px solid rgba(239, 68, 68, 0.2)'
                                  : uploadedFile.status === 'processing'
                                  ? '1px solid rgba(245, 158, 11, 0.2)'
                                  : '1px solid rgba(59, 130, 246, 0.2)',
                              }}
                              icon={
                                uploadedFile.status === 'completed' ? <CheckCircle fontSize="small" /> :
                                uploadedFile.status === 'error' ? <Error fontSize="small" /> :
                                undefined
                              }
                            />
                          </Box>
                        </Box>
                        <IconButton
                          onClick={() => handleDeleteFile(uploadedFile.id)}
                          sx={{
                            color: 'error.main',
                            '&:hover': {
                              background: 'rgba(239, 68, 68, 0.1)',
                            }
                          }}
                          size="small"
                        >
                          <Delete />
                        </IconButton>
                      </Box>

                      {/* Progress Bar */}
                      {(uploadedFile.status === 'uploading' || uploadedFile.status === 'processing') && (
                        <Box sx={{ mb: 3 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.875rem' }}>
                              {uploadedFile.status === 'processing' ? 'Processing with AI OCR...' : 'Uploading...'}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {uploadedFile.progress}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={uploadedFile.progress}
                            sx={{
                              height: 6,
                              borderRadius: 3,
                              background: 'rgba(226, 232, 240, 0.8)',
                              '& .MuiLinearProgress-bar': {
                                borderRadius: 3,
                                background: uploadedFile.status === 'processing'
                                  ? 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)'
                                  : 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                              }
                            }}
                          />
                        </Box>
                      )}

                      {/* Error Message */}
                      {uploadedFile.status === 'error' && uploadedFile.error && (
                        <Alert
                          severity="error"
                          sx={{
                            mb: 3,
                            borderRadius: 2,
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                          }}
                        >
                          {uploadedFile.error}
                        </Alert>
                      )}

                      {/* Document Type Selection */}
                      {uploadedFile.status !== 'completed' && (
                        <Box sx={{ mb: 3 }}>
                          <Typography variant="body2" gutterBottom sx={{ fontWeight: 500, color: 'text.primary' }}>
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
                                  sx={{
                                    cursor: 'pointer',
                                    fontWeight: 500,
                                    fontSize: '0.8125rem',
                                    background: uploadedFile.documentType === type
                                      ? 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)'
                                      : 'rgba(255, 255, 255, 0.8)',
                                    color: uploadedFile.documentType === type ? 'white' : 'text.primary',
                                    border: uploadedFile.documentType === type
                                      ? 'none'
                                      : '1px solid rgba(226, 232, 240, 0.8)',
                                    '&:hover': {
                                      background: uploadedFile.documentType === type
                                        ? 'linear-gradient(135deg, #4338CA 0%, #7C3AED 100%)'
                                        : 'rgba(99, 102, 241, 0.1)',
                                      transform: 'translateY(-1px)',
                                    },
                                    transition: 'all 0.2s ease',
                                  }}
                                  icon={<IconComponent sx={{ fontSize: 16 }} />}
                                  onClick={() => handleDocumentTypeChange(uploadedFile.id, type)}
                                />
                              );
                            })}
                          </Box>
                        </Box>
                      )}

                      {/* Preview Button for completed files */}
                      {uploadedFile.status === 'completed' && uploadedFile.processedResult && (
                        <Button
                          variant="outlined"
                          onClick={() => handlePreviewDocument(uploadedFile.processedResult!)}
                          startIcon={<Description />}
                          sx={{
                            borderRadius: 2,
                            borderColor: 'rgba(16, 185, 129, 0.3)',
                            color: 'success.main',
                            '&:hover': {
                              background: 'rgba(16, 185, 129, 0.1)',
                              borderColor: 'success.main',
                              transform: 'translateY(-1px)',
                            },
                            transition: 'all 0.2s ease',
                          }}
                        >
                          Preview & Edit
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 6 }}>
          <Button
            variant="outlined"
            onClick={() => navigate('/')}
            sx={{
              borderRadius: 3,
              px: 4,
              py: 2,
              fontSize: '1rem',
              fontWeight: 500,
              borderColor: 'rgba(226, 232, 240, 0.8)',
              color: 'text.primary',
              '&:hover': {
                background: 'rgba(99, 102, 241, 0.05)',
                borderColor: '#6366F1',
                transform: 'translateY(-1px)',
                boxShadow: '0 4px 12px rgba(99, 102, 241, 0.15)',
              },
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            ← Back to Application
          </Button>
          <Button
            variant="contained"
            onClick={handleContinue}
            disabled={completedFiles.length === 0 || uploading}
            startIcon={<UploadFile />}
            sx={{
              borderRadius: 3,
              px: 6,
              py: 2,
              fontSize: '1.125rem',
              fontWeight: 600,
              background: completedFiles.length >= totalRequired
                ? 'linear-gradient(135deg, #10B981 0%, #34D399 100%)'
                : 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
              boxShadow: completedFiles.length >= totalRequired
                ? '0 4px 12px rgba(16, 185, 129, 0.25)'
                : '0 4px 12px rgba(99, 102, 241, 0.25)',
              '&:hover': {
                background: completedFiles.length >= totalRequired
                  ? 'linear-gradient(135deg, #047857 0%, #10B981 100%)'
                  : 'linear-gradient(135deg, #4338CA 0%, #7C3AED 100%)',
                transform: 'translateY(-2px)',
                boxShadow: completedFiles.length >= totalRequired
                  ? '0 8px 24px rgba(16, 185, 129, 0.35)'
                  : '0 8px 24px rgba(99, 102, 241, 0.35)',
              },
              '&:disabled': {
                background: '#E2E8F0',
                color: '#64748B',
                boxShadow: 'none',
              },
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            Continue to Quality Control →
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
