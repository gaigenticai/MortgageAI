/**
 * Document Upload Interface - Full Mantine Implementation
 *
 * Comprehensive document upload system with:
 * - Drag & drop file upload with validation
 * - OCR processing and text extraction
 * - Document type classification
 * - Real-time validation and error handling
 * - Progress tracking and status updates
 * - Integration with backend document processing
 */

import React, { useState, useCallback } from 'react';
import {
  Container,
  Paper,
  Text,
  Box,
  Button,
  Alert,
  Grid,
  Card,
  Badge,
  List,
  Title,
  Group,
  Stack,
  ActionIcon,
  Tooltip,
  ThemeIcon,
  SimpleGrid,
  Progress,
  Loader,
  Select,
  Textarea,
  Accordion,
  Table,
  Divider,
  FileInput,
} from '@mantine/core';
import {
  IconUpload,
  IconFile,
  IconCheck,
  IconX,
  IconAlertTriangle,
  IconInfoCircle,
  IconFileText,
  IconScan,
  IconDownload,
  IconEye,
  IconTrash,
  IconRefresh,
  IconCloudUpload,
  IconFileCheck,
  IconClock,
  IconAlertCircle,
  IconFileX,
} from '@tabler/icons-react';
// Removed Dropzone import - using FileInput instead
import { notifications } from '@mantine/notifications';
import { useNavigate } from 'react-router-dom';
import { documentService, DocumentProcessingResult, DOCUMENT_TYPE_CONFIGS } from '../services/documentService';

interface UploadedDocument extends DocumentProcessingResult {
  file: File;
  uploadProgress: number;
  processing: boolean;
}

const DocumentUpload: React.FC = () => {
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [selectedDocumentType, setSelectedDocumentType] = useState<string>('application_form');
  const [processing, setProcessing] = useState(false);

  const documentTypes = Object.entries(DOCUMENT_TYPE_CONFIGS).map(([key, config]) => ({
    value: key,
    label: config.label,
    description: config.description,
  }));

  const handleFileUpload = useCallback(async (files: File[]) => {
    for (const file of files) {
      // Validate file type
      const config = DOCUMENT_TYPE_CONFIGS[selectedDocumentType];
      const isValidType = config.acceptedFormats.some(format => 
        file.name.toLowerCase().endsWith(format.toLowerCase())
      );

      if (!isValidType) {
        notifications.show({
          title: 'Invalid File Type',
          message: `Please upload files in one of these formats: ${config.acceptedFormats.join(', ')}`,
          color: 'red',
          icon: <IconX size={16} />,
        });
        continue;
      }

      // Validate file size
      if (file.size > config.maxSize) {
        notifications.show({
          title: 'File Too Large',
          message: `File size must be less than ${(config.maxSize / (1024 * 1024)).toFixed(1)}MB`,
          color: 'red',
          icon: <IconX size={16} />,
        });
        continue;
      }

      // Add document to processing queue
      const newDoc: UploadedDocument = {
        id: `temp_${Date.now()}_${Math.random()}`,
        documentType: selectedDocumentType,
        fileName: file.name,
        fileSize: file.size,
        extractedText: '',
        structuredData: {},
        confidence: 0,
        processingTime: 0,
        status: 'processing',
      file,
        uploadProgress: 0,
        processing: true,
      };

      setDocuments(prev => [...prev, newDoc]);

      // Start processing
      try {
        setProcessing(true);
        
        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 100));
          setDocuments(prev => prev.map(doc => 
            doc.id === newDoc.id ? { ...doc, uploadProgress: progress } : doc
          ));
        }

        // Process document with OCR
        const result = await documentService.processDocument(file, selectedDocumentType);
        
        setDocuments(prev => prev.map(doc => 
          doc.id === newDoc.id 
            ? { ...doc, ...result, processing: false, uploadProgress: 100 }
            : doc
        ));

        notifications.show({
          title: 'Document Processed',
          message: `${file.name} has been successfully processed`,
          color: 'green',
          icon: <IconCheck size={16} />,
        });

      } catch (error) {
        setDocuments(prev => prev.map(doc => 
          doc.id === newDoc.id 
            ? { 
                ...doc, 
                status: 'error', 
                error: error instanceof Error ? error.message : 'Processing failed',
                processing: false 
              }
            : doc
        ));

        notifications.show({
          title: 'Processing Failed',
          message: `Failed to process ${file.name}`,
          color: 'red',
          icon: <IconX size={16} />,
        });
      } finally {
        setProcessing(false);
      }
    }
  }, [selectedDocumentType]);

  const handleFileInputChange = (files: File[] | null) => {
    if (files && files.length > 0) {
      handleFileUpload(files);
    }
  };

  const handleReprocess = async (docId: string) => {
    const doc = documents.find(d => d.id === docId);
    if (!doc) return;

    setDocuments(prev => prev.map(d => 
      d.id === docId ? { ...d, processing: true, status: 'processing' } : d
    ));

    try {
      const result = await documentService.processDocument(doc.file, doc.documentType);
      
      setDocuments(prev => prev.map(d => 
        d.id === docId ? { ...d, ...result, processing: false } : d
      ));

      notifications.show({
        title: 'Document Reprocessed',
        message: `${doc.fileName} has been reprocessed successfully`,
        color: 'green',
        icon: <IconCheck size={16} />,
      });

    } catch (error) {
      setDocuments(prev => prev.map(d => 
        d.id === docId 
          ? { 
              ...d, 
              status: 'error', 
              error: error instanceof Error ? error.message : 'Reprocessing failed',
              processing: false 
            }
          : d
      ));
    }
  };

  const handleRemoveDocument = (docId: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== docId));
    notifications.show({
      title: 'Document Removed',
      message: 'Document has been removed from the list',
      color: 'blue',
      icon: <IconTrash size={16} />,
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'green';
      case 'processing': return 'blue';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string, processing: boolean) => {
    if (processing) return <Loader size={16} />;
    
    switch (status) {
      case 'completed': return <IconCheck size={16} />;
      case 'error': return <IconX size={16} />;
      default: return <IconClock size={16} />;
    }
  };

  const totalDocuments = documents.length;
  const completedDocuments = documents.filter(doc => doc.status === 'completed').length;
  const errorDocuments = documents.filter(doc => doc.status === 'error').length;

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between">
          <Group>
            <ThemeIcon size="xl" radius={0} color="indigo">
              <IconUpload size={32} />
            </ThemeIcon>
            <div>
              <Title order={1}>Document Upload & Processing</Title>
              <Text c="dimmed">Upload documents for OCR processing and validation</Text>
            </div>
          </Group>
          <Group>
            <Button 
              variant="outline" 
              leftSection={<IconRefresh size={16} />}
              onClick={() => setDocuments([])}
              disabled={processing}
              radius={0}
            >
              Clear All
            </Button>
            <Button 
              leftSection={<IconDownload size={16} />}
              disabled={completedDocuments === 0}
              radius={0}
            >
              Export Results
            </Button>
          </Group>
        </Group>

        {/* Statistics */}
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="lg">
          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Total Documents</Text>
                <Title order={2}>{totalDocuments}</Title>
              </div>
              <ThemeIcon size="xl" color="blue" radius={0}>
                <IconFile size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Completed</Text>
                <Title order={2}>{completedDocuments}</Title>
              </div>
              <ThemeIcon size="xl" color="green" radius={0}>
                <IconFileCheck size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Processing</Text>
                <Title order={2}>{documents.filter(doc => doc.processing).length}</Title>
              </div>
              <ThemeIcon size="xl" color="orange" radius={0}>
                <IconScan size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Errors</Text>
                <Title order={2}>{errorDocuments}</Title>
              </div>
              <ThemeIcon size="xl" color="red" radius={0}>
                <IconFileX size={24} />
              </ThemeIcon>
            </Group>
          </Card>
        </SimpleGrid>

        {/* Upload Area */}
        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="md">Upload Documents</Title>
          
          <Select
            label="Document Type"
            description="Select the type of document you're uploading"
            data={documentTypes}
            value={selectedDocumentType}
            onChange={(value) => setSelectedDocumentType(value || 'application_form')}
            mb="md"
            radius={0}
          />

          <Paper withBorder p="xl" radius={0} style={{ minHeight: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Stack align="center" gap="md">
              <ThemeIcon size={64} color="blue" radius={0}>
                <IconCloudUpload size={32} />
              </ThemeIcon>
              <div style={{ textAlign: 'center' }}>
                <Text size="xl" mb="xs">
                  Upload Documents
                </Text>
                <Text size="sm" c="dimmed" mb="md">
                  Select files for {DOCUMENT_TYPE_CONFIGS[selectedDocumentType]?.label}. 
                  Supported formats: {DOCUMENT_TYPE_CONFIGS[selectedDocumentType]?.acceptedFormats.join(', ')}
                </Text>
                <FileInput
                  placeholder="Click to select files"
                  multiple
                  accept=".pdf,.png,.jpg,.jpeg"
                  onChange={handleFileInputChange}
                  disabled={processing}
                  radius={0}
                  leftSection={<IconUpload size={16} />}
                />
              </div>
            </Stack>
          </Paper>

          {DOCUMENT_TYPE_CONFIGS[selectedDocumentType] && (
            <Alert color="blue" icon={<IconInfoCircle size={16} />} mt="md" radius={0}>
              <Text size="sm">
                <strong>Required fields for {DOCUMENT_TYPE_CONFIGS[selectedDocumentType].label}:</strong>
                <br />
                {DOCUMENT_TYPE_CONFIGS[selectedDocumentType].requiredFields.join(', ')}
              </Text>
            </Alert>
          )}
        </Card>

        {/* Document List */}
        {documents.length > 0 && (
          <Card radius={0} shadow="sm" padding="lg">
            <Title order={3} mb="md">Uploaded Documents</Title>
            
            <Stack gap="md">
              {documents.map((doc) => (
                <Paper key={doc.id} p="md" withBorder radius={0}>
                  <Grid>
                    <Grid.Col span={{ base: 12, md: 8 }}>
                      <Group justify="space-between" mb="sm">
                        <Group>
                          <ThemeIcon 
                            color={getStatusColor(doc.status)} 
                            radius={0}
                            size="lg"
                          >
                            {getStatusIcon(doc.status, doc.processing)}
                          </ThemeIcon>
                          <div>
                            <Text fw={500}>{doc.fileName}</Text>
                            <Text size="sm" c="dimmed">
                              {DOCUMENT_TYPE_CONFIGS[doc.documentType]?.label} • 
                              {(doc.fileSize / 1024).toFixed(1)} KB
                            </Text>
                          </div>
                        </Group>
                        <Badge color={getStatusColor(doc.status)} radius={0}>
                          {doc.status.toUpperCase()}
                        </Badge>
                      </Group>

                      {doc.processing && (
                        <div>
                          <Text size="sm" mb="xs">Upload Progress: {doc.uploadProgress}%</Text>
                          <Progress value={doc.uploadProgress} size="sm" radius={0} />
                        </div>
                      )}

                      {doc.status === 'completed' && (
                        <div>
                          <Group gap="xl" mb="sm">
                            <div>
                              <Text size="xs" c="dimmed">Confidence</Text>
                              <Text fw={500}>{(doc.confidence * 100).toFixed(1)}%</Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">Processing Time</Text>
                              <Text fw={500}>{(doc.processingTime / 1000).toFixed(1)}s</Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">Fields Extracted</Text>
                              <Text fw={500}>{Object.keys(doc.structuredData).length}</Text>
                            </div>
                          </Group>
                          
                          <Accordion radius={0}>
                            <Accordion.Item value="extracted-data">
                              <Accordion.Control>
                                <Group>
                                  <IconFileText size={16} />
                                  <Text>Extracted Data</Text>
                                </Group>
                              </Accordion.Control>
                              <Accordion.Panel>
                                <Table>
                                  <Table.Thead>
                                    <Table.Tr>
                                      <Table.Th>Field</Table.Th>
                                      <Table.Th>Value</Table.Th>
                                    </Table.Tr>
                                  </Table.Thead>
                                  <Table.Tbody>
                                    {Object.entries(doc.structuredData).map(([key, value]) => (
                                      <Table.Tr key={key}>
                                        <Table.Td>{key.replace(/_/g, ' ').toUpperCase()}</Table.Td>
                                        <Table.Td>{String(value)}</Table.Td>
                                      </Table.Tr>
                                    ))}
                                  </Table.Tbody>
                                </Table>
                              </Accordion.Panel>
                            </Accordion.Item>
                            
                            <Accordion.Item value="extracted-text">
                              <Accordion.Control>
                                <Group>
                                  <IconScan size={16} />
                                  <Text>OCR Text</Text>
                                </Group>
                              </Accordion.Control>
                              <Accordion.Panel>
                                <Textarea
                                  value={doc.extractedText}
                                  readOnly
                                  minRows={4}
                                  maxRows={8}
                                  radius={0}
                                />
                              </Accordion.Panel>
                            </Accordion.Item>
                          </Accordion>
                        </div>
                      )}

                      {doc.status === 'error' && doc.error && (
                        <Alert color="red" icon={<IconAlertTriangle size={16} />} radius={0}>
                          <Text size="sm">{doc.error}</Text>
                        </Alert>
                      )}
                    </Grid.Col>
                    
                    <Grid.Col span={{ base: 12, md: 4 }}>
                      <Group justify="flex-end">
                        {doc.status === 'error' && (
                          <Tooltip label="Reprocess Document">
                            <ActionIcon 
                              color="blue" 
                              onClick={() => handleReprocess(doc.id)}
                              disabled={doc.processing}
                              radius={0}
                            >
                              <IconRefresh size={16} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                        
                        {doc.status === 'completed' && (
                          <Tooltip label="View Details">
                            <ActionIcon color="green" radius={0}>
                              <IconEye size={16} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                        
                        <Tooltip label="Remove Document">
                          <ActionIcon 
                            color="red" 
                            onClick={() => handleRemoveDocument(doc.id)}
                            disabled={doc.processing}
                            radius={0}
                          >
                            <IconTrash size={16} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Grid.Col>
                </Grid>
                </Paper>
              ))}
            </Stack>
          </Card>
        )}

        {/* Action Buttons */}
        <Group justify="space-between">
          <Button
            variant="outline" 
            leftSection={<IconFile size={16} />}
            onClick={() => navigate(-1)}
            radius={0}
          >
            Back to Application
          </Button>
          <Group>
            {completedDocuments > 0 && (
          <Button
                leftSection={<IconCheck size={16} />}
                onClick={() => navigate('/quality-control')}
                radius={0}
              >
                Continue to Quality Control
          </Button>
            )}
          </Group>
        </Group>
      </Stack>
    </Container>
  );
};

export default DocumentUpload;