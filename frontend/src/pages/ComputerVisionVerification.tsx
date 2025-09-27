/**
 * Computer Vision Document Verification Interface
 * 
 * Advanced document verification system using computer vision for mortgage applications.
 * Features forgery detection, signature analysis, tampering detection, and authenticity scoring.
 * 
 * Key Features:
 * - Drag & drop document upload with preview
 * - Real-time verification progress tracking
 * - Comprehensive verification results visualization
 * - Reference signature comparison
 * - Batch document processing
 * - Detailed forensic analysis reports
 * - Blockchain-based audit trail
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box,
    Paper,
    Typography,
    Button,
    Alert,
    CircularProgress,
    LinearProgress,
    Card,
    CardContent,
    Grid,
    Chip,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tooltip,
    IconButton,
    Switch,
    FormControlLabel,
    Divider,
    Tab,
    Tabs,
    TabPanel
} from '@mui/material';
import {
    CloudUpload as UploadIcon,
    Security as SecurityIcon,
    CheckCircle as CheckCircleIcon,
    Warning as WarningIcon,
    Error as ErrorIcon,
    Info as InfoIcon,
    ExpandMore as ExpandMoreIcon,
    Download as DownloadIcon,
    Delete as DeleteIcon,
    Refresh as RefreshIcon,
    Timeline as TimelineIcon,
    BarChart as BarChartIcon,
    Fingerprint as FingerprintIcon,
    Shield as ShieldIcon,
    SearchOff as SearchOffIcon,
    BugReport as BugReportIcon,
    Visibility as VisibilityIcon,
    VerifiedUser as VerifiedUserIcon,
    ReportProblem as ReportProblemIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { Line, Bar, Pie, Radar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    RadialLinearScale,
    Title,
    Tooltip as ChartTooltip,
    Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    RadialLinearScale,
    Title,
    ChartTooltip,
    Legend
);

// Interfaces
interface VerificationFile {
    id: string;
    file: File;
    preview?: string;
    type: 'document' | 'reference';
    status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
    progress?: number;
    result?: VerificationResult;
    error?: string;
}

interface TamperingEvidence {
    tampering_type: string;
    confidence: number;
    location: [number, number, number, number];
    description: string;
    technical_details: Record<string, any>;
}

interface VerificationResult {
    verification_id: string;
    document_hash: string;
    verification_status: 'authentic' | 'suspicious' | 'fraudulent' | 'inconclusive' | 'error';
    overall_confidence: number;
    forgery_probability: number;
    signature_authenticity: number;
    tampering_evidence: TamperingEvidence[];
    metadata_analysis: Record<string, any>;
    image_forensics: Record<string, any>;
    blockchain_hash?: string;
    verification_timestamp: string;
    processing_time: number;
}

interface VerificationStats {
    total_verifications: number;
    status_distribution: {
        authentic: number;
        suspicious: number;
        fraudulent: number;
        inconclusive: number;
    };
    average_confidence: number;
    average_processing_time: number;
}

const ComputerVisionVerification: React.FC = () => {
    // State management
    const [files, setFiles] = useState<VerificationFile[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [results, setResults] = useState<VerificationResult[]>([]);
    const [selectedResult, setSelectedResult] = useState<VerificationResult | null>(null);
    const [stats, setStats] = useState<VerificationStats | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
    const [enableBlockchain, setEnableBlockchain] = useState(false);
    const [batchMode, setBatchMode] = useState(false);
    const [currentTab, setCurrentTab] = useState(0);
    const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
    
    // Refs
    const fileInputRef = useRef<HTMLInputElement>(null);
    const referenceInputRef = useRef<HTMLInputElement>(null);

    // WebSocket connection for real-time updates
    useEffect(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/cv-verification`;
        
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected for CV verification updates');
            setWsConnection(ws);
        };
        
        ws.onmessage = (event) => {
            try {
                const update = JSON.parse(event.data);
                handleVerificationUpdate(update);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        ws.onclose = () => {
            console.log('WebSocket connection closed');
            setWsConnection(null);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        return () => {
            ws.close();
        };
    }, []);

    // Load initial statistics
    useEffect(() => {
        fetchVerificationStats();
    }, []);

    // Handle real-time verification updates
    const handleVerificationUpdate = useCallback((update: any) => {
        if (update.type === 'verification_update') {
            const { verificationId, data } = update;
            
            // Update file status
            setFiles(prev => prev.map(file => {
                if (file.id === verificationId) {
                    return {
                        ...file,
                        status: data.status === 'completed' ? 'completed' : 'processing',
                        progress: data.progress || file.progress,
                        result: data.result || file.result,
                        error: data.error || file.error
                    };
                }
                return file;
            }));
            
            // Add to results if completed
            if (data.status === 'completed' && data.result) {
                setResults(prev => [data.result, ...prev]);
            }
        }
    }, []);

    // Fetch verification statistics
    const fetchVerificationStats = async () => {
        try {
            const response = await fetch('/api/cv-verification/status');
            const data = await response.json();
            
            if (data.success) {
                setStats(data.data.statistics);
            }
        } catch (error) {
            console.error('Error fetching verification stats:', error);
        }
    };

    // File drop handlers
    const onDocumentDrop = useCallback((acceptedFiles: File[]) => {
        const newFiles = acceptedFiles.map(file => ({
            id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            file,
            type: 'document' as const,
            status: 'pending' as const,
            preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
        }));
        
        setFiles(prev => [...prev, ...newFiles]);
        setError(null);
    }, []);

    const onReferenceDrop = useCallback((acceptedFiles: File[]) => {
        const newFiles = acceptedFiles.map(file => ({
            id: `ref_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            file,
            type: 'reference' as const,
            status: 'pending' as const,
            preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
        }));
        
        setFiles(prev => [...prev, ...newFiles]);
        setError(null);
    }, []);

    // Dropzone configurations
    const documentDropzone = useDropzone({
        onDrop: onDocumentDrop,
        accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
            'application/pdf': ['.pdf']
        },
        maxFiles: batchMode ? 10 : 1,
        maxSize: 50 * 1024 * 1024 // 50MB
    });

    const referenceDropzone = useDropzone({
        onDrop: onReferenceDrop,
        accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        },
        maxFiles: 5,
        maxSize: 10 * 1024 * 1024 // 10MB
    });

    // Process verification
    const processVerification = async () => {
        const documentFiles = files.filter(f => f.type === 'document' && f.status === 'pending');
        const referenceFiles = files.filter(f => f.type === 'reference' && f.status === 'pending');
        
        if (documentFiles.length === 0) {
            setError('Please upload at least one document to verify');
            return;
        }
        
        setIsProcessing(true);
        setError(null);
        
        try {
            if (batchMode && documentFiles.length > 1) {
                await processBatchVerification(documentFiles, referenceFiles);
            } else {
                await processSingleVerification(documentFiles[0], referenceFiles);
            }
            
            setSuccess('Verification completed successfully');
            fetchVerificationStats();
            
        } catch (error) {
            console.error('Verification error:', error);
            setError(error instanceof Error ? error.message : 'Verification failed');
        } finally {
            setIsProcessing(false);
        }
    };

    // Process single document verification
    const processSingleVerification = async (documentFile: VerificationFile, referenceFiles: VerificationFile[]) => {
        const formData = new FormData();
        formData.append('document', documentFile.file);
        
        referenceFiles.forEach(refFile => {
            formData.append('references', refFile.file);
        });
        
        const metadata = {
            original_filename: documentFile.file.name,
            file_size: documentFile.file.size,
            upload_timestamp: new Date().toISOString()
        };
        
        formData.append('metadata', JSON.stringify(metadata));
        formData.append('include_blockchain', enableBlockchain.toString());
        formData.append('include_details', 'true');
        
        // Subscribe to real-time updates
        if (wsConnection) {
            wsConnection.send(JSON.stringify({
                type: 'subscribe_verification',
                verificationId: documentFile.id
            }));
        }
        
        // Update file status
        setFiles(prev => prev.map(f => 
            f.id === documentFile.id ? { ...f, status: 'uploading', progress: 0 } : f
        ));
        
        const response = await fetch('/api/cv-verification/verify', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Verification failed');
        }
        
        // Update file with result
        setFiles(prev => prev.map(f => 
            f.id === documentFile.id ? { 
                ...f, 
                status: 'completed', 
                progress: 100, 
                result: result.data 
            } : f
        ));
        
        // Add to results
        setResults(prev => [result.data, ...prev]);
    };

    // Process batch verification
    const processBatchVerification = async (documentFiles: VerificationFile[], referenceFiles: VerificationFile[]) => {
        const formData = new FormData();
        
        documentFiles.forEach(file => {
            formData.append('documents', file.file);
        });
        
        const batchMetadata: Record<string, any> = {};
        documentFiles.forEach(file => {
            batchMetadata[file.file.name] = {
                original_filename: file.file.name,
                file_size: file.file.size,
                upload_timestamp: new Date().toISOString()
            };
        });
        
        formData.append('batch_metadata', JSON.stringify(batchMetadata));
        
        // Update all document files to processing status
        setFiles(prev => prev.map(f => 
            documentFiles.some(df => df.id === f.id) 
                ? { ...f, status: 'processing', progress: 0 } 
                : f
        ));
        
        const response = await fetch('/api/cv-verification/batch-verify', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Batch verification failed');
        }
        
        // Update files with results
        const { verification_results } = result.data;
        
        setFiles(prev => prev.map(f => {
            const matchResult = verification_results.find((r: any) => 
                r.filename === f.file.name
            );
            
            if (matchResult && documentFiles.some(df => df.id === f.id)) {
                return {
                    ...f,
                    status: 'completed',
                    progress: 100,
                    result: matchResult
                };
            }
            
            return f;
        }));
        
        // Add results
        setResults(prev => [...verification_results, ...prev]);
    };

    // Remove file
    const removeFile = (fileId: string) => {
        setFiles(prev => {
            const fileToRemove = prev.find(f => f.id === fileId);
            if (fileToRemove?.preview) {
                URL.revokeObjectURL(fileToRemove.preview);
            }
            return prev.filter(f => f.id !== fileId);
        });
    };

    // Clear all files
    const clearFiles = () => {
        files.forEach(file => {
            if (file.preview) {
                URL.revokeObjectURL(file.preview);
            }
        });
        setFiles([]);
        setResults([]);
        setError(null);
        setSuccess(null);
    };

    // Get status icon and color
    const getStatusDisplay = (status: string, confidence?: number) => {
        switch (status) {
            case 'authentic':
                return { 
                    icon: <VerifiedUserIcon />, 
                    color: 'success', 
                    label: 'Authentic',
                    description: 'Document verified as authentic'
                };
            case 'suspicious':
                return { 
                    icon: <WarningIcon />, 
                    color: 'warning', 
                    label: 'Suspicious',
                    description: 'Document shows signs of potential manipulation'
                };
            case 'fraudulent':
                return { 
                    icon: <ReportProblemIcon />, 
                    color: 'error', 
                    label: 'Fraudulent',
                    description: 'Document appears to be fraudulent'
                };
            case 'inconclusive':
                return { 
                    icon: <SearchOffIcon />, 
                    color: 'info', 
                    label: 'Inconclusive',
                    description: 'Unable to determine authenticity conclusively'
                };
            default:
                return { 
                    icon: <ErrorIcon />, 
                    color: 'error', 
                    label: 'Error',
                    description: 'Verification encountered an error'
                };
        }
    };

    // Format confidence score
    const formatConfidence = (confidence: number) => {
        return `${(confidence * 100).toFixed(1)}%`;
    };

    // Format processing time
    const formatProcessingTime = (time: number) => {
        return time < 1 ? `${(time * 1000).toFixed(0)}ms` : `${time.toFixed(1)}s`;
    };

    // Generate forensics visualization data
    const getForensicsChartData = (result: VerificationResult) => {
        const labels = ['Overall Confidence', 'Signature Authenticity', 'Anti-Forgery'];
        const data = [
            result.overall_confidence * 100,
            result.signature_authenticity * 100,
            (1 - result.forgery_probability) * 100
        ];

        return {
            labels,
            datasets: [{
                label: 'Verification Scores',
                data,
                backgroundColor: [
                    'rgba(76, 175, 80, 0.6)',
                    'rgba(33, 150, 243, 0.6)',
                    'rgba(255, 152, 0, 0.6)'
                ],
                borderColor: [
                    'rgba(76, 175, 80, 1)',
                    'rgba(33, 150, 243, 1)',
                    'rgba(255, 152, 0, 1)'
                ],
                borderWidth: 2
            }]
        };
    };

    // Tab panel component
    const TabPanelComponent = ({ children, value, index }: any) => (
        <div hidden={value !== index} style={{ paddingTop: 24 }}>
            {value === index && children}
        </div>
    );

    return (
        <Box sx={{ p: 3 }}>
            {/* Header */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" gutterBottom sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 2,
                    fontWeight: 600,
                    color: 'primary.main'
                }}>
                    <SecurityIcon fontSize="large" />
                    Computer Vision Document Verification
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                    Advanced document authenticity verification using AI-powered computer vision
                </Typography>
            </Box>

            {/* Statistics Cards */}
            {stats && (
                <Grid container spacing={3} sx={{ mb: 4 }}>
                    <Grid item xs={12} sm={6} md={3}>
                        <Card sx={{ textAlign: 'center' }}>
                            <CardContent>
                                <Typography variant="h4" color="primary.main">
                                    {stats.total_verifications}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Total Verifications
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <Card sx={{ textAlign: 'center' }}>
                            <CardContent>
                                <Typography variant="h4" color="success.main">
                                    {formatConfidence(stats.average_confidence)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Average Confidence
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <Card sx={{ textAlign: 'center' }}>
                            <CardContent>
                                <Typography variant="h4" color="info.main">
                                    {formatProcessingTime(stats.average_processing_time)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Avg Processing Time
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <Card sx={{ textAlign: 'center' }}>
                            <CardContent>
                                <Typography variant="h4" color="success.main">
                                    {stats.status_distribution.authentic}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Authentic Documents
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            )}

            {/* Control Options */}
            <Card sx={{ mb: 4 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Verification Options
                    </Typography>
                    <Grid container spacing={3} alignItems="center">
                        <Grid item>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={batchMode}
                                        onChange={(e) => setBatchMode(e.target.checked)}
                                        disabled={isProcessing}
                                    />
                                }
                                label="Batch Mode (Multiple Documents)"
                            />
                        </Grid>
                        <Grid item>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={enableBlockchain}
                                        onChange={(e) => setEnableBlockchain(e.target.checked)}
                                        disabled={isProcessing}
                                    />
                                }
                                label="Blockchain Audit Trail"
                            />
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            {/* Upload Areas */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                {/* Document Upload */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <UploadIcon />
                                Documents to Verify
                            </Typography>
                            
                            <Box
                                {...documentDropzone.getRootProps()}
                                sx={{
                                    border: '2px dashed',
                                    borderColor: documentDropzone.isDragActive ? 'primary.main' : 'grey.300',
                                    borderRadius: 2,
                                    p: 4,
                                    textAlign: 'center',
                                    cursor: 'pointer',
                                    backgroundColor: documentDropzone.isDragActive ? 'primary.50' : 'transparent',
                                    '&:hover': { borderColor: 'primary.main', backgroundColor: 'primary.50' }
                                }}
                            >
                                <input {...documentDropzone.getInputProps()} />
                                <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                                <Typography variant="body1" gutterBottom>
                                    {batchMode 
                                        ? 'Drop documents here or click to select (up to 10 files)'
                                        : 'Drop document here or click to select'
                                    }
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Supported formats: PDF, JPG, PNG, TIFF (max 50MB each)
                                </Typography>
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Reference Signatures Upload */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <FingerprintIcon />
                                Reference Signatures (Optional)
                            </Typography>
                            
                            <Box
                                {...referenceDropzone.getRootProps()}
                                sx={{
                                    border: '2px dashed',
                                    borderColor: referenceDropzone.isDragActive ? 'secondary.main' : 'grey.300',
                                    borderRadius: 2,
                                    p: 4,
                                    textAlign: 'center',
                                    cursor: 'pointer',
                                    backgroundColor: referenceDropzone.isDragActive ? 'secondary.50' : 'transparent',
                                    '&:hover': { borderColor: 'secondary.main', backgroundColor: 'secondary.50' }
                                }}
                            >
                                <input {...referenceDropzone.getInputProps()} />
                                <FingerprintIcon sx={{ fontSize: 48, color: 'secondary.main', mb: 2 }} />
                                <Typography variant="body1" gutterBottom>
                                    Drop reference signatures here (up to 5 files)
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    For signature comparison and verification
                                </Typography>
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* File List */}
            {files.length > 0 && (
                <Card sx={{ mb: 4 }}>
                    <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6">
                                Uploaded Files ({files.length})
                            </Typography>
                            <Button
                                variant="outlined"
                                startIcon={<DeleteIcon />}
                                onClick={clearFiles}
                                disabled={isProcessing}
                                size="small"
                            >
                                Clear All
                            </Button>
                        </Box>
                        
                        <List>
                            {files.map((file) => (
                                <ListItem
                                    key={file.id}
                                    sx={{
                                        border: '1px solid',
                                        borderColor: 'grey.200',
                                        borderRadius: 1,
                                        mb: 1,
                                        backgroundColor: 'background.paper'
                                    }}
                                >
                                    <ListItemIcon>
                                        {file.type === 'document' ? <SecurityIcon /> : <FingerprintIcon />}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                {file.file.name}
                                                <Chip
                                                    label={file.type}
                                                    size="small"
                                                    color={file.type === 'document' ? 'primary' : 'secondary'}
                                                />
                                                {file.status !== 'pending' && (
                                                    <Chip
                                                        label={file.status}
                                                        size="small"
                                                        color={
                                                            file.status === 'completed' ? 'success' :
                                                            file.status === 'error' ? 'error' : 'info'
                                                        }
                                                    />
                                                )}
                                            </Box>
                                        }
                                        secondary={
                                            <Box>
                                                <Typography variant="body2" color="text.secondary">
                                                    {(file.file.size / 1024 / 1024).toFixed(2)} MB
                                                </Typography>
                                                {(file.status === 'processing' || file.status === 'uploading') && (
                                                    <LinearProgress 
                                                        variant="determinate" 
                                                        value={file.progress || 0} 
                                                        sx={{ mt: 1 }}
                                                    />
                                                )}
                                                {file.result && (
                                                    <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        {getStatusDisplay(file.result.verification_status).icon}
                                                        <Typography variant="body2">
                                                            {getStatusDisplay(file.result.verification_status).label} - 
                                                            {formatConfidence(file.result.overall_confidence)} confidence
                                                        </Typography>
                                                    </Box>
                                                )}
                                                {file.error && (
                                                    <Typography variant="body2" color="error">
                                                        Error: {file.error}
                                                    </Typography>
                                                )}
                                            </Box>
                                        }
                                    />
                                    <IconButton
                                        onClick={() => removeFile(file.id)}
                                        disabled={isProcessing}
                                        size="small"
                                    >
                                        <DeleteIcon />
                                    </IconButton>
                                </ListItem>
                            ))}
                        </List>
                    </CardContent>
                </Card>
            )}

            {/* Action Buttons */}
            <Box sx={{ mb: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button
                    variant="contained"
                    size="large"
                    startIcon={isProcessing ? <CircularProgress size={20} /> : <ShieldIcon />}
                    onClick={processVerification}
                    disabled={isProcessing || files.filter(f => f.type === 'document').length === 0}
                    sx={{ minWidth: 200 }}
                >
                    {isProcessing ? 'Verifying...' : 'Start Verification'}
                </Button>
                <Button
                    variant="outlined"
                    size="large"
                    startIcon={<RefreshIcon />}
                    onClick={clearFiles}
                    disabled={isProcessing}
                >
                    Reset
                </Button>
            </Box>

            {/* Alerts */}
            {error && (
                <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}
            {success && (
                <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess(null)}>
                    {success}
                </Alert>
            )}

            {/* Results Section */}
            {results.length > 0 && (
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <TimelineIcon />
                            Verification Results ({results.length})
                        </Typography>

                        <Tabs value={currentTab} onChange={(e, newValue) => setCurrentTab(newValue)} sx={{ mb: 3 }}>
                            <Tab label="Results List" />
                            <Tab label="Analytics" />
                            <Tab label="Detailed Reports" />
                        </Tabs>

                        <TabPanelComponent value={currentTab} index={0}>
                            <TableContainer>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Document</TableCell>
                                            <TableCell>Status</TableCell>
                                            <TableCell>Confidence</TableCell>
                                            <TableCell>Signature Auth</TableCell>
                                            <TableCell>Forgery Risk</TableCell>
                                            <TableCell>Processing Time</TableCell>
                                            <TableCell>Actions</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {results.map((result) => {
                                            const statusDisplay = getStatusDisplay(result.verification_status, result.overall_confidence);
                                            return (
                                                <TableRow key={result.verification_id}>
                                                    <TableCell>
                                                        <Typography variant="body2" noWrap>
                                                            {result.document_hash.substring(0, 16)}...
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>
                                                        <Chip
                                                            icon={statusDisplay.icon}
                                                            label={statusDisplay.label}
                                                            color={statusDisplay.color as any}
                                                            size="small"
                                                        />
                                                    </TableCell>
                                                    <TableCell>
                                                        <Typography
                                                            variant="body2"
                                                            color={
                                                                result.overall_confidence >= 0.8 ? 'success.main' :
                                                                result.overall_confidence >= 0.6 ? 'warning.main' : 'error.main'
                                                            }
                                                        >
                                                            {formatConfidence(result.overall_confidence)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>
                                                        <Typography variant="body2">
                                                            {formatConfidence(result.signature_authenticity)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>
                                                        <Typography
                                                            variant="body2"
                                                            color={
                                                                result.forgery_probability <= 0.2 ? 'success.main' :
                                                                result.forgery_probability <= 0.5 ? 'warning.main' : 'error.main'
                                                            }
                                                        >
                                                            {formatConfidence(result.forgery_probability)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>
                                                        <Typography variant="body2">
                                                            {formatProcessingTime(result.processing_time)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>
                                                        <Tooltip title="View Details">
                                                            <IconButton
                                                                size="small"
                                                                onClick={() => {
                                                                    setSelectedResult(result);
                                                                    setDetailsDialogOpen(true);
                                                                }}
                                                            >
                                                                <VisibilityIcon />
                                                            </IconButton>
                                                        </Tooltip>
                                                    </TableCell>
                                                </TableRow>
                                            );
                                        })}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </TabPanelComponent>

                        <TabPanelComponent value={currentTab} index={1}>
                            {results.length > 0 && (
                                <Grid container spacing={3}>
                                    <Grid item xs={12} md={6}>
                                        <Card>
                                            <CardContent>
                                                <Typography variant="h6" gutterBottom>
                                                    Verification Status Distribution
                                                </Typography>
                                                <Pie
                                                    data={{
                                                        labels: ['Authentic', 'Suspicious', 'Fraudulent', 'Inconclusive'],
                                                        datasets: [{
                                                            data: [
                                                                results.filter(r => r.verification_status === 'authentic').length,
                                                                results.filter(r => r.verification_status === 'suspicious').length,
                                                                results.filter(r => r.verification_status === 'fraudulent').length,
                                                                results.filter(r => r.verification_status === 'inconclusive').length
                                                            ],
                                                            backgroundColor: [
                                                                'rgba(76, 175, 80, 0.8)',
                                                                'rgba(255, 152, 0, 0.8)',
                                                                'rgba(244, 67, 54, 0.8)',
                                                                'rgba(33, 150, 243, 0.8)'
                                                            ]
                                                        }]
                                                    }}
                                                    options={{
                                                        responsive: true,
                                                        plugins: {
                                                            legend: { position: 'bottom' }
                                                        }
                                                    }}
                                                />
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <Card>
                                            <CardContent>
                                                <Typography variant="h6" gutterBottom>
                                                    Confidence Score Distribution
                                                </Typography>
                                                <Bar
                                                    data={{
                                                        labels: results.map((_, i) => `Doc ${i + 1}`),
                                                        datasets: [
                                                            {
                                                                label: 'Overall Confidence',
                                                                data: results.map(r => r.overall_confidence * 100),
                                                                backgroundColor: 'rgba(76, 175, 80, 0.6)'
                                                            },
                                                            {
                                                                label: 'Signature Authenticity',
                                                                data: results.map(r => r.signature_authenticity * 100),
                                                                backgroundColor: 'rgba(33, 150, 243, 0.6)'
                                                            }
                                                        ]
                                                    }}
                                                    options={{
                                                        responsive: true,
                                                        scales: {
                                                            y: { beginAtZero: true, max: 100 }
                                                        }
                                                    }}
                                                />
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>
                            )}
                        </TabPanelComponent>

                        <TabPanelComponent value={currentTab} index={2}>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                                Select a verification result to view detailed forensic analysis reports.
                            </Typography>
                            {selectedResult && (
                                <Card>
                                    <CardContent>
                                        <Typography variant="h6" gutterBottom>
                                            Detailed Analysis: {selectedResult.document_hash.substring(0, 16)}...
                                        </Typography>
                                        
                                        <Grid container spacing={3}>
                                            <Grid item xs={12} md={6}>
                                                <Radar
                                                    data={getForensicsChartData(selectedResult)}
                                                    options={{
                                                        responsive: true,
                                                        scales: {
                                                            r: {
                                                                beginAtZero: true,
                                                                max: 100
                                                            }
                                                        }
                                                    }}
                                                />
                                            </Grid>
                                            <Grid item xs={12} md={6}>
                                                <List>
                                                    <ListItem>
                                                        <ListItemText
                                                            primary="Overall Confidence"
                                                            secondary={formatConfidence(selectedResult.overall_confidence)}
                                                        />
                                                    </ListItem>
                                                    <ListItem>
                                                        <ListItemText
                                                            primary="Signature Authenticity"
                                                            secondary={formatConfidence(selectedResult.signature_authenticity)}
                                                        />
                                                    </ListItem>
                                                    <ListItem>
                                                        <ListItemText
                                                            primary="Forgery Probability"
                                                            secondary={formatConfidence(selectedResult.forgery_probability)}
                                                        />
                                                    </ListItem>
                                                    <ListItem>
                                                        <ListItemText
                                                            primary="Processing Time"
                                                            secondary={formatProcessingTime(selectedResult.processing_time)}
                                                        />
                                                    </ListItem>
                                                </List>
                                            </Grid>
                                        </Grid>
                                    </CardContent>
                                </Card>
                            )}
                        </TabPanelComponent>
                    </CardContent>
                </Card>
            )}

            {/* Detailed Results Dialog */}
            <Dialog
                open={detailsDialogOpen}
                onClose={() => setDetailsDialogOpen(false)}
                maxWidth="lg"
                fullWidth
            >
                <DialogTitle>
                    Detailed Verification Report
                    {selectedResult && (
                        <Chip
                            icon={getStatusDisplay(selectedResult.verification_status).icon}
                            label={getStatusDisplay(selectedResult.verification_status).label}
                            color={getStatusDisplay(selectedResult.verification_status).color as any}
                            sx={{ ml: 2 }}
                        />
                    )}
                </DialogTitle>
                <DialogContent>
                    {selectedResult && (
                        <Box>
                            {/* Summary */}
                            <Card sx={{ mb: 3 }}>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>Summary</Typography>
                                    <Grid container spacing={3}>
                                        <Grid item xs={6} sm={3}>
                                            <Typography variant="subtitle2">Overall Confidence</Typography>
                                            <Typography variant="h6" color={
                                                selectedResult.overall_confidence >= 0.8 ? 'success.main' :
                                                selectedResult.overall_confidence >= 0.6 ? 'warning.main' : 'error.main'
                                            }>
                                                {formatConfidence(selectedResult.overall_confidence)}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6} sm={3}>
                                            <Typography variant="subtitle2">Signature Auth</Typography>
                                            <Typography variant="h6">
                                                {formatConfidence(selectedResult.signature_authenticity)}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6} sm={3}>
                                            <Typography variant="subtitle2">Forgery Risk</Typography>
                                            <Typography variant="h6" color={
                                                selectedResult.forgery_probability <= 0.2 ? 'success.main' :
                                                selectedResult.forgery_probability <= 0.5 ? 'warning.main' : 'error.main'
                                            }>
                                                {formatConfidence(selectedResult.forgery_probability)}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6} sm={3}>
                                            <Typography variant="subtitle2">Processing Time</Typography>
                                            <Typography variant="h6">
                                                {formatProcessingTime(selectedResult.processing_time)}
                                            </Typography>
                                        </Grid>
                                    </Grid>
                                </CardContent>
                            </Card>

                            {/* Tampering Evidence */}
                            {selectedResult.tampering_evidence && selectedResult.tampering_evidence.length > 0 && (
                                <Accordion>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <BugReportIcon />
                                            Tampering Evidence ({selectedResult.tampering_evidence.length})
                                        </Typography>
                                    </AccordionSummary>
                                    <AccordionDetails>
                                        <List>
                                            {selectedResult.tampering_evidence.map((evidence, index) => (
                                                <ListItem key={index} divider>
                                                    <ListItemText
                                                        primary={
                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                <Chip
                                                                    label={evidence.tampering_type}
                                                                    color="warning"
                                                                    size="small"
                                                                />
                                                                <Typography variant="body2">
                                                                    Confidence: {formatConfidence(evidence.confidence)}
                                                                </Typography>
                                                            </Box>
                                                        }
                                                        secondary={
                                                            <Box>
                                                                <Typography variant="body2" sx={{ mt: 1 }}>
                                                                    {evidence.description}
                                                                </Typography>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Location: [{evidence.location.join(', ')}]
                                                                </Typography>
                                                            </Box>
                                                        }
                                                    />
                                                </ListItem>
                                            ))}
                                        </List>
                                    </AccordionDetails>
                                </Accordion>
                            )}

                            {/* Technical Details */}
                            <Accordion>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <InfoIcon />
                                        Technical Analysis
                                    </Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={6}>
                                            <Typography variant="subtitle2" gutterBottom>
                                                Document Hash
                                            </Typography>
                                            <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                                                {selectedResult.document_hash}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={12} md={6}>
                                            <Typography variant="subtitle2" gutterBottom>
                                                Verification Timestamp
                                            </Typography>
                                            <Typography variant="body2">
                                                {new Date(selectedResult.verification_timestamp).toLocaleString()}
                                            </Typography>
                                        </Grid>
                                        {selectedResult.blockchain_hash && (
                                            <Grid item xs={12}>
                                                <Typography variant="subtitle2" gutterBottom>
                                                    Blockchain Hash
                                                </Typography>
                                                <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                                                    {selectedResult.blockchain_hash}
                                                </Typography>
                                            </Grid>
                                        )}
                                    </Grid>
                                </AccordionDetails>
                            </Accordion>
                        </Box>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDetailsDialogOpen(false)}>
                        Close
                    </Button>
                    <Button
                        variant="contained"
                        startIcon={<DownloadIcon />}
                        onClick={() => {
                            if (selectedResult) {
                                const blob = new Blob([JSON.stringify(selectedResult, null, 2)], { type: 'application/json' });
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `verification_report_${selectedResult.verification_id}.json`;
                                a.click();
                                URL.revokeObjectURL(url);
                            }
                        }}
                    >
                        Download Report
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default ComputerVisionVerification;
