/**
 * Quality Control Interface - Full Mantine Implementation
 *
 * Interface for interacting with the Mortgage Application Quality Control Agent with:
 * - Real-time document analysis and validation
 * - Field-level validation results
 * - Anomaly detection and consistency checks
 * - Completeness scoring and remediation suggestions
 * - Integration with backend QC agent
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Text,
  Box,
  Button,
  Alert,
  Progress,
  Grid,
  Card,
  Badge,
  Accordion,
  List,
  Divider,
  Loader,
  Title,
  Group,
  Stack,
  ActionIcon,
  Tooltip,
  ThemeIcon,
  SimpleGrid,
  RingProgress,
  Center,
  Tabs,
} from '@mantine/core';
import {
  IconShield,
  IconCheck,
  IconX,
  IconAlertTriangle,
  IconInfoCircle,
  IconChevronDown,
  IconFileText,
  IconUser,
  IconHome,
  IconBuildingBank,
  IconChartBar,
  IconGavel,
  IconArrowRight,
  IconArrowLeft,
  IconRefresh,
  IconDownload,
  IconEye,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { qualityControlApi, QCResult } from '../services/qualityControlApi';
import { useDemoMode } from '../contexts/DemoModeContext';

const QualityControl: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { isDemoMode } = useDemoMode();
  const [analyzing, setAnalyzing] = useState(true);
  const [qcResult, setQcResult] = useState<QCResult | null>(null);
  const [activeTab, setActiveTab] = useState<string>('overview');

  useEffect(() => {
    const analyzeApplication = async () => {
      setAnalyzing(true);

      try {
        // Get application ID from URL or context
        const applicationId = searchParams.get('application_id') || 'current_application';

        const result = await qualityControlApi.runQualityControl(applicationId);
        setQcResult(result);
      } catch (error) {
        console.error('Failed to run quality control:', error);
        notifications.show({
          title: 'Quality Control Error',
          message: 'Failed to run quality control analysis',
          color: 'red',
          icon: <IconX size={16} />,
        });
      } finally {
        setAnalyzing(false);
      }
    };

    analyzeApplication();
  }, [searchParams]);

  const handleRerunAnalysis = async () => {
    setAnalyzing(true);
    try {
      const applicationId = searchParams.get('application_id') || 'current_application';
      const result = await qualityControlApi.runQualityControl(applicationId);
      setQcResult(result);
      notifications.show({
        title: 'Analysis Complete',
        message: 'Quality control analysis has been updated',
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Analysis Failed',
        message: 'Failed to rerun quality control analysis',
        color: 'red',
        icon: <IconX size={16} />,
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'yellow';
      case 'low': return 'blue';
      default: return 'gray';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <IconX size={16} />;
      case 'high': return <IconAlertTriangle size={16} />;
      case 'medium': return <IconInfoCircle size={16} />;
      case 'low': return <IconCheck size={16} />;
      default: return <IconInfoCircle size={16} />;
    }
  };

  if (analyzing) {
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: '60vh' }}>
          <Stack align="center" gap="md">
            <Loader size="xl" />
            <Title order={3}>Analyzing Application</Title>
            <Text c="dimmed">Running quality control checks...</Text>
          </Stack>
        </Center>
      </Container>
    );
  }

  if (!qcResult) {
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: '60vh' }}>
          <Stack align="center" gap="md">
            <ThemeIcon size="xl" color="red" radius={0}>
              <IconX size={32} />
            </ThemeIcon>
            <Title order={3} c="red">Failed to Load Results</Title>
            <Text c="dimmed">Unable to load quality control results</Text>
            <Button 
              leftSection={<IconRefresh size={16} />} 
              onClick={handleRerunAnalysis}
              radius={0}
            >
              Retry Analysis
            </Button>
          </Stack>
        </Center>
      </Container>
    );
  }

  const overallScore = qcResult.completeness_score;
  const riskLevel = qcResult.risk_assessment?.overall_risk || 'medium';

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between">
          <Group>
            <ThemeIcon size="xl" radius={0} color="indigo">
              <IconShield size={32} />
            </ThemeIcon>
            <div>
              <Title order={1}>Quality Control Analysis</Title>
              <Text c="dimmed">Application ID: {qcResult.application_id}</Text>
            </div>
          </Group>
          <Group>
            <Button 
              variant="outline" 
              leftSection={<IconRefresh size={16} />}
              onClick={handleRerunAnalysis}
              loading={analyzing}
              radius={0}
            >
              Rerun Analysis
            </Button>
            <Button 
              leftSection={<IconDownload size={16} />}
              radius={0}
            >
              Export Report
            </Button>
          </Group>
        </Group>

        {/* Overview Cards */}
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="lg">
          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Overall Score</Text>
                <Title order={2}>{overallScore.toFixed(1)}%</Title>
              </div>
              <RingProgress
                size={60}
                thickness={6}
                sections={[{ value: overallScore, color: overallScore >= 80 ? 'green' : overallScore >= 60 ? 'yellow' : 'red' }]}
              />
            </Group>
            <Progress 
              value={overallScore} 
              color={overallScore >= 80 ? 'green' : overallScore >= 60 ? 'yellow' : 'red'}
              size="sm" 
              mt="md"
              radius={0}
            />
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Status</Text>
                <Badge 
                  color={qcResult.passed ? 'green' : 'red'} 
                  size="lg"
                  radius={0}
                >
                  {qcResult.passed ? 'PASSED' : 'FAILED'}
                </Badge>
              </div>
              <ThemeIcon 
                size="xl" 
                color={qcResult.passed ? 'green' : 'red'} 
                radius={0}
              >
                {qcResult.passed ? <IconCheck size={24} /> : <IconX size={24} />}
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Risk Level</Text>
                <Badge 
                  color={getSeverityColor(riskLevel)} 
                  size="lg"
                  radius={0}
                >
                  {riskLevel.toUpperCase()}
                </Badge>
              </div>
              <ThemeIcon 
                size="xl" 
                color={getSeverityColor(riskLevel)} 
                radius={0}
              >
                {getSeverityIcon(riskLevel)}
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Issues Found</Text>
                <Title order={2}>
                  {qcResult.processing_summary?.critical_issues || 0}
                </Title>
              </div>
              <ThemeIcon size="xl" color="orange" radius={0}>
                <IconAlertTriangle size={24} />
              </ThemeIcon>
            </Group>
          </Card>
        </SimpleGrid>

        {/* Detailed Analysis Tabs */}
        <Tabs value={activeTab} onChange={(value) => setActiveTab(value || 'overview')} radius={0}>
          <Tabs.List>
            <Tabs.Tab value="overview" leftSection={<IconEye size={16} />}>
              Overview
            </Tabs.Tab>
            <Tabs.Tab value="fields" leftSection={<IconFileText size={16} />}>
              Field Validation
            </Tabs.Tab>
            <Tabs.Tab value="anomalies" leftSection={<IconAlertTriangle size={16} />}>
              Anomalies
            </Tabs.Tab>
            <Tabs.Tab value="documents" leftSection={<IconFileText size={16} />}>
              Documents
            </Tabs.Tab>
            <Tabs.Tab value="compliance" leftSection={<IconGavel size={16} />}>
              Compliance
            </Tabs.Tab>
            <Tabs.Tab value="recommendations" leftSection={<IconChartBar size={16} />}>
              Recommendations
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="overview" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 8 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Processing Summary</Title>
                  <SimpleGrid cols={2} spacing="md">
                    <div>
                      <Text size="sm" c="dimmed">Total Fields</Text>
                      <Text size="xl" fw={700}>{qcResult.processing_summary?.total_fields || 0}</Text>
                    </div>
                    <div>
                      <Text size="sm" c="dimmed">Valid Fields</Text>
                      <Text size="xl" fw={700} c="green">{qcResult.processing_summary?.valid_fields || 0}</Text>
                    </div>
                    <div>
                      <Text size="sm" c="dimmed">Invalid Fields</Text>
                      <Text size="xl" fw={700} c="red">{qcResult.processing_summary?.invalid_fields || 0}</Text>
                    </div>
                    <div>
                      <Text size="sm" c="dimmed">Documents Processed</Text>
                      <Text size="xl" fw={700}>{qcResult.processing_summary?.documents_processed || 0}</Text>
                    </div>
                  </SimpleGrid>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Review Details</Title>
                  <Stack gap="sm">
                    <Group justify="space-between">
                      <Text size="sm">QC Officer</Text>
                      <Text size="sm" fw={500}>{qcResult.qc_officer}</Text>
                    </Group>
                    <Group justify="space-between">
                      <Text size="sm">Review Duration</Text>
                      <Text size="sm" fw={500}>{qcResult.review_duration} min</Text>
                    </Group>
                    <Group justify="space-between">
                      <Text size="sm">Reviewed At</Text>
                      <Text size="sm" fw={500}>
                        {new Date(qcResult.reviewed_at).toLocaleString()}
                      </Text>
                    </Group>
                  </Stack>
                </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>

          <Tabs.Panel value="fields" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Field Validation Results</Title>
              <Stack gap="md">
                {qcResult.field_validation?.results?.map((field, index) => (
                  <Paper key={index} p="md" withBorder radius={0}>
                    <Group justify="space-between">
                      <Group>
                        <ThemeIcon 
                          color={field.valid ? 'green' : getSeverityColor(field.severity)} 
                          radius={0}
                        >
                          {field.valid ? <IconCheck size={16} /> : getSeverityIcon(field.severity)}
                        </ThemeIcon>
                        <div>
                          <Text fw={500}>{field.field.replace(/_/g, ' ').toUpperCase()}</Text>
                          {field.error && <Text size="sm" c="red">{field.error}</Text>}
                          {field.suggestion && <Text size="sm" c="blue">{field.suggestion}</Text>}
                        </div>
                      </Group>
                      <Badge 
                        color={field.valid ? 'green' : getSeverityColor(field.severity)}
                        radius={0}
                      >
                        {field.valid ? 'Valid' : field.severity.toUpperCase()}
                      </Badge>
                    </Group>
                  </Paper>
                ))}
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="anomalies" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Anomaly Detection</Title>
              {qcResult.anomaly_check?.anomalies?.length > 0 ? (
                <Stack gap="md">
                  {qcResult.anomaly_check.anomalies.map((anomaly, index) => (
                    <Alert 
                      key={index}
                      color={getSeverityColor(anomaly.severity)}
                      icon={getSeverityIcon(anomaly.severity)}
                      title={`${anomaly.type.replace(/_/g, ' ').toUpperCase()} - ${anomaly.field}`}
                      radius={0}
                    >
                      <Text size="sm" mb="xs">{anomaly.description}</Text>
                      <Text size="xs" c="dimmed">Impact: {anomaly.impact}</Text>
                    </Alert>
                  ))}
                </Stack>
              ) : (
                <Alert color="green" icon={<IconCheck size={16} />} radius={0}>
                  No anomalies detected in the application data.
                </Alert>
              )}
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="documents" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Document Analysis</Title>
              <Stack gap="md">
                {qcResult.document_analysis?.map((doc, index) => (
                  <Paper key={index} p="md" withBorder radius={0}>
                    <Group justify="space-between" mb="sm">
                      <Group>
                        <ThemeIcon color={doc.valid ? 'green' : 'red'} radius={0}>
                          <IconFileText size={16} />
                        </ThemeIcon>
                        <div>
                          <Text fw={500}>{doc.document_type.replace(/_/g, ' ').toUpperCase()}</Text>
                          <Text size="sm" c="dimmed">Status: {doc.processing_status}</Text>
                        </div>
                      </Group>
                      <Group>
                        <Badge color={doc.present ? 'green' : 'red'} radius={0}>
                          {doc.present ? 'Present' : 'Missing'}
                        </Badge>
                        <Badge color={doc.valid ? 'green' : 'red'} radius={0}>
                          {doc.valid ? 'Valid' : 'Invalid'}
                        </Badge>
                      </Group>
                    </Group>
                    <Group justify="space-between">
                      <Text size="sm">Completeness: {doc.completeness}%</Text>
                      <Text size="sm">Confidence: {doc.confidence_score}%</Text>
                    </Group>
                    <Progress 
                      value={doc.completeness} 
                      color={doc.completeness >= 80 ? 'green' : 'yellow'}
                      size="sm" 
                      mt="xs"
                      radius={0}
                    />
                    {doc.issues?.length > 0 && (
                      <List size="sm" mt="sm">
                        {doc.issues.map((issue, issueIndex) => (
                          <List.Item key={issueIndex} icon={<IconAlertTriangle size={12} />}>
                            {issue}
                          </List.Item>
                        ))}
                      </List>
                    )}
                  </Paper>
                ))}
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="compliance" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">AFM Compliance Checks</Title>
              <Stack gap="md">
                {qcResult.compliance_checks?.map((check, index) => (
                  <Paper key={index} p="md" withBorder radius={0}>
                    <Group justify="space-between">
                      <Group>
                        <ThemeIcon color={check.passed ? 'green' : 'red'} radius={0}>
                          <IconGavel size={16} />
                        </ThemeIcon>
                        <div>
                          <Text fw={500}>{check.check_name}</Text>
                          <Text size="sm" c="dimmed">{check.details}</Text>
                          {check.afm_requirement && (
                            <Text size="xs" c="blue">AFM Requirement: {check.afm_requirement}</Text>
                          )}
                        </div>
                      </Group>
                      <Badge color={check.passed ? 'green' : 'red'} radius={0}>
                        {check.passed ? 'PASSED' : 'FAILED'}
                      </Badge>
                    </Group>
                  </Paper>
                ))}
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="recommendations" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Recommendations</Title>
                  <Stack gap="md">
                    {qcResult.recommendations?.map((rec, index) => (
                      <Alert 
                        key={index}
                        color={rec.priority === 'high' ? 'red' : rec.priority === 'medium' ? 'yellow' : 'blue'}
                        icon={getSeverityIcon(rec.priority)}
                        title={rec.type.replace(/_/g, ' ').toUpperCase()}
                        radius={0}
                      >
                        <Text size="sm">{rec.message}</Text>
                        {rec.deadline && (
                          <Text size="xs" c="dimmed" mt="xs">Deadline: {rec.deadline}</Text>
                        )}
                      </Alert>
                    ))}
                  </Stack>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Remediation Instructions</Title>
                  <Stack gap="md">
                    {qcResult.remediation_instructions?.map((instruction, index) => (
                      <Paper key={index} p="md" withBorder radius={0}>
                        <Group justify="space-between" mb="sm">
                          <Badge color={getSeverityColor(instruction.severity)} radius={0}>
                            {instruction.priority.toUpperCase()}
                          </Badge>
                          <Text size="xs" c="dimmed">Est. {instruction.estimated_time}</Text>
                        </Group>
                        <Text fw={500} size="sm" mb="xs">{instruction.issue}</Text>
                        <Text size="sm" c="dimmed" mb="xs">{instruction.instruction}</Text>
                        <Text size="sm" c="blue">{instruction.solution}</Text>
                      </Paper>
                    ))}
                  </Stack>
                </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>
        </Tabs>

        {/* Action Buttons */}
        <Group justify="space-between">
          <Button 
            variant="outline" 
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => navigate(-1)}
            radius={0}
          >
            Back to Application
          </Button>
          <Group>
            {!qcResult.passed && (
              <Button 
                color="orange"
                leftSection={<IconAlertTriangle size={16} />}
                radius={0}
              >
                Request Review
              </Button>
            )}
            <Button 
              leftSection={<IconArrowRight size={16} />}
              disabled={!qcResult.passed}
              onClick={() => navigate('/results')}
              radius={0}
            >
              Continue to Results
            </Button>
          </Group>
        </Group>
      </Stack>
    </Container>
  );
};

export default QualityControl;