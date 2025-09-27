/**
 * Final Results Display - Full Mantine Implementation
 *
 * Comprehensive results display showing analysis from both:
 * - Quality Control Agent (document validation and completeness)
 * - Compliance & Plain-Language Advisor Agent (mortgage advice)
 * - Final application status and next steps
 * - Professional summary with all findings
 */

import React, { useState, useEffect } from 'react';
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
  Divider,
  List,
  Accordion,
  Stepper,
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
  Timeline,
  Progress,
  Loader,
} from '@mantine/core';
import {
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
  IconDownload,
  IconShare,
  IconTrophy,
  IconClock,
  IconShield,
  IconTrendingUp,
  IconEye,
  IconClipboardCheck,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { resultsApi, FinalResults } from '../services/resultsApi';

const ResultsDisplay: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [finalResults, setFinalResults] = useState<FinalResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<string>('overview');

  useEffect(() => {
    loadFinalResults();
  }, []);

  const loadFinalResults = async () => {
    try {
      setLoading(true);
      const applicationId = searchParams.get('application_id') || 'current_application';
      const results = await resultsApi.getFinalResults(applicationId);
      setFinalResults(results);
    } catch (error) {
      console.error('Failed to load final results:', error);
      notifications.show({
        title: 'Loading Error',
        message: 'Failed to load final results',
        color: 'red',
        icon: <IconX size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleExportReport = () => {
    notifications.show({
      title: 'Export Started',
      message: 'Generating comprehensive report...',
      color: 'blue',
      icon: <IconDownload size={16} />,
    });
  };

  const handlePrintReport = () => {
    window.print();
  };

  const handleShareReport = () => {
    notifications.show({
      title: 'Share Report',
      message: 'Report sharing link generated',
      color: 'green',
      icon: <IconShare size={16} />,
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'green';
      case 'conditional': return 'yellow';
      case 'rejected': return 'red';
      case 'pending_review': return 'blue';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved': return <IconCheck size={24} />;
      case 'conditional': return <IconAlertTriangle size={24} />;
      case 'rejected': return <IconX size={24} />;
      case 'pending_review': return <IconClock size={24} />;
      default: return <IconInfoCircle size={24} />;
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'red';
      default: return 'gray';
    }
  };

  const getStepperStatus = (status: string) => {
    switch (status) {
      case 'completed': return 'completed';
      case 'in_progress': return 'loading';
      case 'pending': return undefined;
      default: return undefined;
    }
  };

  if (loading || !finalResults) {
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: '60vh' }}>
          <Stack align="center" gap="md">
            <Loader size="xl" />
            <Title order={3}>Loading Final Results</Title>
            <Text c="dimmed">Compiling comprehensive analysis...</Text>
          </Stack>
        </Center>
      </Container>
    );
  }

  const overallScore = finalResults.overall_score;
  const status = finalResults.overall_status;

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header with Celebration */}
        <Paper p="xl" radius={0} withBorder>
          <Group justify="space-between">
            <Group>
              <ThemeIcon size="xl" color={getStatusColor(status)} radius={0}>
                {status === 'approved' ? <IconTrophy size={32} /> : getStatusIcon(status)}
              </ThemeIcon>
              <div>
                <Title order={1}>Final Application Results</Title>
                <Text c="dimmed">Application ID: {finalResults.application_id}</Text>
                <Badge 
                  size="lg" 
                  color={getStatusColor(status)} 
                  mt="xs"
                  radius={0}
                >
                  {status.replace('_', ' ').toUpperCase()}
                </Badge>
              </div>
            </Group>
            <Group>
              <Tooltip label="Export Report">
                <ActionIcon 
                  size="lg" 
                  variant="outline" 
                  onClick={handleExportReport}
                  radius={0}
                >
                  <IconDownload size={20} />
                </ActionIcon>
              </Tooltip>
              <Tooltip label="Print Report">
                <ActionIcon 
                  size="lg" 
                  variant="outline" 
                  onClick={handlePrintReport}
                  radius={0}
                >
                  <IconFileText size={20} />
                </ActionIcon>
              </Tooltip>
              <Tooltip label="Share Report">
                <ActionIcon 
                  size="lg" 
                  variant="outline" 
                  onClick={handleShareReport}
                  radius={0}
                >
                  <IconShare size={20} />
                </ActionIcon>
              </Tooltip>
            </Group>
          </Group>
        </Paper>

        {/* Key Metrics Overview */}
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
                sections={[{ 
                  value: overallScore, 
                  color: overallScore >= 80 ? 'green' : overallScore >= 60 ? 'yellow' : 'red' 
                }]}
              />
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">AFM Compliance</Text>
                <Title order={2}>{finalResults.afm_compliance.compliance_score.toFixed(1)}%</Title>
              </div>
              <ThemeIcon size="xl" color="indigo" radius={0}>
                <IconGavel size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Quality Control</Text>
                <Title order={2}>{finalResults.quality_control.completeness_score.toFixed(1)}%</Title>
              </div>
              <ThemeIcon size="xl" color="emerald" radius={0}>
                <IconShield size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Risk Profile</Text>
                <Badge 
                  color={getRiskColor(finalResults.afm_compliance.risk_profile)} 
                  size="lg"
                  radius={0}
                >
                  {finalResults.afm_compliance.risk_profile.toUpperCase()}
                </Badge>
              </div>
              <ThemeIcon 
                size="xl" 
                color={getRiskColor(finalResults.afm_compliance.risk_profile)} 
                radius={0}
              >
                <IconTrendingUp size={24} />
              </ThemeIcon>
            </Group>
          </Card>
        </SimpleGrid>

        {/* Processing Timeline */}
        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="md">Processing Timeline</Title>
          <Timeline active={finalResults.processing_timeline.findIndex(step => step.status === 'in_progress')}>
            {finalResults.processing_timeline.map((step, index) => (
              <Timeline.Item
                key={index}
                bullet={
                  step.status === 'completed' ? <IconCheck size={12} /> :
                  step.status === 'in_progress' ? <IconClock size={12} /> :
                  <IconInfoCircle size={12} />
                }
                title={step.step}
              >
                <Text c="dimmed" size="sm">{step.details}</Text>
                {step.timestamp && (
                  <Text size="xs" c="dimmed" mt={4}>
                    {new Date(step.timestamp).toLocaleString()}
                  </Text>
                )}
              </Timeline.Item>
            ))}
          </Timeline>
        </Card>

        {/* Detailed Analysis Tabs */}
        <Tabs value={activeTab} onChange={(value) => setActiveTab(value || 'overview')} radius={0}>
          <Tabs.List>
            <Tabs.Tab value="overview" leftSection={<IconEye size={16} />}>
              Overview
            </Tabs.Tab>
            <Tabs.Tab value="compliance" leftSection={<IconGavel size={16} />}>
              AFM Compliance
            </Tabs.Tab>
            <Tabs.Tab value="quality" leftSection={<IconShield size={16} />}>
              Quality Control
            </Tabs.Tab>
            <Tabs.Tab value="financial" leftSection={<IconChartBar size={16} />}>
              Financial Analysis
            </Tabs.Tab>
            <Tabs.Tab value="lenders" leftSection={<IconBuildingBank size={16} />}>
              Lender Matches
            </Tabs.Tab>
            <Tabs.Tab value="advice" leftSection={<IconClipboardCheck size={16} />}>
              Final Advice
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="overview" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 8 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Executive Summary</Title>
                  <Text mb="md">{finalResults.final_advice.summary}</Text>
                  
                  <Title order={4} mb="sm">Key Highlights</Title>
                  <List spacing="xs" size="sm">
                    {finalResults.final_advice.key_recommendations.slice(0, 3).map((rec, index) => (
                      <List.Item key={index} icon={<IconCheck size={16} color="green" />}>
                        {rec}
                      </List.Item>
                    ))}
                  </List>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Report Details</Title>
                  <Stack gap="sm">
                    <Group justify="space-between">
                      <Text size="sm">Generated</Text>
                      <Text size="sm" fw={500}>
                        {new Date(finalResults.generated_at).toLocaleDateString()}
                      </Text>
                    </Group>
                    <Group justify="space-between">
                      <Text size="sm">Valid Until</Text>
                      <Text size="sm" fw={500}>
                        {new Date(finalResults.valid_until).toLocaleDateString()}
                      </Text>
                    </Group>
                    <Divider />
                    <Group justify="space-between">
                      <Text size="sm">Overall Status</Text>
                      <Badge color={getStatusColor(status)} radius={0}>
                        {status.replace('_', ' ')}
                      </Badge>
                    </Group>
                  </Stack>
                </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>

          <Tabs.Panel value="compliance" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">AFM Compliance Status</Title>
                  <Stack gap="md">
                    <Group justify="space-between">
                      <Text>Compliance Score</Text>
                      <Badge color="indigo" size="lg" radius={0}>
                        {finalResults.afm_compliance.compliance_score.toFixed(1)}%
                      </Badge>
                    </Group>
                    <Progress 
                      value={finalResults.afm_compliance.compliance_score} 
                      color="indigo" 
                      size="lg"
                      radius={0}
                    />
                    
                    <Group justify="space-between">
                      <Text>AFM Status</Text>
                      <Badge 
                        color={finalResults.afm_compliance.afm_status === 'compliant' ? 'green' : 'yellow'} 
                        radius={0}
                      >
                        {finalResults.afm_compliance.afm_status.replace('_', ' ').toUpperCase()}
                      </Badge>
                    </Group>
                    
                    <Group justify="space-between">
                      <Text>Risk Profile</Text>
                      <Badge color={getRiskColor(finalResults.afm_compliance.risk_profile)} radius={0}>
                        {finalResults.afm_compliance.risk_profile.toUpperCase()}
                      </Badge>
                    </Group>
                  </Stack>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Compliance Recommendations</Title>
                  <Stack gap="sm">
                    {finalResults.afm_compliance.recommendations.map((rec, index) => (
                      <Alert key={index} color="blue" icon={<IconInfoCircle size={16} />} radius={0}>
                        <Text size="sm">{rec}</Text>
                      </Alert>
                    ))}
                  </Stack>
                </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>

          <Tabs.Panel value="quality" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Quality Control Summary</Title>
                  <Stack gap="md">
                    <Group justify="space-between">
                      <Text>Completeness Score</Text>
                      <Badge color="emerald" size="lg" radius={0}>
                        {finalResults.quality_control.completeness_score.toFixed(1)}%
                      </Badge>
                    </Group>
                    <Progress 
                      value={finalResults.quality_control.completeness_score} 
                      color="emerald" 
                      size="lg"
                      radius={0}
                    />
                    
                    <Group justify="space-between">
                      <Text>QC Status</Text>
                      <Badge color={finalResults.quality_control.passed ? 'green' : 'red'} radius={0}>
                        {finalResults.quality_control.passed ? 'PASSED' : 'FAILED'}
                      </Badge>
                    </Group>
                    
                    <Group justify="space-between">
                      <Text>Critical Issues</Text>
                      <Badge color={finalResults.quality_control.critical_issues > 0 ? 'red' : 'green'} radius={0}>
                        {finalResults.quality_control.critical_issues}
                      </Badge>
                    </Group>
                  </Stack>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Field Validation</Title>
                  <Stack gap="sm">
                    {finalResults.quality_control.field_validation.map((field, index) => (
                      <Paper key={index} p="sm" withBorder radius={0}>
                        <Group justify="space-between">
                          <Group>
                            <ThemeIcon 
                              size="sm" 
                              color={field.valid ? 'green' : 'red'} 
                              radius={0}
                            >
                              {field.valid ? <IconCheck size={12} /> : <IconX size={12} />}
                            </ThemeIcon>
                            <Text size="sm">{field.field.replace(/_/g, ' ')}</Text>
                          </Group>
                          <Badge 
                            color={field.valid ? 'green' : 'red'} 
                            size="sm"
                            radius={0}
                          >
                            {field.valid ? 'Valid' : field.severity}
                          </Badge>
                        </Group>
                        {field.error && (
                          <Text size="xs" c="red" mt="xs">{field.error}</Text>
                        )}
                      </Paper>
                    ))}
                  </Stack>
                </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>

          <Tabs.Panel value="financial" pt="xl">
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
              <Card radius={0} shadow="sm" padding="lg">
                <Title order={3} mb="md">Financial Ratios</Title>
                <Stack gap="md">
                  <div>
                    <Group justify="space-between" mb="xs">
                      <Text size="sm">Debt-to-Income Ratio</Text>
                      <Text size="sm" fw={500}>{finalResults.financial_analysis.dti_ratio.toFixed(1)}%</Text>
                    </Group>
                    <Progress 
                      value={finalResults.financial_analysis.dti_ratio} 
                      color={finalResults.financial_analysis.dti_ratio <= 30 ? 'green' : 'yellow'}
                      radius={0}
                    />
                  </div>
                  
                  <div>
                    <Group justify="space-between" mb="xs">
                      <Text size="sm">Loan-to-Value Ratio</Text>
                      <Text size="sm" fw={500}>{finalResults.financial_analysis.ltv_ratio.toFixed(1)}%</Text>
                    </Group>
                    <Progress 
                      value={finalResults.financial_analysis.ltv_ratio} 
                      color={finalResults.financial_analysis.ltv_ratio <= 80 ? 'green' : 'yellow'}
                      radius={0}
                    />
                  </div>
                  
                  <div>
                    <Group justify="space-between" mb="xs">
                      <Text size="sm">Affordability Score</Text>
                      <Text size="sm" fw={500}>{finalResults.financial_analysis.affordability_score.toFixed(1)}%</Text>
                    </Group>
                    <Progress 
                      value={finalResults.financial_analysis.affordability_score} 
                      color={finalResults.financial_analysis.affordability_score >= 70 ? 'green' : 'yellow'}
                      radius={0}
                    />
                  </div>
                </Stack>
              </Card>
              
              <Card radius={0} shadow="sm" padding="lg">
                <Title order={3} mb="md">Risk Factors</Title>
                <Stack gap="sm">
                  {finalResults.financial_analysis.risk_factors.map((factor, index) => (
                    <Alert key={index} color="yellow" icon={<IconAlertTriangle size={16} />} radius={0}>
                      <Text size="sm">{factor}</Text>
                    </Alert>
                  ))}
                </Stack>
              </Card>
            </SimpleGrid>
          </Tabs.Panel>

          <Tabs.Panel value="lenders" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Recommended Lenders</Title>
              <Stack gap="md">
                {finalResults.lender_matches.map((lender, index) => (
                  <Paper key={index} p="md" withBorder radius={0}>
                    <Grid>
                      <Grid.Col span={{ base: 12, md: 8 }}>
                        <Group justify="space-between" mb="sm">
                          <div>
                            <Text fw={600} size="lg">{lender.lender}</Text>
                            <Text c="dimmed" size="sm">{lender.product}</Text>
                          </div>
                          <Badge color="green" size="lg" radius={0}>
                            {lender.eligibility_score.toFixed(0)}% Match
                          </Badge>
                        </Group>
                        <Group gap="xl">
                          <div>
                            <Text size="xs" c="dimmed">Interest Rate</Text>
                            <Text fw={500}>{lender.interest_rate.toFixed(2)}%</Text>
                          </div>
                          <div>
                            <Text size="xs" c="dimmed">Max LTV</Text>
                            <Text fw={500}>{lender.max_ltv}%</Text>
                          </div>
                        </Group>
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 4 }}>
                        <Text size="sm" fw={500} mb="xs">Conditions:</Text>
                        <List size="xs">
                          {lender.conditions.map((condition, condIndex) => (
                            <List.Item key={condIndex}>{condition}</List.Item>
                          ))}
                        </List>
                      </Grid.Col>
                    </Grid>
                  </Paper>
                ))}
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="advice" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 8 }}>
                <Stack gap="lg">
                  <Card radius={0} shadow="sm" padding="lg">
                    <Title order={3} mb="md">Summary</Title>
                    <Text>{finalResults.final_advice.summary}</Text>
                  </Card>

                  <Card radius={0} shadow="sm" padding="lg">
                    <Title order={3} mb="md">Key Recommendations</Title>
                    <List spacing="sm">
                      {finalResults.final_advice.key_recommendations.map((rec, index) => (
                        <List.Item key={index} icon={<IconCheck size={16} color="green" />}>
                          {rec}
                        </List.Item>
                      ))}
                    </List>
                  </Card>

                  <Card radius={0} shadow="sm" padding="lg">
                    <Title order={3} mb="md">Next Steps</Title>
                    <List spacing="sm">
                      {finalResults.final_advice.next_steps.map((step, index) => (
                        <List.Item key={index} icon={<IconArrowRight size={16} color="blue" />}>
                          {step}
                        </List.Item>
                      ))}
                    </List>
                  </Card>
                </Stack>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Risk Assessment</Title>
                  <Alert color="blue" icon={<IconInfoCircle size={16} />} radius={0}>
                    <Text size="sm">{finalResults.final_advice.risk_assessment}</Text>
                  </Alert>
                </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>
        </Tabs>

        {/* Next Steps Action Items */}
        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="md">Action Items</Title>
          <Stack gap="sm">
            {finalResults.next_steps.map((step, index) => (
              <Paper key={index} p="md" withBorder radius={0}>
                <Group justify="space-between">
                  <Group>
                    <Badge color={step.priority === 'high' ? 'red' : step.priority === 'medium' ? 'yellow' : 'blue'} radius={0}>
                      {step.priority.toUpperCase()}
                    </Badge>
                    <div>
                      <Text fw={500}>{step.step}</Text>
                      <Text size="sm" c="dimmed">Responsible: {step.responsible_party}</Text>
                    </div>
                  </Group>
                  {step.deadline && (
                    <Text size="sm" c="dimmed">Due: {step.deadline}</Text>
                  )}
                </Group>
              </Paper>
            ))}
          </Stack>
        </Card>

        {/* Action Buttons */}
        <Group justify="space-between">
          <Button 
            variant="outline" 
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => navigate(-1)}
            radius={0}
          >
            Back to Quality Control
          </Button>
          <Group>
            <Button 
              variant="outline"
              leftSection={<IconDownload size={16} />}
              onClick={handleExportReport}
              radius={0}
            >
              Export Full Report
            </Button>
            {status === 'approved' && (
              <Button 
                leftSection={<IconArrowRight size={16} />}
                onClick={() => navigate('/lender-integration')}
                radius={0}
              >
                Proceed to Lender Submission
              </Button>
            )}
          </Group>
        </Group>
      </Stack>
    </Container>
  );
};

export default ResultsDisplay;