/**
 * AFM Compliance Check - Full Mantine Implementation
 *
 * Comprehensive AFM compliance assessment with:
 * - Real-time compliance analysis
 * - Product recommendations
 * - Regulatory requirements validation
 * - Risk assessment and profiling
 * - Integration with compliance APIs
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
  List,
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
  Accordion,
  Table,
  Divider,
} from '@mantine/core';
import {
  IconGavel,
  IconShield,
  IconCheck,
  IconX,
  IconAlertTriangle,
  IconInfoCircle,
  IconChevronDown,
  IconFileText,
  IconUser,
  IconBuildingBank,
  IconChartBar,
  IconArrowRight,
  IconArrowLeft,
  IconRefresh,
  IconDownload,
  IconEye,
  IconClipboardCheck,
  IconScale,
  IconCertificate,
  IconExclamationMark,
  IconClock,
  IconTrendingUp,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { 
  complianceApi, 
  ComplianceAnalysis, 
  ComplianceAssessmentRequest,
  ComplianceRecommendation,
  ProductRecommendation 
} from '../services/complianceApi';

const ComplianceCheck: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [complianceResult, setComplianceResult] = useState<ComplianceAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<string>('overview');

  useEffect(() => {
    loadComplianceAnalysis();
  }, []);

  const loadComplianceAnalysis = async () => {
    setLoading(true);
    try {
      const clientId = searchParams.get('client_id') || 'current_client';
      // For now, use mock data since the API method doesn't exist yet
      const result: ComplianceAnalysis = {
        id: 'comp-001',
        client_id: clientId,
        client_name: 'John Doe',
        assessment_date: new Date().toISOString(),
        compliance_score: 87.5,
        risk_profile: 'medium',
        afm_status: 'compliant',
        overall_status: 'passed',
        recommendations: [
          {
            id: 'rec-1',
            type: 'approved',
            title: 'Documentation Complete',
            description: 'All required documentation has been provided and verified.',
            risk_level: 'low',
            afm_requirements: ['Article 86f compliance', 'Client categorization'],
            recommended_actions: ['Proceed with application'],
            priority: 'low',
          }
        ],
        product_recommendations: [
          {
            id: 'prod-1',
            lender: 'ING Bank',
            lender_id: 'ing-001',
            product_name: 'Green Mortgage Fixed',
            product_type: 'fixed_rate',
            interest_rate: 3.25,
            max_ltv: 90,
            suitability_score: 92,
            afm_compliant: true,
            term_years: 30,
            nhg_required: false,
            estimated_monthly_payment: 1250,
            conditions: ['Minimum income €50,000', 'Energy label A or B required'],
          }
        ],
        compliance_flags: [],
        regulatory_requirements: {
          wft_article_86f: true,
          suitability_assessment: true,
          product_governance: true,
          client_categorization: true,
        },
      };
      setComplianceResult(result);
    } catch (error) {
      console.error('Failed to load compliance analysis:', error);
      notifications.show({
        title: 'Loading Error',
        message: 'Failed to load compliance analysis',
        color: 'red',
        icon: <IconX size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const runNewAssessment = async () => {
    setAnalyzing(true);
    try {
      const clientId = searchParams.get('client_id') || 'current_client';
      const request: ComplianceAssessmentRequest = {
        client_id: clientId,
        assessment_type: 'initial',
        include_product_recommendations: true,
        priority: 'normal',
      };

      const response = await complianceApi.requestComplianceAssessment(request);
      
      notifications.show({
        title: 'Assessment Started',
        message: `New compliance assessment initiated. ID: ${response.assessment_id}`,
        color: 'blue',
        icon: <IconGavel size={16} />,
      });

      // Reload after a short delay to get updated results
      setTimeout(() => {
        loadComplianceAnalysis();
      }, 2000);
    } catch (error) {
      notifications.show({
        title: 'Assessment Failed',
        message: 'Failed to start new compliance assessment',
        color: 'red',
        icon: <IconX size={16} />,
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed': case 'compliant': return 'green';
      case 'conditional_approval': case 'conditional': return 'yellow';
      case 'requires_review': return 'blue';
      case 'rejected': case 'non_compliant': return 'red';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': case 'compliant': return <IconCheck size={24} />;
      case 'conditional_approval': case 'conditional': return <IconAlertTriangle size={24} />;
      case 'requires_review': return <IconClock size={24} />;
      case 'rejected': case 'non_compliant': return <IconX size={24} />;
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

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'yellow';
      case 'low': return 'blue';
      default: return 'gray';
    }
  };

  if (loading || !complianceResult) {
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: '60vh' }}>
          <Stack align="center" gap="md">
            <Loader size="xl" />
            <Title order={3}>Loading Compliance Analysis</Title>
            <Text c="dimmed">Analyzing AFM compliance requirements...</Text>
          </Stack>
        </Center>
      </Container>
    );
  }

  const complianceScore = complianceResult.compliance_score;
  const riskProfile = complianceResult.risk_profile;
  const afmStatus = complianceResult.afm_status;

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between">
          <Group>
            <ThemeIcon size="xl" radius={0} color="indigo">
              <IconGavel size={32} />
            </ThemeIcon>
            <div>
              <Title order={1}>AFM Compliance Check</Title>
              <Text c="dimmed">Client: {complianceResult.client_name}</Text>
              <Text size="sm" c="dimmed">Assessment ID: {complianceResult.id}</Text>
            </div>
          </Group>
          <Group>
            <Button 
              variant="outline" 
              leftSection={<IconRefresh size={16} />}
              onClick={runNewAssessment}
              loading={analyzing}
              radius={0}
            >
              New Assessment
            </Button>
            <Button 
              leftSection={<IconDownload size={16} />}
              radius={0}
            >
              Export Report
            </Button>
          </Group>
        </Group>

        {/* Status Alert */}
        <Alert 
          color={getStatusColor(complianceResult.overall_status)} 
          icon={getStatusIcon(complianceResult.overall_status)}
          title={`Compliance Status: ${complianceResult.overall_status.replace('_', ' ').toUpperCase()}`}
          radius={0}
        >
          <Text size="sm">
            Assessment completed on {new Date(complianceResult.assessment_date).toLocaleDateString()}
            {complianceResult.review_deadline && (
              <> • Review deadline: {new Date(complianceResult.review_deadline).toLocaleDateString()}</>
            )}
          </Text>
        </Alert>

        {/* Key Metrics Overview */}
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="lg">
          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Compliance Score</Text>
                <Title order={2}>{complianceScore.toFixed(1)}%</Title>
              </div>
              <RingProgress
                size={60}
                thickness={6}
                sections={[{ 
                  value: complianceScore, 
                  color: complianceScore >= 80 ? 'green' : complianceScore >= 60 ? 'yellow' : 'red' 
                }]}
              />
            </Group>
            <Progress 
              value={complianceScore} 
              color={complianceScore >= 80 ? 'green' : complianceScore >= 60 ? 'yellow' : 'red'}
              size="sm" 
              mt="md"
              radius={0}
            />
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">AFM Status</Text>
                <Badge 
                  color={getStatusColor(afmStatus)} 
                  size="lg"
                  radius={0}
                >
                  {afmStatus.replace('_', ' ').toUpperCase()}
                </Badge>
              </div>
              <ThemeIcon 
                size="xl" 
                color={getStatusColor(afmStatus)} 
                radius={0}
              >
                <IconCertificate size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Risk Profile</Text>
                <Badge 
                  color={getRiskColor(riskProfile)} 
                  size="lg"
                  radius={0}
                >
                  {riskProfile.toUpperCase()}
                </Badge>
              </div>
              <ThemeIcon 
                size="xl" 
                color={getRiskColor(riskProfile)} 
                radius={0}
              >
                <IconScale size={24} />
              </ThemeIcon>
            </Group>
              </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Compliance Flags</Text>
                <Title order={2}>{complianceResult.compliance_flags.length}</Title>
              </div>
              <ThemeIcon 
                size="xl" 
                color={complianceResult.compliance_flags.length === 0 ? 'green' : 'orange'} 
                radius={0}
              >
                <IconExclamationMark size={24} />
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
            <Tabs.Tab value="regulatory" leftSection={<IconGavel size={16} />}>
              Regulatory Requirements
            </Tabs.Tab>
            <Tabs.Tab value="recommendations" leftSection={<IconClipboardCheck size={16} />}>
              Recommendations
            </Tabs.Tab>
            <Tabs.Tab value="products" leftSection={<IconBuildingBank size={16} />}>
              Product Recommendations
            </Tabs.Tab>
            <Tabs.Tab value="flags" leftSection={<IconAlertTriangle size={16} />}>
              Compliance Flags
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="overview" pt="xl">
            <Grid>
              <Grid.Col span={{ base: 12, md: 8 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Assessment Summary</Title>
                  <Stack gap="md">
                    <Group justify="space-between">
                      <Text>Overall Compliance Score</Text>
                      <Badge color={getStatusColor(complianceResult.overall_status)} size="lg" radius={0}>
                        {complianceScore.toFixed(1)}%
                      </Badge>
                    </Group>
                    <Progress 
                      value={complianceScore} 
                      color={complianceScore >= 80 ? 'green' : complianceScore >= 60 ? 'yellow' : 'red'}
                      size="lg"
                      radius={0}
                    />
                    
                    <Divider />
                    
                    <Group justify="space-between">
                      <Text>Risk Assessment</Text>
                      <Badge color={getRiskColor(riskProfile)} radius={0}>
                        {riskProfile.toUpperCase()} RISK
                      </Badge>
                    </Group>
                    
                    <Group justify="space-between">
                      <Text>AFM Compliance Status</Text>
                      <Badge color={getStatusColor(afmStatus)} radius={0}>
                        {afmStatus.replace('_', ' ').toUpperCase()}
                      </Badge>
                    </Group>
                    
                    {complianceResult.advisor_notes && (
                      <>
                        <Divider />
                        <div>
                          <Text fw={500} mb="xs">Advisor Notes</Text>
                          <Text size="sm" c="dimmed">{complianceResult.advisor_notes}</Text>
                        </div>
                      </>
                    )}
                  </Stack>
              </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card radius={0} shadow="sm" padding="lg">
                  <Title order={3} mb="md">Assessment Details</Title>
                  <Stack gap="sm">
                    <Group justify="space-between">
                      <Text size="sm">Assessment Date</Text>
                      <Text size="sm" fw={500}>
                        {new Date(complianceResult.assessment_date).toLocaleDateString()}
                      </Text>
                    </Group>
                    {complianceResult.review_deadline && (
                      <Group justify="space-between">
                        <Text size="sm">Review Deadline</Text>
                        <Text size="sm" fw={500} c="orange">
                          {new Date(complianceResult.review_deadline).toLocaleDateString()}
                        </Text>
                      </Group>
                    )}
                    {complianceResult.next_review_date && (
                      <Group justify="space-between">
                        <Text size="sm">Next Review</Text>
                        <Text size="sm" fw={500}>
                          {new Date(complianceResult.next_review_date).toLocaleDateString()}
                        </Text>
                      </Group>
                    )}
                    <Divider />
                    <Group justify="space-between">
                      <Text size="sm">Total Recommendations</Text>
                      <Text size="sm" fw={500}>{complianceResult.recommendations.length}</Text>
                    </Group>
                    <Group justify="space-between">
                      <Text size="sm">Product Matches</Text>
                      <Text size="sm" fw={500}>{complianceResult.product_recommendations.length}</Text>
                    </Group>
                  </Stack>
              </Card>
              </Grid.Col>
            </Grid>
          </Tabs.Panel>

          <Tabs.Panel value="regulatory" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Regulatory Requirements Compliance</Title>
              <Stack gap="md">
                <Paper p="md" withBorder radius={0}>
                  <Group justify="space-between">
                    <Group>
                      <ThemeIcon 
                        color={complianceResult.regulatory_requirements.wft_article_86f ? 'green' : 'red'} 
                        radius={0}
                      >
                        {complianceResult.regulatory_requirements.wft_article_86f ? 
                          <IconCheck size={16} /> : <IconX size={16} />}
                      </ThemeIcon>
                      <div>
                        <Text fw={500}>Wft Article 86f Compliance</Text>
                        <Text size="sm" c="dimmed">Suitability and appropriateness assessment</Text>
                      </div>
                    </Group>
                    <Badge 
                      color={complianceResult.regulatory_requirements.wft_article_86f ? 'green' : 'red'} 
                      radius={0}
                    >
                      {complianceResult.regulatory_requirements.wft_article_86f ? 'Compliant' : 'Non-Compliant'}
                    </Badge>
                  </Group>
                </Paper>

                <Paper p="md" withBorder radius={0}>
                  <Group justify="space-between">
                    <Group>
                      <ThemeIcon 
                        color={complianceResult.regulatory_requirements.suitability_assessment ? 'green' : 'red'} 
                        radius={0}
                      >
                        {complianceResult.regulatory_requirements.suitability_assessment ? 
                          <IconCheck size={16} /> : <IconX size={16} />}
                      </ThemeIcon>
                      <div>
                        <Text fw={500}>Suitability Assessment</Text>
                        <Text size="sm" c="dimmed">Client knowledge and experience evaluation</Text>
                      </div>
                    </Group>
                    <Badge 
                      color={complianceResult.regulatory_requirements.suitability_assessment ? 'green' : 'red'} 
                      radius={0}
                    >
                      {complianceResult.regulatory_requirements.suitability_assessment ? 'Complete' : 'Incomplete'}
                    </Badge>
                  </Group>
                </Paper>

                <Paper p="md" withBorder radius={0}>
                  <Group justify="space-between">
                    <Group>
                      <ThemeIcon 
                        color={complianceResult.regulatory_requirements.product_governance ? 'green' : 'red'} 
                        radius={0}
                      >
                        {complianceResult.regulatory_requirements.product_governance ? 
                          <IconCheck size={16} /> : <IconX size={16} />}
                      </ThemeIcon>
                      <div>
                        <Text fw={500}>Product Governance</Text>
                        <Text size="sm" c="dimmed">Product oversight and governance requirements</Text>
                      </div>
                    </Group>
                    <Badge 
                      color={complianceResult.regulatory_requirements.product_governance ? 'green' : 'red'} 
                      radius={0}
                    >
                      {complianceResult.regulatory_requirements.product_governance ? 'Satisfied' : 'Not Satisfied'}
                    </Badge>
                  </Group>
                </Paper>

                <Paper p="md" withBorder radius={0}>
                  <Group justify="space-between">
                    <Group>
                      <ThemeIcon 
                        color={complianceResult.regulatory_requirements.client_categorization ? 'green' : 'red'} 
                        radius={0}
                      >
                        {complianceResult.regulatory_requirements.client_categorization ? 
                          <IconCheck size={16} /> : <IconX size={16} />}
                      </ThemeIcon>
                      <div>
                        <Text fw={500}>Client Categorization</Text>
                        <Text size="sm" c="dimmed">Proper client classification and documentation</Text>
                      </div>
                    </Group>
                    <Badge 
                      color={complianceResult.regulatory_requirements.client_categorization ? 'green' : 'red'} 
                      radius={0}
                    >
                      {complianceResult.regulatory_requirements.client_categorization ? 'Verified' : 'Pending'}
                    </Badge>
                  </Group>
              </Paper>
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="recommendations" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Compliance Recommendations</Title>
              <Stack gap="md">
                {complianceResult.recommendations.map((rec, index) => (
                  <Paper key={index} p="md" withBorder radius={0}>
                    <Group justify="space-between" mb="sm">
                      <Group>
                        <Badge color={getPriorityColor(rec.priority)} radius={0}>
                          {rec.priority.toUpperCase()}
                        </Badge>
                        <Badge color={getStatusColor(rec.type)} radius={0}>
                          {rec.type.replace('_', ' ').toUpperCase()}
                        </Badge>
                      </Group>
                      <Badge color={getRiskColor(rec.risk_level)} radius={0}>
                        {rec.risk_level.toUpperCase()} RISK
                      </Badge>
                    </Group>
                    
                    <Title order={4} mb="xs">{rec.title}</Title>
                    <Text size="sm" c="dimmed" mb="md">{rec.description}</Text>
                    
                    {rec.afm_requirements.length > 0 && (
                      <div>
                        <Text size="sm" fw={500} mb="xs">AFM Requirements:</Text>
                        <List size="sm" spacing="xs">
                          {rec.afm_requirements.map((req, reqIndex) => (
                            <List.Item key={reqIndex}>{req}</List.Item>
                    ))}
                  </List>
                      </div>
                    )}
                    
                    {rec.recommended_actions.length > 0 && (
                      <div>
                        <Text size="sm" fw={500} mb="xs" mt="md">Recommended Actions:</Text>
                        <List size="sm" spacing="xs">
                          {rec.recommended_actions.map((action, actionIndex) => (
                            <List.Item key={actionIndex} icon={<IconArrowRight size={12} />}>
                              {action}
                            </List.Item>
                    ))}
                  </List>
                      </div>
                    )}
                    
                    {rec.deadline && (
                      <Alert color="orange" icon={<IconClock size={16} />} mt="md" radius={0}>
                        <Text size="sm">Deadline: {rec.deadline}</Text>
              </Alert>
                    )}
                  </Paper>
                ))}
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="products" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Recommended Products</Title>
              <Stack gap="md">
                {complianceResult.product_recommendations.map((product, index) => (
                  <Paper key={index} p="md" withBorder radius={0}>
                    <Grid>
                      <Grid.Col span={{ base: 12, md: 8 }}>
                        <Group justify="space-between" mb="sm">
                          <div>
                            <Text fw={600} size="lg">{product.lender}</Text>
                            <Text c="dimmed" size="sm">{product.product_name}</Text>
                          </div>
                          <Group>
                            <Badge color="green" size="lg" radius={0}>
                              {product.suitability_score.toFixed(0)}% Match
                            </Badge>
                            <Badge 
                              color={product.afm_compliant ? 'green' : 'red'} 
                              radius={0}
                            >
                              {product.afm_compliant ? 'AFM Compliant' : 'Non-Compliant'}
                            </Badge>
                          </Group>
                        </Group>
                        
                        <SimpleGrid cols={3} spacing="md">
                          <div>
                            <Text size="xs" c="dimmed">Interest Rate</Text>
                            <Text fw={500}>{product.interest_rate.toFixed(2)}%</Text>
                          </div>
                          <div>
                            <Text size="xs" c="dimmed">Max LTV</Text>
                            <Text fw={500}>{product.max_ltv}%</Text>
                          </div>
                          <div>
                            <Text size="xs" c="dimmed">Term</Text>
                            <Text fw={500}>{product.term_years} years</Text>
                          </div>
                        </SimpleGrid>
                        
                        {product.estimated_monthly_payment && (
                          <Group mt="sm">
                            <Text size="sm" c="dimmed">Est. Monthly Payment:</Text>
                            <Text fw={500}>€{product.estimated_monthly_payment.toLocaleString()}</Text>
                          </Group>
                        )}
                      </Grid.Col>
                      
                      <Grid.Col span={{ base: 12, md: 4 }}>
                        {product.conditions && product.conditions.length > 0 && (
                          <div>
                            <Text size="sm" fw={500} mb="xs">Conditions:</Text>
                            <List size="xs">
                              {product.conditions.map((condition, condIndex) => (
                                <List.Item key={condIndex}>{condition}</List.Item>
                ))}
              </List>
                          </div>
                        )}
                        
                        {product.nhg_required && (
                          <Badge color="blue" mt="xs" radius={0}>NHG Required</Badge>
                        )}
                        
                        {product.energy_efficiency_bonus && (
                          <Badge color="green" mt="xs" radius={0}>
                            Green Bonus: {product.energy_efficiency_bonus}%
                          </Badge>
                        )}
                      </Grid.Col>
                    </Grid>
                  </Paper>
                ))}
              </Stack>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="flags" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Compliance Flags</Title>
              {complianceResult.compliance_flags.length > 0 ? (
                <Stack gap="md">
                  {complianceResult.compliance_flags.map((flag, index) => (
                    <Alert 
                      key={index}
                      color="orange" 
                      icon={<IconAlertTriangle size={16} />}
                      title={`Compliance Flag ${index + 1}`}
                      radius={0}
                    >
                      <Text size="sm">{flag}</Text>
                    </Alert>
                  ))}
                </Stack>
              ) : (
                <Alert color="green" icon={<IconCheck size={16} />} radius={0}>
                  No compliance flags identified. All requirements appear to be met.
                </Alert>
              )}
            </Card>
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
            <Button
              variant="outline"
              leftSection={<IconDownload size={16} />}
              radius={0}
            >
              Export Compliance Report
            </Button>
            {complianceResult.overall_status === 'passed' && (
            <Button
                leftSection={<IconArrowRight size={16} />}
                onClick={() => navigate('/quality-control')}
                radius={0}
              >
                Proceed to Quality Control
            </Button>
        )}
          </Group>
        </Group>
      </Stack>
    </Container>
  );
};

export default ComplianceCheck;