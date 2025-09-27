import React, { useState, useEffect } from 'react';
import {
  Container,
  Card,
  Button,
  Box,
  Grid,
  Badge,
  Alert,
  Avatar,
  Loader,
  Paper,
  Progress,
  List,
  Divider,
  Text,
  Title,
  Group,
  Stack,
} from '@mantine/core';
import {
  IconChartBar,
  IconCheck,
  IconAlertTriangle,
  IconCreditCard,
  IconArrowLeft,
  IconRefresh,
} from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';
import { notifications } from '@mantine/notifications';
import { apiClient } from '../services/apiClient';
import { useClient } from '../contexts/ClientContext';
import { useDemoMode } from '../contexts/DemoModeContext';

const BKRCreditCheck: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(false);
  const [creditReport, setCreditReport] = useState<any>(null);
  const navigate = useNavigate();
  const { currentClientId } = useClient();
  const { isDemoMode } = useDemoMode();

  useEffect(() => {
    if (isDemoMode) {
      // Load demo data immediately in demo mode
      loadCreditReport();
    } else if (currentClientId) {
      // Load real data only if client is selected in production mode
      loadCreditReport();
    }
  }, [currentClientId, isDemoMode]);

  const loadCreditReport = async () => {
    try {
      setLoading(true);
      
      if (isDemoMode) {
        // Return mock data for demo mode
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay
        const mockReport = getMockBKRReport();
        setCreditReport(mockReport);
        return;
      }

      if (!currentClientId) {
        notifications.show({
          title: 'Warning',
          message: 'No client selected',
          color: 'yellow',
          icon: <IconAlertTriangle size={16} />,
        });
        setLoading(false);
        return;
      }

      const report = await apiClient.getBKRReportByClient(currentClientId);
      setCreditReport(report);
    } catch (error) {
      console.error('Failed to load credit report:', error);
      notifications.show({
        title: 'Error',
        message: 'Failed to load BKR credit report',
        color: 'red',
        icon: <IconAlertTriangle size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const performCreditCheck = async () => {
    if (!creditReport) return;
    setChecking(true);
    try {
      if (isDemoMode) {
        // Simulate refresh in demo mode
        await new Promise(resolve => setTimeout(resolve, 2000));
        const refreshedReport = getMockBKRReport();
        setCreditReport(refreshedReport);
        notifications.show({
          title: 'Success',
          message: 'BKR credit check refreshed successfully (Demo)',
          color: 'green',
          icon: <IconCheck size={16} />,
        });
        return;
      }

      const response = await apiClient.refreshBKRReport(creditReport.id);
      if (response.status === 'completed' && response.report) {
        setCreditReport(response.report);
        notifications.show({
          title: 'Success',
          message: 'BKR credit check refreshed successfully',
          color: 'green',
          icon: <IconCheck size={16} />,
        });
      } else {
        notifications.show({
          title: 'Info',
          message: 'Credit check is processing. Please check back later.',
          color: 'blue',
          icon: <IconAlertTriangle size={16} />,
        });
      }
    } catch (error) {
      notifications.show({
        title: 'Error',
        message: 'Failed to refresh BKR credit check',
        color: 'red',
        icon: <IconAlertTriangle size={16} />,
      });
    } finally {
      setChecking(false);
    }
  };

  if (loading) {
    return (
      <Container size="xl" py="xl">
        <Box style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
          <Loader size="lg" />
        </Box>
      </Container>
    );
  }

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <Button
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => navigate(-1)}
            variant="subtle"
            radius={0}
          >
            Back
          </Button>
        </Group>
        
        <Group>
          <Avatar size="xl" radius={0} color="indigo">
            <IconCreditCard size={32} />
          </Avatar>
          <div>
            <Title order={1}>BKR Credit Check</Title>
            <Text c="dimmed">Dutch credit bureau verification and analysis</Text>
          </div>
        </Group>

        {!creditReport ? (
          <Card radius={0} shadow="sm" padding="lg">
            <Title order={3} mb="md">No Credit Report Available</Title>
            <Text c="dimmed" mb="md">
              No BKR credit report found for the selected client. Please ensure a client is selected and try again.
            </Text>
            <Button
              leftSection={<IconRefresh size={16} />}
              onClick={loadCreditReport}
              disabled={!currentClientId}
              radius={0}
            >
              Load Credit Report
            </Button>
          </Card>
        ) : (
          <Stack gap="lg">
            {/* Credit Score Overview */}
            <Grid>
              <Grid.Col span={{ base: 12, md: 3 }}>
                <Paper p="md" radius={0} style={{ textAlign: 'center' }}>
                  <Text size="sm" c="dimmed">Credit Score</Text>
                  <Title order={2} c="indigo" my="sm">
                    {creditReport.creditScore}
                  </Title>
                  <Progress
                    value={(creditReport.creditScore / 850) * 100}
                    color="indigo"
                    radius={0}
                  />
                </Paper>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 3 }}>
                <Paper p="md" radius={0} style={{ textAlign: 'center' }}>
                  <Text size="sm" c="dimmed">Risk Category</Text>
                  <Badge
                    color={creditReport.riskCategory === 'Low' ? 'green' : 'orange'}
                    size="lg"
                    radius={0}
                    mt="sm"
                  >
                    {creditReport.riskCategory}
                  </Badge>
                </Paper>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 3 }}>
                <Paper p="md" radius={0} style={{ textAlign: 'center' }}>
                  <Text size="sm" c="dimmed">Total Debt</Text>
                  <Title order={2} my="sm">
                    €{creditReport.totalDebt?.toLocaleString() || '0'}
                  </Title>
                </Paper>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 3 }}>
                <Paper p="md" radius={0} style={{ textAlign: 'center' }}>
                  <Text size="sm" c="dimmed">Active Loans</Text>
                  <Title order={2} my="sm">
                    {creditReport.activeLoans || 0}
                  </Title>
                </Paper>
              </Grid.Col>
            </Grid>

            {/* Credit Details */}
            <Card radius={0} shadow="sm" padding="lg">
              <Group justify="space-between" mb="md">
                <Title order={3}>Credit Details</Title>
                <Button
                  variant="outline"
                  leftSection={<IconRefresh size={16} />}
                  onClick={performCreditCheck}
                  loading={checking}
                  radius={0}
                >
                  {checking ? 'Refreshing...' : 'Refresh Report'}
                </Button>
              </Group>

              {creditReport.loans && creditReport.loans.length > 0 ? (
                <List spacing="md">
                  {creditReport.loans.map((loan: any, index: number) => (
                    <React.Fragment key={index}>
                      <List.Item
                        icon={<IconChartBar size={20} />}
                      >
                        <div>
                          <Text fw={500}>{loan.type}</Text>
                          <Text size="sm" c="dimmed">
                            Amount: €{loan.amount?.toLocaleString() || '0'}
                          </Text>
                          <Text size="sm" c="dimmed">
                            Monthly Payment: €{loan.monthlyPayment?.toLocaleString() || '0'}
                          </Text>
                          <Badge
                            color={loan.status === 'Active' ? 'green' : 'gray'}
                            size="sm"
                            radius={0}
                            mt="xs"
                          >
                            {loan.status}
                          </Badge>
                        </div>
                      </List.Item>
                      {index < creditReport.loans.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Text c="dimmed">No active loans found.</Text>
              )}
            </Card>

            {/* Report Information */}
            <Alert color="blue" icon={<IconAlertTriangle size={16} />} radius={0}>
              This BKR credit report was generated on{' '}
              {new Date(creditReport.reportDate).toLocaleDateString()}. 
              Credit information is updated regularly and may change.
            </Alert>
          </Stack>
        )}
      </Stack>
    </Container>
  );
};

// Mock data function for demo mode
const getMockBKRReport = () => {
  return {
    id: 'demo-bkr-report-001',
    clientId: 'demo-client-001',
    creditScore: 742,
    riskCategory: 'Low',
    totalDebt: 45000,
    activeLoans: 2,
    reportDate: new Date().toISOString(),
    loans: [
      {
        type: 'Personal Loan',
        amount: 15000,
        monthlyPayment: 285,
        status: 'Active',
        lender: 'ING Bank',
        startDate: '2022-03-15',
        remainingBalance: 12500
      },
      {
        type: 'Car Loan',
        amount: 30000,
        monthlyPayment: 450,
        status: 'Active',
        lender: 'Rabobank',
        startDate: '2021-08-20',
        remainingBalance: 18200
      },
      {
        type: 'Credit Card',
        amount: 2500,
        monthlyPayment: 125,
        status: 'Active',
        lender: 'ABN AMRO',
        startDate: '2020-01-10',
        remainingBalance: 1800
      }
    ],
    creditHistory: {
      totalAccounts: 8,
      closedAccounts: 5,
      onTimePayments: 98.5,
      latePayments: 2,
      defaults: 0
    },
    recommendations: [
      'Maintain current payment schedule to improve credit score',
      'Consider consolidating smaller debts',
      'Monitor credit utilization ratio'
    ]
  };
};

export default BKRCreditCheck;