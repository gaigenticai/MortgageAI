import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  Text,
  Button,
  Box,
  Badge,
  Progress,
  Avatar,
  Stack,
  Group,
  Divider,
  Loader,
  Tooltip,
  Container,
  Title,
  ActionIcon,
  Paper,
  SimpleGrid,
} from '@mantine/core';
import {
  IconGavel,
  IconBuildingBank,
  IconChartBar,
  IconTrendingUp,
  IconUser,
  IconBuilding,
  IconClock,
  IconInfoCircle,
  IconArrowRight,
  IconShield,
  IconCheck,
  IconAlertTriangle,
} from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';
import { notifications } from '@mantine/notifications';
import { apiClient } from '../services/apiClient';
import ComparisonChart from '../components/ComparisonChart';
import { useDemoMode } from '../contexts/DemoModeContext';

interface DashboardMetrics {
  afm_compliance_score: number;
  active_sessions: number;
  pending_reviews: number;
  applications_processed_today: number;
  first_time_right_rate: number;
  average_processing_time: number;
  compliance_alerts: number;
  quality_score: number;
}

interface RecentActivity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'warning' | 'error';
  client_name?: string;
}

const DutchMortgageDashboard: React.FC = () => {
  const navigate = useNavigate();
  const { isDemoMode } = useDemoMode();
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const [metricsData, activityData] = await Promise.all([
          apiClient.getDashboardMetrics(),
          apiClient.getRecentActivity()
        ]);
        
        setMetrics(metricsData);
        setRecentActivity(activityData);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        notifications.show({
          title: 'Error',
          message: 'Failed to load dashboard data',
          color: 'red',
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'green';
      case 'pending': return 'blue';
      case 'warning': return 'yellow';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <IconCheck size={16} />;
      case 'pending': return <IconClock size={16} />;
      case 'warning': return <IconAlertTriangle size={16} />;
      case 'error': return <IconAlertTriangle size={16} />;
      default: return <IconInfoCircle size={16} />;
    }
  };

  if (loading) {
    return (
      <Container size="xl" py="xl">
        <Box style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
          <Stack align="center" gap="md">
            <Loader size="xl" color="indigo" />
            <Text c="dimmed">Loading dashboard...</Text>
          </Stack>
        </Box>
      </Container>
    );
  }

  return (
    <Container size="xl" py="xl">
      {/* Header */}
      <Stack gap="xl">
        <Box>
          <Group justify="space-between" align="center" mb="md">
            <Box>
              <Title order={1} c="dark">
                Dutch Mortgage Dashboard
              </Title>
              <Text c="dimmed" size="lg">
                AFM-compliant mortgage advisory platform
              </Text>
            </Box>
            
            {isDemoMode && (
              <Badge color="amber" variant="filled" size="lg" radius={0}>
                DEMO MODE
              </Badge>
            )}
          </Group>
        </Box>

        {/* Key Metrics */}
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between" mb="xs">
              <Text size="sm" c="dimmed" fw={500}>AFM Compliance Score</Text>
              <ActionIcon variant="subtle" color="emerald" radius={0}>
                <IconShield size={18} />
              </ActionIcon>
            </Group>
            <Text size="xl" fw={700} c="emerald">
              {metrics?.afm_compliance_score || 98}%
            </Text>
            <Progress value={metrics?.afm_compliance_score || 98} color="emerald" size="sm" radius={0} mt="xs" />
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between" mb="xs">
              <Text size="sm" c="dimmed" fw={500}>Active Sessions</Text>
              <ActionIcon variant="subtle" color="indigo" radius={0}>
                <IconUser size={18} />
              </ActionIcon>
            </Group>
            <Text size="xl" fw={700} c="indigo">
              {metrics?.active_sessions || 24}
            </Text>
            <Text size="xs" c="dimmed" mt="xs">
              +12% from yesterday
            </Text>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between" mb="xs">
              <Text size="sm" c="dimmed" fw={500}>Pending Reviews</Text>
              <ActionIcon variant="subtle" color="amber" radius={0}>
                <IconClock size={18} />
              </ActionIcon>
            </Group>
            <Text size="xl" fw={700} c="amber">
              {metrics?.pending_reviews || 8}
            </Text>
            <Text size="xs" c="dimmed" mt="xs">
              Avg. 2.3 hours to complete
            </Text>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between" mb="xs">
              <Text size="sm" c="dimmed" fw={500}>Quality Score</Text>
              <ActionIcon variant="subtle" color="pink" radius={0}>
                <IconChartBar size={18} />
              </ActionIcon>
            </Group>
            <Text size="xl" fw={700} c="pink">
              {metrics?.quality_score || 96}%
            </Text>
            <Progress value={metrics?.quality_score || 96} color="pink" size="sm" radius={0} mt="xs" />
          </Card>
        </SimpleGrid>

        {/* Main Content Grid */}
        <Grid>
          {/* Recent Activity */}
          <Grid.Col span={{ base: 12, lg: 8 }}>
            <Card radius={0} shadow="sm" padding="lg" h="100%">
              <Group justify="space-between" mb="lg">
                <Title order={3}>Recent Activity</Title>
                <Button variant="subtle" rightSection={<IconArrowRight size={16} />} radius={0}>
                  View All
                </Button>
              </Group>
              
              <Stack gap="md">
                {recentActivity.length > 0 ? recentActivity.slice(0, 6).map((activity) => (
                  <Paper key={activity.id} p="md" radius={0} style={{ border: '1px solid #E2E8F0' }}>
                    <Group justify="space-between" align="flex-start">
                      <Group align="flex-start">
                        <Badge 
                          color={getStatusColor(activity.status)} 
                          variant="light" 
                          leftSection={getStatusIcon(activity.status)}
                          radius={0}
                        >
                          {activity.status}
                        </Badge>
                        <Box>
                          <Text fw={500} size="sm">
                            {activity.description}
                          </Text>
                          {activity.client_name && (
                            <Text size="xs" c="dimmed">
                              Client: {activity.client_name}
                            </Text>
                          )}
                        </Box>
                      </Group>
                      <Text size="xs" c="dimmed">
                        {new Date(activity.timestamp).toLocaleTimeString()}
                      </Text>
                    </Group>
                  </Paper>
                )) : (
                  <Box ta="center" py="xl">
                    <Text c="dimmed">No recent activity</Text>
                  </Box>
                )}
              </Stack>
            </Card>
          </Grid.Col>

          {/* Quick Actions */}
          <Grid.Col span={{ base: 12, lg: 4 }}>
            <Card radius={0} shadow="sm" padding="lg" h="100%">
              <Title order={3} mb="lg">Quick Actions</Title>
              
              <Stack gap="md">
                <Button
                  variant="light"
                  color="indigo"
                  leftSection={<IconUser size={18} />}
                  onClick={() => navigate('/afm-client-intake')}
                  radius={0}
                  fullWidth
                  justify="flex-start"
                >
                  New Client Intake
                </Button>
                
                <Button
                  variant="light"
                  color="emerald"
                  leftSection={<IconShield size={18} />}
                  onClick={() => navigate('/compliance')}
                  radius={0}
                  fullWidth
                  justify="flex-start"
                >
                  Compliance Check
                </Button>
                
                <Button
                  variant="light"
                  color="pink"
                  leftSection={<IconChartBar size={18} />}
                  onClick={() => navigate('/quality-control')}
                  radius={0}
                  fullWidth
                  justify="flex-start"
                >
                  Quality Control
                </Button>
                
                <Button
                  variant="light"
                  color="amber"
                  leftSection={<IconBuildingBank size={18} />}
                  onClick={() => navigate('/lender-integration')}
                  radius={0}
                  fullWidth
                  justify="flex-start"
                >
                  Lender Integration
                </Button>

                <Divider my="sm" />

                <Button
                  variant="outline"
                  color="gray"
                  leftSection={<IconTrendingUp size={18} />}
                  onClick={() => navigate('/dutch-market-insights')}
                  radius={0}
                  fullWidth
                  justify="flex-start"
                >
                  Market Insights
                </Button>
              </Stack>
            </Card>
          </Grid.Col>
        </Grid>

        {/* Comparison Chart */}
        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="lg">Performance Comparison</Title>
          <ComparisonChart />
        </Card>
      </Stack>
    </Container>
  );
};

export default DutchMortgageDashboard;