/**
 * Compliance Audit Trail - Full Mantine Implementation
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Card,
  Button,
  Title,
  Group,
  Stack,
  Grid,
  Badge,
  ThemeIcon,
  Text,
  Table,
  Select,
  TextInput,
  ActionIcon,
  Modal,
  Timeline,
  Alert,
  Divider,
  SimpleGrid,
  Pagination,
} from '@mantine/core';
import {
  IconShield,
  IconEye,
  IconDownload,
  IconFilter,
  IconSearch,
  IconCalendar,
  IconUser,
  IconFileText,
  IconAlertTriangle,
  IconCheck,
  IconClock,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

interface AuditEntry {
  id: string;
  timestamp: string;
  user: string;
  action: string;
  resource: string;
  status: 'success' | 'warning' | 'error';
  details: string;
  ipAddress: string;
  userAgent: string;
}

const ComplianceAuditTrail: React.FC = () => {
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEntry, setSelectedEntry] = useState<AuditEntry | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [filters, setFilters] = useState({
    user: '',
    action: '',
    status: '',
    dateFrom: '',
    dateTo: '',
    search: '',
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    loadAuditTrail();
  }, [currentPage, filters]);

  const loadAuditTrail = async () => {
    setLoading(true);
    try {
      // Mock audit trail data
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockEntries: AuditEntry[] = [
        {
          id: '1',
          timestamp: '2024-01-15 14:30:25',
          user: 'john.doe@company.com',
          action: 'Document Upload',
          resource: 'application_123.pdf',
          status: 'success',
          details: 'Successfully uploaded mortgage application document',
          ipAddress: '192.168.1.100',
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        },
        {
          id: '2',
          timestamp: '2024-01-15 14:25:10',
          user: 'jane.smith@company.com',
          action: 'Compliance Check',
          resource: 'AFM_validation_456',
          status: 'warning',
          details: 'AFM compliance check completed with minor warnings',
          ipAddress: '192.168.1.101',
          userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        },
        {
          id: '3',
          timestamp: '2024-01-15 14:20:45',
          user: 'admin@company.com',
          action: 'User Access',
          resource: 'system_login',
          status: 'success',
          details: 'Administrator logged into the system',
          ipAddress: '192.168.1.1',
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        },
        {
          id: '4',
          timestamp: '2024-01-15 14:15:30',
          user: 'bob.wilson@company.com',
          action: 'Data Export',
          resource: 'client_data_789',
          status: 'error',
          details: 'Failed to export client data - insufficient permissions',
          ipAddress: '192.168.1.102',
          userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        },
        {
          id: '5',
          timestamp: '2024-01-15 14:10:15',
          user: 'sarah.jones@company.com',
          action: 'Application Review',
          resource: 'mortgage_app_321',
          status: 'success',
          details: 'Completed review of mortgage application',
          ipAddress: '192.168.1.103',
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        },
      ];
      
      setAuditEntries(mockEntries);
      setTotalPages(5); // Mock pagination
    } catch (error) {
      notifications.show({
        title: 'Load Failed',
        message: 'Failed to load audit trail',
        color: 'red',
        icon: <IconAlertTriangle size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const viewDetails = (entry: AuditEntry) => {
    setSelectedEntry(entry);
    setModalOpen(true);
  };

  const exportAuditLog = async () => {
    try {
      notifications.show({
        title: 'Export Started',
        message: 'Audit log export has been initiated',
        color: 'blue',
        icon: <IconDownload size={16} />,
      });
      
      // Mock export process
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      notifications.show({
        title: 'Export Complete',
        message: 'Audit log has been exported successfully',
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Export Failed',
        message: 'Failed to export audit log',
        color: 'red',
        icon: <IconAlertTriangle size={16} />,
      });
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'green';
      case 'warning': return 'yellow';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <IconCheck size={16} />;
      case 'warning': return <IconAlertTriangle size={16} />;
      case 'error': return <IconAlertTriangle size={16} />;
      default: return <IconClock size={16} />;
    }
  };

  const filteredEntries = auditEntries.filter(entry => {
    return (
      (!filters.user || entry.user.toLowerCase().includes(filters.user.toLowerCase())) &&
      (!filters.action || entry.action.toLowerCase().includes(filters.action.toLowerCase())) &&
      (!filters.status || entry.status === filters.status) &&
      (!filters.search || 
        entry.details.toLowerCase().includes(filters.search.toLowerCase()) ||
        entry.resource.toLowerCase().includes(filters.search.toLowerCase())
      )
    );
  });

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <ThemeIcon size="xl" radius={0} color="indigo">
            <IconShield size={32} />
          </ThemeIcon>
          <div>
            <Title order={1}>Compliance Audit Trail</Title>
            <Text c="dimmed">Complete audit log of system activities and compliance events</Text>
          </div>
        </Group>

        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="lg">
          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Total Events</Text>
                <Title order={2}>{auditEntries.length}</Title>
              </div>
              <ThemeIcon size="xl" color="blue" radius={0}>
                <IconFileText size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Success Rate</Text>
                <Title order={2}>85%</Title>
              </div>
              <ThemeIcon size="xl" color="green" radius={0}>
                <IconCheck size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Active Users</Text>
                <Title order={2}>12</Title>
              </div>
              <ThemeIcon size="xl" color="orange" radius={0}>
                <IconUser size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Compliance Score</Text>
                <Title order={2}>98%</Title>
              </div>
              <ThemeIcon size="xl" color="green" radius={0}>
                <IconShield size={24} />
              </ThemeIcon>
            </Group>
          </Card>
        </SimpleGrid>

        <Card radius={0} shadow="sm" padding="lg">
          <Group justify="space-between" mb="md">
            <Title order={3}>Audit Filters</Title>
            <Button
              leftSection={<IconDownload size={16} />}
              onClick={exportAuditLog}
              variant="light"
              radius={0}
            >
              Export Log
            </Button>
          </Group>
          
          <Grid>
            <Grid.Col span={{ base: 12, md: 3 }}>
              <TextInput
                label="User"
                placeholder="Filter by user"
                value={filters.user}
                onChange={(e) => setFilters(prev => ({ ...prev, user: e.target.value }))}
                leftSection={<IconUser size={16} />}
                radius={0}
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 3 }}>
              <Select
                label="Action"
                placeholder="Filter by action"
                value={filters.action}
                onChange={(value) => setFilters(prev => ({ ...prev, action: value || '' }))}
                data={[
                  { value: '', label: 'All Actions' },
                  { value: 'Document Upload', label: 'Document Upload' },
                  { value: 'Compliance Check', label: 'Compliance Check' },
                  { value: 'User Access', label: 'User Access' },
                  { value: 'Data Export', label: 'Data Export' },
                  { value: 'Application Review', label: 'Application Review' },
                ]}
                radius={0}
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 3 }}>
              <Select
                label="Status"
                placeholder="Filter by status"
                value={filters.status}
                onChange={(value) => setFilters(prev => ({ ...prev, status: value || '' }))}
                data={[
                  { value: '', label: 'All Status' },
                  { value: 'success', label: 'Success' },
                  { value: 'warning', label: 'Warning' },
                  { value: 'error', label: 'Error' },
                ]}
                radius={0}
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 3 }}>
              <TextInput
                label="Search"
                placeholder="Search details..."
                value={filters.search}
                onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                leftSection={<IconSearch size={16} />}
                radius={0}
              />
            </Grid.Col>
          </Grid>
        </Card>

        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="md">Audit Events</Title>
          <Table>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Timestamp</Table.Th>
                <Table.Th>User</Table.Th>
                <Table.Th>Action</Table.Th>
                <Table.Th>Resource</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Actions</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filteredEntries.map((entry) => (
                <Table.Tr key={entry.id}>
                  <Table.Td>
                    <Group>
                      <ThemeIcon size="sm" color="gray" radius={0}>
                        <IconCalendar size={16} />
                      </ThemeIcon>
                      <Text size="sm">{entry.timestamp}</Text>
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" fw={500}>{entry.user}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm">{entry.action}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" c="dimmed">{entry.resource}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Badge 
                      color={getStatusColor(entry.status)} 
                      radius={0}
                      leftSection={getStatusIcon(entry.status)}
                    >
                      {entry.status}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <ActionIcon
                      variant="light"
                      color="blue"
                      onClick={() => viewDetails(entry)}
                      radius={0}
                    >
                      <IconEye size={16} />
                    </ActionIcon>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
          
          <Group justify="center" mt="lg">
            <Pagination
              value={currentPage}
              onChange={setCurrentPage}
              total={totalPages}
              radius={0}
            />
          </Group>
        </Card>

        <Alert color="blue" icon={<IconShield size={16} />} radius={0}>
          <Text size="sm">
            All system activities are logged for compliance purposes. 
            Audit logs are retained for 7 years as per regulatory requirements.
          </Text>
        </Alert>
      </Stack>

      <Modal
        opened={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Audit Event Details"
        size="lg"
        radius={0}
      >
        {selectedEntry && (
          <Stack gap="md">
            <Grid>
              <Grid.Col span={6}>
                <Text size="sm" c="dimmed">Event ID</Text>
                <Text fw={500}>{selectedEntry.id}</Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Text size="sm" c="dimmed">Timestamp</Text>
                <Text fw={500}>{selectedEntry.timestamp}</Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Text size="sm" c="dimmed">User</Text>
                <Text fw={500}>{selectedEntry.user}</Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Text size="sm" c="dimmed">Status</Text>
                <Badge 
                  color={getStatusColor(selectedEntry.status)} 
                  radius={0}
                  leftSection={getStatusIcon(selectedEntry.status)}
                >
                  {selectedEntry.status}
                </Badge>
              </Grid.Col>
              <Grid.Col span={12}>
                <Text size="sm" c="dimmed">Action</Text>
                <Text fw={500}>{selectedEntry.action}</Text>
              </Grid.Col>
              <Grid.Col span={12}>
                <Text size="sm" c="dimmed">Resource</Text>
                <Text fw={500}>{selectedEntry.resource}</Text>
              </Grid.Col>
              <Grid.Col span={12}>
                <Text size="sm" c="dimmed">Details</Text>
                <Text>{selectedEntry.details}</Text>
              </Grid.Col>
            </Grid>
            
            <Divider />
            
            <Title order={4}>Technical Details</Title>
            <Grid>
              <Grid.Col span={6}>
                <Text size="sm" c="dimmed">IP Address</Text>
                <Text fw={500}>{selectedEntry.ipAddress}</Text>
              </Grid.Col>
              <Grid.Col span={12}>
                <Text size="sm" c="dimmed">User Agent</Text>
                <Text size="sm" style={{ wordBreak: 'break-all' }}>{selectedEntry.userAgent}</Text>
              </Grid.Col>
            </Grid>
          </Stack>
        )}
      </Modal>
    </Container>
  );
};

export default ComplianceAuditTrail;