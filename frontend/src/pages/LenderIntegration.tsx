/**
 * Lender Integration - Full Mantine Implementation
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
  Alert,
  Badge,
  ThemeIcon,
  Text,
  Table,
  Switch,
  ActionIcon,
  Modal,
  TextInput,
  Textarea,
  Select,
  Progress,
  Divider,
  SimpleGrid,
} from '@mantine/core';
import {
  IconBuilding,
  IconCheck,
  IconX,
  IconSettings,
  IconRefresh,
  IconPlus,
  IconEdit,
  IconTrash,
  IconApi,
  IconKey,
  IconDatabase,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

interface Lender {
  id: string;
  name: string;
  status: 'connected' | 'disconnected' | 'error';
  apiEndpoint: string;
  lastSync: string;
  applications: number;
  responseTime: number;
}

const LenderIntegration: React.FC = () => {
  const [lenders, setLenders] = useState<Lender[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingLender, setEditingLender] = useState<Lender | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    apiEndpoint: '',
    apiKey: '',
    description: '',
  });

  useEffect(() => {
    loadLenders();
  }, []);

  const loadLenders = async () => {
    setLoading(true);
    try {
      // Mock lender data
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setLenders([
        {
          id: '1',
          name: 'ING Bank',
          status: 'connected',
          apiEndpoint: 'https://api.ing.nl/mortgage',
          lastSync: '2024-01-15 14:30',
          applications: 45,
          responseTime: 250,
        },
        {
          id: '2',
          name: 'ABN AMRO',
          status: 'connected',
          apiEndpoint: 'https://api.abnamro.nl/mortgage',
          lastSync: '2024-01-15 14:25',
          applications: 32,
          responseTime: 180,
        },
        {
          id: '3',
          name: 'Rabobank',
          status: 'error',
          apiEndpoint: 'https://api.rabobank.nl/mortgage',
          lastSync: '2024-01-15 12:15',
          applications: 28,
          responseTime: 0,
        },
        {
          id: '4',
          name: 'SNS Bank',
          status: 'disconnected',
          apiEndpoint: 'https://api.snsbank.nl/mortgage',
          lastSync: '2024-01-14 16:45',
          applications: 15,
          responseTime: 0,
        },
      ]);
    } catch (error) {
      notifications.show({
        title: 'Load Failed',
        message: 'Failed to load lender integrations',
        color: 'red',
        icon: <IconX size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleLenderStatus = async (lenderId: string) => {
    const lender = lenders.find(l => l.id === lenderId);
    if (!lender) return;

    try {
      const newStatus = lender.status === 'connected' ? 'disconnected' : 'connected';
      
      setLenders(prev => prev.map(l => 
        l.id === lenderId 
          ? { ...l, status: newStatus, lastSync: new Date().toLocaleString() }
          : l
      ));

      notifications.show({
        title: 'Status Updated',
        message: `${lender.name} has been ${newStatus}`,
        color: newStatus === 'connected' ? 'green' : 'orange',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Update Failed',
        message: 'Failed to update lender status',
        color: 'red',
        icon: <IconX size={16} />,
      });
    }
  };

  const syncLender = async (lenderId: string) => {
    const lender = lenders.find(l => l.id === lenderId);
    if (!lender) return;

    try {
      setLenders(prev => prev.map(l => 
        l.id === lenderId 
          ? { ...l, lastSync: 'Syncing...' }
          : l
      ));

      await new Promise(resolve => setTimeout(resolve, 2000));

      setLenders(prev => prev.map(l => 
        l.id === lenderId 
          ? { ...l, lastSync: new Date().toLocaleString(), status: 'connected' }
          : l
      ));

      notifications.show({
        title: 'Sync Complete',
        message: `${lender.name} has been synchronized`,
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Sync Failed',
        message: 'Failed to synchronize with lender',
        color: 'red',
        icon: <IconX size={16} />,
      });
    }
  };

  const openAddModal = () => {
    setEditingLender(null);
    setFormData({ name: '', apiEndpoint: '', apiKey: '', description: '' });
    setModalOpen(true);
  };

  const openEditModal = (lender: Lender) => {
    setEditingLender(lender);
    setFormData({
      name: lender.name,
      apiEndpoint: lender.apiEndpoint,
      apiKey: '••••••••',
      description: '',
    });
    setModalOpen(true);
  };

  const saveLender = async () => {
    if (!formData.name || !formData.apiEndpoint) {
      notifications.show({
        title: 'Missing Information',
        message: 'Please fill in all required fields',
        color: 'red',
        icon: <IconX size={16} />,
      });
      return;
    }

    try {
      if (editingLender) {
        setLenders(prev => prev.map(l => 
          l.id === editingLender.id 
            ? { ...l, name: formData.name, apiEndpoint: formData.apiEndpoint }
            : l
        ));
      } else {
        const newLender: Lender = {
          id: Date.now().toString(),
          name: formData.name,
          status: 'disconnected',
          apiEndpoint: formData.apiEndpoint,
          lastSync: 'Never',
          applications: 0,
          responseTime: 0,
        };
        setLenders(prev => [...prev, newLender]);
      }

      setModalOpen(false);
      notifications.show({
        title: 'Lender Saved',
        message: `${formData.name} has been ${editingLender ? 'updated' : 'added'}`,
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Save Failed',
        message: 'Failed to save lender configuration',
        color: 'red',
        icon: <IconX size={16} />,
      });
    }
  };

  const deleteLender = async (lenderId: string) => {
    const lender = lenders.find(l => l.id === lenderId);
    if (!lender) return;

    setLenders(prev => prev.filter(l => l.id !== lenderId));
    notifications.show({
      title: 'Lender Removed',
      message: `${lender.name} has been removed`,
      color: 'orange',
      icon: <IconTrash size={16} />,
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'green';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  const connectedLenders = lenders.filter(l => l.status === 'connected').length;
  const totalApplications = lenders.reduce((sum, l) => sum + l.applications, 0);
  const avgResponseTime = lenders.length > 0 
    ? Math.round(lenders.reduce((sum, l) => sum + l.responseTime, 0) / lenders.length)
    : 0;

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <ThemeIcon size="xl" radius={0} color="indigo">
            <IconBuilding size={32} />
          </ThemeIcon>
          <div>
            <Title order={1}>Lender Integration</Title>
            <Text c="dimmed">Manage connections with mortgage lenders and banks</Text>
          </div>
        </Group>

        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="lg">
          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Connected Lenders</Text>
                <Title order={2}>{connectedLenders}</Title>
              </div>
              <ThemeIcon size="xl" color="green" radius={0}>
                <IconCheck size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Total Applications</Text>
                <Title order={2}>{totalApplications}</Title>
              </div>
              <ThemeIcon size="xl" color="blue" radius={0}>
                <IconDatabase size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">Avg Response Time</Text>
                <Title order={2}>{avgResponseTime}ms</Title>
              </div>
              <ThemeIcon size="xl" color="orange" radius={0}>
                <IconApi size={24} />
              </ThemeIcon>
            </Group>
          </Card>

          <Card radius={0} shadow="sm" padding="lg">
            <Group justify="space-between">
              <div>
                <Text size="sm" c="dimmed">System Health</Text>
                <Badge color="green" radius={0}>Operational</Badge>
              </div>
              <ThemeIcon size="xl" color="green" radius={0}>
                <IconSettings size={24} />
              </ThemeIcon>
            </Group>
          </Card>
        </SimpleGrid>

        <Card radius={0} shadow="sm" padding="lg">
          <Group justify="space-between" mb="md">
            <Title order={3}>Lender Connections</Title>
            <Group>
              <Button
                leftSection={<IconRefresh size={16} />}
                onClick={loadLenders}
                loading={loading}
                variant="light"
                radius={0}
              >
                Refresh
              </Button>
              <Button
                leftSection={<IconPlus size={16} />}
                onClick={openAddModal}
                radius={0}
              >
                Add Lender
              </Button>
            </Group>
          </Group>

          <Table>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Lender</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Last Sync</Table.Th>
                <Table.Th>Applications</Table.Th>
                <Table.Th>Response Time</Table.Th>
                <Table.Th>Actions</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {lenders.map((lender) => (
                <Table.Tr key={lender.id}>
                  <Table.Td>
                    <Group>
                      <ThemeIcon size="sm" color="blue" radius={0}>
                        <IconBuilding size={16} />
                      </ThemeIcon>
                      <div>
                        <Text fw={500}>{lender.name}</Text>
                        <Text size="xs" c="dimmed">{lender.apiEndpoint}</Text>
                      </div>
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Badge color={getStatusColor(lender.status)} radius={0}>
                      {lender.status}
                    </Badge>
                  </Table.Td>
                  <Table.Td>{lender.lastSync}</Table.Td>
                  <Table.Td>{lender.applications}</Table.Td>
                  <Table.Td>{lender.responseTime > 0 ? `${lender.responseTime}ms` : '-'}</Table.Td>
                  <Table.Td>
                    <Group gap="xs">
                      <Switch
                        checked={lender.status === 'connected'}
                        onChange={() => toggleLenderStatus(lender.id)}
                        size="sm"
                      />
                      <ActionIcon
                        variant="light"
                        color="blue"
                        onClick={() => syncLender(lender.id)}
                        radius={0}
                      >
                        <IconRefresh size={16} />
                      </ActionIcon>
                      <ActionIcon
                        variant="light"
                        color="orange"
                        onClick={() => openEditModal(lender)}
                        radius={0}
                      >
                        <IconEdit size={16} />
                      </ActionIcon>
                      <ActionIcon
                        variant="light"
                        color="red"
                        onClick={() => deleteLender(lender.id)}
                        radius={0}
                      >
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Card>

        <Alert color="blue" icon={<IconApi size={16} />} radius={0}>
          <Text size="sm">
            Lender integrations allow automatic submission of mortgage applications and real-time status updates. 
            Ensure API credentials are properly configured for each lender.
          </Text>
        </Alert>
      </Stack>

      <Modal
        opened={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editingLender ? 'Edit Lender' : 'Add New Lender'}
        radius={0}
      >
        <Stack gap="md">
          <TextInput
            label="Lender Name"
            placeholder="Bank Name"
            value={formData.name}
            onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
            radius={0}
            required
          />
          <TextInput
            label="API Endpoint"
            placeholder="https://api.bank.com/mortgage"
            value={formData.apiEndpoint}
            onChange={(e) => setFormData(prev => ({ ...prev, apiEndpoint: e.target.value }))}
            radius={0}
            required
          />
          <TextInput
            label="API Key"
            placeholder="Enter API key"
            value={formData.apiKey}
            onChange={(e) => setFormData(prev => ({ ...prev, apiKey: e.target.value }))}
            leftSection={<IconKey size={16} />}
            radius={0}
            type="password"
          />
          <Textarea
            label="Description"
            placeholder="Additional notes about this lender"
            value={formData.description}
            onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
            radius={0}
            rows={3}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setModalOpen(false)} radius={0}>
              Cancel
            </Button>
            <Button onClick={saveLender} radius={0}>
              {editingLender ? 'Update' : 'Add'} Lender
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Container>
  );
};

export default LenderIntegration;