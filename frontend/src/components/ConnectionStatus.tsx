/**
 * Connection Status Component
 * 
 * Shows the current connection status to various services
 */

import React, { useState, useEffect } from 'react';
import {
  Paper,
  Group,
  Badge,
  Text,
  Stack,
  Button,
  Title,
  Divider,
  Alert,
  Code
} from '@mantine/core';
import {
  IconWifi,
  IconWifiOff,
  IconRefresh,
  IconAlertTriangle,
  IconServer,
  IconDatabase,
  IconBrain
} from '@tabler/icons-react';

interface ConnectionStatusProps {
  className?: string;
}

interface ServiceStatus {
  name: string;
  status: 'connected' | 'disconnected' | 'error' | 'checking';
  url: string;
  lastChecked?: Date;
  error?: string;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ className }) => {
  const [services, setServices] = useState<ServiceStatus[]>([
    {
      name: 'Backend API',
      status: 'checking',
      url: 'http://localhost:3000',
    },
    {
      name: 'WebSocket Chat',
      status: 'checking',
      url: 'ws://localhost:8005',
    },
    {
      name: 'AI Agents',
      status: 'checking',
      url: 'http://localhost:8000',
    }
  ]);

  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkServiceStatus = async (service: ServiceStatus): Promise<ServiceStatus> => {
    try {
      if (service.name === 'WebSocket Chat') {
        // Test WebSocket connection
        return new Promise((resolve) => {
          const ws = new WebSocket(service.url + '?token=demo-token');
          
          const timeout = setTimeout(() => {
            ws.close();
            resolve({
              ...service,
              status: 'disconnected',
              lastChecked: new Date(),
              error: 'Connection timeout'
            });
          }, 5000);

          ws.onopen = () => {
            clearTimeout(timeout);
            ws.close();
            resolve({
              ...service,
              status: 'connected',
              lastChecked: new Date(),
              error: undefined
            });
          };

          ws.onerror = () => {
            clearTimeout(timeout);
            resolve({
              ...service,
              status: 'error',
              lastChecked: new Date(),
              error: 'WebSocket connection failed'
            });
          };
        });
      } else {
        // Test HTTP services
        const response = await fetch(service.url + '/health', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (response.ok) {
          return {
            ...service,
            status: 'connected',
            lastChecked: new Date(),
            error: undefined
          };
        } else {
          return {
            ...service,
            status: 'error',
            lastChecked: new Date(),
            error: `HTTP ${response.status}: ${response.statusText}`
          };
        }
      }
    } catch (error) {
      return {
        ...service,
        status: 'disconnected',
        lastChecked: new Date(),
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  };

  const checkAllServices = async () => {
    setIsRefreshing(true);
    
    const updatedServices = await Promise.all(
      services.map(service => checkServiceStatus(service))
    );
    
    setServices(updatedServices);
    setIsRefreshing(false);
  };

  useEffect(() => {
    checkAllServices();
  }, []);

  const getStatusColor = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'connected':
        return 'green';
      case 'disconnected':
        return 'orange';
      case 'error':
        return 'red';
      case 'checking':
        return 'blue';
      default:
        return 'gray';
    }
  };

  const getStatusIcon = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'connected':
        return <IconWifi size={16} />;
      case 'disconnected':
      case 'error':
        return <IconWifiOff size={16} />;
      case 'checking':
        return <IconRefresh size={16} />;
      default:
        return <IconServer size={16} />;
    }
  };

  const getServiceIcon = (serviceName: string) => {
    switch (serviceName) {
      case 'Backend API':
        return <IconServer size={20} />;
      case 'WebSocket Chat':
        return <IconBrain size={20} />;
      case 'AI Agents':
        return <IconDatabase size={20} />;
      default:
        return <IconServer size={20} />;
    }
  };

  const hasErrors = services.some(s => s.status === 'error' || s.status === 'disconnected');

  return (
    <Paper className={className} p="md" radius="md" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={4}>Service Status</Title>
        <Button
          size="xs"
          variant="outline"
          leftSection={<IconRefresh size={14} />}
          onClick={checkAllServices}
          loading={isRefreshing}
        >
          Refresh
        </Button>
      </Group>

      {hasErrors && (
        <Alert
          icon={<IconAlertTriangle size={16} />}
          color="orange"
          mb="md"
        >
          <Text size="sm">
            Some services are not responding. The AI chat may not work properly until all services are connected.
          </Text>
        </Alert>
      )}

      <Stack gap="sm">
        {services.map((service, index) => (
          <Group key={index} justify="space-between" wrap="nowrap">
            <Group gap="sm" style={{ flex: 1 }}>
              {getServiceIcon(service.name)}
              <div style={{ flex: 1 }}>
                <Text fw={500} size="sm">{service.name}</Text>
                <Text size="xs" c="dimmed">
                  <Code>{service.url}</Code>
                </Text>
                {service.error && (
                  <Text size="xs" c="red">
                    {service.error}
                  </Text>
                )}
              </div>
            </Group>
            <Badge
              color={getStatusColor(service.status)}
              variant="light"
              leftSection={getStatusIcon(service.status)}
            >
              {service.status}
            </Badge>
          </Group>
        ))}
      </Stack>

      <Divider my="md" />

      <Text size="xs" c="dimmed">
        Last checked: {services[0]?.lastChecked?.toLocaleTimeString() || 'Never'}
      </Text>
    </Paper>
  );
};

export default ConnectionStatus;