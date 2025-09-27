/**
 * Dashboard - Mantine Version (Simplified)
 */

import React from 'react';
import { Container, Card, Text, Title, Stack, Button, Group } from '@mantine/core';
import { IconDashboard } from '@tabler/icons-react';

const Dashboard: React.FC = () => {
  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <IconDashboard size={32} />
          <Title order={1}>Dashboard</Title>
        </Group>

        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="lg">Overview</Title>
          <Text c="dimmed" ta="center" py="xl">
            Dashboard functionality will be implemented here
          </Text>
        </Card>
      </Stack>
    </Container>
  );
};

export default Dashboard;