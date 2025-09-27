/**
 * AFM Compliance Advisor - Mantine Version (Simplified)
 */

import React from 'react';
import { Container, Card, Text, Title, Stack, Button, Group } from '@mantine/core';
import { IconShield } from '@tabler/icons-react';

const AFMComplianceAdvisor: React.FC = () => {
  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <IconShield size={32} />
          <Title order={1}>AFM Compliance Advisor</Title>
        </Group>

        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="lg">Compliance Guidance</Title>
          <Text c="dimmed" ta="center" py="xl">
            AFM Compliance Advisor functionality will be implemented here
          </Text>
        </Card>
      </Stack>
    </Container>
  );
};

export default AFMComplianceAdvisor;