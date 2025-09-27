/**
 * Document Preview Component - Mantine Version
 */

import React from 'react';
import { Card, Text, Box, Group, Button } from '@mantine/core';
import { IconEye, IconDownload } from '@tabler/icons-react';

interface DocumentPreviewProps {
  document?: any;
}

const DocumentPreview: React.FC<DocumentPreviewProps> = ({ document }) => {
  return (
    <Card radius={0} shadow="sm" padding="lg">
      <Group justify="space-between" mb="md">
        <Text size="lg" fw={600}>Document Preview</Text>
        <Group>
          <Button variant="light" leftSection={<IconEye size={16} />} radius={0}>
            View
          </Button>
          <Button variant="light" leftSection={<IconDownload size={16} />} radius={0}>
            Download
          </Button>
        </Group>
      </Group>
      
      {document ? (
        <Box>
          <Text size="sm" c="dimmed">
            {document.name || 'Document'}
          </Text>
        </Box>
      ) : (
        <Text c="dimmed" ta="center" py="xl">
          No document selected
        </Text>
      )}
    </Card>
  );
};

export default DocumentPreview;