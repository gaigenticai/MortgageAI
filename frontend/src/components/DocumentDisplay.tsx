/**
 * Document Display Component - Mantine Version
 */

import React from 'react';
import { Card, Text, Box, Group, Badge } from '@mantine/core';
import { IconFileText } from '@tabler/icons-react';

interface DocumentDisplayProps {
  documents?: any[];
}

const DocumentDisplay: React.FC<DocumentDisplayProps> = ({ documents = [] }) => {
  return (
    <Card radius={0} shadow="sm" padding="lg">
      <Group mb="md">
        <IconFileText size={24} />
        <Text size="lg" fw={600}>Documents</Text>
      </Group>
      
      {documents.length > 0 ? (
        <Box>
          {documents.map((doc, index) => (
            <Box key={index} p="sm" style={{ border: '1px solid #E2E8F0', marginBottom: 8 }}>
              <Group justify="space-between">
                <Text size="sm">{doc.name || `Document ${index + 1}`}</Text>
                <Badge color="green" radius={0}>Processed</Badge>
              </Group>
            </Box>
          ))}
        </Box>
      ) : (
        <Text c="dimmed" ta="center" py="xl">
          No documents uploaded
        </Text>
      )}
    </Card>
  );
};

export default DocumentDisplay;