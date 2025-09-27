/**
 * Before vs After AI Comparison Chart - Mantine Version
 *
 * Shows a bar chart comparing mortgage processing metrics before and after AI implementation
 * Demonstrates the value proposition of MortgageAI to prospects
 */

import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import {
  Card,
  Text,
  Box,
  Loader,
  Alert,
  Badge,
  Stack,
  Group,
} from '@mantine/core';
import { IconTrendingUp, IconChartBar } from '@tabler/icons-react';
import { apiClient } from '../services/apiClient';

interface ComparisonData {
  metric: string;
  before: number;
  after: number;
  improvement: number;
  unit: string;
}

const ComparisonChart: React.FC = () => {
  const [data, setData] = useState<ComparisonData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchComparisonData = async () => {
      try {
        setLoading(true);
        setError(null);
        // Use demo data since getComparisonData doesn't exist yet
        const comparisonData = [
          {
            metric: 'Processing Time',
            before: 45,
            after: 12,
            improvement: 73,
            unit: 'minutes'
          },
          {
            metric: 'Compliance Score',
            before: 78,
            after: 98,
            improvement: 26,
            unit: '%'
          },
          {
            metric: 'Error Rate',
            before: 15,
            after: 2,
            improvement: 87,
            unit: '%'
          },
          {
            metric: 'Customer Satisfaction',
            before: 72,
            after: 94,
            improvement: 31,
            unit: '%'
          }
        ];
        setData(comparisonData);
      } catch (err) {
        console.error('Error fetching comparison data:', err);
        setError('Failed to load comparison data');
        
        // Fallback demo data
        setData([
          {
            metric: 'Processing Time',
            before: 45,
            after: 12,
            improvement: 73,
            unit: 'minutes'
          },
          {
            metric: 'Compliance Score',
            before: 78,
            after: 98,
            improvement: 26,
            unit: '%'
          },
          {
            metric: 'Error Rate',
            before: 15,
            after: 2,
            improvement: 87,
            unit: '%'
          },
          {
            metric: 'Customer Satisfaction',
            before: 72,
            after: 94,
            improvement: 31,
            unit: '%'
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchComparisonData();
  }, []);

  const formatTooltipValue = (value: number, name: string, props: any) => {
    const unit = props.payload?.unit || '';
    return [`${value}${unit}`, name === 'before' ? 'Before AI' : 'After AI'];
  };

  if (loading) {
    return (
      <Box style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <Stack align="center" gap="md">
          <Loader size="lg" color="indigo" />
          <Text c="dimmed">Loading comparison data...</Text>
        </Stack>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert color="red" title="Error" radius={0}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header with improvement badges */}
      <Group justify="space-between" mb="lg">
        <Group>
          <IconChartBar size={24} color="#6366F1" />
          <Text size="lg" fw={600}>
            Performance Improvement
          </Text>
        </Group>
        
        <Group gap="xs">
          {data.slice(0, 2).map((item) => (
            <Badge
              key={item.metric}
              color="emerald"
              variant="filled"
              leftSection={<IconTrendingUp size={12} />}
              radius={0}
            >
              {item.improvement}% improvement
            </Badge>
          ))}
        </Group>
      </Group>

      {/* Chart */}
      <Box style={{ height: 400, width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
            barCategoryGap="20%"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
            <XAxis 
              dataKey="metric" 
              tick={{ fontSize: 12, fill: '#64748B' }}
              axisLine={{ stroke: '#CBD5E1' }}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: '#64748B' }}
              axisLine={{ stroke: '#CBD5E1' }}
            />
            <Tooltip
              formatter={formatTooltipValue}
              contentStyle={{
                backgroundColor: '#FFFFFF',
                border: '1px solid #E2E8F0',
                borderRadius: 0,
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              }}
              labelStyle={{ color: '#374151', fontWeight: 600 }}
            />
            <Legend 
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="rect"
            />
            <Bar 
              dataKey="before" 
              name="Before AI"
              fill="#EF4444"
              radius={[0, 0, 0, 0]}
            />
            <Bar 
              dataKey="after" 
              name="After AI"
              fill="#10B981"
              radius={[0, 0, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </Box>

      {/* Summary metrics */}
      <Group justify="center" mt="lg" gap="xl">
        <Box ta="center">
          <Text size="xl" fw={700} c="emerald">
            73%
          </Text>
          <Text size="sm" c="dimmed">
            Faster Processing
          </Text>
        </Box>
        <Box ta="center">
          <Text size="xl" fw={700} c="indigo">
            98%
          </Text>
          <Text size="sm" c="dimmed">
            Compliance Score
          </Text>
        </Box>
        <Box ta="center">
          <Text size="xl" fw={700} c="pink">
            87%
          </Text>
          <Text size="sm" c="dimmed">
            Error Reduction
          </Text>
        </Box>
      </Group>
    </Box>
  );
};

export default ComparisonChart;