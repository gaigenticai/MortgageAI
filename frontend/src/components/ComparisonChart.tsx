/**
 * Before vs After AI Comparison Chart
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
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import { TrendingUp, Assessment } from '@mui/icons-material';
import { apiClient } from '../services/apiClient';

interface ComparisonData {
  name: string;
  before: number;
  after: number;
  improvement: number;
}

interface ComparisonChartProps {
  id?: string;
}

const ComparisonChart: React.FC<ComparisonChartProps> = ({ id }) => {
  const [data, setData] = useState<ComparisonData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadComparisonData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Load before AI data from local fixture
        const beforeData = await import('../demo-data/comparison.json');

        // Load after AI data from API (or demo data if in demo mode)
        const afterMetrics = await apiClient.getDashboardMetrics();

        // Combine data for chart
        const combinedData: ComparisonData[] = beforeData.default.metrics.map((metric: any) => ({
          name: metric.name,
          before: metric.before,
          after: afterMetrics[metric.name.toLowerCase().replace(/\s+/g, '_') as keyof typeof afterMetrics] || metric.after,
          improvement: metric.improvement
        }));

        setData(combinedData);
      } catch (err) {
        console.error('Failed to load comparison data:', err);
        setError('Failed to load comparison data');
      } finally {
        setLoading(false);
      }
    };

    loadComparisonData();
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const beforeValue = payload.find((p: any) => p.dataKey === 'before')?.value;
      const afterValue = payload.find((p: any) => p.dataKey === 'after')?.value;
      const improvement = afterValue - beforeValue;

      return (
        <Box
          sx={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            p: 2,
            border: '1px solid #e0e0e0',
            borderRadius: 2,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
            minWidth: 200
          }}
        >
          <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
            {label}
          </Typography>
          <Typography variant="body2" sx={{ color: '#8884d8', mb: 0.5 }}>
            Before AI: {beforeValue}%
          </Typography>
          <Typography variant="body2" sx={{ color: '#82ca9d', mb: 0.5 }}>
            After AI: {afterValue}%
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: improvement > 0 ? '#4caf50' : '#f44336',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 0.5
            }}
          >
            <TrendingUp fontSize="small" />
            Improvement: +{improvement.toFixed(1)}%
          </Typography>
        </Box>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card id={id} sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CardContent>
          <CircularProgress size={40} />
          <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
            Loading comparison data...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card id={id} sx={{ height: 400 }}>
        <CardContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card id={id} sx={{ height: 400 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Assessment sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            AI Impact Comparison
          </Typography>
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Mortgage processing metrics before and after AI implementation
        </Typography>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
            barCategoryGap="25%"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              label={{
                value: 'Percentage (%)',
                angle: -90,
                position: 'insideLeft',
                style: { textAnchor: 'middle', fontSize: 12 }
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
              iconType="rect"
            />
            <Bar
              dataKey="before"
              name="Before AI"
              radius={[2, 2, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell key={`before-${index}`} fill="#8884d8" />
              ))}
            </Bar>
            <Bar
              dataKey="after"
              name="After AI"
              radius={[2, 2, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell key={`after-${index}`} fill="#82ca9d" />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ComparisonChart;
