/**
 * NHG Eligibility Check - Full Mantine Implementation
 */

import React, { useState } from 'react';
import {
  Container,
  Card,
  TextInput,
  Button,
  Title,
  Group,
  Stack,
  Grid,
  Alert,
  Badge,
  ThemeIcon,
  Text,
  NumberInput,
  Select,
  Checkbox,
  Divider,
  SimpleGrid,
  List,
} from '@mantine/core';
import {
  IconShield,
  IconCheck,
  IconX,
  IconInfoCircle,
  IconCalculator,
  IconHome,
  IconCurrencyEuro,
  IconUser,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

const NHGEligibilityCheck: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    purchasePrice: 0,
    income: 0,
    propertyType: '',
    isFirstTime: false,
    hasPartner: false,
    partnerIncome: 0,
  });
  const [results, setResults] = useState<any>(null);

  const checkEligibility = async () => {
    if (!formData.purchasePrice || !formData.income) {
      notifications.show({
        title: 'Missing Information',
        message: 'Please fill in all required fields',
        color: 'red',
        icon: <IconX size={16} />,
      });
      return;
    }

    setLoading(true);
    try {
      // Mock NHG eligibility check
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const totalIncome = formData.income + (formData.hasPartner ? formData.partnerIncome : 0);
      const nhgLimit = 435000; // 2024 NHG limit
      const isEligible = formData.purchasePrice <= nhgLimit && totalIncome >= 30000;
      
      setResults({
        eligible: isEligible,
        nhgLimit,
        purchasePrice: formData.purchasePrice,
        totalIncome,
        guaranteeCoverage: isEligible ? Math.min(formData.purchasePrice * 0.9, 391500) : 0,
        monthlyPremium: isEligible ? Math.round((formData.purchasePrice * 0.007) / 12) : 0,
        requirements: [
          { text: `Purchase price ≤ €${nhgLimit.toLocaleString()}`, met: formData.purchasePrice <= nhgLimit },
          { text: 'Minimum income €30,000', met: totalIncome >= 30000 },
          { text: 'Property for own residence', met: true },
          { text: 'Maximum 100% financing', met: true },
        ]
      });

      notifications.show({
        title: 'Eligibility Check Complete',
        message: isEligible ? 'You are eligible for NHG!' : 'Unfortunately, you do not meet NHG requirements',
        color: isEligible ? 'green' : 'red',
        icon: isEligible ? <IconCheck size={16} /> : <IconX size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Check Failed',
        message: 'Failed to perform NHG eligibility check',
        color: 'red',
        icon: <IconX size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <ThemeIcon size="xl" radius={0} color="green">
            <IconShield size={32} />
          </ThemeIcon>
          <div>
            <Title order={1}>NHG Eligibility Check</Title>
            <Text c="dimmed">National Mortgage Guarantee (Nationale Hypotheek Garantie) eligibility assessment</Text>
          </div>
        </Group>

        <Alert color="blue" icon={<IconInfoCircle size={16} />} radius={0}>
          <Text size="sm">
            NHG provides protection against residual debt if you can no longer pay your mortgage. 
            The 2024 NHG limit is €435,000 for existing homes and €459,000 for new builds.
          </Text>
        </Alert>

        <Card radius={0} shadow="sm" padding="lg">
          <Title order={3} mb="md">Property & Income Information</Title>
          <Grid>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <NumberInput
                label="Purchase Price"
                placeholder="350000"
                value={formData.purchasePrice}
                onChange={(value) => setFormData(prev => ({ ...prev, purchasePrice: Number(value) || 0 }))}
                leftSection={<IconCurrencyEuro size={16} />}
                thousandSeparator=","
                radius={0}
                required
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Select
                label="Property Type"
                placeholder="Select property type"
                value={formData.propertyType}
                onChange={(value) => setFormData(prev => ({ ...prev, propertyType: value || '' }))}
                data={[
                  { value: 'existing', label: 'Existing Home' },
                  { value: 'new', label: 'New Construction' },
                  { value: 'apartment', label: 'Apartment' },
                ]}
                leftSection={<IconHome size={16} />}
                radius={0}
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <NumberInput
                label="Annual Gross Income"
                placeholder="50000"
                value={formData.income}
                onChange={(value) => setFormData(prev => ({ ...prev, income: Number(value) || 0 }))}
                leftSection={<IconUser size={16} />}
                thousandSeparator=","
                radius={0}
                required
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Stack gap="sm" mt={25}>
                <Checkbox
                  label="First-time buyer"
                  checked={formData.isFirstTime}
                  onChange={(e) => setFormData(prev => ({ ...prev, isFirstTime: e.target.checked }))}
                  radius={0}
                />
                <Checkbox
                  label="I have a partner"
                  checked={formData.hasPartner}
                  onChange={(e) => setFormData(prev => ({ ...prev, hasPartner: e.target.checked }))}
                  radius={0}
                />
              </Stack>
            </Grid.Col>
            {formData.hasPartner && (
              <Grid.Col span={{ base: 12, md: 6 }}>
                <NumberInput
                  label="Partner's Annual Gross Income"
                  placeholder="40000"
                  value={formData.partnerIncome}
                  onChange={(value) => setFormData(prev => ({ ...prev, partnerIncome: Number(value) || 0 }))}
                  leftSection={<IconUser size={16} />}
                  thousandSeparator=","
                  radius={0}
                />
              </Grid.Col>
            )}
          </Grid>
          <Group justify="flex-end" mt="lg">
            <Button
              leftSection={<IconCalculator size={16} />}
              onClick={checkEligibility}
              loading={loading}
              radius={0}
            >
              Check NHG Eligibility
            </Button>
          </Group>
        </Card>

        {results && (
          <>
            <Card radius={0} shadow="sm" padding="lg">
              <Group justify="space-between" mb="md">
                <Title order={3}>Eligibility Result</Title>
                <Badge 
                  color={results.eligible ? 'green' : 'red'} 
                  size="lg" 
                  radius={0}
                  leftSection={results.eligible ? <IconCheck size={16} /> : <IconX size={16} />}
                >
                  {results.eligible ? 'Eligible' : 'Not Eligible'}
                </Badge>
              </Group>
              
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg">
                <div>
                  <Text size="sm" c="dimmed">Purchase Price</Text>
                  <Title order={4}>€{results.purchasePrice.toLocaleString()}</Title>
                </div>
                <div>
                  <Text size="sm" c="dimmed">NHG Limit</Text>
                  <Title order={4}>€{results.nhgLimit.toLocaleString()}</Title>
                </div>
                <div>
                  <Text size="sm" c="dimmed">Total Income</Text>
                  <Title order={4}>€{results.totalIncome.toLocaleString()}</Title>
                </div>
              </SimpleGrid>
            </Card>

            {results.eligible && (
              <Card radius={0} shadow="sm" padding="lg">
                <Title order={3} mb="md">NHG Benefits</Title>
                <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
                  <div>
                    <Text size="sm" c="dimmed">Guarantee Coverage</Text>
                    <Title order={4} c="green">€{results.guaranteeCoverage.toLocaleString()}</Title>
                    <Text size="xs" c="dimmed">Up to 90% of purchase price</Text>
                  </div>
                  <div>
                    <Text size="sm" c="dimmed">Monthly Premium</Text>
                    <Title order={4} c="blue">€{results.monthlyPremium}</Title>
                    <Text size="xs" c="dimmed">0.7% annually of loan amount</Text>
                  </div>
                </SimpleGrid>
              </Card>
            )}

            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Requirements Check</Title>
              <List spacing="sm">
                {results.requirements.map((req: any, index: number) => (
                  <List.Item
                    key={index}
                    icon={
                      <ThemeIcon 
                        color={req.met ? 'green' : 'red'} 
                        size="sm" 
                        radius={0}
                      >
                        {req.met ? <IconCheck size={12} /> : <IconX size={12} />}
                      </ThemeIcon>
                    }
                  >
                    <Text c={req.met ? 'green' : 'red'}>{req.text}</Text>
                  </List.Item>
                ))}
              </List>
            </Card>

            <Alert 
              color={results.eligible ? 'green' : 'orange'} 
              icon={<IconInfoCircle size={16} />} 
              radius={0}
            >
              <Text size="sm">
                {results.eligible 
                  ? 'You meet the basic NHG requirements. Contact a mortgage advisor for the complete application process.'
                  : 'You do not currently meet NHG requirements. Consider adjusting your purchase price or improving your income situation.'
                }
              </Text>
            </Alert>
          </>
        )}
      </Stack>
    </Container>
  );
};

export default NHGEligibilityCheck;