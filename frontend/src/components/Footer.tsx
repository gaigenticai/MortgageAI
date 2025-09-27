/**
 * Professional Footer Component for MortgageAI - Mantine Version
 */

import React from 'react';
import {
  Box,
  Container,
  Text,
  Group,
  Stack,
  Anchor,
  Divider,
  ActionIcon,
  Grid,
  Badge,
} from '@mantine/core';
import {
  IconBrandGithub,
  IconBrandLinkedin,
  IconMail,
  IconPhone,
  IconMapPin,
  IconBrain,
  IconShield,
  IconCertificate,
} from '@tabler/icons-react';

const Footer: React.FC = () => {
  return (
    <Box
      component="footer"
      style={{
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
        color: 'white',
        borderRadius: 0,
        marginTop: 'auto',
      }}
    >
      <Container size="xl" py="xl">
        <Grid>
          {/* Company Info */}
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Stack gap="md">
              <Group>
                <Box
                  style={{
                    width: 40,
                    height: 40,
                    background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: 0,
                  }}
                >
                  <IconBrain size={24} color="white" />
                </Box>
                <Text size="xl" fw={700} c="white">
                  MortgageAI
                </Text>
              </Group>
              
              <Text size="sm" c="gray.3" style={{ lineHeight: 1.6 }}>
                Advanced AI-powered mortgage advisory platform specialized for the Dutch market. 
                Fully compliant with AFM regulations and integrated with major Dutch lenders.
              </Text>

              <Group gap="xs">
                <Badge color="emerald" variant="filled" radius={0}>
                  <IconShield size={12} style={{ marginRight: 4 }} />
                  AFM Compliant
                </Badge>
                <Badge color="indigo" variant="filled" radius={0}>
                  <IconCertificate size={12} style={{ marginRight: 4 }} />
                  NHG Certified
                </Badge>
              </Group>
            </Stack>
          </Grid.Col>

          {/* Quick Links */}
          <Grid.Col span={{ base: 12, sm: 6, md: 2 }}>
            <Stack gap="sm">
              <Text fw={600} c="white" size="sm">
                Platform
              </Text>
              <Stack gap="xs">
                <Anchor href="/dashboard" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Dashboard
                </Anchor>
                <Anchor href="/application" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Applications
                </Anchor>
                <Anchor href="/compliance" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Compliance
                </Anchor>
                <Anchor href="/quality-control" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Quality Control
                </Anchor>
              </Stack>
            </Stack>
          </Grid.Col>

          {/* Dutch Services */}
          <Grid.Col span={{ base: 12, sm: 6, md: 2 }}>
            <Stack gap="sm">
              <Text fw={600} c="white" size="sm">
                Dutch Services
              </Text>
              <Stack gap="xs">
                <Anchor href="/afm-compliance-advisor" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  AFM Compliance
                </Anchor>
                <Anchor href="/bkr-credit-check" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  BKR Credit Check
                </Anchor>
                <Anchor href="/nhg-eligibility-check" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  NHG Eligibility
                </Anchor>
                <Anchor href="/dutch-market-insights" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Market Insights
                </Anchor>
              </Stack>
            </Stack>
          </Grid.Col>

          {/* Support */}
          <Grid.Col span={{ base: 12, sm: 6, md: 2 }}>
            <Stack gap="sm">
              <Text fw={600} c="white" size="sm">
                Support
              </Text>
              <Stack gap="xs">
                <Anchor href="/docs" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Documentation
                </Anchor>
                <Anchor href="/api" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  API Reference
                </Anchor>
                <Anchor href="/help" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Help Center
                </Anchor>
                <Anchor href="/contact" c="gray.3" size="sm" style={{ textDecoration: 'none' }}>
                  Contact Us
                </Anchor>
              </Stack>
            </Stack>
          </Grid.Col>

          {/* Contact Info */}
          <Grid.Col span={{ base: 12, sm: 6, md: 2 }}>
            <Stack gap="sm">
              <Text fw={600} c="white" size="sm">
                Contact
              </Text>
              <Stack gap="xs">
                <Group gap="xs">
                  <IconMail size={16} color="#9CA3AF" />
                  <Text size="sm" c="gray.3">
                    info@mortgageai.nl
                  </Text>
                </Group>
                <Group gap="xs">
                  <IconPhone size={16} color="#9CA3AF" />
                  <Text size="sm" c="gray.3">
                    +31 20 123 4567
                  </Text>
                </Group>
                <Group gap="xs">
                  <IconMapPin size={16} color="#9CA3AF" />
                  <Text size="sm" c="gray.3">
                    Amsterdam, Netherlands
                  </Text>
                </Group>
              </Stack>

              {/* Social Links */}
              <Group gap="xs" mt="sm">
                <ActionIcon variant="subtle" color="gray" radius={0}>
                  <IconBrandGithub size={18} />
                </ActionIcon>
                <ActionIcon variant="subtle" color="gray" radius={0}>
                  <IconBrandLinkedin size={18} />
                </ActionIcon>
                <ActionIcon variant="subtle" color="gray" radius={0}>
                  <IconMail size={18} />
                </ActionIcon>
              </Group>
            </Stack>
          </Grid.Col>
        </Grid>

        <Divider my="xl" color="gray.7" />

        {/* Bottom Section */}
        <Group justify="space-between" align="center">
          <Text size="sm" c="gray.4">
            © 2024 MortgageAI by Gaigentic. All rights reserved.
          </Text>
          
          <Group gap="md">
            <Anchor href="/privacy" c="gray.4" size="sm" style={{ textDecoration: 'none' }}>
              Privacy Policy
            </Anchor>
            <Anchor href="/terms" c="gray.4" size="sm" style={{ textDecoration: 'none' }}>
              Terms of Service
            </Anchor>
            <Anchor href="/cookies" c="gray.4" size="sm" style={{ textDecoration: 'none' }}>
              Cookie Policy
            </Anchor>
          </Group>
        </Group>

        {/* Compliance Notice */}
        <Box
          mt="md"
          p="md"
          style={{
            background: 'rgba(99, 102, 241, 0.1)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            borderRadius: 0,
          }}
        >
          <Text size="xs" c="gray.3" ta="center">
            <IconShield size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
            This platform is fully compliant with Dutch AFM (Autoriteit Financiële Markten) regulations. 
            All mortgage advice is provided in accordance with Wft (Wet op het financieel toezicht) requirements.
          </Text>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;