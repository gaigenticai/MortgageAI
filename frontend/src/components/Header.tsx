/**
 * World-Class Header Component for MortgageAI - Mantine Version
 *
 * Features:
 * - Modern design with square borders
 * - Professional branding with animations
 * - Smooth micro-interactions
 * - Responsive navigation with modern UX
 */

import React, { useState } from 'react';
import {
  AppShell,
  Group,
  Button,
  Text,
  Box,
  Burger,
  Drawer,
  Stack,
  Switch,
  TextInput,
  Avatar,
  Badge,
  Transition,
  UnstyledButton,
  rem,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import {
  IconHome,
  IconFileText,
  IconUpload,
  IconShield,
  IconBrain,
  IconSettings,
  IconSearch,
  IconMenu2,
  IconScan,
} from '@tabler/icons-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useDemoMode } from '../contexts/DemoModeContext';

const Header: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isDemoMode, toggleDemoMode } = useDemoMode();
  const [opened, { toggle, close }] = useDisclosure(false);
  const [searchValue, setSearchValue] = useState('');

  const navigationItems = [
    { path: '/dashboard', label: 'Dashboard', icon: IconHome },
    { path: '/ai-mortgage-advisor-chat', label: 'AI Chat', icon: IconBrain },
    { path: '/document-ocr-processor', label: 'OCR Processor', icon: IconScan },
    { path: '/application', label: 'Application', icon: IconFileText },
    { path: '/documents', label: 'Documents', icon: IconUpload },
    { path: '/compliance', label: 'Compliance', icon: IconShield },
    { path: '/quality-control', label: 'Quality Control', icon: IconBrain },
    { path: '/settings', label: 'Settings', icon: IconSettings },
  ];

  const handleNavigation = (path: string) => {
    navigate(path);
    close();
  };

  const isActivePath = (path: string) => {
    return location.pathname === path || (path === '/dashboard' && location.pathname === '/');
  };

  return (
    <>
      <Box component="header" style={{ height: 80, 
        borderRadius: 0,
        border: 'none',
        borderBottom: '1px solid #E2E8F0',
        background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%)',
        backdropFilter: 'blur(20px)',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
      }}>
        <Group h="100%" px="xl" justify="space-between" style={{ maxWidth: '100%', width: '100%' }}>
          {/* Logo and Brand */}
          <Group>
            <Box
              style={{
                width: 48,
                height: 48,
                background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: 0,
                boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)',
              }}
            >
              <Text c="white" fw={700} size="xl">M</Text>
            </Box>
            <Box>
              <Text 
                size="xl" 
                fw={700} 
                style={{
                  background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                MortgageAI
              </Text>
              <Text size="xs" c="dimmed">Dutch AFM Compliant</Text>
            </Box>
          </Group>

          {/* Desktop Navigation */}
          <Group visibleFrom="lg" gap="xs">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = isActivePath(item.path);
              
              return (
                <Button
                  key={item.path}
                  variant={isActive ? "filled" : "subtle"}
                  color={isActive ? "indigo" : "gray"}
                  leftSection={<Icon size={18} />}
                  onClick={() => handleNavigation(item.path)}
                  radius={0}
                  style={{
                    fontWeight: isActive ? 600 : 500,
                    transition: 'all 0.2s ease',
                    transform: isActive ? 'translateY(-1px)' : 'none',
                  }}
                >
                  {item.label}
                </Button>
              );
            })}
          </Group>

          {/* Right Side Controls */}
          <Group>
            {/* Search */}
            <TextInput
              placeholder="Search..."
              leftSection={<IconSearch size={16} />}
              value={searchValue}
              onChange={(e) => setSearchValue(e.currentTarget.value)}
              radius={0}
              style={{ width: rem(200) }}
              visibleFrom="md"
            />

            {/* Demo Mode Toggle */}
            <Switch
              label="Demo Mode"
              checked={isDemoMode}
              onChange={toggleDemoMode}
              color="indigo"
              size="sm"
              visibleFrom="sm"
            />

            {/* Demo Mode Badge */}
            {isDemoMode && (
              <Badge color="amber" variant="filled" radius={0}>
                DEMO
              </Badge>
            )}

            {/* User Avatar */}
            <Avatar
              src={null}
              alt="User"
              color="indigo"
              radius={0}
              style={{ cursor: 'pointer' }}
            >
              U
            </Avatar>

            {/* Mobile Menu Burger */}
            <Burger
              opened={opened}
              onClick={toggle}
              hiddenFrom="lg"
              size="sm"
            />
          </Group>
        </Group>
      </Box>

      {/* Mobile Drawer */}
      <Drawer
        opened={opened}
        onClose={close}
        title="Navigation"
        padding="md"
        size="sm"
        radius={0}
        hiddenFrom="lg"
      >
        <Stack gap="xs">
          {/* Mobile Search */}
          <TextInput
            placeholder="Search..."
            leftSection={<IconSearch size={16} />}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
            radius={0}
            mb="md"
          />

          {/* Mobile Navigation Items */}
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = isActivePath(item.path);
            
            return (
              <Button
                key={item.path}
                variant={isActive ? "filled" : "subtle"}
                color={isActive ? "indigo" : "gray"}
                leftSection={<Icon size={18} />}
                onClick={() => handleNavigation(item.path)}
                radius={0}
                fullWidth
                justify="flex-start"
                style={{
                  fontWeight: isActive ? 600 : 500,
                }}
              >
                {item.label}
              </Button>
            );
          })}

          {/* Mobile Demo Mode Toggle */}
          <Switch
            label="Demo Mode"
            checked={isDemoMode}
            onChange={toggleDemoMode}
            color="indigo"
            size="sm"
            mt="md"
          />
        </Stack>
      </Drawer>
    </>
  );
};

export default Header;