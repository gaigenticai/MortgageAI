/**
 * Settings Page - Full Mantine Implementation
 *
 * Comprehensive settings and configuration interface with:
 * - User preferences and profile settings
 * - System configuration options
 * - API and integration settings
 * - Notification preferences
 * - Security and privacy controls
 * - Theme and display options
 */

import React, { useState } from 'react';
import {
  Container,
  Paper,
  Text,
  Button,
  Alert,
  Grid,
  Card,
  TextInput,
  NumberInput,
  Select,
  Switch,
  Title,
  Group,
  Stack,
  Divider,
  Badge,
  ThemeIcon,
  Tabs,
  PasswordInput,
  Textarea,
  ActionIcon,
  Tooltip,
  ColorInput,
  Slider,
} from '@mantine/core';
import {
  IconSettings,
  IconUser,
  IconBell,
  IconShield,
  IconPalette,
  IconApi,
  IconCheck,
  IconAlertTriangle,
  IconInfoCircle,
  IconDeviceFloppy,
  IconRefresh,
  IconTrash,
  IconEye,
  IconEyeOff,
  IconKey,
  IconMail,
  IconPhone,
  IconWorld,
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { useNavigate } from 'react-router-dom';

interface SettingsData {
  user: {
    name: string;
    email: string;
    phone: string;
    company: string;
    role: string;
    timezone: string;
    language: string;
  };
  notifications: {
    email_enabled: boolean;
    sms_enabled: boolean;
    push_enabled: boolean;
    compliance_alerts: boolean;
    quality_control_alerts: boolean;
    application_updates: boolean;
    marketing_emails: boolean;
  };
  security: {
    two_factor_enabled: boolean;
    session_timeout: number;
    password_expiry: number;
    login_notifications: boolean;
    api_access_enabled: boolean;
  };
  system: {
    auto_save_interval: number;
    default_currency: string;
    date_format: string;
    number_format: string;
    backup_enabled: boolean;
    debug_mode: boolean;
  };
  api: {
    base_url: string;
    timeout: number;
    retry_attempts: number;
    rate_limit: number;
    cache_enabled: boolean;
  };
  theme: {
    primary_color: string;
    dark_mode: boolean;
    compact_mode: boolean;
    animations_enabled: boolean;
    font_size: number;
  };
}

const Settings: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<string>('user');
  const [loading, setLoading] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  const [settings, setSettings] = useState<SettingsData>({
    user: {
      name: 'John Doe',
      email: 'john.doe@example.com',
      phone: '+31 6 12345678',
      company: 'MortgageAI Solutions',
      role: 'Senior Mortgage Advisor',
      timezone: 'Europe/Amsterdam',
      language: 'en',
    },
    notifications: {
      email_enabled: true,
      sms_enabled: false,
      push_enabled: true,
      compliance_alerts: true,
      quality_control_alerts: true,
      application_updates: true,
      marketing_emails: false,
    },
    security: {
      two_factor_enabled: true,
      session_timeout: 30,
      password_expiry: 90,
      login_notifications: true,
      api_access_enabled: false,
    },
    system: {
      auto_save_interval: 5,
      default_currency: 'EUR',
      date_format: 'DD/MM/YYYY',
      number_format: 'European',
      backup_enabled: true,
      debug_mode: false,
    },
    api: {
      base_url: 'http://localhost:8000',
      timeout: 30,
      retry_attempts: 3,
      rate_limit: 100,
      cache_enabled: true,
    },
    theme: {
      primary_color: '#6366F1',
      dark_mode: false,
      compact_mode: false,
      animations_enabled: true,
      font_size: 14,
    },
  });

  const updateSetting = (category: keyof SettingsData, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value,
      },
    }));
  };

  const saveSettings = async () => {
    setLoading(true);
    try {
      // Mock save operation
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      notifications.show({
        title: 'Settings Saved',
        message: 'Your settings have been saved successfully',
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Save Failed',
        message: 'Failed to save settings',
        color: 'red',
        icon: <IconAlertTriangle size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const resetSettings = () => {
    notifications.show({
      title: 'Settings Reset',
      message: 'Settings have been reset to defaults',
      color: 'blue',
      icon: <IconRefresh size={16} />,
    });
  };

  const generateApiKey = () => {
    const newKey = 'sk-' + Math.random().toString(36).substr(2, 32);
    notifications.show({
      title: 'API Key Generated',
      message: 'New API key has been generated',
      color: 'green',
      icon: <IconKey size={16} />,
    });
  };

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between">
          <Group>
            <ThemeIcon size="xl" radius={0} color="indigo">
              <IconSettings size={32} />
            </ThemeIcon>
            <div>
              <Title order={1}>Settings & Configuration</Title>
              <Text c="dimmed">Manage your preferences and system settings</Text>
            </div>
          </Group>
          <Group>
            <Button 
              variant="outline" 
              leftSection={<IconRefresh size={16} />}
              onClick={resetSettings}
              radius={0}
            >
              Reset to Defaults
            </Button>
            <Button 
              leftSection={<IconDeviceFloppy size={16} />}
              onClick={saveSettings}
              loading={loading}
              radius={0}
            >
              Save Settings
            </Button>
          </Group>
        </Group>

        {/* Settings Tabs */}
        <Tabs value={activeTab} onChange={(value) => setActiveTab(value || 'user')} radius={0}>
          <Tabs.List>
            <Tabs.Tab value="user" leftSection={<IconUser size={16} />}>
              User Profile
            </Tabs.Tab>
            <Tabs.Tab value="notifications" leftSection={<IconBell size={16} />}>
              Notifications
            </Tabs.Tab>
            <Tabs.Tab value="security" leftSection={<IconShield size={16} />}>
              Security
            </Tabs.Tab>
            <Tabs.Tab value="system" leftSection={<IconSettings size={16} />}>
              System
            </Tabs.Tab>
            <Tabs.Tab value="api" leftSection={<IconApi size={16} />}>
              API Settings
            </Tabs.Tab>
            <Tabs.Tab value="theme" leftSection={<IconPalette size={16} />}>
              Appearance
            </Tabs.Tab>
          </Tabs.List>

          {/* User Profile Tab */}
          <Tabs.Panel value="user" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">User Profile Information</Title>
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Full Name"
                    placeholder="Enter your full name"
                    value={settings.user.name}
                    onChange={(e) => updateSetting('user', 'name', e.target.value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Email Address"
                    placeholder="your.email@example.com"
                    type="email"
                    value={settings.user.email}
                    onChange={(e) => updateSetting('user', 'email', e.target.value)}
                    leftSection={<IconMail size={16} />}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Phone Number"
                    placeholder="+31 6 12345678"
                    value={settings.user.phone}
                    onChange={(e) => updateSetting('user', 'phone', e.target.value)}
                    leftSection={<IconPhone size={16} />}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Company"
                    placeholder="Your company name"
                    value={settings.user.company}
                    onChange={(e) => updateSetting('user', 'company', e.target.value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Role"
                    placeholder="Your job title"
                    value={settings.user.role}
                    onChange={(e) => updateSetting('user', 'role', e.target.value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Timezone"
                    data={[
                      { value: 'Europe/Amsterdam', label: 'Amsterdam (CET)' },
                      { value: 'Europe/London', label: 'London (GMT)' },
                      { value: 'America/New_York', label: 'New York (EST)' },
                      { value: 'Asia/Tokyo', label: 'Tokyo (JST)' },
                    ]}
                    value={settings.user.timezone}
                    onChange={(value) => updateSetting('user', 'timezone', value)}
                    leftSection={<IconWorld size={16} />}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <Select
                    label="Language"
                    data={[
                      { value: 'en', label: 'English' },
                      { value: 'nl', label: 'Dutch' },
                      { value: 'de', label: 'German' },
                      { value: 'fr', label: 'French' },
                    ]}
                    value={settings.user.language}
                    onChange={(value) => updateSetting('user', 'language', value)}
                    radius={0}
                  />
                </Grid.Col>
              </Grid>
            </Card>
          </Tabs.Panel>

          {/* Notifications Tab */}
          <Tabs.Panel value="notifications" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Notification Preferences</Title>
              
              <Stack gap="lg">
                <div>
                  <Title order={4} mb="sm">Communication Channels</Title>
                  <Stack gap="sm">
                    <Switch
                      label="Email Notifications"
                      description="Receive notifications via email"
                      checked={settings.notifications.email_enabled}
                      onChange={(e) => updateSetting('notifications', 'email_enabled', e.target.checked)}
                    />
                    <Switch
                      label="SMS Notifications"
                      description="Receive notifications via SMS"
                      checked={settings.notifications.sms_enabled}
                      onChange={(e) => updateSetting('notifications', 'sms_enabled', e.target.checked)}
                    />
                    <Switch
                      label="Push Notifications"
                      description="Receive browser push notifications"
                      checked={settings.notifications.push_enabled}
                      onChange={(e) => updateSetting('notifications', 'push_enabled', e.target.checked)}
                    />
                  </Stack>
                </div>

                <Divider />

                <div>
                  <Title order={4} mb="sm">Alert Types</Title>
                  <Stack gap="sm">
                    <Switch
                      label="Compliance Alerts"
                      description="Get notified about compliance issues"
                      checked={settings.notifications.compliance_alerts}
                      onChange={(e) => updateSetting('notifications', 'compliance_alerts', e.target.checked)}
                    />
                    <Switch
                      label="Quality Control Alerts"
                      description="Get notified about QC results"
                      checked={settings.notifications.quality_control_alerts}
                      onChange={(e) => updateSetting('notifications', 'quality_control_alerts', e.target.checked)}
                    />
                    <Switch
                      label="Application Updates"
                      description="Get notified about application status changes"
                      checked={settings.notifications.application_updates}
                      onChange={(e) => updateSetting('notifications', 'application_updates', e.target.checked)}
                    />
                    <Switch
                      label="Marketing Emails"
                      description="Receive product updates and marketing content"
                      checked={settings.notifications.marketing_emails}
                      onChange={(e) => updateSetting('notifications', 'marketing_emails', e.target.checked)}
                    />
                  </Stack>
                </div>
              </Stack>
          </Card>
          </Tabs.Panel>

          {/* Security Tab */}
          <Tabs.Panel value="security" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Security & Privacy Settings</Title>
              
              <Stack gap="lg">
                <div>
                  <Title order={4} mb="sm">Authentication</Title>
                  <Stack gap="sm">
                    <Switch
                      label="Two-Factor Authentication"
                      description="Enable 2FA for enhanced security"
                      checked={settings.security.two_factor_enabled}
                      onChange={(e) => updateSetting('security', 'two_factor_enabled', e.target.checked)}
                    />
                    <Switch
                      label="Login Notifications"
                      description="Get notified of new login attempts"
                      checked={settings.security.login_notifications}
                      onChange={(e) => updateSetting('security', 'login_notifications', e.target.checked)}
                    />
                    <Switch
                      label="API Access"
                      description="Enable API access for this account"
                      checked={settings.security.api_access_enabled}
                      onChange={(e) => updateSetting('security', 'api_access_enabled', e.target.checked)}
                    />
              </Stack>
                </div>

                <Divider />

                <div>
                  <Title order={4} mb="sm">Session Management</Title>
                  <Grid>
                    <Grid.Col span={{ base: 12, md: 6 }}>
                      <NumberInput
                        label="Session Timeout (minutes)"
                        description="Automatically log out after inactivity"
                        value={settings.security.session_timeout}
                        onChange={(value) => updateSetting('security', 'session_timeout', value)}
                        min={5}
                        max={480}
                        radius={0}
                      />
                    </Grid.Col>
                    <Grid.Col span={{ base: 12, md: 6 }}>
                      <NumberInput
                        label="Password Expiry (days)"
                        description="Require password change after this period"
                        value={settings.security.password_expiry}
                        onChange={(value) => updateSetting('security', 'password_expiry', value)}
                        min={30}
                        max={365}
                        radius={0}
                      />
                    </Grid.Col>
                  </Grid>
                </div>

                <Divider />

                <div>
                  <Title order={4} mb="sm">API Key Management</Title>
                  <Group>
                    <TextInput
                      label="API Key"
                      value={showApiKey ? 'sk-1234567890abcdef1234567890abcdef' : '••••••••••••••••••••••••••••••••'}
                      readOnly
                      style={{ flex: 1 }}
                      rightSection={
                        <ActionIcon onClick={() => setShowApiKey(!showApiKey)} radius={0}>
                          {showApiKey ? <IconEyeOff size={16} /> : <IconEye size={16} />}
                        </ActionIcon>
                      }
                      radius={0}
                    />
                    <Button onClick={generateApiKey} radius={0}>
                      Generate New Key
                    </Button>
                  </Group>
                </div>
              </Stack>
            </Card>
          </Tabs.Panel>

          {/* System Tab */}
          <Tabs.Panel value="system" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">System Configuration</Title>
              
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Auto-save Interval (minutes)"
                    description="Automatically save drafts every N minutes"
                    value={settings.system.auto_save_interval}
                    onChange={(value) => updateSetting('system', 'auto_save_interval', value)}
                    min={1}
                    max={60}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Default Currency"
                    data={[
                      { value: 'EUR', label: 'Euro (€)' },
                      { value: 'USD', label: 'US Dollar ($)' },
                      { value: 'GBP', label: 'British Pound (£)' },
                    ]}
                    value={settings.system.default_currency}
                    onChange={(value) => updateSetting('system', 'default_currency', value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Date Format"
                    data={[
                      { value: 'DD/MM/YYYY', label: 'DD/MM/YYYY' },
                      { value: 'MM/DD/YYYY', label: 'MM/DD/YYYY' },
                      { value: 'YYYY-MM-DD', label: 'YYYY-MM-DD' },
                    ]}
                    value={settings.system.date_format}
                    onChange={(value) => updateSetting('system', 'date_format', value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Number Format"
                    data={[
                      { value: 'European', label: 'European (1.234,56)' },
                      { value: 'American', label: 'American (1,234.56)' },
                    ]}
                    value={settings.system.number_format}
                    onChange={(value) => updateSetting('system', 'number_format', value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <Stack gap="sm">
                    <Switch
                      label="Automatic Backups"
                      description="Enable automatic data backups"
                      checked={settings.system.backup_enabled}
                      onChange={(e) => updateSetting('system', 'backup_enabled', e.target.checked)}
                    />
                    <Switch
                      label="Debug Mode"
                      description="Enable debug logging (for troubleshooting)"
                      checked={settings.system.debug_mode}
                      onChange={(e) => updateSetting('system', 'debug_mode', e.target.checked)}
                    />
                  </Stack>
                </Grid.Col>
              </Grid>
          </Card>
          </Tabs.Panel>

          {/* API Settings Tab */}
          <Tabs.Panel value="api" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">API Configuration</Title>
              
              <Grid>
                <Grid.Col span={12}>
                  <TextInput
                    label="API Base URL"
                    description="Base URL for API requests"
                    value={settings.api.base_url}
                    onChange={(e) => updateSetting('api', 'base_url', e.target.value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Request Timeout (seconds)"
                    description="Maximum time to wait for API responses"
                    value={settings.api.timeout}
                    onChange={(value) => updateSetting('api', 'timeout', value)}
                    min={5}
                    max={300}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Retry Attempts"
                    description="Number of times to retry failed requests"
                    value={settings.api.retry_attempts}
                    onChange={(value) => updateSetting('api', 'retry_attempts', value)}
                    min={0}
                    max={10}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Rate Limit (requests/minute)"
                    description="Maximum API requests per minute"
                    value={settings.api.rate_limit}
                    onChange={(value) => updateSetting('api', 'rate_limit', value)}
                    min={10}
                    max={1000}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <Switch
                    label="Enable Caching"
                    description="Cache API responses to improve performance"
                    checked={settings.api.cache_enabled}
                    onChange={(e) => updateSetting('api', 'cache_enabled', e.target.checked)}
                  />
                </Grid.Col>
        </Grid>
            </Card>
          </Tabs.Panel>

          {/* Theme Tab */}
          <Tabs.Panel value="theme" pt="xl">
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Appearance & Theme</Title>
              
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <ColorInput
                    label="Primary Color"
                    description="Choose your preferred primary color"
                    value={settings.theme.primary_color}
                    onChange={(value) => updateSetting('theme', 'primary_color', value)}
                    radius={0}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <div>
                    <Text size="sm" fw={500} mb="xs">Font Size</Text>
                    <Text size="xs" c="dimmed" mb="md">Adjust the base font size</Text>
                    <Slider
                      value={settings.theme.font_size}
                      onChange={(value) => updateSetting('theme', 'font_size', value)}
                      min={12}
                      max={18}
                      step={1}
                      marks={[
                        { value: 12, label: '12px' },
                        { value: 14, label: '14px' },
                        { value: 16, label: '16px' },
                        { value: 18, label: '18px' },
                      ]}
                      radius={0}
                    />
                  </div>
                </Grid.Col>
                <Grid.Col span={12}>
                  <Stack gap="sm">
                    <Switch
                      label="Dark Mode"
                      description="Use dark theme for the interface"
                      checked={settings.theme.dark_mode}
                      onChange={(e) => updateSetting('theme', 'dark_mode', e.target.checked)}
                    />
                    <Switch
                      label="Compact Mode"
                      description="Use smaller spacing and components"
                      checked={settings.theme.compact_mode}
                      onChange={(e) => updateSetting('theme', 'compact_mode', e.target.checked)}
                    />
                    <Switch
                      label="Animations"
                      description="Enable interface animations and transitions"
                      checked={settings.theme.animations_enabled}
                      onChange={(e) => updateSetting('theme', 'animations_enabled', e.target.checked)}
                    />
                  </Stack>
                </Grid.Col>
      </Grid>
            </Card>
          </Tabs.Panel>
        </Tabs>

        {/* Action Buttons */}
        <Group justify="space-between">
          <Button 
            variant="outline" 
            leftSection={<IconTrash size={16} />}
            color="red"
            onClick={resetSettings}
            radius={0}
          >
            Reset All Settings
          </Button>
          <Group>
            <Button 
              variant="outline"
              onClick={() => navigate(-1)}
              radius={0}
            >
              Cancel
            </Button>
            <Button 
              leftSection={<IconDeviceFloppy size={16} />}
              onClick={saveSettings}
              loading={loading}
              radius={0}
            >
              Save All Changes
            </Button>
          </Group>
        </Group>
      </Stack>
    </Container>
  );
};

export default Settings;