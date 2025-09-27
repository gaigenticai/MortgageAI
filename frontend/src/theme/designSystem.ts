import { SxProps, Theme } from '@mui/material';

export const palette = {
  background: '#F5F6FA',
  surface: '#FFFFFF',
  primary: '#6366F1',
  primaryLight: '#A5B4FF',
  primaryDark: '#4338CA',
  accentBlue: '#3B82F6',
  accentGreen: '#22C55E',
  accentOrange: '#F97316',
  accentRed: '#EF4444',
  textPrimary: '#111827',
  textSecondary: '#6B7280',
  border: '#E5E7EB',
} as const;

export const shadows = {
  xl: '0 30px 60px rgba(15, 23, 42, 0.12)',
  lg: '0 20px 45px rgba(15, 23, 42, 0.10)',
  md: '0 12px 30px rgba(15, 23, 42, 0.08)',
  sm: '0 6px 15px rgba(15, 23, 42, 0.06)',
  xs: '0 2px 8px rgba(15, 23, 42, 0.05)'
};

export const card = {
  background: palette.surface,
  borderRadius: '0 !important',
  border: `1px solid ${palette.border}`,
  boxShadow: shadows.sm,
};

export const tile = {
  ...card,
  borderRadius: 0,
  padding: '20px',
};

export const softBadge = (color: string): SxProps<Theme> => ({
  px: 1.5,
  py: 0.5,
  borderRadius: 0,
  fontSize: '0.75rem',
  fontWeight: 600,
  backgroundColor: `${color}1A`,
  color,
});

export const iconContainer = (color: string): SxProps<Theme> => ({
  width: 48,
  height: 48,
  borderRadius: 0,
  backgroundColor: `${color}1A`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color,
});

export const sectionHeading = {
  fontWeight: 700,
  fontSize: '1.6rem',
  color: palette.textPrimary,
  letterSpacing: '-0.02em',
};

export const helperText = {
  color: palette.textSecondary,
  fontSize: '0.95rem',
};

export const toolbarButton = {
  textTransform: 'none',
  fontWeight: 600,
  fontSize: '0.9rem',
  borderRadius: 0,
  px: 2.5,
  py: 0.75,
};


