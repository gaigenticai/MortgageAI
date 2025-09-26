/**
 * World-Class Header Component for MortgageAI
 *
 * Features:
 * - Modern glassmorphism design
 * - Advanced gradients and effects
 * - Professional branding with animations
 * - Smooth micro-interactions
 * - Responsive navigation with modern UX
 */

import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  useTheme,
  useMediaQuery,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Chip,
  Fade
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home,
  Description,
  Assessment,
  CheckCircle,
  Upload,
  Star,
  Psychology,
  Security,
  TrendingUp,
  Settings,
  Switch,
  FormControlLabel
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useDemoMode } from '../contexts/DemoModeContext';

const Header: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();
  const { isDemoMode, toggleDemoMode } = useDemoMode();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    handleClose();
  };

  const navigationItems = [
    { path: '/', label: 'Dashboard', icon: Home, gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' },
    { path: '/application', label: 'Application', icon: Description, gradient: 'linear-gradient(135deg, #ec4899 0%, #f472b6 100%)' },
    { path: '/documents', label: 'Documents', icon: Upload, gradient: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)' },
    { path: '/compliance', label: 'Compliance', icon: Security, gradient: 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)' },
    { path: '/quality-control', label: 'Quality Control', icon: Psychology, gradient: 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)' },
    { path: '/settings', label: 'Settings', icon: Settings, gradient: 'linear-gradient(135deg, #64748b 0%, #94a3b8 100%)' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        background: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(226, 232, 240, 0.8)',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.2) 50%, transparent 100%)',
        },
      }}
    >
      <Toolbar sx={{ py: 2 }}>
        {/* Logo and Brand */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            mr: 4,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'scale(1.02)',
              '& .logo-icon': {
                transform: 'rotate(5deg) scale(1.1)',
              },
            }
          }}
          onClick={() => navigate('/')}
        >
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: 3,
              background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mr: 3,
              boxShadow: '0 4px 12px rgba(99, 102, 241, 0.25), 0 2px 4px rgba(0, 0, 0, 0.1)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
            className="logo-icon"
          >
            <Psychology sx={{ color: 'white', fontSize: 24 }} />
          </Box>
          <Box>
            <Typography
              variant="h5"
              component="div"
              sx={{
                fontWeight: 700,
                color: 'text.primary',
                lineHeight: 1.2,
                letterSpacing: '-0.025em',
                background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              MortgageAI
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: 'text.secondary',
                fontWeight: 500,
                lineHeight: 1,
                fontSize: '0.75rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              AI Solution
            </Typography>
          </Box>
        </Box>

        {/* Desktop Navigation */}
        {!isMobile && (
          <Box sx={{ display: 'flex', gap: 1, flexGrow: 1 }}>
            {navigationItems.map((item) => {
              const IconComponent = item.icon;
              const active = isActive(item.path);
              return (
                <Button
                  key={item.path}
                  startIcon={<IconComponent />}
                  onClick={() => handleNavigate(item.path)}
                  sx={{
                    color: active ? '#FFFFFF' : 'text.secondary',
                    background: active
                      ? `linear-gradient(135deg, ${item.gradient.split('linear-gradient(135deg, ')[1].split(' 0%')[0]} 0%, ${item.gradient.split('%, ')[1].split(' 100%)')[0]} 100%)`
                      : 'rgba(255, 255, 255, 0.6)',
                    borderRadius: 3,
                    px: 3,
                    py: 1.5,
                    fontWeight: 500,
                    fontSize: '0.875rem',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    border: active
                      ? 'none'
                      : '1px solid rgba(226, 232, 240, 0.8)',
                    backdropFilter: 'blur(10px)',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      background: active
                        ? `linear-gradient(135deg, ${item.gradient.split('linear-gradient(135deg, ')[1].split(' 0%')[0]} 0%, ${item.gradient.split('%, ')[1].split(' 100%)')[0]} 100%)`
                        : 'rgba(255, 255, 255, 0.9)',
                      boxShadow: active
                        ? '0 8px 20px rgba(0, 0, 0, 0.15)'
                        : '0 4px 12px rgba(0, 0, 0, 0.1)',
                    },
                  }}
                >
                  {item.label}
                </Button>
              );
            })}
          </Box>
        )}

        {/* Mobile Menu */}
        {isMobile && (
          <>
            <Box sx={{ flexGrow: 1 }} />
            <IconButton
              size="large"
              edge="start"
              aria-label="menu"
              onClick={handleMenu}
              sx={{
                color: 'text.secondary',
                borderRadius: 3,
                background: 'rgba(255, 255, 255, 0.6)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(226, 232, 240, 0.8)',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.9)',
                  transform: 'translateY(-1px)',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                }
              }}
            >
              <MenuIcon />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
              PaperProps={{
                sx: {
                  background: 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(226, 232, 240, 0.8)',
                  borderRadius: 3,
                  mt: 2,
                  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15), 0 4px 10px rgba(0, 0, 0, 0.1)',
                  minWidth: 200,
                }
              }}
            >
              {navigationItems.map((item) => {
                const IconComponent = item.icon;
                const active = isActive(item.path);
                return (
                  <MenuItem
                    key={item.path}
                    onClick={() => handleNavigate(item.path)}
                    sx={{
                      background: active
                        ? `rgba(${item.gradient.split('linear-gradient(135deg, ')[1].split(' 0%')[0].replace('#', '').split(',').map(c => parseInt(c, 16))}, 0.1)`
                        : 'transparent',
                      borderRadius: 2,
                      mx: 1,
                      my: 0.5,
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        background: 'rgba(226, 232, 240, 0.8)',
                        transform: 'translateY(-1px)',
                      }
                    }}
                  >
                    <Box
                      sx={{
                        width: 32,
                        height: 32,
                        borderRadius: 2,
                        background: active
                          ? `rgba(${item.gradient.split('linear-gradient(135deg, ')[1].split(' 0%')[0].replace('#', '').split(',').map(c => parseInt(c, 16))}, 0.15)`
                          : 'rgba(226, 232, 240, 0.6)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mr: 2,
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <IconComponent sx={{ color: active ? '#6366F1' : 'text.secondary', fontSize: 18 }} />
                    </Box>
                    <Typography sx={{ fontWeight: active ? 600 : 500, color: 'text.primary' }}>
                      {item.label}
                    </Typography>
                  </MenuItem>
                );
              })}
            </Menu>
          </>
        )}

        {/* Status Indicator */}
        <Box
          sx={{
            ml: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            px: 3,
            py: 1.5,
            borderRadius: 3,
            background: 'rgba(16, 185, 129, 0.1)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              background: 'rgba(16, 185, 129, 0.15)',
              transform: 'translateY(-1px)',
              boxShadow: '0 4px 12px rgba(16, 185, 129, 0.15)',
            },
          }}
        >
          <Box
            sx={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              bgcolor: 'success.main',
              boxShadow: '0 0 0 2px rgba(16, 185, 129, 0.2)',
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.5 },
              },
            }}
          />
          <Typography
            variant="body2"
            sx={{
              color: 'success.main',
              fontWeight: 600,
              display: { xs: 'none', sm: 'block' },
              fontSize: '0.875rem',
            }}
          >
            All Systems Operational
          </Typography>

          {/* Demo Mode Toggle */}
          <FormControlLabel
            className="demo-toggle"
            control={
              <Switch
                checked={isDemoMode}
                onChange={toggleDemoMode}
                color="primary"
                size="small"
              />
            }
            label={
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 600,
                  color: isDemoMode ? 'primary.main' : 'text.secondary',
                  fontSize: '0.875rem',
                  display: { xs: 'none', md: 'block' }
                }}
              >
                Demo Mode
              </Typography>
            }
            sx={{
              ml: 2,
              mr: { xs: 1, md: 0 },
              '& .MuiFormControlLabel-label': {
                fontSize: '0.875rem',
              }
            }}
          />
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
