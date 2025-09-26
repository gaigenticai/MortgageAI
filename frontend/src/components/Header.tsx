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
  Settings
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Header: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();
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
      elevation={1}
      sx={{
        background: 'background.paper',
        borderBottom: '1px solid #e0e0e0',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
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
            transition: 'transform 0.2s ease',
            '&:hover': {
              transform: 'scale(1.05)',
            }
          }}
          onClick={() => navigate('/')}
        >
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: 2,
              background: 'primary.main',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mr: 2,
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
            }}
          >
            <Psychology sx={{ color: 'white', fontSize: 20 }} />
          </Box>
          <Box>
            <Typography
              variant="h6"
              component="div"
              sx={{
                fontWeight: 600,
                color: 'text.primary',
                lineHeight: 1.2,
                letterSpacing: '0em',
              }}
            >
              MortgageAI
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: 'text.secondary',
                fontWeight: 400,
                lineHeight: 1,
                letterSpacing: '0em'
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
                    color: active ? 'primary.main' : 'text.secondary',
                    background: active ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
                    borderRadius: 2,
                    px: 2,
                    py: 1,
                    fontWeight: 500,
                    fontSize: '0.875rem',
                    transition: 'all 0.2s ease',
                    border: active ? '1px solid rgba(25, 118, 210, 0.2)' : '1px solid transparent',
                    '&:hover': {
                      transform: 'translateY(-1px)',
                      background: 'rgba(25, 118, 210, 0.04)',
                      border: '1px solid rgba(25, 118, 210, 0.3)',
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
                borderRadius: 2,
                '&:hover': {
                  background: 'rgba(0, 0, 0, 0.04)',
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
                  background: 'background.paper',
                  border: '1px solid #e0e0e0',
                  borderRadius: 2,
                  mt: 1,
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
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
                      background: active ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
                      borderRadius: 2,
                      mx: 1,
                      my: 0.5,
                      '&:hover': {
                        background: 'rgba(0, 0, 0, 0.04)',
                      }
                    }}
                  >
                    <Box
                      sx={{
                        width: 24,
                        height: 24,
                        borderRadius: 1,
                        background: active ? 'rgba(25, 118, 210, 0.1)' : 'transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mr: 2,
                      }}
                    >
                      <IconComponent sx={{ color: active ? 'primary.main' : 'text.secondary', fontSize: 16 }} />
                    </Box>
                    <Typography sx={{ fontWeight: active ? 600 : 400, color: 'text.primary' }}>
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
            gap: 1,
            px: 2,
            py: 1,
            borderRadius: 2,
            background: 'rgba(76, 175, 80, 0.1)',
            border: '1px solid rgba(76, 175, 80, 0.2)',
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: 'success.main',
            }}
          />
          <Typography
            variant="body2"
            sx={{
              color: 'success.main',
              fontWeight: 500,
              display: { xs: 'none', sm: 'block' }
            }}
          >
            Online
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
