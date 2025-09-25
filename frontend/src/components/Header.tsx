/**
 * Professional Header Component for MortgageAI
 *
 * Features:
 * - Modern navigation
 * - Responsive design
 * - Professional branding
 * - Clean typography
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
  MenuItem
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home,
  Description,
  Assessment,
  CheckCircle,
  Upload
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
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/application', label: 'Application', icon: Description },
    { path: '/documents', label: 'Documents', icon: Upload },
    { path: '/compliance', label: 'Compliance', icon: CheckCircle },
    { path: '/quality-control', label: 'Quality Control', icon: Assessment },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar
      position="static"
      elevation={2}
      sx={{
        bgcolor: 'primary.main',
        backgroundImage: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
      }}
    >
      <Toolbar>
        {/* Logo and Brand */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            mr: 4
          }}
          onClick={() => navigate('/')}
        >
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: 2,
              bgcolor: 'rgba(255, 255, 255, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mr: 2
            }}
          >
            <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
              AI
            </Typography>
          </Box>
          <Box>
            <Typography
              variant="h6"
              component="div"
              sx={{
                fontWeight: 'bold',
                color: 'white',
                lineHeight: 1.2
              }}
            >
              MortgageAI
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: 'rgba(255, 255, 255, 0.8)',
                lineHeight: 1
              }}
            >
              Agentic AI Solution
            </Typography>
          </Box>
        </Box>

        {/* Desktop Navigation */}
        {!isMobile && (
          <Box sx={{ display: 'flex', gap: 1, flexGrow: 1 }}>
            {navigationItems.map((item) => {
              const IconComponent = item.icon;
              return (
                <Button
                  key={item.path}
                  color="inherit"
                  startIcon={<IconComponent />}
                  onClick={() => handleNavigate(item.path)}
                  sx={{
                    color: 'white',
                    opacity: isActive(item.path) ? 1 : 0.8,
                    backgroundColor: isActive(item.path) ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      opacity: 1,
                    },
                    px: 2,
                    py: 1,
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
              color="inherit"
              aria-label="menu"
              onClick={handleMenu}
              sx={{ color: 'white' }}
            >
              <MenuIcon />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              {navigationItems.map((item) => {
                const IconComponent = item.icon;
                return (
                  <MenuItem
                    key={item.path}
                    onClick={() => handleNavigate(item.path)}
                    sx={{
                      backgroundColor: isActive(item.path) ? 'rgba(25, 118, 210, 0.1)' : 'transparent',
                    }}
                  >
                    <IconComponent sx={{ mr: 1 }} />
                    {item.label}
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
            gap: 1
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: 'success.main',
              boxShadow: '0 0 8px rgba(76, 175, 80, 0.5)',
            }}
          />
          <Typography
            variant="caption"
            sx={{
              color: 'rgba(255, 255, 255, 0.8)',
              display: { xs: 'none', sm: 'block' }
            }}
          >
            System Online
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
