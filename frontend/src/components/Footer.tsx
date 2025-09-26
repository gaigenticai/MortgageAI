/**
 * Professional Footer Component for MortgageAI
 */

import React from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Link,
  Divider,
  IconButton
} from '@mui/material';
import {
  GitHub,
  LinkedIn,
  Email,
  Phone,
  LocationOn,
  Psychology
} from '@mui/icons-material';

const Footer: React.FC = () => {
  return (
    <Box
      component="footer"
      sx={{
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
        color: 'white',
        py: 8,
        mt: 'auto',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.3) 50%, transparent 100%)',
        }
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Company Info */}
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Box sx={{
                width: 48,
                height: 48,
                borderRadius: 2,
                background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2,
                boxShadow: '0 4px 12px rgba(99, 102, 241, 0.25)',
              }}>
                <Psychology sx={{ color: 'white', fontSize: 24 }} />
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 700, color: 'white', lineHeight: 1.2 }}>
                  MortgageAI
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '0.75rem' }}>
                  by Gaigentic
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" sx={{ mb: 3, color: 'rgba(255, 255, 255, 0.8)', lineHeight: 1.6 }}>
              Premium AI solution for enhancing mortgage advice quality and application accuracy.
              Powered by advanced compliance and quality control agents.
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <IconButton
                color="inherit"
                size="small"
                sx={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  '&:hover': {
                    background: 'rgba(99, 102, 241, 0.2)',
                    transform: 'translateY(-1px)',
                  },
                  transition: 'all 0.2s ease',
                }}
              >
                <GitHub />
              </IconButton>
              <IconButton
                color="inherit"
                size="small"
                sx={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  '&:hover': {
                    background: 'rgba(236, 72, 153, 0.2)',
                    transform: 'translateY(-1px)',
                  },
                  transition: 'all 0.2s ease',
                }}
              >
                <LinkedIn />
              </IconButton>
            </Box>
          </Grid>

          {/* Quick Links */}
          <Grid item xs={12} md={2}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'white', mb: 3 }}>
              Features
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#6366F1',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                Compliance Agent
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#6366F1',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                QC Agent
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#10B981',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                Document Processing
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#EC4899',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                API Integration
              </Link>
            </Box>
          </Grid>

          {/* Support */}
          <Grid item xs={12} md={2}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'white', mb: 3 }}>
              Support
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#6366F1',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                Documentation
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#6366F1',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                API Reference
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#10B981',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                Help Center
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  '&:hover': {
                    color: '#EC4899',
                    textDecoration: 'none',
                  },
                  transition: 'color 0.2s ease',
                }}
              >
                Contact Us
              </Link>
            </Box>
          </Grid>

          {/* Contact Info */}
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'white', mb: 3 }}>
              Contact Information
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box sx={{
                  width: 32,
                  height: 32,
                  borderRadius: 2,
                  background: 'rgba(99, 102, 241, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  <Email sx={{ fontSize: 16, color: '#6366F1' }} />
                </Box>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '0.875rem' }}>
                  support@mortgage-ai.com
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box sx={{
                  width: 32,
                  height: 32,
                  borderRadius: 2,
                  background: 'rgba(16, 185, 129, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  <Phone sx={{ fontSize: 16, color: '#10B981' }} />
                </Box>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '0.875rem' }}>
                  +31 20 123 4567
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box sx={{
                  width: 32,
                  height: 32,
                  borderRadius: 2,
                  background: 'rgba(236, 72, 153, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  <LocationOn sx={{ fontSize: 16, color: '#EC4899' }} />
                </Box>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '0.875rem' }}>
                  Amsterdam, Netherlands
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ my: 6, background: 'linear-gradient(90deg, transparent 0%, rgba(226, 232, 240, 0.3) 50%, transparent 100%)' }} />

        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 3
        }}>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.875rem' }}>
            © 2024 MortgageAI by Gaigentic. All rights reserved.
          </Typography>
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <Link
              href="#"
              sx={{
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '0.875rem',
                fontWeight: 500,
                '&:hover': {
                  color: '#6366F1',
                  textDecoration: 'none',
                },
                transition: 'color 0.2s ease',
              }}
            >
              Privacy Policy
            </Link>
            <Link
              href="#"
              sx={{
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '0.875rem',
                fontWeight: 500,
                '&:hover': {
                  color: '#6366F1',
                  textDecoration: 'none',
                },
                transition: 'color 0.2s ease',
              }}
            >
              Terms of Service
            </Link>
            <Link
              href="#"
              sx={{
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '0.875rem',
                fontWeight: 500,
                '&:hover': {
                  color: '#EC4899',
                  textDecoration: 'none',
                },
                transition: 'color 0.2s ease',
              }}
            >
              AFM Compliance
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
