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
  LocationOn
} from '@mui/icons-material';

const Footer: React.FC = () => {
  return (
    <Box
      component="footer"
      sx={{
        bgcolor: 'grey.900',
        color: 'white',
        py: 6,
        mt: 'auto'
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Company Info */}
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
              MortgageAI
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Agentic AI Solution for Enhancing Mortgage Advice Quality and Application Accuracy.
              Powered by advanced compliance and quality control agents.
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <IconButton color="inherit" size="small">
                <GitHub />
              </IconButton>
              <IconButton color="inherit" size="small">
                <LinkedIn />
              </IconButton>
            </Box>
          </Grid>

          {/* Quick Links */}
          <Grid item xs={12} md={2}>
            <Typography variant="h6" gutterBottom>
              Features
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Link href="#" color="inherit" underline="hover">
                Compliance Agent
              </Link>
              <Link href="#" color="inherit" underline="hover">
                QC Agent
              </Link>
              <Link href="#" color="inherit" underline="hover">
                Document Processing
              </Link>
              <Link href="#" color="inherit" underline="hover">
                API Integration
              </Link>
            </Box>
          </Grid>

          {/* Support */}
          <Grid item xs={12} md={2}>
            <Typography variant="h6" gutterBottom>
              Support
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Link href="#" color="inherit" underline="hover">
                Documentation
              </Link>
              <Link href="#" color="inherit" underline="hover">
                API Reference
              </Link>
              <Link href="#" color="inherit" underline="hover">
                Help Center
              </Link>
              <Link href="#" color="inherit" underline="hover">
                Contact Us
              </Link>
            </Box>
          </Grid>

          {/* Contact Info */}
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              Contact Information
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Email fontSize="small" />
                <Typography variant="body2">
                  support@mortgage-ai.com
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Phone fontSize="small" />
                <Typography variant="body2">
                  +31 20 123 4567
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <LocationOn fontSize="small" />
                <Typography variant="body2">
                  Amsterdam, Netherlands
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ my: 3, bgcolor: 'grey.700' }} />

        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 2
        }}>
          <Typography variant="body2" color="grey.400">
            © 2024 MortgageAI. All rights reserved.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Link href="#" color="grey.400" underline="hover" variant="body2">
              Privacy Policy
            </Link>
            <Link href="#" color="grey.400" underline="hover" variant="body2">
              Terms of Service
            </Link>
            <Link href="#" color="grey.400" underline="hover" variant="body2">
              AFM Compliance
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
