/**
 * MortgageAI Dutch Mortgage Advisory Platform
 *
 * Specialized for Dutch AFM-compliant mortgage advice and application processing
 * Integration with Dutch lenders, BKR, NHG, and AFM compliance
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { SnackbarProvider } from 'notistack';
import { ClientProvider } from './contexts/ClientContext';

// Components
import Header from './components/Header';
import Footer from './components/Footer';

// Dutch Mortgage Specific Pages
import DutchMortgageDashboard from './pages/DutchMortgageDashboard';
import AFMClientIntake from './pages/AFMClientIntake';
import AFMComplianceAdvisor from './pages/AFMComplianceAdvisor';
import DutchMortgageApplication from './pages/DutchMortgageApplication';
import BKRCreditCheck from './pages/BKRCreditCheck';
import NHGEligibilityCheck from './pages/NHGEligibilityCheck';
import LenderIntegration from './pages/LenderIntegration';
import ComplianceAuditTrail from './pages/ComplianceAuditTrail';
import DutchMarketInsights from './pages/DutchMarketInsights';

// Gaigentic Brand Colors (as specified in previous prompt)
const theme = createTheme({
  palette: {
    primary: {
      main: '#6366F1', // Gaigentic primary indigo
      light: '#818CF8',
      dark: '#4338CA',
    },
    secondary: {
      main: '#EC4899', // Gaigentic secondary pink
      light: '#F472B6',
      dark: '#BE185D',
    },
    background: {
      default: '#FAFBFC', // Ultra-light gray background
      paper: '#FFFFFF', // White surfaces
    },
    text: {
      primary: '#0F172A', // Dark text
      secondary: '#64748B', // Gray text
    },
    success: {
      main: '#10B981', // Emerald for AFM compliance
      light: '#34D399',
      dark: '#047857',
    },
    warning: {
      main: '#F59E0B', // Amber for warnings
      light: '#FCD34D',
      dark: '#D97706',
    },
    error: {
      main: '#EF4444', // Red for errors
      light: '#F87171',
      dark: '#DC2626',
    },
  },
  typography: {
    fontFamily: '"Inter", "SF Pro Display", "-apple-system", "BlinkMacSystemFont", "Segoe UI", sans-serif',
    h1: {
      fontSize: '2.25rem',
      fontWeight: 700,
      letterSpacing: '-0.025em',
      color: '#0F172A',
    },
    h2: {
      fontSize: '1.875rem',
      fontWeight: 600,
      letterSpacing: '-0.025em',
      color: '#0F172A',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '-0.025em',
      color: '#0F172A',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      color: '#374151',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
      color: '#6B7280',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 500,
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4338CA 0%, #7C3AED 100%)',
            boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            backgroundColor: '#F8FAFC',
            border: '1px solid #E2E8F0',
            transition: 'all 0.2s ease',
            '&:hover': {
              backgroundColor: '#F1F5F9',
              borderColor: '#CBD5E1',
            },
            '&.Mui-focused': {
              backgroundColor: '#FFFFFF',
              borderColor: '#6366F1',
              boxShadow: '0 0 0 3px rgba(99, 102, 241, 0.1)',
            },
          },
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider
        maxSnack={3}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <ClientProvider>
          <Router>
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              minHeight: '100vh',
              backgroundColor: 'background.default'
            }}>
              <Header />

              <Box component="main" sx={{
                flexGrow: 1,
                py: { xs: 2, md: 4 },
                px: { xs: 1, md: 2 },
              }}>
                <Routes>
                  <Route path="/" element={<DutchMortgageDashboard />} />
                  <Route path="/afm-client-intake" element={<AFMClientIntake />} />
                  <Route path="/afm-compliance-advisor" element={<AFMComplianceAdvisor />} />
                  <Route path="/mortgage-application" element={<DutchMortgageApplication />} />
                  <Route path="/bkr-credit-check" element={<BKRCreditCheck />} />
                  <Route path="/nhg-eligibility" element={<NHGEligibilityCheck />} />
                  <Route path="/lender-integration" element={<LenderIntegration />} />
                  <Route path="/compliance-audit" element={<ComplianceAuditTrail />} />
                  <Route path="/market-insights" element={<DutchMarketInsights />} />
                </Routes>
              </Box>

              <Footer />
            </Box>
          </Router>
        </ClientProvider>
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;


