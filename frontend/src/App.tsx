/**
 * MortgageAI Professional Frontend Application
 *
 * Features:
 * - Mortgage application form with validation
 * - Document upload with OCR preview
 * - Real-time compliance checking
 * - Quality control dashboard
 * - Explain-back dialogue system
 * - Professional UI/UX design
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { SnackbarProvider } from 'notistack';

// Components
import Header from './components/Header';
import Footer from './components/Footer';
import Dashboard from './pages/Dashboard';
import ApplicationForm from './pages/ApplicationForm';
import ComplianceCheck from './pages/ComplianceCheck';
import QualityControl from './pages/QualityControl';
import DocumentUpload from './pages/DocumentUpload';
import ResultsDisplay from './pages/ResultsDisplay';
import Settings from './pages/Settings';

// Ultra-modern theme configuration inspired by top Dribbble designs
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // Professional blue
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e', // Professional red
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#ffffff', // Clean white background
      paper: '#ffffff', // White cards
    },
    text: {
      primary: '#212121', // Dark text
      secondary: '#757575', // Gray text
    },
    success: {
      main: '#4caf50', // Standard green
      light: '#81c784',
      dark: '#388e3c',
    },
    warning: {
      main: '#ff9800', // Standard orange
      light: '#ffb74d',
      dark: '#f57c00',
    },
    error: {
      main: '#f44336', // Standard red
      light: '#e57373',
      dark: '#d32f2f',
    },
    info: {
      main: '#2196f3', // Standard blue
      light: '#64b5f6',
      dark: '#1976d2',
    },
  },
  typography: {
    fontFamily: '"Inter", "Poppins", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
      letterSpacing: '0em',
      color: 'text.primary',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
      letterSpacing: '0em',
      color: 'text.primary',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
      letterSpacing: '0em',
      color: 'text.primary',
    },
    h4: {
      fontSize: '1.6rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.3rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '1.15rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: '#ffffff',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#f1f1f1',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#c1c1c1',
            borderRadius: '4px',
            '&:hover': {
              background: '#a8a8a8',
            },
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '0.875rem',
          padding: '8px 16px',
          transition: 'all 0.2s ease',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.15)',
          },
        },
        contained: {
          background: 'primary.main',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            background: 'primary.dark',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          },
        },
        outlined: {
          borderWidth: '1px',
          '&:hover': {
            borderWidth: '1px',
            background: 'rgba(25, 118, 210, 0.04)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          background: 'background.paper',
          border: '1px solid #e0e0e0',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.2s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'background.paper',
          border: '1px solid #e0e0e0',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          fontWeight: 500,
          fontSize: '0.8125rem',
          background: '#f5f5f5',
          border: '1px solid #e0e0e0',
          transition: 'all 0.2s ease',
          '&:hover': {
            background: '#eeeeee',
          },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          height: 6,
          background: '#e0e0e0',
        },
        bar: {
          borderRadius: 4,
          background: 'primary.main',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 4,
            background: 'background.paper',
            border: '1px solid #e0e0e0',
            transition: 'all 0.2s ease',
            '&:hover': {
              background: '#fafafa',
              border: '1px solid #bdbdbd',
            },
            '&.Mui-focused': {
              background: 'background.paper',
              border: '2px solid primary.main',
              boxShadow: '0 0 0 3px rgba(25, 118, 210, 0.1)',
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
        <Router>
          <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            position: 'relative',
            overflowX: 'hidden',
            background: 'background.default'
          }}>
            <Header />
            <Box
              component="main"
              sx={{
                flexGrow: 1,
                py: { xs: 2, md: 4 },
                px: { xs: 1, md: 2 },
                position: 'relative',
                zIndex: 1,
              }}
            >
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/application" element={<ApplicationForm />} />
                <Route path="/documents" element={<DocumentUpload />} />
                <Route path="/compliance" element={<ComplianceCheck />} />
                <Route path="/quality-control" element={<QualityControl />} />
                <Route path="/results" element={<ResultsDisplay />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Box>
            <Footer />
          </Box>
        </Router>
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;

