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

// Professional theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.07)',
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
            bgcolor: 'background.default'
          }}>
            <Header />
            <Box component="main" sx={{ flexGrow: 1, py: 4 }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/application" element={<ApplicationForm />} />
                <Route path="/documents" element={<DocumentUpload />} />
                <Route path="/compliance" element={<ComplianceCheck />} />
                <Route path="/quality-control" element={<QualityControl />} />
                <Route path="/results" element={<ResultsDisplay />} />
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
