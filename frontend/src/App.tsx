/**
 * MortgageAI Dutch Mortgage Advisory Platform
 *
 * Specialized for Dutch AFM-compliant mortgage advice and application processing
 * Integration with Dutch lenders, BKR, NHG, and AFM compliance
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { MantineProvider, createTheme, Box } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import { ClientProvider } from './contexts/ClientContext';
import { DemoModeProvider } from './contexts/DemoModeContext';

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

// Additional Pages for Navigation
import ApplicationForm from './pages/ApplicationForm';
import DocumentUpload from './pages/DocumentUpload';
import ComplianceCheck from './pages/ComplianceCheck';
import QualityControl from './pages/QualityControl';
import Settings from './pages/Settings';

// AI Features
import AIMortgageAdvisorChat from './pages/AIMortgageAdvisorChat';
import DocumentOCRProcessor from './pages/DocumentOCRProcessor';

// Gaigentic Brand Colors - Mantine Theme with SQUARE BORDERS
const theme = createTheme({
  fontFamily: '"Inter", "SF Pro Display", "-apple-system", "BlinkMacSystemFont", "Segoe UI", sans-serif',
  primaryColor: 'indigo',
  defaultRadius: 0, // This ensures ALL components have square borders by default
  colors: {
    indigo: ['#EEF2FF', '#E0E7FF', '#C7D2FE', '#A5B4FC', '#818CF8', '#6366F1', '#4338CA', '#3730A3', '#312E81', '#1E1B4B'],
    pink: ['#FDF2F8', '#FCE7F3', '#FBCFE8', '#F9A8D4', '#F472B6', '#EC4899', '#DB2777', '#BE185D', '#9D174D', '#831843'],
    emerald: ['#ECFDF5', '#D1FAE5', '#A7F3D0', '#6EE7B7', '#34D399', '#10B981', '#059669', '#047857', '#065F46', '#064E3B'],
    amber: ['#FFFBEB', '#FEF3C7', '#FDE68A', '#FCD34D', '#FBBF24', '#F59E0B', '#D97706', '#B45309', '#92400E', '#78350F'],
    red: ['#FEF2F2', '#FEE2E2', '#FECACA', '#FCA5A5', '#F87171', '#EF4444', '#DC2626', '#B91C1C', '#991B1B', '#7F1D1D'],
  },
  components: {
    Card: {
      defaultProps: {
        radius: 0,
      },
      styles: {
        root: {
          borderRadius: '0 !important',
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
    Button: {
      defaultProps: {
        radius: 0,
      },
      styles: {
        root: {
          borderRadius: '0 !important',
          fontWeight: 500,
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-1px)',
          },
        },
      },
    },
    TextInput: {
      defaultProps: {
        radius: 0,
      },
      styles: {
        input: {
          borderRadius: '0 !important',
          backgroundColor: '#F8FAFC',
          border: '1px solid #E2E8F0',
          '&:focus': {
            backgroundColor: '#FFFFFF',
            borderColor: '#6366F1',
          },
        },
      },
    },
    Paper: {
      defaultProps: {
        radius: 0,
      },
      styles: {
        root: {
          borderRadius: '0 !important',
        },
      },
    },
  },
});

function App() {
  return (
    <MantineProvider theme={theme}>
      <Notifications />
      <DemoModeProvider>
        <ClientProvider>
          <Router>
            <Box style={{ 
              minHeight: '100vh', 
              backgroundColor: '#FAFBFC',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <Header />
              <Box component="main" style={{ flex: 1, paddingTop: 80 }}>
                <Routes>
                  <Route path="/" element={<DutchMortgageDashboard />} />
                  <Route path="/dashboard" element={<DutchMortgageDashboard />} />
                  <Route path="/afm-client-intake" element={<AFMClientIntake />} />
                  <Route path="/afm-compliance-advisor" element={<AFMComplianceAdvisor />} />
                  <Route path="/dutch-mortgage-application" element={<DutchMortgageApplication />} />
                  <Route path="/bkr-credit-check" element={<BKRCreditCheck />} />
                  <Route path="/nhg-eligibility-check" element={<NHGEligibilityCheck />} />
                  <Route path="/lender-integration" element={<LenderIntegration />} />
                  <Route path="/compliance-audit-trail" element={<ComplianceAuditTrail />} />
                  <Route path="/dutch-market-insights" element={<DutchMarketInsights />} />
                  <Route path="/application" element={<ApplicationForm />} />
                  <Route path="/documents" element={<DocumentUpload />} />
                  <Route path="/compliance" element={<ComplianceCheck />} />
                  <Route path="/quality-control" element={<QualityControl />} />
                  <Route path="/ai-mortgage-advisor-chat" element={<AIMortgageAdvisorChat />} />
                  <Route path="/document-ocr-processor" element={<DocumentOCRProcessor />} />
                  <Route path="/settings" element={<Settings />} />
                </Routes>
              </Box>
              <Footer />
            </Box>
          </Router>
        </ClientProvider>
      </DemoModeProvider>
    </MantineProvider>
  );
}

export default App;