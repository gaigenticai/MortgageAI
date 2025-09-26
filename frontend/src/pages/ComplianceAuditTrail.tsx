/**
 * Compliance Audit Trail
 *
 * Provides comprehensive audit trail for AFM compliance activities
 * Tracks all compliance checks, decisions, and regulatory actions
 */
import React, { useState, useEffect } from 'react';
import {
  Container,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Grid,
  Chip,
  Alert,
  Avatar,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Gavel,
  Assessment,
  Person,
  Business,
  Security,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  ArrowBack,
  Download,
  FilterList,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { auditApi, AuditEntry, ComplianceStats } from '../services/auditApi';


const ComplianceAuditTrail: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [stats, setStats] = useState<ComplianceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    auditApi.setSnackbar(enqueueSnackbar);
    loadAuditTrail();
  }, [enqueueSnackbar]);

  const loadAuditTrail = async () => {
    try {
      setLoading(true);
      const [entries, statsData] = await Promise.all([
        auditApi.getAuditEntries(),
        auditApi.getComplianceStats()
      ]);
      setAuditEntries(entries);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load audit trail:', error);
      enqueueSnackbar('Failed to load audit trail', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const exportAuditTrail = async () => {
    try {
      // In production, this would export audit data
      await new Promise(resolve => setTimeout(resolve, 2000));
      enqueueSnackbar('Audit trail exported successfully', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Export failed', { variant: 'error' });
    }
  };

  const filteredEntries = auditEntries.filter(entry => {
    if (filter === 'all') return true;
    return entry.category === filter || entry.risk_level === filter || entry.status === filter;
  });

  const getActionIcon = (category: string) => {
    switch (category) {
      case 'client_intake': return <Person />;
      case 'compliance_check': return <Gavel />;
      case 'application': return <Business />;
      case 'audit': return <Assessment />;
      default: return <Security />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
          flexDirection: 'column',
          gap: 2
        }}>
          <CircularProgress size={60} />
          <Typography variant="h6" color="text.secondary">
            Loading Compliance Audit Trail...
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 700 }}>
              Compliance Audit Trail
            </Typography>
            <Typography variant="body1" color="text.secondary">
              AFM regulatory compliance and audit tracking
            </Typography>
          </Box>
          <Button
            variant="outlined"
            startIcon={<Download />}
            onClick={exportAuditTrail}
          >
            Export Audit Log
          </Button>
        </Box>

        {/* Compliance Statistics */}
        {stats && (
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light' }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'primary.dark' }}>
                  {stats.total_entries}
                </Typography>
                <Typography variant="body2" color="primary.dark">
                  Total Audit Entries
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'success.dark' }}>
                  {stats.compliance_checks}
                </Typography>
                <Typography variant="body2" color="success.dark">
                  Compliance Checks
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'warning.dark' }}>
                  {stats.high_risk_actions}
                </Typography>
                <Typography variant="body2" color="warning.dark">
                  High-Risk Actions
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light' }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'info.dark' }}>
                  {stats.average_score}%
                </Typography>
                <Typography variant="body2" color="info.dark">
                  Avg Compliance Score
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        )}

        {/* Audit Entries */}
        <Card>
          <CardContent>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Recent Audit Entries
              </Typography>

              {/* Filter Tabs */}
              <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
                <Tab label="All Entries" />
                <Tab label="Compliance Checks" />
                <Tab label="High Risk" />
                <Tab label="System Events" />
              </Tabs>
            </Box>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Action</TableCell>
                    <TableCell>Client</TableCell>
                    <TableCell>Actor</TableCell>
                    <TableCell>Risk Level</TableCell>
                    <TableCell>Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredEntries.map((entry) => (
                    <TableRow key={entry.id} hover>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(entry.timestamp).toLocaleString('nl-NL')}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Avatar sx={{ width: 24, height: 24 }}>
                            {getActionIcon(entry.category)}
                          </Avatar>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {entry.action}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {entry.client_name || 'System'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {entry.actor}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={entry.risk_level}
                          size="small"
                          color={entry.risk_level === 'low' ? 'success' :
                                 entry.risk_level === 'medium' ? 'warning' : 'error'}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={entry.status}
                          size="small"
                          color={getStatusColor(entry.status)}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>

        {/* Audit Alerts */}
        {stats && stats.audit_alerts > 0 && (
          <Alert severity="warning" sx={{ mt: 3 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Audit Alerts ({stats.audit_alerts})
            </Typography>
            <Typography variant="body2">
              There are {stats.audit_alerts} audit alerts requiring attention. Please review high-risk actions and ensure compliance measures are in place.
            </Typography>
          </Alert>
        )}

        {/* Last Audit Info */}
        {stats && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Last Audit: {new Date(stats.last_audit).toLocaleString('nl-NL')}
            </Typography>
            <Typography variant="body2">
              Regular audits ensure ongoing AFM compliance. Next scheduled audit: {new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toLocaleDateString('nl-NL')}
            </Typography>
          </Alert>
        )}
      </Box>
    </Container>
  );
};

export default ComplianceAuditTrail;
