/**
 * Dutch Market Insights
 *
 * Provides real-time market intelligence for Dutch mortgage market
 * Includes interest rates, lender competition, and regulatory updates
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
  LinearProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  Euro,
  AccountBalance,
  ArrowBack,
  Refresh,
  Info as InfoIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { marketApi, MarketInsights } from '../services/marketApi';


const DutchMarketInsights: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const [insights, setInsights] = useState<MarketInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    marketApi.setSnackbar(enqueueSnackbar);
    loadMarketInsights();
  }, [enqueueSnackbar]);

  const loadMarketInsights = async () => {
    try {
      setLoading(true);
      const marketData = await marketApi.getMarketInsights();
      setInsights(marketData);
    } catch (error) {
      console.error('Failed to load market insights:', error);
      enqueueSnackbar('Failed to load market insights', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const refreshInsights = async () => {
    setRefreshing(true);
    try {
      const refreshedData = await marketApi.refreshMarketData();
      setInsights(refreshedData);
      enqueueSnackbar('Market insights refreshed', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Failed to refresh market insights', { variant: 'error' });
    } finally {
      setRefreshing(false);
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
            Loading Dutch Market Insights...
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
              Dutch Market Insights
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Real-time Dutch mortgage market intelligence and regulatory updates
            </Typography>
          </Box>
          <Button
            variant="outlined"
            startIcon={refreshing ? <CircularProgress size={16} /> : <Refresh />}
            onClick={refreshInsights}
            disabled={refreshing}
          >
            {refreshing ? 'Refreshing...' : 'Refresh Data'}
          </Button>
        </Box>

        {/* Market Summary */}
        {insights && (
          <Alert
            severity={insights.market_summary.overall_trend === 'bullish' ? 'success' :
                     insights.market_summary.overall_trend === 'bearish' ? 'warning' : 'info'}
            sx={{ mb: 4 }}
          >
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Market Summary - {insights.market_summary.overall_trend.toUpperCase()}
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              {insights.market_summary.forecast_3m}
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              {insights.market_summary.key_drivers.map((driver, index) => (
                <Chip key={index} label={driver} size="small" color="primary" variant="outlined" />
              ))}
            </Box>
          </Alert>
        )}

        {/* Market Indicators */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {insights?.indicators.map((indicator) => (
            <Grid item xs={12} sm={6} md={3} key={indicator.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      {indicator.name}
                    </Typography>
                    {indicator.trend === 'up' && <TrendingUp sx={{ color: 'success.main', fontSize: 16 }} />}
                    {indicator.trend === 'down' && <TrendingDown sx={{ color: 'error.main', fontSize: 16 }} />}
                  </Box>

                  <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                    {indicator.value}{indicator.unit}
                  </Typography>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography
                      variant="body2"
                      sx={{
                        color: indicator.change >= 0 ? 'success.main' : 'error.main',
                        fontWeight: 600
                      }}
                    >
                      {indicator.change >= 0 ? '+' : ''}{indicator.change}{indicator.unit}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      vs last period
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Tabs for different views */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
              <Tab label="Lender Rates" />
              <Tab label="Regulatory Updates" />
              <Tab label="Risk Factors" />
            </Tabs>

            {/* Lender Rates Tab */}
            {activeTab === 0 && (
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Current Lender Interest Rates
                </Typography>

                <List>
                  {insights?.lender_rates.map((rate, index) => (
                    <React.Fragment key={rate.lender}>
                      <ListItem>
                        <ListItemIcon>
                          <AccountBalance />
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              {rate.lender}
                            </Typography>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                              <Typography variant="body2">
                                10yr Fixed: {rate.fixed_10yr}%
                              </Typography>
                              <Typography variant="body2">
                                20yr Fixed: {rate.fixed_20yr}%
                              </Typography>
                              <Typography variant="body2">
                                Variable: {rate.variable}%
                              </Typography>
                            </Box>
                          }
                        />
                        <Typography variant="caption" color="text.secondary">
                          Updated: {new Date(rate.last_updated).toLocaleDateString('nl-NL')}
                        </Typography>
                      </ListItem>
                      {index < (insights.lender_rates.length - 1) && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </Box>
            )}

            {/* Regulatory Updates Tab */}
            {activeTab === 1 && (
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Recent Regulatory Updates
                </Typography>

                <List>
                  {insights?.regulatory_updates.map((update, index) => (
                    <React.Fragment key={update.id}>
                      <ListItem>
                        <ListItemIcon>
                          {update.impact === 'high' ? <WarningIcon sx={{ color: 'error.main' }} /> :
                           update.impact === 'medium' ? <InfoIcon sx={{ color: 'warning.main' }} /> :
                           <InfoIcon sx={{ color: 'info.main' }} />}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {update.title}
                              </Typography>
                              <Chip
                                label={update.category.toUpperCase()}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                              <Chip
                                label={update.impact}
                                size="small"
                                color={update.impact === 'high' ? 'error' :
                                       update.impact === 'medium' ? 'warning' : 'info'}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                {update.summary}
                              </Typography>
                              <Typography variant="caption">
                                {new Date(update.date).toLocaleDateString('nl-NL')}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < (insights.regulatory_updates.length - 1) && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </Box>
            )}

            {/* Risk Factors Tab */}
            {activeTab === 2 && (
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Market Risk Factors
                </Typography>

                <Alert severity="warning" sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    Current Risk Assessment
                  </Typography>
                  <Typography variant="body2">
                    Monitor these factors closely as they may impact mortgage availability and pricing.
                  </Typography>
                </Alert>

                <List>
                  {insights?.market_summary.risk_factors.map((risk, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <WarningIcon sx={{ color: 'warning.main' }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {risk}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default DutchMarketInsights;
