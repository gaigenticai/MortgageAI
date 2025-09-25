import React from 'react';
import { Container, Typography, Paper } from '@mui/material';

const QualityControl: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold' }}>
          QualityControl
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary">
          This feature is under development. Implementation coming soon.
        </Typography>
      </Paper>
    </Container>
  );
};

export default QualityControl;
