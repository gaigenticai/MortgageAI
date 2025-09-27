import React, { ReactNode } from 'react';
import { Box, Container } from '@mui/material';
import { palette } from '../../theme/designSystem';

interface PageSurfaceProps {
  children: ReactNode;
  maxWidth?: 'xl' | 'lg' | 'md' | 'sm' | 'xs' | false;
}

const PageSurface: React.FC<PageSurfaceProps> = ({ children, maxWidth = 'xl' }) => {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        backgroundColor: palette.background,
        py: { xs: 4, md: 6 },
        px: { xs: 2, md: 4 }
      }}
    >
      <Container maxWidth={maxWidth}>
        {children}
      </Container>
    </Box>
  );
};

export default PageSurface;

