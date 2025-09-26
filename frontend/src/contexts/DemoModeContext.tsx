import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

/**
 * Demo Mode Context
 * Provides global demo mode state management for showcasing MortgageAI without requiring authentication
 */

interface DemoModeContextType {
  isDemoMode: boolean;
  toggleDemoMode: () => void;
  setDemoMode: (enabled: boolean) => void;
}

const DemoModeContext = createContext<DemoModeContextType | undefined>(undefined);

interface DemoModeProviderProps {
  children: ReactNode;
}

export const DemoModeProvider: React.FC<DemoModeProviderProps> = ({ children }) => {
  // Initialize demo mode from localStorage or default to false
  const [isDemoMode, setIsDemoMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('mortgageai_demo_mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Persist demo mode changes to localStorage
  useEffect(() => {
    localStorage.setItem('mortgageai_demo_mode', JSON.stringify(isDemoMode));
  }, [isDemoMode]);

  const toggleDemoMode = () => {
    setIsDemoMode(prev => !prev);
  };

  const setDemoMode = (enabled: boolean) => {
    setIsDemoMode(enabled);
  };

  const value: DemoModeContextType = {
    isDemoMode,
    toggleDemoMode,
    setDemoMode,
  };

  return (
    <DemoModeContext.Provider value={value}>
      {children}
    </DemoModeContext.Provider>
  );
};

export const useDemoMode = (): DemoModeContextType => {
  const context = useContext(DemoModeContext);
  if (context === undefined) {
    throw new Error('useDemoMode must be used within a DemoModeProvider');
  }
  return context;
};
