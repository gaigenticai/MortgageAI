/**
 * Client Context for managing current client and application state
 *
 * Provides centralized state management for client and application IDs
 * throughout the application
 */
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface ClientContextType {
  currentClientId: string | null;
  currentApplicationId: string | null;
  setCurrentClientId: (id: string | null) => void;
  setCurrentApplicationId: (id: string | null) => void;
  clearCurrentIds: () => void;
}

const ClientContext = createContext<ClientContextType | undefined>(undefined);

interface ClientProviderProps {
  children: ReactNode;
}

export const ClientProvider: React.FC<ClientProviderProps> = ({ children }) => {
  const [currentClientId, setCurrentClientId] = useState<string | null>(null);
  const [currentApplicationId, setCurrentApplicationId] = useState<string | null>(null);

  // Initialize from localStorage or environment variables
  useEffect(() => {
    const storedClientId = localStorage.getItem('current_client_id');
    const storedApplicationId = localStorage.getItem('current_application_id');

    // Use stored values or fallback to environment variables
    const clientId = storedClientId || process.env.REACT_APP_DEFAULT_CLIENT_ID || null;
    const applicationId = storedApplicationId || process.env.REACT_APP_DEFAULT_APPLICATION_ID || null;

    setCurrentClientId(clientId);
    setCurrentApplicationId(applicationId);
  }, []);

  // Persist to localStorage when values change
  useEffect(() => {
    if (currentClientId) {
      localStorage.setItem('current_client_id', currentClientId);
    } else {
      localStorage.removeItem('current_client_id');
    }
  }, [currentClientId]);

  useEffect(() => {
    if (currentApplicationId) {
      localStorage.setItem('current_application_id', currentApplicationId);
    } else {
      localStorage.removeItem('current_application_id');
    }
  }, [currentApplicationId]);

  const clearCurrentIds = () => {
    setCurrentClientId(null);
    setCurrentApplicationId(null);
    localStorage.removeItem('current_client_id');
    localStorage.removeItem('current_application_id');
  };

  const value: ClientContextType = {
    currentClientId,
    currentApplicationId,
    setCurrentClientId,
    setCurrentApplicationId,
    clearCurrentIds,
  };

  return (
    <ClientContext.Provider value={value}>
      {children}
    </ClientContext.Provider>
  );
};

export const useClient = (): ClientContextType => {
  const context = useContext(ClientContext);
  if (context === undefined) {
    throw new Error('useClient must be used within a ClientProvider');
  }
  return context;
};
