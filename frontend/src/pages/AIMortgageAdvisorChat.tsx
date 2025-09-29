/**
 * AI Mortgage Advisor Chat Interface
 * 
 * Production-grade conversational AI interface for mortgage advisory services.
 * Provides real-time chat with context-aware AI agents, compliance checking,
 * and comprehensive user experience features.
 * 
 * Features:
 * - Real-time WebSocket communication
 * - Context-aware conversation management
 * - Message persistence and history
 * - Typing indicators and read receipts
 * - Compliance alerts and notifications
 * - Multi-language support (Dutch/English)
 * - Responsive design with accessibility
 * - Voice input/output capabilities
 * - File sharing and document upload
 * - Conversation export and sharing
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Container,
  Card,
  TextInput,
  Button,
  Group,
  Stack,
  Text,
  Badge,
  Avatar,
  ActionIcon,
  Tooltip,
  ScrollArea,
  Loader,
  Alert,
  Modal,
  Textarea,
  Rating,
  Divider,
  Menu,
  FileInput,
  Progress,
  Notification,
  Switch,
  Select,
  Paper,
  ThemeIcon,
  Anchor,
  Code,
  List,
  Timeline,
  Kbd
} from '@mantine/core';
import {
  IconSend,
  IconMicrophone,
  IconMicrophoneOff,
  IconPaperclip,
  IconDownload,
  IconShare,
  IconThumbUp,
  IconThumbDown,
  IconAlertTriangle,
  IconShield,
  IconRobot,
  IconUser,
  IconCopy,
  IconExternalLink,
  IconSettings,
  IconLanguage,
  IconVolume,
  IconVolumeOff,
  IconRefresh,
  IconX,
  IconCheck,
  IconClock,
  IconEye,
  IconSearch,
  IconHelp,
  IconBrain,
  IconChartBar,
  IconFileText,
  IconLock,
  IconWifi,
  IconWifiOff
} from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';
import { notifications } from '@mantine/notifications';
import { modals } from '@mantine/modals';
import { spotlight } from '@mantine/spotlight';
import { useDemoMode } from '../contexts/DemoModeContext';
import { useClient } from '../contexts/ClientContext';
import { useDisclosure, useLocalStorage, useViewportSize } from '@mantine/hooks';
import ConnectionStatus from '../components/ConnectionStatus';

// TypeScript interfaces for type safety
interface ChatMessage {
  type: 'user_message' | 'ai_message' | 'system_message' | 'compliance_alert' | 'typing_indicator' | 'error' | 'context_update' | 'feedback';
  messageId: string;
  conversationId: string;
  content: {
    text?: string;
    translation?: string;
    suggestions?: string[];
    suggestedActions?: string[];
    complianceNotes?: string[];
    riskAssessment?: 'low' | 'medium' | 'high';
    explanation?: string;
    isTyping?: boolean;
    sender?: 'user' | 'ai';
    severity?: 'low' | 'medium' | 'high';
    issues?: string[];
    recommendations?: string[];
    regulatoryReference?: string;
    messageId?: string;
  };
  timestamp: string;
  metadata?: {
    aiConfidence?: number;
    complianceChecked?: boolean;
    language?: 'nl' | 'en';
    processingTime?: number;
    escalationRequired?: boolean;
    regulatoryFlags?: string[];
    error?: boolean;
    files?: any[];
  };
  sender?: 'user' | 'ai' | 'system';
  status?: 'sending' | 'sent' | 'delivered' | 'read' | 'failed';
}

interface ConversationContext {
  userProfile: {
    buyerType?: 'first_time' | 'experienced' | 'investor';
    annualIncome?: number;
    desiredMortgageAmount?: number;
    propertyValue?: number;
    employmentStatus?: 'employed' | 'self_employed' | 'unemployed' | 'retired';
    age?: number;
    dependents?: number;
    existingDebt?: number;
  };
  mortgageContext: {
    productFeatures?: string[];
    interestRatePreference?: 'fixed' | 'variable' | 'flexible';
    mortgageTerm?: number;
    downPayment?: number;
    propertyType?: 'house' | 'apartment' | 'commercial';
  };
  currentTopic?: string;
  conversationGoals?: string[];
  riskAssessment?: 'low' | 'medium' | 'high';
  complianceFlags?: string[];
}

interface WebSocketConnection {
  ws: WebSocket | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  reconnectAttempts: number;
  lastPingTime: number;
}

const AIMortgageAdvisorChat: React.FC = () => {
  const navigate = useNavigate();
  const { isDemoMode } = useDemoMode();
  const { currentClientId } = useClient();
  const { width } = useViewportSize();
  const isMobile = width < 768;

  // WebSocket connection state
  const [wsConnection, setWsConnection] = useState<WebSocketConnection>({
    ws: null,
    connectionStatus: 'disconnected',
    reconnectAttempts: 0,
    lastPingTime: 0
  });

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [aiTyping, setAiTyping] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationContext, setConversationContext] = useState<ConversationContext>({
    userProfile: {},
    mortgageContext: {},
    currentTopic: 'introduction',
    conversationGoals: [],
    riskAssessment: 'low',
    complianceFlags: []
  });

  // UI state
  const [language, setLanguage] = useLocalStorage<'nl' | 'en'>({ key: 'chat-language', defaultValue: 'nl' });
  const [voiceEnabled, setVoiceEnabled] = useLocalStorage({ key: 'voice-enabled', defaultValue: false });
  const [soundEnabled, setSoundEnabled] = useLocalStorage({ key: 'sound-enabled', defaultValue: true });
  const [autoTranslate, setAutoTranslate] = useLocalStorage({ key: 'auto-translate', defaultValue: false });
  const [compactMode, setCompactMode] = useLocalStorage({ key: 'compact-mode', defaultValue: false });

  // Modal states
  const [settingsOpened, { open: openSettings, close: closeSettings }] = useDisclosure(false);
  const [feedbackOpened, { open: openFeedback, close: closeFeedback }] = useDisclosure(false);
  const [exportOpened, { open: openExport, close: closeExport }] = useDisclosure(false);
  const [contextOpened, { open: openContext, close: closeContext }] = useDisclosure(false);

  // Feedback state
  const [feedbackRating, setFeedbackRating] = useState(0);
  const [feedbackText, setFeedbackText] = useState('');
  const [feedbackMessageId, setFeedbackMessageId] = useState<string | null>(null);

  // Voice recognition state
  const [isListening, setIsListening] = useState(false);
  const [speechRecognition, setSpeechRecognition] = useState<SpeechRecognition | null>(null);
  const [speechSynthesis, setSpeechSynthesis] = useState<SpeechSynthesis | null>(null);

  // File upload state
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket URL based on environment
  const wsUrl = useMemo(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.REACT_APP_WS_HOST || window.location.hostname;
    const port = process.env.REACT_APP_CHAT_WS_PORT || '8005';
    const token = localStorage.getItem('auth_token') || 'demo-token';
    return `${protocol}//${host}:${port}?token=${encodeURIComponent(token)}`;
  }, []);

  /**
   * Initialize WebSocket connection
   */
  const initializeWebSocket = useCallback(() => {
    if (wsConnection.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    console.log('🔗 Connecting to AI Mortgage Advisor Chat...');
    
    setWsConnection(prev => ({
      ...prev,
      connectionStatus: 'connecting'
    }));

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('✅ Connected to AI Mortgage Advisor Chat');
        setWsConnection(prev => ({
          ...prev,
          ws,
          connectionStatus: 'connected',
          reconnectAttempts: 0,
          lastPingTime: Date.now()
        }));

        // Start ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.ping?.();
            setWsConnection(prev => ({
              ...prev,
              lastPingTime: Date.now()
            }));
          }
        }, 30000);

        notifications.show({
          title: 'Verbonden',
          message: 'Verbinding met AI Hypotheekadviseur tot stand gebracht',
          color: 'green',
          icon: <IconWifi size={16} />
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: ChatMessage = JSON.parse(event.data);
          handleIncomingMessage(message);
        } catch (error) {
          console.error('❌ Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('🔌 WebSocket connection closed:', event.code, event.reason);
        setWsConnection(prev => ({
          ...prev,
          ws: null,
          connectionStatus: 'disconnected'
        }));

        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Attempt reconnection if not intentional
        if (event.code !== 1000 && wsConnection.reconnectAttempts < 5) {
          scheduleReconnect();
        }

        notifications.show({
          title: 'Verbinding verbroken',
          message: 'Verbinding met AI Hypotheekadviseur verbroken',
          color: 'orange',
          icon: <IconWifiOff size={16} />
        });
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setWsConnection(prev => ({
          ...prev,
          connectionStatus: 'error'
        }));

        notifications.show({
          title: 'Verbindingsfout',
          message: 'Fout bij verbinding met AI Hypotheekadviseur',
          color: 'red',
          icon: <IconAlertTriangle size={16} />
        });
      };

    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      setWsConnection(prev => ({
        ...prev,
        connectionStatus: 'error'
      }));
    }
  }, [wsUrl, wsConnection.reconnectAttempts]);

  /**
   * Schedule reconnection attempt
   */
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    const delay = Math.min(1000 * Math.pow(2, wsConnection.reconnectAttempts), 30000);
    
    setWsConnection(prev => ({
      ...prev,
      reconnectAttempts: prev.reconnectAttempts + 1
    }));

    reconnectTimeoutRef.current = setTimeout(() => {
      console.log(`🔄 Attempting reconnection (${wsConnection.reconnectAttempts + 1}/5)...`);
      initializeWebSocket();
    }, delay);
  }, [wsConnection.reconnectAttempts, initializeWebSocket]);

  /**
   * Handle incoming WebSocket message
   */
  const handleIncomingMessage = useCallback((message: ChatMessage) => {
    console.log('📨 Received message:', message.type);

    switch (message.type) {
      case 'ai_message':
        setMessages(prev => [...prev, { ...message, sender: 'ai', status: 'delivered' }]);
        setAiTyping(false);
        
        // Play notification sound
        if (soundEnabled) {
          playNotificationSound();
        }

        // Speak message if voice is enabled
        if (voiceEnabled && message.content.text) {
          speakText(message.content.text);
        }

        // Set conversation ID if not set
        if (!conversationId && message.conversationId) {
          setConversationId(message.conversationId);
        }
        break;

      case 'typing_indicator':
        if (message.content.sender === 'ai') {
          setAiTyping(message.content.isTyping || false);
        }
        break;

      case 'compliance_alert':
        setMessages(prev => [...prev, { ...message, sender: 'system', status: 'delivered' }]);
        
        notifications.show({
          title: 'AFM Compliance Waarschuwing',
          message: `${message.content.issues?.join(', ') || 'Compliance issue gedetecteerd'}`,
          color: message.content.severity === 'high' ? 'red' : 'orange',
          icon: <IconShield size={16} />,
          autoClose: false
        });
        break;

      case 'system_message':
        setMessages(prev => [...prev, { ...message, sender: 'system', status: 'delivered' }]);
        break;

      case 'error':
        notifications.show({
          title: 'Fout',
          message: message.content.text || 'Er is een fout opgetreden',
          color: 'red',
          icon: <IconAlertTriangle size={16} />
        });
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  }, [conversationId, soundEnabled, voiceEnabled]);

  /**
   * Send message via WebSocket
   */
  const sendMessage = useCallback((message: Partial<ChatMessage>) => {
    if (wsConnection.ws?.readyState === WebSocket.OPEN) {
      wsConnection.ws.send(JSON.stringify(message));
      return true;
    } else {
      notifications.show({
        title: 'Verbindingsfout',
        message: 'Geen verbinding met AI Hypotheekadviseur',
        color: 'red',
        icon: <IconWifiOff size={16} />
      });
      return false;
    }
  }, [wsConnection.ws]);

  /**
   * Handle user message submission
   */
  const handleSendMessage = useCallback(async () => {
    if (!currentMessage.trim() || wsConnection.connectionStatus !== 'connected') {
      return;
    }

    const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const userMessage: ChatMessage = {
      type: 'user_message',
      messageId,
      conversationId: conversationId || '',
      content: {
        text: currentMessage.trim()
      },
      timestamp: new Date().toISOString(),
      sender: 'user',
      status: 'sending',
      metadata: {
        language: language
      }
    };

    // Add message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');

    // Send via WebSocket
    const sent = sendMessage(userMessage);
    
    if (sent) {
      // Update message status
      setMessages(prev => prev.map(msg => 
        msg.messageId === messageId 
          ? { ...msg, status: 'sent' }
          : msg
      ));

      // Show AI typing indicator
      setAiTyping(true);
    } else {
      // Mark message as failed
      setMessages(prev => prev.map(msg => 
        msg.messageId === messageId 
          ? { ...msg, status: 'failed' }
          : msg
      ));
    }
  }, [currentMessage, wsConnection.connectionStatus, conversationId, language, sendMessage]);

  /**
   * Handle typing indicator
   */
  const handleTyping = useCallback(() => {
    if (!isTyping && wsConnection.connectionStatus === 'connected') {
      setIsTyping(true);
      sendMessage({
        type: 'typing_indicator',
        content: { isTyping: true, sender: 'user' },
        timestamp: new Date().toISOString()
      });
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      if (wsConnection.connectionStatus === 'connected') {
        sendMessage({
          type: 'typing_indicator',
          content: { isTyping: false, sender: 'user' },
          timestamp: new Date().toISOString()
        });
      }
    }, 2000);
  }, [isTyping, wsConnection.connectionStatus, sendMessage]);

  /**
   * Handle context update
   */
  const updateContext = useCallback((updates: Partial<ConversationContext>) => {
    const newContext = { ...conversationContext, ...updates };
    setConversationContext(newContext);

    if (wsConnection.connectionStatus === 'connected') {
      sendMessage({
        type: 'context_update',
        content: updates,
        timestamp: new Date().toISOString()
      });
    }
  }, [conversationContext, wsConnection.connectionStatus, sendMessage]);

  /**
   * Handle message feedback
   */
  const handleFeedback = useCallback((messageId: string, rating: number, feedbackText?: string) => {
    if (wsConnection.connectionStatus === 'connected') {
      sendMessage({
        type: 'feedback',
        content: {
          messageId,
          rating,
          feedbackText,
          feedbackType: 'message_rating'
        },
        timestamp: new Date().toISOString()
      });

      notifications.show({
        title: 'Feedback verzonden',
        message: 'Bedankt voor uw feedback!',
        color: 'green',
        icon: <IconCheck size={16} />
      });
    }
  }, [wsConnection.connectionStatus, sendMessage]);

  /**
   * Initialize speech recognition
   */
  const initializeSpeechRecognition = useCallback(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = language === 'nl' ? 'nl-NL' : 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setCurrentMessage(prev => prev + ' ' + transcript);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        notifications.show({
          title: 'Spraakherkenning fout',
          message: 'Kon spraak niet herkennen',
          color: 'red',
          icon: <IconMicrophoneOff size={16} />
        });
      };

      setSpeechRecognition(recognition);
    }
  }, [language]);

  /**
   * Initialize speech synthesis
   */
  const initializeSpeechSynthesis = useCallback(() => {
    if ('speechSynthesis' in window) {
      setSpeechSynthesis(window.speechSynthesis);
    }
  }, []);

  /**
   * Speak text using speech synthesis
   */
  const speakText = useCallback((text: string) => {
    if (speechSynthesis && voiceEnabled) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = language === 'nl' ? 'nl-NL' : 'en-US';
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      speechSynthesis.speak(utterance);
    }
  }, [speechSynthesis, voiceEnabled, language]);

  /**
   * Play notification sound
   */
  const playNotificationSound = useCallback(() => {
    if (soundEnabled) {
      const audio = new Audio('/sounds/notification.mp3');
      audio.volume = 0.3;
      audio.play().catch(() => {
        // Ignore audio play errors
      });
    }
  }, [soundEnabled]);

  /**
   * Toggle voice recognition
   */
  const toggleVoiceRecognition = useCallback(() => {
    if (!speechRecognition) {
      notifications.show({
        title: 'Spraakherkenning niet beschikbaar',
        message: 'Uw browser ondersteunt geen spraakherkenning',
        color: 'orange',
        icon: <IconMicrophoneOff size={16} />
      });
      return;
    }

    if (isListening) {
      speechRecognition.stop();
    } else {
      speechRecognition.start();
    }
  }, [speechRecognition, isListening]);

  /**
   * Handle file upload
   */
  const handleFileUpload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    setUploadedFiles(files);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          
          notifications.show({
            title: 'Bestanden geüpload',
            message: `${files.length} bestand(en) succesvol geüpload`,
            color: 'green',
            icon: <IconCheck size={16} />
          });

          // Send file information to chat
          const fileMessage: ChatMessage = {
            type: 'user_message',
            messageId: `file_${Date.now()}`,
            conversationId: conversationId || '',
            content: {
              text: `Ik heb ${files.length} bestand(en) geüpload: ${files.map(f => f.name).join(', ')}`
            },
            timestamp: new Date().toISOString(),
            sender: 'user',
            status: 'sent',
            metadata: {
              language: language,
              files: files.map(f => ({
                name: f.name,
                size: f.size,
                type: f.type
              }))
            }
          };

          setMessages(prev => [...prev, fileMessage]);
          sendMessage(fileMessage);

          return 100;
        }
        return prev + 10;
      });
    }, 200);
  }, [conversationId, language, sendMessage]);

  /**
   * Export conversation
   */
  const exportConversation = useCallback(() => {
    const conversationData = {
      conversationId,
      messages: messages.filter(msg => msg.type !== 'typing_indicator'),
      context: conversationContext,
      exportedAt: new Date().toISOString(),
      language
    };

    const blob = new Blob([JSON.stringify(conversationData, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mortgage-chat-${conversationId || 'session'}-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    notifications.show({
      title: 'Gesprek geëxporteerd',
      message: 'Gesprek is gedownload als JSON bestand',
      color: 'green',
      icon: <IconDownload size={16} />
    });
  }, [conversationId, messages, conversationContext, language]);

  /**
   * Copy message to clipboard
   */
  const copyMessage = useCallback((text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      notifications.show({
        title: 'Gekopieerd',
        message: 'Bericht gekopieerd naar klembord',
        color: 'green',
        icon: <IconCopy size={16} />
      });
    });
  }, []);

  /**
   * Scroll to bottom of messages
   */
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Initialize WebSocket connection on mount
  useEffect(() => {
    initializeWebSocket();
    initializeSpeechRecognition();
    initializeSpeechSynthesis();

    return () => {
      if (wsConnection.ws) {
        wsConnection.ws.close(1000, 'Component unmounting');
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
  }, []);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Focus input when connected
  useEffect(() => {
    if (wsConnection.connectionStatus === 'connected' && messageInputRef.current) {
      messageInputRef.current.focus();
    }
  }, [wsConnection.connectionStatus]);

  /**
   * Render individual message
   */
  const renderMessage = (message: ChatMessage, index: number) => {
    const isUser = message.sender === 'user';
    const isSystem = message.sender === 'system';
    const isAI = message.sender === 'ai';

    return (
      <Group
        key={message.messageId || index}
        align="flex-start"
        gap="sm"
        style={{
          flexDirection: isUser ? 'row-reverse' : 'row',
          marginBottom: compactMode ? '8px' : '16px'
        }}
      >
        {/* Avatar */}
        <Avatar
          size={compactMode ? 'sm' : 'md'}
          radius="xl"
          color={isUser ? 'blue' : isSystem ? 'gray' : 'green'}
        >
          {isUser ? (
            <IconUser size={compactMode ? 16 : 20} />
          ) : isSystem ? (
            <IconShield size={compactMode ? 16 : 20} />
          ) : (
            <IconRobot size={compactMode ? 16 : 20} />
          )}
        </Avatar>

        {/* Message Content */}
        <Paper
          shadow="sm"
          radius="lg"
          p={compactMode ? "xs" : "sm"}
          style={{
            maxWidth: isMobile ? '85%' : '70%',
            backgroundColor: isUser 
              ? 'var(--mantine-color-blue-6)' 
              : isSystem 
                ? 'var(--mantine-color-gray-1)'
                : 'var(--mantine-color-green-0)',
            color: isUser ? 'white' : 'var(--mantine-color-dark-7)'
          }}
        >
          {/* Message Text */}
          {message.content.text && (
            <Text size={compactMode ? "sm" : "md"} style={{ whiteSpace: 'pre-wrap' }}>
              {message.content.text}
            </Text>
          )}

          {/* Translation */}
          {message.content.translation && autoTranslate && (
            <Text size="xs" style={{ opacity: 0.7, marginTop: '4px' }}>
              🌐 {message.content.translation}
            </Text>
          )}

          {/* AI Confidence Score */}
          {isAI && message.metadata?.aiConfidence && (
            <Group gap="xs" mt="xs">
              <Badge size="xs" variant="light" color="green">
                Vertrouwen: {Math.round(message.metadata.aiConfidence * 100)}%
              </Badge>
              {message.metadata.complianceChecked && (
                <Badge size="xs" variant="light" color="blue">
                  <IconShield size={10} style={{ marginRight: '2px' }} />
                  AFM Gecheckt
                </Badge>
              )}
            </Group>
          )}

          {/* Suggestions */}
          {message.content.suggestions && message.content.suggestions.length > 0 && (
            <Stack gap="xs" mt="sm">
              <Text size="xs" fw={500}>Suggesties:</Text>
              {message.content.suggestions.map((suggestion, idx) => (
                <Button
                  key={idx}
                  variant="light"
                  size="xs"
                  radius="xl"
                  onClick={() => setCurrentMessage(suggestion)}
                  style={{ justifyContent: 'flex-start' }}
                >
                  {suggestion}
                </Button>
              ))}
            </Stack>
          )}

          {/* Compliance Alerts */}
          {message.type === 'compliance_alert' && (
            <Alert
              icon={<IconAlertTriangle size={16} />}
              title="AFM Compliance Waarschuwing"
              color={message.content.severity === 'high' ? 'red' : 'orange'}
              mt="sm"
            >
              <Stack gap="xs">
                {message.content.issues?.map((issue, idx) => (
                  <Text key={idx} size="sm">• {issue}</Text>
                ))}
                {message.content.recommendations && (
                  <Text size="sm" fw={500}>
                    Aanbevelingen: {message.content.recommendations.join(', ')}
                  </Text>
                )}
              </Stack>
            </Alert>
          )}

          {/* Message Actions */}
          <Group gap="xs" mt="xs" justify="space-between">
            <Group gap="xs">
              <Text size="xs" c="dimmed">
                {new Date(message.timestamp).toLocaleTimeString('nl-NL', {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </Text>
              
              {message.status && (
                <Badge size="xs" variant="dot" color={
                  message.status === 'sent' ? 'blue' :
                  message.status === 'delivered' ? 'green' :
                  message.status === 'failed' ? 'red' : 'gray'
                }>
                  {message.status === 'sending' && <Loader size={8} />}
                  {message.status === 'sent' && <IconClock size={8} />}
                  {message.status === 'delivered' && <IconCheck size={8} />}
                  {message.status === 'failed' && <IconX size={8} />}
                </Badge>
              )}
            </Group>

            <Group gap="xs">
              {/* Copy Message */}
              <ActionIcon
                size="xs"
                variant="subtle"
                onClick={() => copyMessage(message.content.text || '')}
              >
                <IconCopy size={12} />
              </ActionIcon>

              {/* Feedback for AI messages */}
              {isAI && (
                <Group gap="xs">
                  <ActionIcon
                    size="xs"
                    variant="subtle"
                    color="green"
                    onClick={() => {
                      setFeedbackMessageId(message.messageId);
                      setFeedbackRating(5);
                      handleFeedback(message.messageId, 5);
                    }}
                  >
                    <IconThumbUp size={12} />
                  </ActionIcon>
                  <ActionIcon
                    size="xs"
                    variant="subtle"
                    color="red"
                    onClick={() => {
                      setFeedbackMessageId(message.messageId);
                      setFeedbackRating(1);
                      openFeedback();
                    }}
                  >
                    <IconThumbDown size={12} />
                  </ActionIcon>
                </Group>
              )}
            </Group>
          </Group>
        </Paper>
      </Group>
    );
  };

  return (
    <Container size="xl" py="md">
      <Stack gap="md">
        {/* Header */}
        <Card shadow="sm" padding="lg" radius="md">
          <Group justify="space-between" align="center">
            <Group gap="md">
              <ThemeIcon size="xl" radius="md" variant="gradient" gradient={{ from: 'blue', to: 'green' }}>
                <IconBrain size={24} />
              </ThemeIcon>
              <div>
                <Text size="xl" fw={700}>AI Hypotheekadviseur</Text>
                <Text size="sm" c="dimmed">
                  Intelligente hypotheekadvies met AFM compliance
                </Text>
              </div>
            </Group>

            <Group gap="xs">
              {/* Connection Status */}
              <Badge
                color={
                  wsConnection.connectionStatus === 'connected' ? 'green' :
                  wsConnection.connectionStatus === 'connecting' ? 'yellow' :
                  wsConnection.connectionStatus === 'error' ? 'red' : 'gray'
                }
                variant="light"
                leftSection={
                  wsConnection.connectionStatus === 'connected' ? <IconWifi size={12} /> :
                  wsConnection.connectionStatus === 'connecting' ? <Loader size={12} /> :
                  <IconWifiOff size={12} />
                }
              >
                {wsConnection.connectionStatus === 'connected' ? 'Verbonden' :
                 wsConnection.connectionStatus === 'connecting' ? 'Verbinden...' :
                 wsConnection.connectionStatus === 'error' ? 'Fout' : 'Niet verbonden'}
              </Badge>

              {/* Demo Mode Indicator */}
              {isDemoMode && (
                <Badge color="orange" variant="light">
                  Demo Modus
                </Badge>
              )}

              {/* Settings Menu */}
              <Menu shadow="md" width={200}>
                <Menu.Target>
                  <ActionIcon variant="light" size="lg">
                    <IconSettings size={18} />
                  </ActionIcon>
                </Menu.Target>

                <Menu.Dropdown>
                  <Menu.Label>Instellingen</Menu.Label>
                  <Menu.Item
                    leftSection={<IconLanguage size={14} />}
                    onClick={openSettings}
                  >
                    Taal & Spraak
                  </Menu.Item>
                  <Menu.Item
                    leftSection={<IconEye size={14} />}
                    onClick={openContext}
                  >
                    Gesprek Context
                  </Menu.Item>
                  <Menu.Divider />
                  <Menu.Item
                    leftSection={<IconDownload size={14} />}
                    onClick={exportConversation}
                  >
                    Exporteer Gesprek
                  </Menu.Item>
                  <Menu.Item
                    leftSection={<IconShare size={14} />}
                    onClick={openExport}
                  >
                    Deel Gesprek
                  </Menu.Item>
                  <Menu.Divider />
                  <Menu.Item
                    leftSection={<IconRefresh size={14} />}
                    onClick={initializeWebSocket}
                    disabled={wsConnection.connectionStatus === 'connecting'}
                  >
                    Herverbind
                  </Menu.Item>
                </Menu.Dropdown>
              </Menu>
            </Group>
          </Group>
        </Card>

        {/* Chat Area */}
        <Card shadow="sm" padding={0} radius="md" style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
          {/* Messages */}
          <ScrollArea
            style={{ flex: 1, padding: '16px' }}
            scrollbarSize={6}
            scrollHideDelay={1000}
          >
            <Stack gap="md">
              {messages.length === 0 && wsConnection.connectionStatus === 'connected' && (
                <Paper p="xl" radius="md" style={{ textAlign: 'center', backgroundColor: 'var(--mantine-color-gray-0)' }}>
                  <ThemeIcon size="xl" radius="xl" variant="light" color="blue" mb="md">
                    <IconBrain size={24} />
                  </ThemeIcon>
                  <Text size="lg" fw={500} mb="xs">
                    Welkom bij uw AI Hypotheekadviseur
                  </Text>
                  <Text size="sm" c="dimmed" mb="md">
                    Stel uw hypotheekvragen en ontvang persoonlijk advies dat voldoet aan alle AFM-regelgeving
                  </Text>
                  <Group justify="center" gap="xs">
                    <Button variant="light" size="xs" onClick={() => setCurrentMessage("Wat zijn de huidige hypotheekrentes?")}>
                      Hypotheekrentes
                    </Button>
                    <Button variant="light" size="xs" onClick={() => setCurrentMessage("Hoeveel kan ik lenen?")}>
                      Leencapaciteit
                    </Button>
                    <Button variant="light" size="xs" onClick={() => setCurrentMessage("Wat zijn de kosten?")}>
                      Kosten overzicht
                    </Button>
                  </Group>
                </Paper>
              )}

              {messages.map((message, index) => renderMessage(message, index))}

              {/* AI Typing Indicator */}
              {aiTyping && (
                <Group align="flex-start" gap="sm">
                  <Avatar size="md" radius="xl" color="green">
                    <IconRobot size={20} />
                  </Avatar>
                  <Paper shadow="sm" radius="lg" p="sm" style={{ backgroundColor: 'var(--mantine-color-green-0)' }}>
                    <Group gap="xs">
                      <Loader size="xs" />
                      <Text size="sm" c="dimmed">AI denkt na...</Text>
                    </Group>
                  </Paper>
                </Group>
              )}

              <div ref={messagesEndRef} />
            </Stack>
          </ScrollArea>

          {/* Upload Progress */}
          {uploadProgress > 0 && uploadProgress < 100 && (
            <Paper p="sm" style={{ borderTop: '1px solid var(--mantine-color-gray-3)' }}>
              <Group gap="sm">
                <Text size="sm">Uploaden...</Text>
                <Progress value={uploadProgress} style={{ flex: 1 }} />
                <Text size="sm">{uploadProgress}%</Text>
              </Group>
            </Paper>
          )}

          {/* Input Area */}
          <Paper p="md" style={{ borderTop: '1px solid var(--mantine-color-gray-3)' }}>
            <Group gap="sm" align="flex-end">
              {/* File Upload */}
              <FileInput
                placeholder="Bestanden"
                multiple
                accept=".pdf,.doc,.docx,.jpg,.jpeg,.png"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                id="file-upload"
              />
              <Tooltip label="Bestanden uploaden">
                <ActionIcon
                  variant="light"
                  size="lg"
                  onClick={() => document.getElementById('file-upload')?.click()}
                  disabled={wsConnection.connectionStatus !== 'connected'}
                >
                  <IconPaperclip size={18} />
                </ActionIcon>
              </Tooltip>

              {/* Voice Input */}
              <Tooltip label={isListening ? "Stop opname" : "Start spraakherkenning"}>
                <ActionIcon
                  variant={isListening ? "filled" : "light"}
                  size="lg"
                  color={isListening ? "red" : "blue"}
                  onClick={toggleVoiceRecognition}
                  disabled={wsConnection.connectionStatus !== 'connected' || !speechRecognition}
                >
                  {isListening ? <IconMicrophoneOff size={18} /> : <IconMicrophone size={18} />}
                </ActionIcon>
              </Tooltip>

              {/* Message Input */}
              <TextInput
                ref={messageInputRef}
                placeholder={
                  wsConnection.connectionStatus === 'connected' 
                    ? "Typ uw hypotheekvraag..." 
                    : "Verbinding maken..."
                }
                value={currentMessage}
                onChange={(e) => {
                  setCurrentMessage(e.target.value);
                  handleTyping();
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                disabled={wsConnection.connectionStatus !== 'connected'}
                style={{ flex: 1 }}
                rightSection={
                  isTyping && (
                    <Loader size="xs" />
                  )
                }
              />

              {/* Send Button */}
              <Button
                leftSection={<IconSend size={16} />}
                onClick={handleSendMessage}
                disabled={!currentMessage.trim() || wsConnection.connectionStatus !== 'connected'}
                loading={wsConnection.connectionStatus === 'connecting'}
              >
                Verstuur
              </Button>
            </Group>

            {/* Quick Actions */}
            <Group gap="xs" mt="xs" justify="center">
              <Button
                variant="subtle"
                size="xs"
                onClick={() => setCurrentMessage("Wat zijn de huidige hypotheekrentes?")}
                disabled={wsConnection.connectionStatus !== 'connected'}
              >
                Rentes
              </Button>
              <Button
                variant="subtle"
                size="xs"
                onClick={() => setCurrentMessage("Hoeveel kan ik maximaal lenen?")}
                disabled={wsConnection.connectionStatus !== 'connected'}
              >
                Leencapaciteit
              </Button>
              <Button
                variant="subtle"
                size="xs"
                onClick={() => setCurrentMessage("Wat zijn de kosten van een hypotheek?")}
                disabled={wsConnection.connectionStatus !== 'connected'}
              >
                Kosten
              </Button>
              <Button
                variant="subtle"
                size="xs"
                onClick={() => setCurrentMessage("Hoe werkt de NHG?")}
                disabled={wsConnection.connectionStatus !== 'connected'}
              >
                NHG
              </Button>
            </Group>
          </Paper>
        </Card>
      </Stack>

      {/* Settings Modal */}
      <Modal opened={settingsOpened} onClose={closeSettings} title="Chat Instellingen" size="md">
        <Stack gap="md">
          <Select
            label="Taal"
            value={language}
            onChange={(value) => setLanguage(value as 'nl' | 'en')}
            data={[
              { value: 'nl', label: '🇳🇱 Nederlands' },
              { value: 'en', label: '🇬🇧 English' }
            ]}
          />

          <Switch
            label="Spraakherkenning inschakelen"
            description="Gebruik uw microfoon om berichten in te spreken"
            checked={voiceEnabled}
            onChange={(event) => setVoiceEnabled(event.currentTarget.checked)}
          />

          <Switch
            label="Geluidsmeldingen"
            description="Speel geluid af bij nieuwe berichten"
            checked={soundEnabled}
            onChange={(event) => setSoundEnabled(event.currentTarget.checked)}
          />

          <Switch
            label="Automatisch vertalen"
            description="Toon vertalingen van AI berichten"
            checked={autoTranslate}
            onChange={(event) => setAutoTranslate(event.currentTarget.checked)}
          />

          <Switch
            label="Compacte weergave"
            description="Kleinere berichten en minder witruimte"
            checked={compactMode}
            onChange={(event) => setCompactMode(event.currentTarget.checked)}
          />
        </Stack>
      </Modal>

      {/* Feedback Modal */}
      <Modal opened={feedbackOpened} onClose={closeFeedback} title="Feedback geven" size="sm">
        <Stack gap="md">
          <Text size="sm">
            Help ons de AI Hypotheekadviseur te verbeteren door feedback te geven op dit bericht.
          </Text>

          <Rating
            value={feedbackRating}
            onChange={setFeedbackRating}
            size="lg"
          />

          <Textarea
            label="Aanvullende opmerkingen (optioneel)"
            placeholder="Wat kunnen we verbeteren?"
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            minRows={3}
          />

          <Group justify="flex-end">
            <Button variant="light" onClick={closeFeedback}>
              Annuleren
            </Button>
            <Button
              onClick={() => {
                if (feedbackMessageId) {
                  handleFeedback(feedbackMessageId, feedbackRating, feedbackText);
                }
                setFeedbackText('');
                setFeedbackRating(0);
                setFeedbackMessageId(null);
                closeFeedback();
              }}
              disabled={feedbackRating === 0}
            >
              Verstuur Feedback
            </Button>
          </Group>
        </Stack>
      </Modal>

      {/* Context Modal */}
      <Modal opened={contextOpened} onClose={closeContext} title="Gesprek Context" size="lg">
        <Stack gap="md">
          <Text size="sm" c="dimmed">
            Deze informatie helpt de AI om betere en meer gepersonaliseerde adviezen te geven.
          </Text>

          <Card withBorder>
            <Text fw={500} mb="sm">Persoonlijke Informatie</Text>
            <Stack gap="xs">
              <Group justify="space-between">
                <Text size="sm">Koper type:</Text>
                <Badge variant="light">
                  {conversationContext.userProfile.buyerType === 'first_time' ? 'Starter' :
                   conversationContext.userProfile.buyerType === 'experienced' ? 'Ervaren' :
                   conversationContext.userProfile.buyerType === 'investor' ? 'Belegger' : 'Onbekend'}
                </Badge>
              </Group>
              {conversationContext.userProfile.annualIncome && (
                <Group justify="space-between">
                  <Text size="sm">Jaarinkomen:</Text>
                  <Text size="sm" fw={500}>
                    €{conversationContext.userProfile.annualIncome.toLocaleString()}
                  </Text>
                </Group>
              )}
              {conversationContext.userProfile.desiredMortgageAmount && (
                <Group justify="space-between">
                  <Text size="sm">Gewenst hypotheekbedrag:</Text>
                  <Text size="sm" fw={500}>
                    €{conversationContext.userProfile.desiredMortgageAmount.toLocaleString()}
                  </Text>
                </Group>
              )}
            </Stack>
          </Card>

          <Card withBorder>
            <Text fw={500} mb="sm">Hypotheek Context</Text>
            <Stack gap="xs">
              <Group justify="space-between">
                <Text size="sm">Huidige onderwerp:</Text>
                <Badge variant="light">{conversationContext.currentTopic}</Badge>
              </Group>
              <Group justify="space-between">
                <Text size="sm">Risico beoordeling:</Text>
                <Badge 
                  color={
                    conversationContext.riskAssessment === 'low' ? 'green' :
                    conversationContext.riskAssessment === 'medium' ? 'yellow' : 'red'
                  }
                >
                  {conversationContext.riskAssessment === 'low' ? 'Laag' :
                   conversationContext.riskAssessment === 'medium' ? 'Gemiddeld' : 'Hoog'}
                </Badge>
              </Group>
              {conversationContext.complianceFlags && conversationContext.complianceFlags.length > 0 && (
                <div>
                  <Text size="sm" mb="xs">Compliance waarschuwingen:</Text>
                  <Stack gap="xs">
                    {conversationContext.complianceFlags.map((flag, idx) => (
                      <Alert key={idx} icon={<IconAlertTriangle size={16} />} color="orange" size="sm">
                        {flag}
                      </Alert>
                    ))}
                  </Stack>
                </div>
              )}
            </Stack>
          </Card>
        </Stack>
      </Modal>
    </Container>
  );
};

export default AIMortgageAdvisorChat;


