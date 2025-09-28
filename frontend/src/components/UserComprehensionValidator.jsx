import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent
} from './ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from './ui/tabs';
import {
  Button
} from './ui/button';
import {
  Badge
} from './ui/badge';
import {
  Progress
} from './ui/progress';
import {
  Alert,
  AlertDescription,
  AlertTitle
} from './ui/alert';
import {
  Input
} from './ui/input';
import {
  Label
} from './ui/label';
import {
  RadioGroup,
  RadioGroupItem
} from './ui/radio-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from './ui/select';
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  Brain,
  Target,
  BookOpen,
  Award,
  TrendingUp,
  TrendingDown,
  Clock,
  Users,
  BarChart3,
  Activity,
  Lightbulb,
  RefreshCw,
  Play,
  Pause,
  SkipForward,
  Eye,
  Star,
  Trophy,
  Zap,
  Shield,
  Gauge,
  Flag,
  Settings,
  HelpCircle,
  CheckSquare,
  ArrowRight,
  Timer,
  Smile,
  Frown,
  Meh
} from 'lucide-react';

/**
 * User Comprehension Validator Component
 * Advanced UI for adaptive testing, knowledge assessment, and learning path optimization
 */
const UserComprehensionValidator = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [assessments, setAssessments] = useState([]);
  const [learningPaths, setLearningPaths] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [userProgress, setUserProgress] = useState({});
  const [analytics, setAnalytics] = useState({});
  const [gamification, setGamification] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Assessment session state
  const [activeSession, setActiveSession] = useState(null);
  const [sessionProgress, setSessionProgress] = useState(0);
  const [questionsAnswered, setQuestionsAnswered] = useState(0);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [confidenceLevel, setConfidenceLevel] = useState(3);
  const [questionStartTime, setQuestionStartTime] = useState(null);

  // Learning state
  const [selectedModule, setSelectedModule] = useState(null);
  const [moduleProgress, setModuleProgress] = useState({});

  // WebSocket connection
  const wsRef = useRef(null);

  // Initialize component
  useEffect(() => {
    initializeComponent();
    setupWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  /**
   * Initialize component data
   */
  const initializeComponent = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadAssessments(),
        loadLearningPaths(),
        loadUserProgress(),
        loadAnalytics(),
        loadGamification()
      ]);
    } catch (err) {
      setError('Failed to initialize User Comprehension Validator');
      console.error('Initialization error:', err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Setup WebSocket connection for real-time updates
   */
  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8016`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to User Comprehension Validator');
        wsRef.current.send(JSON.stringify({ type: 'subscribe_assessments' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_learning_progress' }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (err) {
          console.error('WebSocket message error:', err);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected, attempting to reconnect...');
        setTimeout(setupWebSocket, 5000);
      };
    } catch (err) {
      console.error('WebSocket setup error:', err);
    }
  };

  /**
   * Handle WebSocket messages
   */
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'assessment_session_created':
        setActiveSession(data.data.session_id);
        showNotification('Assessment session started', 'success');
        break;
      case 'question_answered':
        setQuestionsAnswered(prev => prev + 1);
        showNotification(data.data.is_correct ? 'Correct answer!' : 'Incorrect answer', 
          data.data.is_correct ? 'success' : 'warning');
        break;
      case 'assessment_completed':
        loadAssessments();
        showNotification(`Assessment completed! Score: ${(data.data.overall_score * 100).toFixed(0)}%`, 'success');
        setActiveSession(null);
        break;
      case 'learning_path_created':
        loadLearningPaths();
        showNotification('Personalized learning path created', 'success');
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load assessments
   */
  const loadAssessments = async () => {
    try {
      const response = await fetch('/api/user-comprehension/assessments?limit=10');
      const data = await response.json();
      
      if (data.success) {
        setAssessments(data.assessments);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load assessments:', err);
    }
  };

  /**
   * Load learning paths
   */
  const loadLearningPaths = async () => {
    try {
      const response = await fetch('/api/user-comprehension/learning-paths');
      const data = await response.json();
      
      if (data.success) {
        setLearningPaths(data.learning_paths);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load learning paths:', err);
    }
  };

  /**
   * Load user progress
   */
  const loadUserProgress = async () => {
    try {
      const userId = 'demo_user_123'; // In real app, get from auth context
      const response = await fetch(`/api/user-comprehension/progress/${userId}`);
      const data = await response.json();
      
      if (data.success) {
        setUserProgress(data.progress_summary);
        setModuleProgress(data.module_progress);
      }
    } catch (err) {
      console.error('Failed to load user progress:', err);
    }
  };

  /**
   * Load analytics
   */
  const loadAnalytics = async () => {
    try {
      const response = await fetch('/api/user-comprehension/analytics/dashboard');
      const data = await response.json();
      
      if (data.success) {
        setAnalytics(data.analytics);
      }
    } catch (err) {
      console.error('Failed to load analytics:', err);
    }
  };

  /**
   * Load gamification data
   */
  const loadGamification = async () => {
    try {
      const userId = 'demo_user_123';
      const response = await fetch(`/api/user-comprehension/gamification/${userId}`);
      const data = await response.json();
      
      if (data.success) {
        setGamification(data.gamification);
      }
    } catch (err) {
      console.error('Failed to load gamification data:', err);
    }
  };

  /**
   * Start new assessment
   */
  const startAssessment = async (assessmentGoals = ['mortgage_basics']) => {
    setLoading(true);
    try {
      const response = await fetch('/api/user-comprehension/assessments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: 'demo_user_123',
          assessment_goals: assessmentGoals,
          user_preferences: {
            learning_style: 'multimodal',
            max_questions: 15,
            adaptive_algorithm: 'cat'
          }
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setActiveSession(data.session_id);
        setQuestionsAnswered(0);
        setSessionProgress(0);
        await getNextQuestion(data.session_id);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Failed to start assessment: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get next question
   */
  const getNextQuestion = async (sessionId) => {
    try {
      const response = await fetch(`/api/user-comprehension/assessments/${sessionId}/next-question`);
      const data = await response.json();
      
      if (data.success) {
        if (data.question) {
          setCurrentQuestion(data.question);
          setCurrentAnswer('');
          setConfidenceLevel(3);
          setQuestionStartTime(Date.now());
        } else {
          // Assessment complete
          await completeAssessment(sessionId);
        }
      }
    } catch (err) {
      showNotification(`Failed to get next question: ${err.message}`, 'error');
    }
  };

  /**
   * Submit answer
   */
  const submitAnswer = async () => {
    if (!currentQuestion || !activeSession) return;

    setLoading(true);
    try {
      const responseTime = Math.round((Date.now() - questionStartTime) / 1000);
      
      const response = await fetch(`/api/user-comprehension/assessments/${activeSession}/responses`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question_id: currentQuestion.question_id,
          user_answer: currentAnswer,
          confidence_level: confidenceLevel,
          response_time: responseTime
        })
      });

      const data = await response.json();
      
      if (data.success) {
        // Show feedback
        const isCorrect = data.feedback.is_correct;
        showNotification(
          isCorrect ? 'Correct! Well done.' : 'Incorrect. Review the explanation.',
          isCorrect ? 'success' : 'warning'
        );
        
        // Update progress
        setQuestionsAnswered(prev => prev + 1);
        setSessionProgress(prev => prev + (100 / 15)); // Assuming max 15 questions
        
        // Get next question after brief delay
        setTimeout(() => {
          getNextQuestion(activeSession);
        }, 2000);
      }
    } catch (err) {
      showNotification(`Failed to submit answer: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Complete assessment
   */
  const completeAssessment = async (sessionId) => {
    try {
      const response = await fetch(`/api/user-comprehension/assessments/${sessionId}/complete`, {
        method: 'POST'
      });

      const data = await response.json();
      
      if (data.success) {
        setCurrentQuestion(null);
        setActiveSession(null);
        loadAssessments();
        showNotification('Assessment completed successfully!', 'success');
      }
    } catch (err) {
      showNotification(`Failed to complete assessment: ${err.message}`, 'error');
    }
  };

  /**
   * Show notification
   */
  const showNotification = (message, type = 'info') => {
    console.log(`${type.toUpperCase()}: ${message}`);
  };

  /**
   * Get comprehension level color
   */
  const getComprehensionLevelColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'excellent': return 'text-green-600';
      case 'good': return 'text-blue-600';
      case 'adequate': return 'text-yellow-600';
      case 'basic': return 'text-orange-600';
      case 'insufficient': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  /**
   * Get comprehension badge variant
   */
  const getComprehensionBadgeVariant = (level) => {
    switch (level?.toLowerCase()) {
      case 'excellent':
      case 'good': return 'default';
      case 'adequate': return 'secondary';
      case 'basic': return 'outline';
      case 'insufficient': return 'destructive';
      default: return 'outline';
    }
  };

  /**
   * Format timestamp
   */
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  /**
   * Render Dashboard Tab
   */
  const renderDashboard = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Brain className="h-4 w-4 text-blue-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Assessments</p>
                <p className="text-2xl font-bold">{userProgress.total_assessments || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Avg Score</p>
                <p className="text-2xl font-bold">
                  {userProgress.avg_assessment_score 
                    ? `${(userProgress.avg_assessment_score * 100).toFixed(0)}%`
                    : '0%'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <BookOpen className="h-4 w-4 text-purple-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Modules Mastered</p>
                <p className="text-2xl font-bold">{userProgress.modules_mastered || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Award className="h-4 w-4 text-yellow-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Achievement Level</p>
                <p className="text-2xl font-bold">{gamification.level?.level || 'Novice'}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Current Assessment */}
      {activeSession ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>Active Assessment</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">Progress</span>
                <span className="text-sm font-medium">{questionsAnswered}/15 questions</span>
              </div>
              <Progress value={sessionProgress} className="h-2" />
              
              {currentQuestion && (
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold mb-2">{currentQuestion.question_text}</h4>
                    {currentQuestion.question_context && (
                      <p className="text-sm text-gray-600 mb-3">{currentQuestion.question_context}</p>
                    )}
                    
                    {currentQuestion.question_type === 'multiple_choice' && (
                      <RadioGroup value={currentAnswer} onValueChange={setCurrentAnswer}>
                        {currentQuestion.options?.map((option, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <RadioGroupItem value={option} id={`option-${index}`} />
                            <Label htmlFor={`option-${index}`}>{option}</Label>
                          </div>
                        ))}
                      </RadioGroup>
                    )}
                    
                    {currentQuestion.question_type === 'numerical_input' && (
                      <Input
                        type="number"
                        placeholder="Enter your answer"
                        value={currentAnswer}
                        onChange={(e) => setCurrentAnswer(e.target.value)}
                      />
                    )}
                    
                    {currentQuestion.question_type === 'true_false' && (
                      <RadioGroup value={currentAnswer} onValueChange={setCurrentAnswer}>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="True" id="true" />
                          <Label htmlFor="true">True</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="False" id="false" />
                          <Label htmlFor="false">False</Label>
                        </div>
                      </RadioGroup>
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="flex-1">
                      <Label>Confidence Level</Label>
                      <Select value={confidenceLevel.toString()} onValueChange={(value) => setConfidenceLevel(parseInt(value))}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">Very Unsure</SelectItem>
                          <SelectItem value="2">Unsure</SelectItem>
                          <SelectItem value="3">Neutral</SelectItem>
                          <SelectItem value="4">Confident</SelectItem>
                          <SelectItem value="5">Very Confident</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-end">
                      <Button 
                        onClick={submitAnswer} 
                        disabled={!currentAnswer || loading}
                      >
                        {loading ? <RefreshCw className="h-4 w-4 animate-spin mr-2" /> : <CheckSquare className="h-4 w-4 mr-2" />}
                        Submit Answer
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Start Knowledge Assessment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4">
              <Button onClick={() => startAssessment(['mortgage_basics'])} className="h-20">
                <div className="text-center">
                  <BookOpen className="h-6 w-6 mx-auto mb-1" />
                  <div className="text-sm">Mortgage Basics</div>
                </div>
              </Button>
              <Button onClick={() => startAssessment(['risk_understanding'])} className="h-20">
                <div className="text-center">
                  <Shield className="h-6 w-6 mx-auto mb-1" />
                  <div className="text-sm">Risk Understanding</div>
                </div>
              </Button>
              <Button onClick={() => startAssessment(['costs_and_fees'])} className="h-20">
                <div className="text-center">
                  <Target className="h-6 w-6 mx-auto mb-1" />
                  <div className="text-sm">Costs & Fees</div>
                </div>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Activity */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Recent Assessments</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {assessments.slice(0, 5).map((assessment) => (
                <div key={assessment.assessment_id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <p className="font-medium">Knowledge Assessment</p>
                    <p className="text-sm text-gray-500">
                      Score: {(assessment.overall_score * 100).toFixed(0)}% • 
                      Level: {assessment.overall_comprehension_level}
                    </p>
                    <p className="text-xs text-gray-400">
                      {formatTimestamp(assessment.assessment_timestamp)}
                    </p>
                  </div>
                  <Badge variant={getComprehensionBadgeVariant(assessment.overall_comprehension_level)}>
                    {assessment.overall_comprehension_level}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Trophy className="h-5 w-5" />
              <span>Achievements</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {gamification.badges?.map((badge, index) => (
                <div key={index} className="flex items-center space-x-3 p-3 border rounded-lg">
                  <Award className="h-5 w-5 text-yellow-500" />
                  <div>
                    <p className="font-medium">{badge.name}</p>
                    <p className="text-sm text-gray-500">{badge.description}</p>
                  </div>
                </div>
              ))}
              
              <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
                <Star className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="font-medium">Total Points: {gamification.points || 0}</p>
                  <p className="text-sm text-gray-500">
                    Level: {gamification.level?.level || 'Novice'} • 
                    Next: {gamification.level?.points_to_next || 0} points
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  /**
   * Render Assessment Tab
   */
  const renderAssessment = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Knowledge Assessment</h3>
        <Button onClick={() => startAssessment(['mortgage_basics', 'risk_understanding'])}>
          <Brain className="h-4 w-4 mr-2" />
          Start Comprehensive Assessment
        </Button>
      </div>

      {/* Assessment Types */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <BookOpen className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <h4 className="font-semibold mb-2">Mortgage Basics</h4>
            <p className="text-sm text-gray-500 mb-3">Fundamental mortgage concepts</p>
            <Button variant="outline" size="sm" onClick={() => startAssessment(['mortgage_basics'])}>
              Start Assessment
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <Shield className="h-8 w-8 mx-auto mb-2 text-red-500" />
            <h4 className="font-semibold mb-2">Risk Understanding</h4>
            <p className="text-sm text-gray-500 mb-3">Mortgage risks and mitigation</p>
            <Button variant="outline" size="sm" onClick={() => startAssessment(['risk_understanding'])}>
              Start Assessment
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <Target className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <h4 className="font-semibold mb-2">Costs & Fees</h4>
            <p className="text-sm text-gray-500 mb-3">Understanding mortgage costs</p>
            <Button variant="outline" size="sm" onClick={() => startAssessment(['costs_and_fees'])}>
              Start Assessment
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <Flag className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <h4 className="font-semibold mb-2">Regulations</h4>
            <p className="text-sm text-gray-500 mb-3">Dutch mortgage regulations</p>
            <Button variant="outline" size="sm" onClick={() => startAssessment(['regulatory_requirements'])}>
              Start Assessment
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Assessment History */}
      <Card>
        <CardHeader>
          <CardTitle>Assessment History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {assessments.map((assessment) => (
              <div key={assessment.assessment_id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className={`w-3 h-3 rounded-full ${getComprehensionLevelColor(assessment.overall_comprehension_level).replace('text-', 'bg-')}`} />
                  <div>
                    <p className="font-medium">Knowledge Assessment</p>
                    <p className="text-sm text-gray-500">
                      {formatTimestamp(assessment.assessment_timestamp)} • 
                      Duration: {assessment.session_duration_minutes || 0} min
                    </p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="text-lg font-bold">{(assessment.overall_score * 100).toFixed(0)}%</p>
                  <Badge variant={getComprehensionBadgeVariant(assessment.overall_comprehension_level)}>
                    {assessment.overall_comprehension_level}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Learning Tab
   */
  const renderLearning = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Learning Paths</h3>
        <Button variant="outline">
          <Lightbulb className="h-4 w-4 mr-2" />
          Create Learning Path
        </Button>
      </div>

      {/* Learning Modules */}
      <div className="grid md:grid-cols-2 gap-4">
        {[
          { id: 'mortgage_fundamentals', name: 'Mortgage Fundamentals', progress: 85, time: 30, difficulty: 'Beginner' },
          { id: 'interest_rate_mechanics', name: 'Interest Rate Mechanics', progress: 60, time: 25, difficulty: 'Intermediate' },
          { id: 'risk_assessment_training', name: 'Risk Assessment Training', progress: 0, time: 40, difficulty: 'Intermediate' },
          { id: 'regulatory_compliance', name: 'Regulatory Compliance', progress: 0, time: 35, difficulty: 'Advanced' }
        ].map((module) => (
          <Card key={module.id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="font-semibold">{module.name}</h4>
                  <p className="text-sm text-gray-500">
                    {module.time} min • {module.difficulty}
                  </p>
                </div>
                <Badge variant={module.progress > 0 ? 'default' : 'outline'}>
                  {module.progress > 0 ? 'In Progress' : 'Not Started'}
                </Badge>
              </div>
              
              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{module.progress}%</span>
                </div>
                <Progress value={module.progress} className="h-2" />
              </div>
              
              <div className="flex space-x-2">
                <Button variant="outline" size="sm" className="flex-1">
                  {module.progress > 0 ? (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Continue
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Start
                    </>
                  )}
                </Button>
                <Button variant="ghost" size="sm">
                  <Eye className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Learning Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Lightbulb className="h-5 w-5" />
            <span>Personalized Recommendations</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              "Focus on interest rate mechanics to improve understanding",
              "Complete risk assessment training for better comprehension",
              "Practice cost calculation exercises",
              "Review regulatory requirements for compliance"
            ].map((recommendation, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 border rounded-lg">
                <ArrowRight className="h-4 w-4 text-blue-500" />
                <span className="text-sm">{recommendation}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Analytics Tab
   */
  const renderAnalytics = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Learning Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Performance Trends</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">This Week</span>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">+8.5%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">This Month</span>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">+15.2%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Response Time</span>
                <div className="flex items-center space-x-2">
                  <TrendingDown className="h-4 w-4 text-blue-500" />
                  <span className="text-blue-600 font-medium">-12s</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Gauge className="h-5 w-5" />
              <span>Domain Mastery</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { domain: 'Mortgage Basics', score: 0.85, color: 'text-green-600' },
                { domain: 'Interest Rates', score: 0.72, color: 'text-blue-600' },
                { domain: 'Risk Understanding', score: 0.68, color: 'text-yellow-600' },
                { domain: 'Costs & Fees', score: 0.55, color: 'text-orange-600' }
              ].map((item) => (
                <div key={item.domain} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{item.domain}</span>
                    <span className={`font-medium ${item.color}`}>
                      {(item.score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Progress value={item.score * 100} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Learning Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-blue-600">47</p>
              <p className="text-sm text-gray-500">Questions Answered</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-green-600">89%</p>
              <p className="text-sm text-gray-500">Accuracy Rate</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-purple-600">3.2h</p>
              <p className="text-sm text-gray-500">Total Learning Time</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-orange-600">5</p>
              <p className="text-sm text-gray-500">Learning Streak</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (loading && assessments.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading User Comprehension Validator...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center space-x-3">
          <Brain className="h-8 w-8 text-blue-600" />
          <span>User Comprehension Validator</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Adaptive testing, knowledge assessment, and personalized learning path optimization
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="assessment" className="flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>Assessment</span>
          </TabsTrigger>
          <TabsTrigger value="learning" className="flex items-center space-x-2">
            <BookOpen className="h-4 w-4" />
            <span>Learning</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="assessment">
          {renderAssessment()}
        </TabsContent>

        <TabsContent value="learning">
          {renderLearning()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default UserComprehensionValidator;