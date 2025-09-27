/**
 * Selenium MCP (Model Context Protocol) Integration
 * Advanced browser-based testing capabilities with Selenium WebDriver for cross-browser testing
 * 
 * Features:
 * - Cross-browser testing (Chrome, Firefox, Safari, Edge)
 * - Advanced element interaction and manipulation
 * - Parallel test execution across multiple browsers
 * - Integration testing with real user workflows
 * - Form validation and submission testing
 * - JavaScript execution and DOM manipulation
 * - Cookie and session management
 * - File upload and download testing
 * - Window and frame handling
 * - Comprehensive error handling and reporting
 */

const { Builder, By, Key, until, Select } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
const firefox = require('selenium-webdriver/firefox');
const edge = require('selenium-webdriver/edge');
const fs = require('fs').promises;
const path = require('path');

class SeleniumMCP {
    constructor(config = {}) {
        this.config = {
            browsers: config.browsers || ['chrome'],
            headless: config.headless !== false,
            timeout: config.timeout || 30000,
            screenshotPath: config.screenshotPath || './testing/selenium-screenshots',
            reportsPath: config.reportsPath || './testing/selenium-reports',
            parallelExecution: config.parallelExecution !== false,
            ...config
        };
        
        this.drivers = {};
        this.testResults = [];
        this.performanceMetrics = {};
    }

    /**
     * Initialize Selenium WebDriver for specified browsers
     */
    async initialize() {
        try {
            console.log('🚀 Initializing Selenium MCP...');
            
            // Ensure directories exist
            await this.ensureDirectories();
            
            // Initialize drivers for each browser
            for (const browserName of this.config.browsers) {
                try {
                    const driver = await this.createDriver(browserName);
                    this.drivers[browserName] = driver;
                    console.log(`✅ ${browserName} driver initialized`);
                } catch (error) {
                    console.error(`❌ Failed to initialize ${browserName} driver:`, error.message);
                }
            }

            const initializedBrowsers = Object.keys(this.drivers);
            if (initializedBrowsers.length === 0) {
                throw new Error('No browsers could be initialized');
            }

            console.log(`✅ Selenium MCP initialized with browsers: ${initializedBrowsers.join(', ')}`);
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize Selenium MCP:', error);
            throw error;
        }
    }

    /**
     * Create WebDriver for specific browser
     */
    async createDriver(browserName) {
        let builder = new Builder();

        switch (browserName.toLowerCase()) {
            case 'chrome':
                const chromeOptions = new chrome.Options();
                if (this.config.headless) {
                    chromeOptions.addArguments('--headless');
                }
                chromeOptions.addArguments(
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--window-size=1920,1080'
                );
                builder = builder.forBrowser('chrome').setChromeOptions(chromeOptions);
                break;

            case 'firefox':
                const firefoxOptions = new firefox.Options();
                if (this.config.headless) {
                    firefoxOptions.addArguments('--headless');
                }
                firefoxOptions.addArguments('--width=1920', '--height=1080');
                builder = builder.forBrowser('firefox').setFirefoxOptions(firefoxOptions);
                break;

            case 'edge':
                const edgeOptions = new edge.Options();
                if (this.config.headless) {
                    edgeOptions.addArguments('--headless');
                }
                edgeOptions.addArguments('--window-size=1920,1080');
                builder = builder.forBrowser('MicrosoftEdge').setEdgeOptions(edgeOptions);
                break;

            default:
                throw new Error(`Unsupported browser: ${browserName}`);
        }

        const driver = await builder.build();
        await driver.manage().setTimeouts({
            implicit: this.config.timeout,
            pageLoad: this.config.timeout,
            script: this.config.timeout
        });

        return driver;
    }

    /**
     * Ensure required directories exist
     */
    async ensureDirectories() {
        const dirs = [
            this.config.screenshotPath,
            this.config.reportsPath,
            path.join(this.config.screenshotPath, 'chrome'),
            path.join(this.config.screenshotPath, 'firefox'),
            path.join(this.config.screenshotPath, 'edge')
        ];

        for (const dir of dirs) {
            try {
                await fs.mkdir(dir, { recursive: true });
            } catch (error) {
                // Directory might already exist
            }
        }
    }

    /**
     * Run tests across all browsers
     */
    async runCrossBrowserTests(testSuite) {
        try {
            console.log(`🌐 Running cross-browser tests across ${Object.keys(this.drivers).length} browsers`);
            
            const results = {};
            const browserNames = Object.keys(this.drivers);

            if (this.config.parallelExecution && browserNames.length > 1) {
                // Run tests in parallel
                const promises = browserNames.map(browser => 
                    this.runTestsForBrowser(browser, testSuite)
                );
                const parallelResults = await Promise.allSettled(promises);
                
                browserNames.forEach((browser, index) => {
                    const result = parallelResults[index];
                    results[browser] = result.status === 'fulfilled' ? result.value : {
                        success: false,
                        error: result.reason.message
                    };
                });
            } else {
                // Run tests sequentially
                for (const browser of browserNames) {
                    results[browser] = await this.runTestsForBrowser(browser, testSuite);
                }
            }

            // Analyze cross-browser compatibility
            const compatibility = this.analyzeCrossBrowserCompatibility(results);
            
            console.log(`✅ Cross-browser testing completed`);
            
            return {
                success: true,
                results,
                compatibility,
                summary: this.generateTestSummary(results)
            };
        } catch (error) {
            console.error(`❌ Cross-browser testing failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Run tests for specific browser
     */
    async runTestsForBrowser(browserName, testSuite) {
        const driver = this.drivers[browserName];
        if (!driver) {
            return {
                success: false,
                error: `Driver not available for ${browserName}`
            };
        }

        try {
            console.log(`🧪 Running tests for ${browserName}...`);
            
            const browserResults = {
                browser: browserName,
                tests: [],
                startTime: Date.now(),
                success: true
            };

            for (const test of testSuite.tests) {
                const testResult = await this.runSingleTest(driver, browserName, test);
                browserResults.tests.push(testResult);
                
                if (!testResult.success) {
                    browserResults.success = false;
                }
            }

            browserResults.endTime = Date.now();
            browserResults.duration = browserResults.endTime - browserResults.startTime;
            
            console.log(`✅ ${browserName} tests completed: ${browserResults.tests.filter(t => t.success).length}/${browserResults.tests.length} passed`);
            
            return browserResults;
        } catch (error) {
            console.error(`❌ ${browserName} testing failed: ${error.message}`);
            return {
                browser: browserName,
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Run single test
     */
    async runSingleTest(driver, browserName, test) {
        const testResult = {
            testName: test.name,
            browser: browserName,
            startTime: Date.now(),
            success: true,
            steps: [],
            screenshots: [],
            errors: []
        };

        try {
            console.log(`🔍 Running test: ${test.name} on ${browserName}`);

            for (let i = 0; i < test.steps.length; i++) {
                const step = test.steps[i];
                const stepResult = await this.executeTestStep(driver, browserName, step, i + 1);
                
                testResult.steps.push(stepResult);
                
                if (stepResult.screenshot) {
                    testResult.screenshots.push(stepResult.screenshot);
                }
                
                if (!stepResult.success) {
                    testResult.success = false;
                    testResult.errors.push(stepResult.error);
                }
            }

            testResult.endTime = Date.now();
            testResult.duration = testResult.endTime - testResult.startTime;
            
            console.log(`${testResult.success ? '✅' : '❌'} Test ${test.name} on ${browserName}: ${testResult.success ? 'PASSED' : 'FAILED'}`);
            
            return testResult;
        } catch (error) {
            testResult.success = false;
            testResult.error = error.message;
            testResult.endTime = Date.now();
            testResult.duration = testResult.endTime - testResult.startTime;
            
            console.error(`❌ Test ${test.name} on ${browserName} failed: ${error.message}`);
            return testResult;
        }
    }

    /**
     * Execute individual test step
     */
    async executeTestStep(driver, browserName, step, stepNumber) {
        const stepResult = {
            stepNumber,
            action: step.action,
            success: true,
            startTime: Date.now()
        };

        try {
            switch (step.action) {
                case 'navigate':
                    await driver.get(step.url);
                    stepResult.url = step.url;
                    
                    if (step.waitFor) {
                        await driver.wait(until.elementLocated(By.css(step.waitFor)), 10000);
                    }
                    break;

                case 'click':
                    const clickElement = await driver.wait(until.elementLocated(By.css(step.selector)), 10000);
                    await driver.wait(until.elementIsVisible(clickElement), 5000);
                    await clickElement.click();
                    stepResult.selector = step.selector;
                    break;

                case 'type':
                    const typeElement = await driver.wait(until.elementLocated(By.css(step.selector)), 10000);
                    await typeElement.clear();
                    await typeElement.sendKeys(step.text);
                    stepResult.selector = step.selector;
                    stepResult.text = step.text;
                    break;

                case 'select':
                    const selectElement = await driver.wait(until.elementLocated(By.css(step.selector)), 10000);
                    const select = new Select(selectElement);
                    
                    if (step.value) {
                        await select.selectByValue(step.value);
                    } else if (step.text) {
                        await select.selectByVisibleText(step.text);
                    } else if (step.index !== undefined) {
                        await select.selectByIndex(step.index);
                    }
                    
                    stepResult.selector = step.selector;
                    stepResult.value = step.value || step.text || step.index;
                    break;

                case 'wait':
                    if (step.selector) {
                        await driver.wait(until.elementLocated(By.css(step.selector)), step.timeout || 10000);
                        stepResult.selector = step.selector;
                    } else {
                        await driver.sleep(step.timeout || 1000);
                        stepResult.timeout = step.timeout;
                    }
                    break;

                case 'assert_text':
                    const textElement = await driver.wait(until.elementLocated(By.css(step.selector)), 10000);
                    const actualText = await textElement.getText();
                    const expectedText = step.expectedText;
                    
                    if (expectedText && !actualText.includes(expectedText)) {
                        throw new Error(`Text assertion failed. Expected: "${expectedText}", Actual: "${actualText}"`);
                    }
                    
                    stepResult.selector = step.selector;
                    stepResult.expectedText = expectedText;
                    stepResult.actualText = actualText;
                    break;

                case 'assert_element':
                    try {
                        await driver.wait(until.elementLocated(By.css(step.selector)), 5000);
                        stepResult.selector = step.selector;
                        stepResult.elementFound = true;
                    } catch {
                        if (step.shouldExist !== false) {
                            throw new Error(`Element not found: ${step.selector}`);
                        }
                        stepResult.elementFound = false;
                    }
                    break;

                case 'screenshot':
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const filename = `${step.name || `step_${stepNumber}`}_${browserName}_${timestamp}.png`;
                    const filepath = path.join(this.config.screenshotPath, browserName, filename);
                    
                    await driver.takeScreenshot().then(data => 
                        fs.writeFile(filepath, data, 'base64')
                    );
                    
                    stepResult.screenshot = {
                        filename,
                        filepath
                    };
                    break;

                case 'execute_script':
                    const scriptResult = await driver.executeScript(step.script, ...(step.args || []));
                    stepResult.script = step.script;
                    stepResult.result = scriptResult;
                    break;

                case 'switch_frame':
                    if (step.frameSelector) {
                        const frameElement = await driver.findElement(By.css(step.frameSelector));
                        await driver.switchTo().frame(frameElement);
                    } else if (step.frameIndex !== undefined) {
                        await driver.switchTo().frame(step.frameIndex);
                    } else {
                        await driver.switchTo().defaultContent();
                    }
                    stepResult.frameAction = step.frameSelector || step.frameIndex || 'default';
                    break;

                case 'handle_alert':
                    const alert = await driver.switchTo().alert();
                    const alertText = await alert.getText();
                    
                    if (step.action === 'accept') {
                        await alert.accept();
                    } else if (step.action === 'dismiss') {
                        await alert.dismiss();
                    } else if (step.action === 'sendKeys') {
                        await alert.sendKeys(step.text);
                        await alert.accept();
                    }
                    
                    stepResult.alertText = alertText;
                    stepResult.alertAction = step.action;
                    break;

                default:
                    throw new Error(`Unknown action: ${step.action}`);
            }

            stepResult.endTime = Date.now();
            stepResult.duration = stepResult.endTime - stepResult.startTime;
            
            return stepResult;
        } catch (error) {
            stepResult.success = false;
            stepResult.error = error.message;
            stepResult.endTime = Date.now();
            stepResult.duration = stepResult.endTime - stepResult.startTime;
            
            return stepResult;
        }
    }

    /**
     * Test mortgage application workflow
     */
    async testMortgageApplicationWorkflow(baseUrl) {
        try {
            console.log('🏠 Testing mortgage application workflow...');
            
            const workflow = {
                name: 'Mortgage Application Complete Workflow',
                description: 'End-to-end testing of mortgage application process',
                steps: [
                    { action: 'navigate', url: `${baseUrl}/mortgage-application`, waitFor: 'form[data-testid="mortgage-form"]' },
                    { action: 'screenshot', name: 'application_start' },
                    
                    // Personal Information
                    { action: 'type', selector: 'input[name="firstName"]', text: 'Jan' },
                    { action: 'type', selector: 'input[name="lastName"]', text: 'de Vries' },
                    { action: 'type', selector: 'input[name="email"]', text: 'jan.devries@example.nl' },
                    { action: 'type', selector: 'input[name="phone"]', text: '+31612345678' },
                    { action: 'type', selector: 'input[name="bsn"]', text: '123456789' },
                    
                    // Financial Information
                    { action: 'type', selector: 'input[name="grossIncome"]', text: '75000' },
                    { action: 'type', selector: 'input[name="netIncome"]', text: '4500' },
                    { action: 'select', selector: 'select[name="employmentType"]', value: 'permanent' },
                    { action: 'type', selector: 'input[name="employmentYears"]', text: '8' },
                    
                    // Property Information
                    { action: 'type', selector: 'input[name="propertyValue"]', text: '450000' },
                    { action: 'type', selector: 'input[name="loanAmount"]', text: '400000' },
                    { action: 'select', selector: 'select[name="propertyType"]', value: 'house' },
                    { action: 'type', selector: 'input[name="postcode"]', text: '1012AB' },
                    
                    { action: 'screenshot', name: 'form_filled' },
                    
                    // Submit application
                    { action: 'click', selector: 'button[type="submit"]' },
                    { action: 'wait', selector: '.application-success, .application-error', timeout: 15000 },
                    
                    { action: 'screenshot', name: 'application_result' },
                    
                    // Verify success message
                    { action: 'assert_element', selector: '.application-success' },
                    { action: 'assert_text', selector: '.application-success', expectedText: 'Application submitted successfully' }
                ]
            };

            const results = await this.runCrossBrowserTests({ tests: [workflow] });
            
            console.log('✅ Mortgage application workflow testing completed');
            return results;
        } catch (error) {
            console.error('❌ Mortgage application workflow test failed:', error);
            throw error;
        }
    }

    /**
     * Test BKR/NHG integration workflow
     */
    async testBKRNHGIntegrationWorkflow(baseUrl) {
        try {
            console.log('🔍 Testing BKR/NHG integration workflow...');
            
            const workflow = {
                name: 'BKR/NHG Integration Workflow',
                description: 'Testing BKR credit check and NHG eligibility integration',
                steps: [
                    { action: 'navigate', url: `${baseUrl}/bkr-nhg-integration` },
                    { action: 'screenshot', name: 'bkr_nhg_start' },
                    
                    // BKR Credit Check
                    { action: 'type', selector: 'input[name="bsn"]', text: '123456789' },
                    { action: 'click', selector: 'button[data-testid="bkr-check-btn"]' },
                    { action: 'wait', selector: '.bkr-results, .bkr-error', timeout: 20000 },
                    
                    { action: 'screenshot', name: 'bkr_results' },
                    
                    // Verify BKR results
                    { action: 'assert_element', selector: '.bkr-results' },
                    { action: 'assert_text', selector: '.bkr-score', expectedText: 'Credit Score' },
                    
                    // NHG Eligibility Check
                    { action: 'type', selector: 'input[name="propertyValue"]', text: '350000' },
                    { action: 'type', selector: 'input[name="loanAmount"]', text: '315000' },
                    { action: 'click', selector: 'button[data-testid="nhg-check-btn"]' },
                    { action: 'wait', selector: '.nhg-results', timeout: 15000 },
                    
                    { action: 'screenshot', name: 'nhg_results' },
                    
                    // Verify NHG results
                    { action: 'assert_element', selector: '.nhg-eligible, .nhg-not-eligible' },
                    { action: 'assert_text', selector: '.nhg-cost-benefit', expectedText: 'Cost-Benefit Analysis' }
                ]
            };

            const results = await this.runCrossBrowserTests({ tests: [workflow] });
            
            console.log('✅ BKR/NHG integration workflow testing completed');
            return results;
        } catch (error) {
            console.error('❌ BKR/NHG integration workflow test failed:', error);
            throw error;
        }
    }

    /**
     * Test compliance audit trail workflow
     */
    async testComplianceAuditTrailWorkflow(baseUrl) {
        try {
            console.log('📋 Testing compliance audit trail workflow...');
            
            const workflow = {
                name: 'Compliance Audit Trail Workflow',
                description: 'Testing compliance logging and audit trail functionality',
                steps: [
                    { action: 'navigate', url: `${baseUrl}/compliance-audit-trail` },
                    { action: 'screenshot', name: 'compliance_dashboard' },
                    
                    // View audit events
                    { action: 'click', selector: 'button[data-testid="view-audit-events"]' },
                    { action: 'wait', selector: '.audit-events-table' },
                    { action: 'assert_element', selector: '.audit-events-table tbody tr' },
                    
                    // Filter audit events
                    { action: 'select', selector: 'select[name="eventType"]', value: 'compliance_violation' },
                    { action: 'click', selector: 'button[data-testid="apply-filter"]' },
                    { action: 'wait', timeout: 2000 },
                    
                    { action: 'screenshot', name: 'filtered_audit_events' },
                    
                    // View investigation details
                    { action: 'click', selector: '.audit-event-row:first-child .view-details-btn' },
                    { action: 'wait', selector: '.investigation-details' },
                    { action: 'assert_text', selector: '.investigation-details h3', expectedText: 'Investigation Details' },
                    
                    { action: 'screenshot', name: 'investigation_details' },
                    
                    // Generate compliance report
                    { action: 'click', selector: 'button[data-testid="generate-report"]' },
                    { action: 'wait', selector: '.report-generation-status', timeout: 30000 },
                    { action: 'assert_text', selector: '.report-generation-status', expectedText: 'Report generated successfully' }
                ]
            };

            const results = await this.runCrossBrowserTests({ tests: [workflow] });
            
            console.log('✅ Compliance audit trail workflow testing completed');
            return results;
        } catch (error) {
            console.error('❌ Compliance audit trail workflow test failed:', error);
            throw error;
        }
    }

    /**
     * Test all mortgage system workflows
     */
    async testAllMortgageWorkflows(baseUrl) {
        try {
            console.log('🏢 Testing all mortgage system workflows...');
            
            const allResults = {};
            
            // Test mortgage application workflow
            allResults.mortgageApplication = await this.testMortgageApplicationWorkflow(baseUrl);
            
            // Test BKR/NHG integration workflow
            allResults.bkrNhgIntegration = await this.testBKRNHGIntegrationWorkflow(baseUrl);
            
            // Test compliance audit trail workflow
            allResults.complianceAuditTrail = await this.testComplianceAuditTrailWorkflow(baseUrl);
            
            // Test risk assessment workflow
            allResults.riskAssessment = await this.testRiskAssessmentWorkflow(baseUrl);
            
            // Test document authenticity workflow
            allResults.documentAuthenticity = await this.testDocumentAuthenticityWorkflow(baseUrl);
            
            // Test NLP content analyzer workflow
            allResults.nlpContentAnalyzer = await this.testNLPContentAnalyzerWorkflow(baseUrl);
            
            // Test mortgage advice generator workflow
            allResults.mortgageAdviceGenerator = await this.testMortgageAdviceGeneratorWorkflow(baseUrl);
            
            // Test user comprehension validator workflow
            allResults.userComprehensionValidator = await this.testUserComprehensionValidatorWorkflow(baseUrl);
            
            // Test Dutch market intelligence workflow
            allResults.dutchMarketIntelligence = await this.testDutchMarketIntelligenceWorkflow(baseUrl);
            
            // Generate comprehensive report
            const comprehensiveReport = await this.generateComprehensiveTestReport(allResults);
            
            console.log('✅ All mortgage system workflows testing completed');
            return {
                success: true,
                results: allResults,
                report: comprehensiveReport
            };
        } catch (error) {
            console.error('❌ All workflows testing failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Test risk assessment workflow
     */
    async testRiskAssessmentWorkflow(baseUrl) {
        const workflow = {
            name: 'Risk Assessment Engine Workflow',
            steps: [
                { action: 'navigate', url: `${baseUrl}/risk-assessment-engine` },
                { action: 'screenshot', name: 'risk_assessment_start' },
                { action: 'click', selector: 'button[data-testid="start-assessment"]' },
                { action: 'wait', selector: '.risk-assessment-form' },
                { action: 'type', selector: 'input[name="loanAmount"]', text: '400000' },
                { action: 'type', selector: 'input[name="propertyValue"]', text: '500000' },
                { action: 'click', selector: 'button[data-testid="calculate-risk"]' },
                { action: 'wait', selector: '.risk-results', timeout: 20000 },
                { action: 'screenshot', name: 'risk_results' },
                { action: 'assert_element', selector: '.risk-score' },
                { action: 'assert_text', selector: '.risk-level', expectedText: 'Risk Level' }
            ]
        };
        return await this.runCrossBrowserTests({ tests: [workflow] });
    }

    /**
     * Test document authenticity workflow
     */
    async testDocumentAuthenticityWorkflow(baseUrl) {
        const workflow = {
            name: 'Document Authenticity Checker Workflow',
            steps: [
                { action: 'navigate', url: `${baseUrl}/document-authenticity-checker` },
                { action: 'screenshot', name: 'document_checker_start' },
                { action: 'assert_element', selector: '.document-upload-area' },
                { action: 'click', selector: 'button[data-testid="analyze-sample"]' },
                { action: 'wait', selector: '.verification-results', timeout: 30000 },
                { action: 'screenshot', name: 'verification_results' },
                { action: 'assert_element', selector: '.authenticity-score' },
                { action: 'assert_text', selector: '.verification-status', expectedText: 'Verification' }
            ]
        };
        return await this.runCrossBrowserTests({ tests: [workflow] });
    }

    /**
     * Test NLP content analyzer workflow
     */
    async testNLPContentAnalyzerWorkflow(baseUrl) {
        const workflow = {
            name: 'NLP Content Analyzer Workflow',
            steps: [
                { action: 'navigate', url: `${baseUrl}/nlp-content-analyzer` },
                { action: 'screenshot', name: 'nlp_analyzer_start' },
                { action: 'type', selector: 'textarea[name="content"]', text: 'Dit is een voorbeeld van een hypotheekdocument voor analyse.' },
                { action: 'click', selector: 'button[data-testid="analyze-content"]' },
                { action: 'wait', selector: '.analysis-results', timeout: 25000 },
                { action: 'screenshot', name: 'nlp_analysis_results' },
                { action: 'assert_element', selector: '.semantic-analysis' },
                { action: 'assert_element', selector: '.named-entities' }
            ]
        };
        return await this.runCrossBrowserTests({ tests: [workflow] });
    }

    /**
     * Test mortgage advice generator workflow
     */
    async testMortgageAdviceGeneratorWorkflow(baseUrl) {
        const workflow = {
            name: 'Mortgage Advice Generator Workflow',
            steps: [
                { action: 'navigate', url: `${baseUrl}/mortgage-advice-generator` },
                { action: 'screenshot', name: 'advice_generator_start' },
                { action: 'type', selector: 'input[name="income"]', text: '60000' },
                { action: 'type', selector: 'input[name="savings"]', text: '80000' },
                { action: 'select', selector: 'select[name="riskTolerance"]', value: 'moderate' },
                { action: 'click', selector: 'button[data-testid="generate-advice"]' },
                { action: 'wait', selector: '.advice-results', timeout: 30000 },
                { action: 'screenshot', name: 'advice_results' },
                { action: 'assert_element', selector: '.product-recommendations' },
                { action: 'assert_text', selector: '.suitability-assessment', expectedText: 'Suitability Assessment' }
            ]
        };
        return await this.runCrossBrowserTests({ tests: [workflow] });
    }

    /**
     * Test user comprehension validator workflow
     */
    async testUserComprehensionValidatorWorkflow(baseUrl) {
        const workflow = {
            name: 'User Comprehension Validator Workflow',
            steps: [
                { action: 'navigate', url: `${baseUrl}/user-comprehension-validator` },
                { action: 'screenshot', name: 'comprehension_validator_start' },
                { action: 'click', selector: 'button[data-testid="start-assessment"]' },
                { action: 'wait', selector: '.assessment-question' },
                { action: 'click', selector: 'input[type="radio"]:first-child' },
                { action: 'click', selector: 'button[data-testid="submit-answer"]' },
                { action: 'wait', selector: '.question-feedback', timeout: 10000 },
                { action: 'screenshot', name: 'assessment_feedback' },
                { action: 'assert_element', selector: '.comprehension-score' }
            ]
        };
        return await this.runCrossBrowserTests({ tests: [workflow] });
    }

    /**
     * Test Dutch market intelligence workflow
     */
    async testDutchMarketIntelligenceWorkflow(baseUrl) {
        const workflow = {
            name: 'Dutch Market Intelligence Workflow',
            steps: [
                { action: 'navigate', url: `${baseUrl}/dutch-market-intelligence` },
                { action: 'screenshot', name: 'market_intelligence_start' },
                { action: 'click', selector: 'button[data-testid="generate-report"]' },
                { action: 'wait', selector: '.intelligence-report', timeout: 30000 },
                { action: 'screenshot', name: 'intelligence_report' },
                { action: 'assert_element', selector: '.market-trends' },
                { action: 'assert_element', selector: '.predictive-insights' }
            ]
        };
        return await this.runCrossBrowserTests({ tests: [workflow] });
    }

    /**
     * Analyze cross-browser compatibility
     */
    analyzeCrossBrowserCompatibility(results) {
        const compatibility = {
            overallCompatibility: true,
            browserResults: {},
            commonIssues: [],
            browserSpecificIssues: {}
        };

        for (const [browser, result] of Object.entries(results)) {
            const browserSuccess = result.success && result.results && result.results.success;
            compatibility.browserResults[browser] = {
                success: browserSuccess,
                testsCount: result.results?.tests?.[0]?.steps?.length || 0,
                passedSteps: result.results?.tests?.[0]?.steps?.filter(s => s.success).length || 0,
                failedSteps: result.results?.tests?.[0]?.steps?.filter(s => !s.success).length || 0
            };

            if (!browserSuccess) {
                compatibility.overallCompatibility = false;
                compatibility.browserSpecificIssues[browser] = result.error || 'Unknown error';
            }
        }

        return compatibility;
    }

    /**
     * Generate test summary
     */
    generateTestSummary(results) {
        const summary = {
            totalBrowsers: Object.keys(results).length,
            successfulBrowsers: 0,
            failedBrowsers: 0,
            totalTests: 0,
            totalSteps: 0,
            passedSteps: 0,
            failedSteps: 0
        };

        for (const result of Object.values(results)) {
            if (result.success) {
                summary.successfulBrowsers++;
            } else {
                summary.failedBrowsers++;
            }

            if (result.results && result.results.tests) {
                summary.totalTests += result.results.tests.length;
                
                result.results.tests.forEach(test => {
                    if (test.steps) {
                        summary.totalSteps += test.steps.length;
                        summary.passedSteps += test.steps.filter(s => s.success).length;
                        summary.failedSteps += test.steps.filter(s => !s.success).length;
                    }
                });
            }
        }

        return summary;
    }

    /**
     * Generate comprehensive test report
     */
    async generateComprehensiveTestReport(allResults) {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const reportPath = path.join(this.config.reportsPath, `selenium_comprehensive_report_${timestamp}.html`);

            let html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Selenium MCP Comprehensive Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: #2563eb; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .success { color: #059669; }
        .error { color: #dc2626; }
        .warning { color: #d97706; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
        th { background: #f9fafb; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8fafc; border-radius: 8px; text-align: center; }
        .workflow-card { border: 1px solid #e5e7eb; padding: 15px; margin: 10px 0; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Selenium MCP Comprehensive Test Report</h1>
        <p>Generated on: ${new Date().toLocaleString()}</p>
        <p>Testing Framework: Selenium WebDriver with Cross-Browser Support</p>
    </div>

    <div class="section">
        <h2>📊 Overall Test Summary</h2>
        <div class="metric">
            <h3>${Object.keys(allResults).length}</h3>
            <p>Workflows Tested</p>
        </div>
        <div class="metric">
            <h3 class="success">${Object.values(allResults).filter(r => r.success).length}</h3>
            <p>Successful Workflows</p>
        </div>
        <div class="metric">
            <h3 class="error">${Object.values(allResults).filter(r => !r.success).length}</h3>
            <p>Failed Workflows</p>
        </div>
        <div class="metric">
            <h3>${this.config.browsers.length}</h3>
            <p>Browsers Tested</p>
        </div>
    </div>
            `;

            // Add workflow results
            for (const [workflowName, workflowResult] of Object.entries(allResults)) {
                html += `
    <div class="section">
        <h2>🔧 ${workflowName.replace(/([A-Z])/g, ' $1').trim()}</h2>
        <div class="workflow-card">
            <h3>Status: <span class="${workflowResult.success ? 'success' : 'error'}">${workflowResult.success ? '✅ PASSED' : '❌ FAILED'}</span></h3>
            
            ${workflowResult.results ? `
            <h4>Browser Results:</h4>
            <table>
                <tr>
                    <th>Browser</th>
                    <th>Status</th>
                    <th>Tests</th>
                    <th>Steps Passed</th>
                    <th>Steps Failed</th>
                </tr>
                ${Object.entries(workflowResult.results).map(([browser, result]) => `
                <tr>
                    <td>${browser}</td>
                    <td class="${result.success ? 'success' : 'error'}">${result.success ? '✅ PASSED' : '❌ FAILED'}</td>
                    <td>${result.tests ? result.tests.length : 0}</td>
                    <td class="success">${result.tests ? result.tests.reduce((acc, test) => acc + (test.steps ? test.steps.filter(s => s.success).length : 0), 0) : 0}</td>
                    <td class="error">${result.tests ? result.tests.reduce((acc, test) => acc + (test.steps ? test.steps.filter(s => !s.success).length : 0), 0) : 0}</td>
                </tr>
                `).join('')}
            </table>
            ` : ''}
            
            ${workflowResult.error ? `<p class="error">Error: ${workflowResult.error}</p>` : ''}
        </div>
    </div>
                `;
            }

            html += `
    <div class="section">
        <h2>🔧 Test Configuration</h2>
        <table>
            <tr><th>Setting</th><th>Value</th></tr>
            <tr><td>Browsers</td><td>${this.config.browsers.join(', ')}</td></tr>
            <tr><td>Headless Mode</td><td>${this.config.headless}</td></tr>
            <tr><td>Parallel Execution</td><td>${this.config.parallelExecution}</td></tr>
            <tr><td>Timeout</td><td>${this.config.timeout}ms</td></tr>
            <tr><td>Screenshots Path</td><td>${this.config.screenshotPath}</td></tr>
        </table>
    </div>
</body>
</html>
            `;

            await fs.writeFile(reportPath, html);
            console.log(`✅ Comprehensive test report generated: ${reportPath}`);
            
            return {
                success: true,
                reportPath,
                summary: this.generateOverallSummary(allResults)
            };
        } catch (error) {
            console.error(`❌ Comprehensive report generation failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Generate overall summary
     */
    generateOverallSummary(allResults) {
        const summary = {
            totalWorkflows: Object.keys(allResults).length,
            successfulWorkflows: Object.values(allResults).filter(r => r.success).length,
            failedWorkflows: Object.values(allResults).filter(r => !r.success).length,
            browsersSupported: this.config.browsers,
            overallSuccess: Object.values(allResults).every(r => r.success)
        };

        summary.successRate = (summary.successfulWorkflows / summary.totalWorkflows) * 100;
        
        return summary;
    }

    /**
     * Clean up all resources
     */
    async cleanup() {
        try {
            console.log('🧹 Cleaning up Selenium MCP...');
            
            const cleanupPromises = Object.values(this.drivers).map(driver => 
                driver.quit().catch(err => console.error('Driver cleanup error:', err))
            );
            
            await Promise.allSettled(cleanupPromises);
            this.drivers = {};
            
            console.log('✅ Selenium MCP cleanup completed');
        } catch (error) {
            console.error('❌ Selenium MCP cleanup failed:', error);
        }
    }
}

module.exports = SeleniumMCP;