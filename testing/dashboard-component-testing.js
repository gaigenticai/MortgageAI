/**
 * Comprehensive Dashboard Component Testing
 * This script performs thorough component testing of the dashboard tab 
 * to ensure all components are properly connected to the backend
 */

const { chromium } = require('playwright');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class DashboardComponentTester {
    constructor() {
        this.baseUrl = 'http://localhost:5173';
        this.apiBaseUrl = 'http://localhost:3000';
        this.browser = null;
        this.page = null;
        this.testResults = {
            componentTests: [],
            apiConnectivityTests: [],
            dataExchangeTests: [],
            interactionTests: [],
            overallStatus: 'PENDING',
            totalTests: 0,
            passedTests: 0,
            failedTests: 0,
            timestamp: new Date().toISOString()
        };
    }

    async initialize() {
        console.log('🚀 Initializing Dashboard Component Tester...');
        
        // Launch browser
        this.browser = await chromium.launch({ 
            headless: false,
            devtools: true,
            args: [
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        });
        
        this.page = await this.browser.newPage();
        
        // Set viewport
        await this.page.setViewportSize({ width: 1920, height: 1080 });
        
        // Enable request interception to monitor API calls
        await this.page.route('**/*', route => {
            const url = route.request().url();
            if (url.includes('/api/')) {
                console.log(`📡 API Call: ${route.request().method()} ${url}`);
            }
            route.continue();
        });
        
        console.log('✅ Browser initialized successfully');
    }

    async testBackendConnectivity() {
        console.log('\n🔗 Testing Backend Connectivity...');
        
        const endpoints = [
            '/health',
            '/api/dashboard/metrics',
            '/api/dashboard/agent-status', 
            '/api/dashboard/lender-status',
            '/api/dashboard/recent-activity'
        ];

        for (const endpoint of endpoints) {
            try {
                const response = await axios.get(`${this.apiBaseUrl}${endpoint}`, {
                    timeout: 5000
                });
                
                this.testResults.apiConnectivityTests.push({
                    endpoint,
                    status: 'PASS',
                    statusCode: response.status,
                    responseTime: Date.now(),
                    hasData: !!response.data
                });
                
                console.log(`✅ ${endpoint}: ${response.status}`);
            } catch (error) {
                this.testResults.apiConnectivityTests.push({
                    endpoint,
                    status: 'FAIL',
                    error: error.message,
                    statusCode: error.response?.status || 'TIMEOUT'
                });
                
                console.log(`❌ ${endpoint}: ${error.message}`);
            }
        }
    }

    async navigateToDashboard() {
        console.log('\n🌐 Navigating to Dashboard...');
        
        try {
            await this.page.goto(this.baseUrl, { 
                waitUntil: 'networkidle',
                timeout: 30000 
            });
            
            // Wait for React to load
            await this.page.waitForSelector('[data-testid], .mantine-Card-root, .mantine-Container-root', { 
                timeout: 15000 
            });
            
            await this.page.screenshot({ 
                path: './testing/screenshots/dashboard-loaded.png',
                fullPage: true 
            });
            
            console.log('✅ Dashboard loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to load dashboard:', error.message);
            return false;
        }
    }

    async testDashboardComponents() {
        console.log('\n📊 Testing Dashboard Components...');
        
        const componentTests = [
            {
                name: 'Header Component',
                selector: 'header, [role="banner"]',
                expectedText: 'MortgageAI'
            },
            {
                name: 'AFM Compliance Score Card',
                selector: '[data-testid="afm-compliance"], .mantine-Card-root:has-text("AFM Compliance")',
                expectedElements: ['text', 'progress', 'percentage']
            },
            {
                name: 'Active Sessions Card',
                selector: '[data-testid="active-sessions"], .mantine-Card-root:has-text("Active Sessions")',
                expectedElements: ['number', 'icon']
            },
            {
                name: 'Pending Reviews Card',
                selector: '[data-testid="pending-reviews"], .mantine-Card-root:has-text("Pending Reviews")',
                expectedElements: ['number', 'icon']
            },
            {
                name: 'Quality Score Card',
                selector: '[data-testid="quality-score"], .mantine-Card-root:has-text("Quality Score")',
                expectedElements: ['percentage', 'progress']
            },
            {
                name: 'Recent Activity Section',
                selector: '[data-testid="recent-activity"], .mantine-Card-root:has-text("Recent Activity")',
                expectedElements: ['list', 'activities']
            },
            {
                name: 'Quick Actions Section',
                selector: '[data-testid="quick-actions"], .mantine-Card-root:has-text("Quick Actions")',
                expectedElements: ['buttons', 'navigation']
            },
            {
                name: 'Comparison Chart Component',
                selector: '[data-testid="comparison-chart"], .recharts-wrapper',
                expectedElements: ['chart', 'bars', 'data']
            }
        ];

        for (const test of componentTests) {
            try {
                const startTime = Date.now();
                
                // Wait for component to be visible
                const element = await this.page.waitForSelector(test.selector, { 
                    timeout: 10000,
                    state: 'visible'
                });

                if (element) {
                    // Take screenshot of component
                    await element.screenshot({ 
                        path: `./testing/screenshots/component-${test.name.toLowerCase().replace(/\s+/g, '-')}.png` 
                    });

                    // Test for expected text if provided
                    if (test.expectedText) {
                        const textContent = await element.textContent();
                        const hasExpectedText = textContent.includes(test.expectedText);
                        
                        this.testResults.componentTests.push({
                            name: test.name,
                            status: hasExpectedText ? 'PASS' : 'FAIL',
                            selector: test.selector,
                            responseTime: Date.now() - startTime,
                            hasExpectedText,
                            actualText: textContent.substring(0, 100)
                        });
                    } else {
                        this.testResults.componentTests.push({
                            name: test.name,
                            status: 'PASS',
                            selector: test.selector,
                            responseTime: Date.now() - startTime,
                            elementFound: true
                        });
                    }
                    
                    console.log(`✅ ${test.name}: Component found and visible`);
                } else {
                    throw new Error('Element not found');
                }
            } catch (error) {
                this.testResults.componentTests.push({
                    name: test.name,
                    status: 'FAIL',
                    selector: test.selector,
                    error: error.message
                });
                
                console.log(`❌ ${test.name}: ${error.message}`);
            }
        }
    }

    async testDataExchange() {
        console.log('\n📡 Testing Data Exchange Between Frontend and Backend...');
        
        // Monitor network requests
        const apiCalls = [];
        
        this.page.on('response', response => {
            const url = response.url();
            if (url.includes('/api/')) {
                apiCalls.push({
                    url,
                    status: response.status(),
                    contentType: response.headers()['content-type'],
                    timestamp: Date.now()
                });
            }
        });

        // Refresh the page to trigger API calls
        await this.page.reload({ waitUntil: 'networkidle' });
        
        // Wait for all API calls to complete
        await this.page.waitForTimeout(5000);

        // Analyze API calls
        const expectedAPICalls = [
            '/api/dashboard/metrics',
            '/api/dashboard/recent-activity'
        ];

        for (const expectedCall of expectedAPICalls) {
            const found = apiCalls.find(call => call.url.includes(expectedCall));
            
            if (found) {
                this.testResults.dataExchangeTests.push({
                    apiCall: expectedCall,
                    status: found.status === 200 ? 'PASS' : 'FAIL',
                    statusCode: found.status,
                    contentType: found.contentType,
                    timestamp: found.timestamp
                });
                
                console.log(`✅ ${expectedCall}: Status ${found.status}`);
            } else {
                this.testResults.dataExchangeTests.push({
                    apiCall: expectedCall,
                    status: 'FAIL',
                    error: 'API call not made'
                });
                
                console.log(`❌ ${expectedCall}: API call not made`);
            }
        }

        // Test data rendering in components
        await this.testDataRendering();
    }

    async testDataRendering() {
        console.log('\n🎨 Testing Data Rendering in Components...');
        
        const dataRenderingTests = [
            {
                name: 'AFM Compliance Score Value',
                selector: '[data-testid="afm-compliance"] .mantine-Text-root, .mantine-Card-root:has-text("AFM Compliance") .mantine-Text-root',
                expectedPattern: /\d+%/,
                description: 'Should display percentage value'
            },
            {
                name: 'Active Sessions Count',
                selector: '[data-testid="active-sessions"] .mantine-Text-root, .mantine-Card-root:has-text("Active Sessions") .mantine-Text-root',
                expectedPattern: /\d+/,
                description: 'Should display numeric value'
            },
            {
                name: 'Recent Activity Items',
                selector: '[data-testid="recent-activity"] .mantine-Paper-root, .mantine-Card-root:has-text("Recent Activity") .mantine-Paper-root',
                expectedCount: { min: 1, max: 10 },
                description: 'Should display activity items'
            },
            {
                name: 'Chart Data Visualization',
                selector: '.recharts-bar, .recharts-line',
                expectedCount: { min: 2, max: 20 },
                description: 'Should display chart elements'
            }
        ];

        for (const test of dataRenderingTests) {
            try {
                if (test.expectedPattern) {
                    const elements = await this.page.$$eval(test.selector, els => 
                        els.map(el => el.textContent)
                    );
                    
                    const hasValidData = elements.some(text => test.expectedPattern.test(text));
                    
                    this.testResults.dataExchangeTests.push({
                        name: test.name,
                        status: hasValidData ? 'PASS' : 'FAIL',
                        description: test.description,
                        foundData: elements.join(', ').substring(0, 100),
                        hasValidPattern: hasValidData
                    });
                    
                    console.log(`${hasValidData ? '✅' : '❌'} ${test.name}: ${hasValidData ? 'Valid data found' : 'No valid data'}`);
                } else if (test.expectedCount) {
                    const elements = await this.page.$$(test.selector);
                    const count = elements.length;
                    const isValidCount = count >= test.expectedCount.min && count <= test.expectedCount.max;
                    
                    this.testResults.dataExchangeTests.push({
                        name: test.name,
                        status: isValidCount ? 'PASS' : 'FAIL',
                        description: test.description,
                        elementCount: count,
                        expectedRange: test.expectedCount,
                        isValidCount
                    });
                    
                    console.log(`${isValidCount ? '✅' : '❌'} ${test.name}: Found ${count} elements`);
                }
            } catch (error) {
                this.testResults.dataExchangeTests.push({
                    name: test.name,
                    status: 'FAIL',
                    error: error.message,
                    description: test.description
                });
                
                console.log(`❌ ${test.name}: ${error.message}`);
            }
        }
    }

    async testComponentInteractions() {
        console.log('\n🖱️  Testing Component Interactions...');
        
        const interactionTests = [
            {
                name: 'Quick Action Button - New Client Intake',
                selector: 'button:has-text("New Client Intake"), .mantine-Button-root:has-text("New Client")',
                action: 'click',
                expectedNavigation: '/afm-client-intake'
            },
            {
                name: 'Quick Action Button - Compliance Check',
                selector: 'button:has-text("Compliance Check"), .mantine-Button-root:has-text("Compliance")',
                action: 'click',
                expectedNavigation: '/compliance'
            },
            {
                name: 'Quick Action Button - Quality Control',
                selector: 'button:has-text("Quality Control"), .mantine-Button-root:has-text("Quality")',
                action: 'click',
                expectedNavigation: '/quality-control'
            },
            {
                name: 'Recent Activity View All',
                selector: 'button:has-text("View All"), .mantine-Button-root:has-text("View All")',
                action: 'click',
                description: 'Should expand or navigate to detailed view'
            }
        ];

        for (const test of interactionTests) {
            try {
                // Navigate back to dashboard first
                await this.page.goto(this.baseUrl);
                await this.page.waitForLoadState('networkidle');
                
                const element = await this.page.waitForSelector(test.selector, { 
                    timeout: 10000,
                    state: 'visible'
                });

                if (element) {
                    const startUrl = this.page.url();
                    
                    // Perform the interaction
                    await element.click();
                    
                    // Wait for navigation or state change
                    await this.page.waitForTimeout(2000);
                    
                    const endUrl = this.page.url();
                    const hasNavigated = startUrl !== endUrl;
                    
                    // Check if expected navigation occurred
                    if (test.expectedNavigation) {
                        const navigatedCorrectly = endUrl.includes(test.expectedNavigation);
                        
                        this.testResults.interactionTests.push({
                            name: test.name,
                            status: navigatedCorrectly ? 'PASS' : 'FAIL',
                            startUrl,
                            endUrl,
                            expectedNavigation: test.expectedNavigation,
                            navigatedCorrectly
                        });
                        
                        console.log(`${navigatedCorrectly ? '✅' : '❌'} ${test.name}: ${navigatedCorrectly ? 'Navigation successful' : 'Navigation failed'}`);
                    } else {
                        this.testResults.interactionTests.push({
                            name: test.name,
                            status: 'PASS',
                            description: test.description,
                            interactionPerformed: true,
                            hasNavigated
                        });
                        
                        console.log(`✅ ${test.name}: Interaction performed successfully`);
                    }
                } else {
                    throw new Error('Interactive element not found');
                }
            } catch (error) {
                this.testResults.interactionTests.push({
                    name: test.name,
                    status: 'FAIL',
                    error: error.message
                });
                
                console.log(`❌ ${test.name}: ${error.message}`);
            }
        }
    }

    async testResponsiveDesign() {
        console.log('\n📱 Testing Responsive Design...');
        
        const viewports = [
            { name: 'Desktop', width: 1920, height: 1080 },
            { name: 'Tablet', width: 768, height: 1024 },
            { name: 'Mobile', width: 375, height: 667 }
        ];

        for (const viewport of viewports) {
            try {
                await this.page.setViewportSize({ width: viewport.width, height: viewport.height });
                await this.page.goto(this.baseUrl);
                await this.page.waitForLoadState('networkidle');
                
                // Take screenshot for visual verification
                await this.page.screenshot({ 
                    path: `./testing/screenshots/responsive-${viewport.name.toLowerCase()}.png`,
                    fullPage: true 
                });
                
                // Check if key components are still visible
                const criticalComponents = [
                    '.mantine-Card-root',
                    'header, [role="banner"]',
                    '.mantine-Button-root'
                ];
                
                let visibleComponents = 0;
                for (const selector of criticalComponents) {
                    const elements = await this.page.$$(selector);
                    if (elements.length > 0) {
                        visibleComponents++;
                    }
                }
                
                const allComponentsVisible = visibleComponents === criticalComponents.length;
                
                this.testResults.componentTests.push({
                    name: `Responsive Design - ${viewport.name}`,
                    status: allComponentsVisible ? 'PASS' : 'FAIL',
                    viewport: `${viewport.width}x${viewport.height}`,
                    visibleComponents,
                    totalComponents: criticalComponents.length
                });
                
                console.log(`${allComponentsVisible ? '✅' : '❌'} ${viewport.name}: ${visibleComponents}/${criticalComponents.length} components visible`);
            } catch (error) {
                console.log(`❌ ${viewport.name}: ${error.message}`);
            }
        }
        
        // Reset to desktop viewport
        await this.page.setViewportSize({ width: 1920, height: 1080 });
    }

    calculateResults() {
        const allTests = [
            ...this.testResults.componentTests,
            ...this.testResults.apiConnectivityTests,
            ...this.testResults.dataExchangeTests,
            ...this.testResults.interactionTests
        ];
        
        this.testResults.totalTests = allTests.length;
        this.testResults.passedTests = allTests.filter(test => test.status === 'PASS').length;
        this.testResults.failedTests = allTests.filter(test => test.status === 'FAIL').length;
        
        const passRate = (this.testResults.passedTests / this.testResults.totalTests) * 100;
        
        if (passRate >= 90) {
            this.testResults.overallStatus = 'EXCELLENT';
        } else if (passRate >= 75) {
            this.testResults.overallStatus = 'GOOD';
        } else if (passRate >= 50) {
            this.testResults.overallStatus = 'NEEDS_IMPROVEMENT';
        } else {
            this.testResults.overallStatus = 'CRITICAL';
        }
    }

    async generateReport() {
        console.log('\n📄 Generating Comprehensive Test Report...');
        
        this.calculateResults();
        
        const reportHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard Component Testing Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .status-excellent { color: #10b981; font-weight: bold; }
        .status-good { color: #3b82f6; font-weight: bold; }
        .status-needs-improvement { color: #f59e0b; font-weight: bold; }
        .status-critical { color: #ef4444; font-weight: bold; }
        .pass { color: #10b981; }
        .fail { color: #ef4444; }
        .metric { display: inline-block; margin: 15px; padding: 20px; background: #f8fafc; border-radius: 8px; text-align: center; min-width: 120px; }
        .metric h3 { margin: 0; font-size: 2em; }
        .section { margin: 30px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #e5e7eb; padding: 12px; text-align: left; }
        th { background: #f9fafb; font-weight: 600; }
        .screenshot { max-width: 300px; margin: 10px; border: 1px solid #e5e7eb; border-radius: 8px; }
        pre { background: #f1f5f9; padding: 20px; border-radius: 8px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎭 Dashboard Component Testing Report</h1>
        <p>Comprehensive testing of MortgageAI Dashboard components and backend connectivity</p>
        <p>Generated: ${this.testResults.timestamp}</p>
    </div>

    <div class="section">
        <h2>📊 Overall Results</h2>
        <div class="metric">
            <h3 class="status-${this.testResults.overallStatus.toLowerCase().replace('_', '-')}">${this.testResults.overallStatus}</h3>
            <p>Overall Status</p>
        </div>
        <div class="metric">
            <h3>${this.testResults.totalTests}</h3>
            <p>Total Tests</p>
        </div>
        <div class="metric">
            <h3 class="pass">${this.testResults.passedTests}</h3>
            <p>Passed</p>
        </div>
        <div class="metric">
            <h3 class="fail">${this.testResults.failedTests}</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3>${((this.testResults.passedTests / this.testResults.totalTests) * 100).toFixed(1)}%</h3>
            <p>Success Rate</p>
        </div>
    </div>

    <div class="section">
        <h2>🧩 Component Tests</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Status</th>
                <th>Response Time</th>
                <th>Details</th>
            </tr>
            ${this.testResults.componentTests.map(test => `
            <tr>
                <td>${test.name}</td>
                <td class="${test.status.toLowerCase()}">${test.status}</td>
                <td>${test.responseTime || 'N/A'}ms</td>
                <td>${test.error || test.description || 'Component rendered successfully'}</td>
            </tr>
            `).join('')}
        </table>
    </div>

    <div class="section">
        <h2>🔗 API Connectivity Tests</h2>
        <table>
            <tr>
                <th>Endpoint</th>
                <th>Status</th>
                <th>Status Code</th>
                <th>Has Data</th>
            </tr>
            ${this.testResults.apiConnectivityTests.map(test => `
            <tr>
                <td>${test.endpoint}</td>
                <td class="${test.status.toLowerCase()}">${test.status}</td>
                <td>${test.statusCode}</td>
                <td>${test.hasData ? '✅' : '❌'}</td>
            </tr>
            `).join('')}
        </table>
    </div>

    <div class="section">
        <h2>📡 Data Exchange Tests</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Description</th>
                <th>Details</th>
            </tr>
            ${this.testResults.dataExchangeTests.map(test => `
            <tr>
                <td>${test.name || test.apiCall}</td>
                <td class="${test.status.toLowerCase()}">${test.status}</td>
                <td>${test.description || 'API call verification'}</td>
                <td>${test.error || test.foundData || test.contentType || 'Success'}</td>
            </tr>
            `).join('')}
        </table>
    </div>

    <div class="section">
        <h2>🖱️ Interaction Tests</h2>
        <table>
            <tr>
                <th>Interaction</th>
                <th>Status</th>
                <th>Expected Navigation</th>
                <th>Result</th>
            </tr>
            ${this.testResults.interactionTests.map(test => `
            <tr>
                <td>${test.name}</td>
                <td class="${test.status.toLowerCase()}">${test.status}</td>
                <td>${test.expectedNavigation || 'N/A'}</td>
                <td>${test.error || (test.navigatedCorrectly ? 'Navigation successful' : test.description || 'Interaction completed')}</td>
            </tr>
            `).join('')}
        </table>
    </div>

    <div class="section">
        <h2>📱 Test Configuration</h2>
        <pre>${JSON.stringify({
            baseUrl: this.baseUrl,
            apiBaseUrl: this.apiBaseUrl,
            browser: 'Chromium',
            viewport: '1920x1080',
            testTypes: ['Component Rendering', 'API Connectivity', 'Data Exchange', 'User Interactions', 'Responsive Design']
        }, null, 2)}</pre>
    </div>

    <div class="section">
        <h2>🔍 Recommendations</h2>
        ${this.generateRecommendations()}
    </div>
</body>
</html>
        `;

        const reportPath = './testing/dashboard-component-test-report.html';
        await fs.writeFile(reportPath, reportHTML);
        
        // Also save JSON results
        await fs.writeFile('./testing/dashboard-test-results.json', JSON.stringify(this.testResults, null, 2));
        
        console.log(`✅ Report generated: ${reportPath}`);
    }

    generateRecommendations() {
        const recommendations = [];
        
        if (this.testResults.failedTests > 0) {
            recommendations.push('<li><strong>Fix Failed Tests:</strong> Review and address the failed test cases to ensure all components are working correctly.</li>');
        }
        
        if (this.testResults.apiConnectivityTests.some(test => test.status === 'FAIL')) {
            recommendations.push('<li><strong>API Connectivity:</strong> Some API endpoints are not responding correctly. Check backend service status and network connectivity.</li>');
        }
        
        if (this.testResults.dataExchangeTests.some(test => test.status === 'FAIL')) {
            recommendations.push('<li><strong>Data Exchange:</strong> Issues found with data exchange between frontend and backend. Verify API responses and frontend data handling.</li>');
        }
        
        if (this.testResults.interactionTests.some(test => test.status === 'FAIL')) {
            recommendations.push('<li><strong>User Interactions:</strong> Some interactive elements are not working as expected. Check navigation and event handlers.</li>');
        }
        
        const passRate = (this.testResults.passedTests / this.testResults.totalTests) * 100;
        if (passRate < 85) {
            recommendations.push('<li><strong>Overall Quality:</strong> Consider implementing additional error handling and improving component reliability.</li>');
        }
        
        if (recommendations.length === 0) {
            recommendations.push('<li><strong>Excellent Work!</strong> All dashboard components are functioning correctly and properly connected to the backend.</li>');
        }
        
        return `<ul>${recommendations.join('')}</ul>`;
    }

    async cleanup() {
        if (this.browser) {
            await this.browser.close();
        }
        console.log('🧹 Cleanup completed');
    }

    async runFullTestSuite() {
        try {
            console.log('🎯 Starting Full Dashboard Component Test Suite...');
            
            await this.initialize();
            
            // Test backend connectivity first
            await this.testBackendConnectivity();
            
            // Navigate to dashboard
            const dashboardLoaded = await this.navigateToDashboard();
            if (!dashboardLoaded) {
                throw new Error('Failed to load dashboard');
            }
            
            // Run all component tests
            await this.testDashboardComponents();
            await this.testDataExchange();
            await this.testComponentInteractions();
            await this.testResponsiveDesign();
            
            // Generate final report
            await this.generateReport();
            
            console.log(`\n🎉 Testing completed! Results: ${this.testResults.passedTests}/${this.testResults.totalTests} tests passed`);
            console.log(`📊 Overall Status: ${this.testResults.overallStatus}`);
            
            return this.testResults;
        } catch (error) {
            console.error('❌ Test suite failed:', error.message);
            throw error;
        } finally {
            await this.cleanup();
        }
    }
}

// Main execution
async function main() {
    const tester = new DashboardComponentTester();
    
    try {
        const results = await tester.runFullTestSuite();
        
        // Exit with appropriate code
        const exitCode = results.overallStatus === 'CRITICAL' ? 1 : 0;
        process.exit(exitCode);
    } catch (error) {
        console.error('Fatal error:', error);
        process.exit(1);
    }
}

// Export for use in other scripts
module.exports = { DashboardComponentTester };

// Run if this is the main module
if (require.main === module) {
    main();
}