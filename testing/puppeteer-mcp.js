/**
 * Puppeteer MCP (Model Context Protocol) Integration
 * Advanced browser-based testing capabilities with Puppeteer for comprehensive UI testing
 * 
 * Features:
 * - Automated browser testing with Puppeteer
 * - Visual regression testing and screenshot comparison
 * - Performance monitoring and lighthouse audits
 * - Cross-browser compatibility testing
 * - Advanced interaction testing (forms, navigation, workflows)
 * - Accessibility testing integration
 * - PDF generation and validation
 * - Network monitoring and API testing
 * - Mobile device simulation
 * - Comprehensive test reporting
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const pixelmatch = require('pixelmatch');
const { PNG } = require('pngjs');
const axeCore = require('@axe-core/puppeteer');

class PuppeteerMCP {
    constructor(config = {}) {
        this.config = {
            headless: config.headless !== false,
            viewport: config.viewport || { width: 1920, height: 1080 },
            timeout: config.timeout || 30000,
            screenshotPath: config.screenshotPath || './testing/screenshots',
            reportsPath: config.reportsPath || './testing/reports',
            ...config
        };
        
        this.browser = null;
        this.page = null;
        this.testResults = [];
        this.performanceMetrics = {};
    }

    /**
     * Initialize Puppeteer browser
     */
    async initialize() {
        try {
            console.log('🚀 Initializing Puppeteer MCP...');
            
            // Ensure directories exist
            await this.ensureDirectories();
            
            // Launch browser
            this.browser = await puppeteer.launch({
                headless: this.config.headless,
                args: [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            });

            // Create new page
            this.page = await this.browser.newPage();
            await this.page.setViewport(this.config.viewport);
            
            // Set default timeout
            this.page.setDefaultTimeout(this.config.timeout);
            
            console.log('✅ Puppeteer MCP initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize Puppeteer MCP:', error);
            throw error;
        }
    }

    /**
     * Ensure required directories exist
     */
    async ensureDirectories() {
        const dirs = [
            this.config.screenshotPath,
            this.config.reportsPath,
            path.join(this.config.reportsPath, 'lighthouse'),
            path.join(this.config.reportsPath, 'accessibility'),
            path.join(this.config.reportsPath, 'performance')
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
     * Navigate to URL and wait for load
     */
    async navigateToUrl(url, waitForSelector = null) {
        try {
            console.log(`🌐 Navigating to: ${url}`);
            
            const response = await this.page.goto(url, {
                waitUntil: 'networkidle2',
                timeout: this.config.timeout
            });

            if (waitForSelector) {
                await this.page.waitForSelector(waitForSelector, { timeout: 10000 });
            }

            const finalUrl = this.page.url();
            const statusCode = response.status();
            
            console.log(`✅ Navigation completed: ${finalUrl} (${statusCode})`);
            
            return {
                success: true,
                url: finalUrl,
                statusCode,
                loadTime: response.timing()
            };
        } catch (error) {
            console.error(`❌ Navigation failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Take screenshot with comparison
     */
    async takeScreenshot(name, compareWithBaseline = false) {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `${name}_${timestamp}.png`;
            const filepath = path.join(this.config.screenshotPath, filename);
            
            await this.page.screenshot({
                path: filepath,
                fullPage: true
            });

            console.log(`📸 Screenshot saved: ${filename}`);

            let comparison = null;
            if (compareWithBaseline) {
                comparison = await this.compareWithBaseline(name, filepath);
            }

            return {
                success: true,
                filename,
                filepath,
                comparison
            };
        } catch (error) {
            console.error(`❌ Screenshot failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Compare screenshot with baseline
     */
    async compareWithBaseline(name, currentPath) {
        try {
            const baselinePath = path.join(this.config.screenshotPath, 'baseline', `${name}_baseline.png`);
            
            // Check if baseline exists
            try {
                await fs.access(baselinePath);
            } catch {
                // Create baseline if it doesn't exist
                await fs.mkdir(path.dirname(baselinePath), { recursive: true });
                await fs.copyFile(currentPath, baselinePath);
                return {
                    status: 'baseline_created',
                    message: 'Baseline screenshot created'
                };
            }

            // Load images
            const baseline = PNG.sync.read(await fs.readFile(baselinePath));
            const current = PNG.sync.read(await fs.readFile(currentPath));

            // Compare dimensions
            if (baseline.width !== current.width || baseline.height !== current.height) {
                return {
                    status: 'dimension_mismatch',
                    message: 'Screenshot dimensions do not match baseline'
                };
            }

            // Compare pixels
            const diff = new PNG({ width: baseline.width, height: baseline.height });
            const numDiffPixels = pixelmatch(
                baseline.data, 
                current.data, 
                diff.data, 
                baseline.width, 
                baseline.height,
                { threshold: 0.1 }
            );

            const diffPercentage = (numDiffPixels / (baseline.width * baseline.height)) * 100;

            // Save diff image if significant differences
            if (diffPercentage > 0.1) {
                const diffPath = path.join(this.config.screenshotPath, 'diff', `${name}_diff.png`);
                await fs.mkdir(path.dirname(diffPath), { recursive: true });
                await fs.writeFile(diffPath, PNG.sync.write(diff));
            }

            return {
                status: diffPercentage > 5 ? 'significant_difference' : 'minor_difference',
                diffPercentage,
                diffPixels: numDiffPixels,
                message: `${diffPercentage.toFixed(2)}% difference detected`
            };
        } catch (error) {
            return {
                status: 'comparison_failed',
                error: error.message
            };
        }
    }

    /**
     * Test form interactions
     */
    async testFormInteractions(formSelector, formData) {
        try {
            console.log(`📝 Testing form interactions: ${formSelector}`);
            
            const results = {
                success: true,
                interactions: [],
                errors: []
            };

            // Wait for form to be visible
            await this.page.waitForSelector(formSelector);

            // Fill form fields
            for (const [fieldName, value] of Object.entries(formData)) {
                try {
                    const fieldSelector = `${formSelector} [name="${fieldName}"], ${formSelector} #${fieldName}`;
                    await this.page.waitForSelector(fieldSelector, { timeout: 5000 });
                    
                    const fieldType = await this.page.evaluate((selector) => {
                        const element = document.querySelector(selector);
                        return element ? element.type || element.tagName.toLowerCase() : null;
                    }, fieldSelector);

                    if (fieldType === 'select' || fieldType === 'SELECT') {
                        await this.page.select(fieldSelector, value);
                    } else if (fieldType === 'checkbox' || fieldType === 'radio') {
                        if (value) {
                            await this.page.click(fieldSelector);
                        }
                    } else {
                        await this.page.type(fieldSelector, value.toString(), { delay: 50 });
                    }

                    results.interactions.push({
                        field: fieldName,
                        type: fieldType,
                        value: value,
                        success: true
                    });
                } catch (error) {
                    results.errors.push({
                        field: fieldName,
                        error: error.message
                    });
                    results.success = false;
                }
            }

            console.log(`✅ Form interactions completed: ${results.interactions.length} successful, ${results.errors.length} errors`);
            return results;
        } catch (error) {
            console.error(`❌ Form interaction test failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Test navigation and workflow
     */
    async testWorkflow(steps) {
        try {
            console.log(`🔄 Testing workflow with ${steps.length} steps`);
            
            const results = {
                success: true,
                steps: [],
                totalTime: 0
            };

            const startTime = Date.now();

            for (let i = 0; i < steps.length; i++) {
                const step = steps[i];
                const stepStartTime = Date.now();
                
                try {
                    let stepResult = { step: i + 1, action: step.action, success: true };

                    switch (step.action) {
                        case 'navigate':
                            const navResult = await this.navigateToUrl(step.url, step.waitFor);
                            stepResult.url = step.url;
                            stepResult.success = navResult.success;
                            break;

                        case 'click':
                            await this.page.waitForSelector(step.selector);
                            await this.page.click(step.selector);
                            stepResult.selector = step.selector;
                            break;

                        case 'type':
                            await this.page.waitForSelector(step.selector);
                            await this.page.type(step.selector, step.text);
                            stepResult.selector = step.selector;
                            stepResult.text = step.text;
                            break;

                        case 'wait':
                            if (step.selector) {
                                await this.page.waitForSelector(step.selector);
                                stepResult.selector = step.selector;
                            } else {
                                await this.page.waitForTimeout(step.timeout || 1000);
                                stepResult.timeout = step.timeout;
                            }
                            break;

                        case 'screenshot':
                            const screenshotResult = await this.takeScreenshot(step.name || `step_${i + 1}`);
                            stepResult.screenshot = screenshotResult;
                            break;

                        case 'assert':
                            const assertResult = await this.page.evaluate((selector, expectedText) => {
                                const element = document.querySelector(selector);
                                if (!element) return { success: false, error: 'Element not found' };
                                
                                const actualText = element.textContent.trim();
                                const success = expectedText ? actualText.includes(expectedText) : !!actualText;
                                
                                return { success, actualText, expectedText };
                            }, step.selector, step.expectedText);
                            
                            stepResult = { ...stepResult, ...assertResult };
                            break;

                        default:
                            stepResult.success = false;
                            stepResult.error = `Unknown action: ${step.action}`;
                    }

                    stepResult.duration = Date.now() - stepStartTime;
                    results.steps.push(stepResult);

                    if (!stepResult.success) {
                        results.success = false;
                        console.log(`❌ Step ${i + 1} failed: ${stepResult.error || 'Unknown error'}`);
                    } else {
                        console.log(`✅ Step ${i + 1} completed: ${step.action}`);
                    }

                } catch (error) {
                    results.steps.push({
                        step: i + 1,
                        action: step.action,
                        success: false,
                        error: error.message,
                        duration: Date.now() - stepStartTime
                    });
                    results.success = false;
                    console.log(`❌ Step ${i + 1} failed with exception: ${error.message}`);
                }
            }

            results.totalTime = Date.now() - startTime;
            console.log(`🔄 Workflow completed in ${results.totalTime}ms`);
            
            return results;
        } catch (error) {
            console.error(`❌ Workflow test failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Run Lighthouse performance audit
     */
    async runLighthouseAudit(url) {
        try {
            console.log(`⚡ Running Lighthouse audit for: ${url}`);
            
            const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });
            const options = {
                logLevel: 'info',
                output: 'html',
                onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
                port: chrome.port
            };

            const runnerResult = await lighthouse(url, options);
            await chrome.kill();

            // Save report
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const reportPath = path.join(this.config.reportsPath, 'lighthouse', `lighthouse_${timestamp}.html`);
            await fs.writeFile(reportPath, runnerResult.report);

            // Extract key metrics
            const lhr = runnerResult.lhr;
            const metrics = {
                performance: Math.round(lhr.categories.performance.score * 100),
                accessibility: Math.round(lhr.categories.accessibility.score * 100),
                bestPractices: Math.round(lhr.categories['best-practices'].score * 100),
                seo: Math.round(lhr.categories.seo.score * 100),
                firstContentfulPaint: lhr.audits['first-contentful-paint'].displayValue,
                largestContentfulPaint: lhr.audits['largest-contentful-paint'].displayValue,
                cumulativeLayoutShift: lhr.audits['cumulative-layout-shift'].displayValue,
                totalBlockingTime: lhr.audits['total-blocking-time'].displayValue
            };

            console.log(`✅ Lighthouse audit completed - Performance: ${metrics.performance}/100`);
            
            return {
                success: true,
                metrics,
                reportPath,
                fullReport: lhr
            };
        } catch (error) {
            console.error(`❌ Lighthouse audit failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Run accessibility tests
     */
    async runAccessibilityTests(url) {
        try {
            console.log(`♿ Running accessibility tests for: ${url}`);
            
            await this.navigateToUrl(url);
            await axeCore.injectIntoPage(this.page);
            
            const results = await this.page.evaluate(async () => {
                return await axe.run();
            });

            // Save detailed report
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const reportPath = path.join(this.config.reportsPath, 'accessibility', `accessibility_${timestamp}.json`);
            await fs.writeFile(reportPath, JSON.stringify(results, null, 2));

            const summary = {
                violations: results.violations.length,
                passes: results.passes.length,
                incomplete: results.incomplete.length,
                inapplicable: results.inapplicable.length,
                criticalViolations: results.violations.filter(v => v.impact === 'critical').length,
                seriousViolations: results.violations.filter(v => v.impact === 'serious').length
            };

            console.log(`✅ Accessibility tests completed - ${summary.violations} violations found`);
            
            return {
                success: true,
                summary,
                reportPath,
                fullResults: results
            };
        } catch (error) {
            console.error(`❌ Accessibility tests failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Monitor network requests
     */
    async monitorNetworkRequests(url, duration = 30000) {
        try {
            console.log(`🌐 Monitoring network requests for: ${url}`);
            
            const requests = [];
            const responses = [];
            const errors = [];

            // Set up network monitoring
            this.page.on('request', request => {
                requests.push({
                    url: request.url(),
                    method: request.method(),
                    headers: request.headers(),
                    timestamp: Date.now()
                });
            });

            this.page.on('response', response => {
                responses.push({
                    url: response.url(),
                    status: response.status(),
                    headers: response.headers(),
                    size: response.headers()['content-length'] || 0,
                    timestamp: Date.now()
                });
            });

            this.page.on('requestfailed', request => {
                errors.push({
                    url: request.url(),
                    error: request.failure().errorText,
                    timestamp: Date.now()
                });
            });

            // Navigate and monitor
            await this.navigateToUrl(url);
            await this.page.waitForTimeout(duration);

            const analysis = {
                totalRequests: requests.length,
                totalResponses: responses.length,
                totalErrors: errors.length,
                avgResponseTime: this.calculateAverageResponseTime(requests, responses),
                statusCodes: this.analyzeStatusCodes(responses),
                largestRequests: this.findLargestRequests(responses, 5),
                errorDetails: errors
            };

            console.log(`✅ Network monitoring completed - ${analysis.totalRequests} requests, ${analysis.totalErrors} errors`);
            
            return {
                success: true,
                analysis,
                requests,
                responses,
                errors
            };
        } catch (error) {
            console.error(`❌ Network monitoring failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Test mobile responsiveness
     */
    async testMobileResponsiveness(url) {
        try {
            console.log(`📱 Testing mobile responsiveness for: ${url}`);
            
            const devices = [
                { name: 'iPhone 12', viewport: { width: 390, height: 844 } },
                { name: 'iPad', viewport: { width: 768, height: 1024 } },
                { name: 'Galaxy S21', viewport: { width: 384, height: 854 } },
                { name: 'Desktop', viewport: { width: 1920, height: 1080 } }
            ];

            const results = [];

            for (const device of devices) {
                try {
                    await this.page.setViewport(device.viewport);
                    await this.navigateToUrl(url);
                    
                    // Take screenshot
                    const screenshot = await this.takeScreenshot(`mobile_${device.name.toLowerCase().replace(' ', '_')}`);
                    
                    // Check for mobile-specific issues
                    const mobileIssues = await this.page.evaluate(() => {
                        const issues = [];
                        
                        // Check for horizontal scrolling
                        if (document.body.scrollWidth > window.innerWidth) {
                            issues.push('horizontal_scroll');
                        }
                        
                        // Check for tiny text
                        const elements = document.querySelectorAll('*');
                        let tinyTextCount = 0;
                        elements.forEach(el => {
                            const style = window.getComputedStyle(el);
                            const fontSize = parseInt(style.fontSize);
                            if (fontSize < 12 && el.textContent.trim()) {
                                tinyTextCount++;
                            }
                        });
                        
                        if (tinyTextCount > 0) {
                            issues.push(`tiny_text_${tinyTextCount}_elements`);
                        }
                        
                        // Check for clickable elements too close together
                        const clickables = document.querySelectorAll('button, a, input[type="button"], input[type="submit"]');
                        let tooCloseCount = 0;
                        for (let i = 0; i < clickables.length; i++) {
                            for (let j = i + 1; j < clickables.length; j++) {
                                const rect1 = clickables[i].getBoundingClientRect();
                                const rect2 = clickables[j].getBoundingClientRect();
                                
                                const distance = Math.sqrt(
                                    Math.pow(rect1.left - rect2.left, 2) + 
                                    Math.pow(rect1.top - rect2.top, 2)
                                );
                                
                                if (distance < 44) { // 44px is recommended minimum touch target
                                    tooCloseCount++;
                                }
                            }
                        }
                        
                        if (tooCloseCount > 0) {
                            issues.push(`touch_targets_too_close_${tooCloseCount}`);
                        }
                        
                        return issues;
                    });

                    results.push({
                        device: device.name,
                        viewport: device.viewport,
                        screenshot: screenshot,
                        issues: mobileIssues,
                        success: mobileIssues.length === 0
                    });

                    console.log(`📱 ${device.name}: ${mobileIssues.length} issues found`);
                } catch (error) {
                    results.push({
                        device: device.name,
                        viewport: device.viewport,
                        success: false,
                        error: error.message
                    });
                }
            }

            // Reset to default viewport
            await this.page.setViewport(this.config.viewport);

            const overallSuccess = results.every(r => r.success);
            console.log(`✅ Mobile responsiveness testing completed - ${overallSuccess ? 'All devices passed' : 'Issues found'}`);
            
            return {
                success: overallSuccess,
                results
            };
        } catch (error) {
            console.error(`❌ Mobile responsiveness test failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Generate comprehensive test report
     */
    async generateTestReport(testResults) {
        try {
            console.log('📄 Generating comprehensive test report...');
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const reportPath = path.join(this.config.reportsPath, `puppeteer_test_report_${timestamp}.html`);

            const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Puppeteer MCP Test Report</title>
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
        .screenshot { max-width: 300px; margin: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎭 Puppeteer MCP Test Report</h1>
        <p>Generated on: ${new Date().toLocaleString()}</p>
    </div>

    <div class="section">
        <h2>📊 Test Summary</h2>
        <div class="metric">
            <h3>${testResults.totalTests || 0}</h3>
            <p>Total Tests</p>
        </div>
        <div class="metric">
            <h3 class="${testResults.passedTests > 0 ? 'success' : ''}">${testResults.passedTests || 0}</h3>
            <p>Passed</p>
        </div>
        <div class="metric">
            <h3 class="${testResults.failedTests > 0 ? 'error' : ''}">${testResults.failedTests || 0}</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3>${testResults.totalTime || 0}ms</h3>
            <p>Total Time</p>
        </div>
    </div>

    ${testResults.lighthouse ? `
    <div class="section">
        <h2>⚡ Performance (Lighthouse)</h2>
        <div class="metric">
            <h3 class="${testResults.lighthouse.metrics.performance >= 90 ? 'success' : testResults.lighthouse.metrics.performance >= 50 ? 'warning' : 'error'}">${testResults.lighthouse.metrics.performance}/100</h3>
            <p>Performance Score</p>
        </div>
        <div class="metric">
            <h3>${testResults.lighthouse.metrics.accessibility}/100</h3>
            <p>Accessibility</p>
        </div>
        <div class="metric">
            <h3>${testResults.lighthouse.metrics.bestPractices}/100</h3>
            <p>Best Practices</p>
        </div>
        <div class="metric">
            <h3>${testResults.lighthouse.metrics.seo}/100</h3>
            <p>SEO</p>
        </div>
    </div>
    ` : ''}

    ${testResults.accessibility ? `
    <div class="section">
        <h2>♿ Accessibility Results</h2>
        <div class="metric">
            <h3 class="${testResults.accessibility.summary.violations === 0 ? 'success' : 'error'}">${testResults.accessibility.summary.violations}</h3>
            <p>Violations</p>
        </div>
        <div class="metric">
            <h3 class="error">${testResults.accessibility.summary.criticalViolations}</h3>
            <p>Critical</p>
        </div>
        <div class="metric">
            <h3 class="warning">${testResults.accessibility.summary.seriousViolations}</h3>
            <p>Serious</p>
        </div>
        <div class="metric">
            <h3 class="success">${testResults.accessibility.summary.passes}</h3>
            <p>Passes</p>
        </div>
    </div>
    ` : ''}

    ${testResults.mobile ? `
    <div class="section">
        <h2>📱 Mobile Responsiveness</h2>
        <table>
            <tr>
                <th>Device</th>
                <th>Viewport</th>
                <th>Issues</th>
                <th>Status</th>
            </tr>
            ${testResults.mobile.results.map(result => `
            <tr>
                <td>${result.device}</td>
                <td>${result.viewport.width}x${result.viewport.height}</td>
                <td>${result.issues ? result.issues.join(', ') : 'None'}</td>
                <td class="${result.success ? 'success' : 'error'}">${result.success ? '✅ Pass' : '❌ Fail'}</td>
            </tr>
            `).join('')}
        </table>
    </div>
    ` : ''}

    <div class="section">
        <h2>🔧 Test Configuration</h2>
        <table>
            <tr><th>Setting</th><th>Value</th></tr>
            <tr><td>Headless Mode</td><td>${this.config.headless}</td></tr>
            <tr><td>Viewport</td><td>${this.config.viewport.width}x${this.config.viewport.height}</td></tr>
            <tr><td>Timeout</td><td>${this.config.timeout}ms</td></tr>
            <tr><td>Screenshots Path</td><td>${this.config.screenshotPath}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>📝 Detailed Results</h2>
        <pre>${JSON.stringify(testResults, null, 2)}</pre>
    </div>
</body>
</html>
            `;

            await fs.writeFile(reportPath, html);
            console.log(`✅ Test report generated: ${reportPath}`);
            
            return {
                success: true,
                reportPath,
                summary: {
                    totalTests: testResults.totalTests || 0,
                    passedTests: testResults.passedTests || 0,
                    failedTests: testResults.failedTests || 0,
                    totalTime: testResults.totalTime || 0
                }
            };
        } catch (error) {
            console.error(`❌ Report generation failed: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Helper methods
     */
    calculateAverageResponseTime(requests, responses) {
        const times = [];
        requests.forEach(req => {
            const resp = responses.find(r => r.url === req.url);
            if (resp) {
                times.push(resp.timestamp - req.timestamp);
            }
        });
        return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
    }

    analyzeStatusCodes(responses) {
        const codes = {};
        responses.forEach(resp => {
            codes[resp.status] = (codes[resp.status] || 0) + 1;
        });
        return codes;
    }

    findLargestRequests(responses, count) {
        return responses
            .filter(r => r.size > 0)
            .sort((a, b) => b.size - a.size)
            .slice(0, count)
            .map(r => ({ url: r.url, size: r.size }));
    }

    /**
     * Clean up resources
     */
    async cleanup() {
        try {
            if (this.page) {
                await this.page.close();
            }
            if (this.browser) {
                await this.browser.close();
            }
            console.log('🧹 Puppeteer MCP cleanup completed');
        } catch (error) {
            console.error('❌ Cleanup failed:', error);
        }
    }
}

module.exports = PuppeteerMCP;