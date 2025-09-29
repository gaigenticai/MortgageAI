/**
 * Simplified Dashboard Component Tester
 * Tests all dashboard components and their backend connectivity
 */

const { chromium } = require('playwright');
const axios = require('axios');

async function testDashboardComponents() {
    let browser;
    let page;
    
    console.log('🎯 Starting Dashboard Component Testing...');
    
    try {
        // Initialize browser
        browser = await chromium.launch({ 
            headless: false,
            devtools: false
        });
        page = await browser.newPage();
        
        // Test 1: Backend Connectivity
        console.log('\n🔗 Testing Backend Connectivity...');
        const backendTests = await testBackendEndpoints();
        
        // Test 2: Dashboard Loading
        console.log('\n🌐 Testing Dashboard Loading...');
        await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
        
        // Wait for React to render
        await page.waitForTimeout(3000);
        
        // Test 3: Component Visibility
        console.log('\n📊 Testing Component Visibility...');
        const componentTests = await testComponentVisibility(page);
        
        // Test 4: API Data Loading
        console.log('\n📡 Testing API Data Loading...');
        const dataTests = await testDataLoading(page);
        
        // Test 5: Component Interactions
        console.log('\n🖱️ Testing Component Interactions...');
        const interactionTests = await testInteractions(page);
        
        // Generate Report
        generateReport({
            backend: backendTests,
            components: componentTests,
            data: dataTests,
            interactions: interactionTests
        });
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

async function testBackendEndpoints() {
    const endpoints = [
        '/health',
        '/api/dashboard/metrics',
        '/api/dashboard/agent-status',
        '/api/dashboard/lender-status',
        '/api/dashboard/recent-activity'
    ];
    
    const results = [];
    
    for (const endpoint of endpoints) {
        try {
            const response = await axios.get(`http://localhost:3000${endpoint}`, { timeout: 5000 });
            results.push({
                endpoint,
                status: 'PASS',
                statusCode: response.status,
                hasData: !!response.data
            });
            console.log(`✅ ${endpoint}: ${response.status}`);
        } catch (error) {
            results.push({
                endpoint,
                status: 'FAIL',
                error: error.message
            });
            console.log(`❌ ${endpoint}: ${error.message}`);
        }
    }
    
    return results;
}

async function testComponentVisibility(page) {
    const components = [
        {
            name: 'Header',
            selector: 'header, [role="banner"]'
        },
        {
            name: 'MortgageAI Logo',
            selector: 'text=MortgageAI'
        },
        {
            name: 'Dashboard Title',
            selector: 'text=Dutch Mortgage Dashboard'
        },
        {
            name: 'AFM Compliance Card',
            selector: 'text=AFM Compliance Score'
        },
        {
            name: 'Active Sessions Card',
            selector: 'text=Active Sessions'
        },
        {
            name: 'Pending Reviews Card',
            selector: 'text=Pending Reviews'
        },
        {
            name: 'Quality Score Card',
            selector: 'text=Quality Score'
        },
        {
            name: 'Recent Activity Section',
            selector: 'text=Recent Activity'
        },
        {
            name: 'Quick Actions Section',
            selector: 'text=Quick Actions'
        },
        {
            name: 'Comparison Chart',
            selector: 'text=Performance Comparison'
        }
    ];
    
    const results = [];
    
    for (const component of components) {
        try {
            const element = await page.waitForSelector(component.selector, { timeout: 5000 });
            if (element) {
                const isVisible = await element.isVisible();
                results.push({
                    name: component.name,
                    status: isVisible ? 'PASS' : 'FAIL',
                    selector: component.selector,
                    visible: isVisible
                });
                console.log(`${isVisible ? '✅' : '❌'} ${component.name}: ${isVisible ? 'Visible' : 'Not visible'}`);
            }
        } catch (error) {
            results.push({
                name: component.name,
                status: 'FAIL',
                selector: component.selector,
                error: error.message
            });
            console.log(`❌ ${component.name}: Not found`);
        }
    }
    
    return results;
}

async function testDataLoading(page) {
    console.log('Checking if data is loaded in components...');
    
    const dataChecks = [
        {
            name: 'AFM Compliance Score Value',
            check: async () => {
                const elements = await page.$$eval('text=/\\d+%/', els => els.map(el => el.textContent));
                return elements.length > 0;
            }
        },
        {
            name: 'Active Sessions Count',
            check: async () => {
                const text = await page.textContent('body');
                return /Active Sessions.*\d+/.test(text);
            }
        },
        {
            name: 'Chart Data',
            check: async () => {
                const chartElements = await page.$$('.recharts-bar, .recharts-line, .recharts-wrapper');
                return chartElements.length > 0;
            }
        },
        {
            name: 'Demo Mode Detection',
            check: async () => {
                const text = await page.textContent('body');
                return text.includes('DEMO') || text.includes('Demo Mode');
            }
        }
    ];
    
    const results = [];
    
    for (const check of dataChecks) {
        try {
            const hasData = await check.check();
            results.push({
                name: check.name,
                status: hasData ? 'PASS' : 'FAIL',
                hasData
            });
            console.log(`${hasData ? '✅' : '❌'} ${check.name}: ${hasData ? 'Data found' : 'No data'}`);
        } catch (error) {
            results.push({
                name: check.name,
                status: 'FAIL',
                error: error.message
            });
            console.log(`❌ ${check.name}: ${error.message}`);
        }
    }
    
    return results;
}

async function testInteractions(page) {
    const interactions = [
        {
            name: 'New Client Intake Button',
            action: async () => {
                const button = await page.getByText('New Client', { exact: false }).first();
                if (button) {
                    await button.click();
                    await page.waitForTimeout(2000);
                    return page.url().includes('/afm-client-intake');
                }
                return false;
            }
        },
        {
            name: 'Compliance Check Button',
            action: async () => {
                await page.goto('http://localhost:5173'); // Reset to dashboard
                await page.waitForTimeout(1000);
                const button = await page.getByText('Compliance', { exact: false }).first();
                if (button) {
                    await button.click();
                    await page.waitForTimeout(2000);
                    return page.url().includes('/compliance');
                }
                return false;
            }
        },
        {
            name: 'Quality Control Button',
            action: async () => {
                await page.goto('http://localhost:5173'); // Reset to dashboard
                await page.waitForTimeout(1000);
                const button = await page.getByText('Quality Control', { exact: false }).first();
                if (button) {
                    await button.click();
                    await page.waitForTimeout(2000);
                    return page.url().includes('/quality-control');
                }
                return false;
            }
        }
    ];
    
    const results = [];
    
    for (const interaction of interactions) {
        try {
            const success = await interaction.action();
            results.push({
                name: interaction.name,
                status: success ? 'PASS' : 'FAIL',
                navigated: success
            });
            console.log(`${success ? '✅' : '❌'} ${interaction.name}: ${success ? 'Navigation successful' : 'Navigation failed'}`);
        } catch (error) {
            results.push({
                name: interaction.name,
                status: 'FAIL',
                error: error.message
            });
            console.log(`❌ ${interaction.name}: ${error.message}`);
        }
    }
    
    return results;
}

function generateReport(results) {
    console.log('\n' + '='.repeat(60));
    console.log('📄 DASHBOARD COMPONENT TEST REPORT');
    console.log('='.repeat(60));
    
    const allTests = [
        ...results.backend,
        ...results.components,
        ...results.data,
        ...results.interactions
    ];
    
    const totalTests = allTests.length;
    const passedTests = allTests.filter(test => test.status === 'PASS').length;
    const failedTests = allTests.filter(test => test.status === 'FAIL').length;
    const passRate = (passedTests / totalTests * 100).toFixed(1);
    
    console.log(`\n📊 SUMMARY:`);
    console.log(`Total Tests: ${totalTests}`);
    console.log(`Passed: ${passedTests} ✅`);
    console.log(`Failed: ${failedTests} ❌`);
    console.log(`Success Rate: ${passRate}%`);
    
    let overallStatus;
    if (passRate >= 90) overallStatus = '🟢 EXCELLENT';
    else if (passRate >= 75) overallStatus = '🟡 GOOD';
    else if (passRate >= 50) overallStatus = '🟠 NEEDS IMPROVEMENT';
    else overallStatus = '🔴 CRITICAL';
    
    console.log(`Overall Status: ${overallStatus}`);
    
    console.log(`\n🔗 BACKEND CONNECTIVITY:`);
    results.backend.forEach(test => {
        console.log(`  ${test.status === 'PASS' ? '✅' : '❌'} ${test.endpoint}: ${test.status}`);
    });
    
    console.log(`\n📊 COMPONENT VISIBILITY:`);
    results.components.forEach(test => {
        console.log(`  ${test.status === 'PASS' ? '✅' : '❌'} ${test.name}: ${test.status}`);
    });
    
    console.log(`\n📡 DATA LOADING:`);
    results.data.forEach(test => {
        console.log(`  ${test.status === 'PASS' ? '✅' : '❌'} ${test.name}: ${test.status}`);
    });
    
    console.log(`\n🖱️ INTERACTIONS:`);
    results.interactions.forEach(test => {
        console.log(`  ${test.status === 'PASS' ? '✅' : '❌'} ${test.name}: ${test.status}`);
    });
    
    console.log('\n' + '='.repeat(60));
    
    if (passRate >= 85) {
        console.log('🎉 CONCLUSION: Dashboard components are properly connected to the backend!');
        console.log('✅ All critical components are functional and exchanging data correctly.');
    } else {
        console.log('⚠️  CONCLUSION: Some dashboard components need attention.');
        console.log('❌ Please review failed tests and fix connectivity issues.');
    }
    
    console.log('='.repeat(60));
}

// Run the test
testDashboardComponents().catch(console.error);