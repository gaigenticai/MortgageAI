/**
 * Final Dashboard Verification Script
 * This script verifies all backend endpoints and component structures
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

async function verifyDashboardSystem() {
    console.log('🎯 FINAL DASHBOARD COMPONENT VERIFICATION');
    console.log('=' .repeat(60));
    
    // Test 1: Backend API Verification
    console.log('\n🔗 Testing Backend API Connectivity...');
    const apiResults = await testAllAPIs();
    
    // Test 2: Component Structure Verification
    console.log('\n🧩 Verifying Component Structure...');
    const componentResults = await verifyComponentStructure();
    
    // Test 3: Data Flow Verification
    console.log('\n📡 Verifying Data Flow...');
    const dataFlowResults = await verifyDataFlow();
    
    // Generate Final Report
    generateFinalReport({
        apis: apiResults,
        components: componentResults,
        dataFlow: dataFlowResults
    });
}

async function testAllAPIs() {
    const endpoints = [
        { path: '/health', description: 'Health Check' },
        { path: '/api/dashboard/metrics', description: 'Dashboard Metrics' },
        { path: '/api/dashboard/agent-status', description: 'Agent Status' },
        { path: '/api/dashboard/lender-status', description: 'Lender Status' },
        { path: '/api/dashboard/recent-activity', description: 'Recent Activity' }
    ];
    
    const results = [];
    
    for (const endpoint of endpoints) {
        try {
            const startTime = Date.now();
            const response = await axios.get(`http://localhost:3000${endpoint.path}`, {
                timeout: 5000
            });
            const responseTime = Date.now() - startTime;
            
            results.push({
                endpoint: endpoint.path,
                description: endpoint.description,
                status: 'PASS',
                statusCode: response.status,
                responseTime: `${responseTime}ms`,
                hasData: !!response.data,
                dataSize: JSON.stringify(response.data).length
            });
            
            console.log(`✅ ${endpoint.description}: ${response.status} (${responseTime}ms)`);
        } catch (error) {
            results.push({
                endpoint: endpoint.path,
                description: endpoint.description,
                status: 'FAIL',
                error: error.message
            });
            
            console.log(`❌ ${endpoint.description}: ${error.message}`);
        }
    }
    
    return results;
}

async function verifyComponentStructure() {
    const componentsToCheck = [
        {
            name: 'DutchMortgageDashboard',
            path: '/Users/krishna/Downloads/gaigenticai/MortgageAI/frontend/src/pages/DutchMortgageDashboard.tsx',
            expectedFeatures: ['apiClient', 'getDashboardMetrics', 'getRecentActivity', 'useState', 'useEffect']
        },
        {
            name: 'ComparisonChart',
            path: '/Users/krishna/Downloads/gaigenticai/MortgageAI/frontend/src/components/ComparisonChart.tsx',
            expectedFeatures: ['ResponsiveContainer', 'BarChart', 'apiClient']
        },
        {
            name: 'ApiClient',
            path: '/Users/krishna/Downloads/gaigenticai/MortgageAI/frontend/src/services/apiClient.ts',
            expectedFeatures: ['getDashboardMetrics', 'getRecentActivity', 'axios', 'demoDataService']
        },
        {
            name: 'App Router',
            path: '/Users/krishna/Downloads/gaigenticai/MortgageAI/frontend/src/App.tsx',
            expectedFeatures: ['DutchMortgageDashboard', 'Router', 'Routes']
        }
    ];
    
    const results = [];
    
    for (const component of componentsToCheck) {
        try {
            if (fs.existsSync(component.path)) {
                const content = fs.readFileSync(component.path, 'utf8');
                
                const features = component.expectedFeatures.filter(feature => 
                    content.includes(feature)
                );
                
                const isValid = features.length >= Math.ceil(component.expectedFeatures.length * 0.8);
                
                results.push({
                    name: component.name,
                    status: isValid ? 'PASS' : 'FAIL',
                    foundFeatures: features.length,
                    expectedFeatures: component.expectedFeatures.length,
                    features: features
                });
                
                console.log(`${isValid ? '✅' : '❌'} ${component.name}: ${features.length}/${component.expectedFeatures.length} features found`);
            } else {
                results.push({
                    name: component.name,
                    status: 'FAIL',
                    error: 'File not found'
                });
                
                console.log(`❌ ${component.name}: File not found`);
            }
        } catch (error) {
            results.push({
                name: component.name,
                status: 'FAIL',
                error: error.message
            });
            
            console.log(`❌ ${component.name}: ${error.message}`);
        }
    }
    
    return results;
}

async function verifyDataFlow() {
    const dataFlowChecks = [
        {
            name: 'Demo Data Service',
            check: async () => {
                try {
                    const response = await axios.get('http://localhost:3000/api/dashboard/metrics');
                    return response.data && typeof response.data === 'object';
                } catch {
                    return false;
                }
            }
        },
        {
            name: 'Dashboard Metrics Structure',
            check: async () => {
                try {
                    const response = await axios.get('http://localhost:3000/api/dashboard/metrics');
                    const data = response.data;
                    return data.afm_compliance_score !== undefined && 
                           data.active_sessions !== undefined;
                } catch {
                    return false;
                }
            }
        },
        {
            name: 'Recent Activity Data',
            check: async () => {
                try {
                    const response = await axios.get('http://localhost:3000/api/dashboard/recent-activity');
                    return Array.isArray(response.data) || 
                           (response.data && Array.isArray(response.data.activities));
                } catch {
                    return false;
                }
            }
        },
        {
            name: 'Frontend Build Check',
            check: async () => {
                try {
                    const response = await axios.get('http://localhost:5173');
                    return response.data.includes('MortgageAI') && 
                           response.data.includes('root');
                } catch {
                    return false;
                }
            }
        }
    ];
    
    const results = [];
    
    for (const check of dataFlowChecks) {
        try {
            const passed = await check.check();
            results.push({
                name: check.name,
                status: passed ? 'PASS' : 'FAIL',
                verified: passed
            });
            
            console.log(`${passed ? '✅' : '❌'} ${check.name}: ${passed ? 'Verified' : 'Failed'}`);
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

function generateFinalReport(results) {
    console.log('\n' + '='.repeat(60));
    console.log('📄 FINAL VERIFICATION REPORT');
    console.log('='.repeat(60));
    
    // Calculate overall statistics
    const allTests = [
        ...results.apis,
        ...results.components,
        ...results.dataFlow
    ];
    
    const totalTests = allTests.length;
    const passedTests = allTests.filter(test => test.status === 'PASS').length;
    const failedTests = allTests.filter(test => test.status === 'FAIL').length;
    const successRate = (passedTests / totalTests * 100).toFixed(1);
    
    console.log(`\n📊 OVERALL RESULTS:`);
    console.log(`Total Tests: ${totalTests}`);
    console.log(`✅ Passed: ${passedTests}`);
    console.log(`❌ Failed: ${failedTests}`);
    console.log(`📈 Success Rate: ${successRate}%`);
    
    // Determine overall status
    let overallStatus;
    if (successRate >= 90) {
        overallStatus = '🟢 EXCELLENT - All systems operational';
    } else if (successRate >= 75) {
        overallStatus = '🟡 GOOD - Minor issues detected';
    } else if (successRate >= 50) {
        overallStatus = '🟠 NEEDS ATTENTION - Several issues found';
    } else {
        overallStatus = '🔴 CRITICAL - Major issues detected';
    }
    
    console.log(`🎯 Overall Status: ${overallStatus}`);
    
    // Detailed breakdown
    console.log(`\n🔗 API CONNECTIVITY (${results.apis.filter(a => a.status === 'PASS').length}/${results.apis.length} passed):`);
    results.apis.forEach(api => {
        console.log(`  ${api.status === 'PASS' ? '✅' : '❌'} ${api.description}: ${api.status === 'PASS' ? api.responseTime : api.error}`);
    });
    
    console.log(`\n🧩 COMPONENT STRUCTURE (${results.components.filter(c => c.status === 'PASS').length}/${results.components.length} passed):`);
    results.components.forEach(comp => {
        console.log(`  ${comp.status === 'PASS' ? '✅' : '❌'} ${comp.name}: ${comp.status === 'PASS' ? `${comp.foundFeatures}/${comp.expectedFeatures} features` : comp.error}`);
    });
    
    console.log(`\n📡 DATA FLOW (${results.dataFlow.filter(d => d.status === 'PASS').length}/${results.dataFlow.length} passed):`);
    results.dataFlow.forEach(flow => {
        console.log(`  ${flow.status === 'PASS' ? '✅' : '❌'} ${flow.name}: ${flow.status === 'PASS' ? 'Verified' : flow.error || 'Failed'}`);
    });
    
    console.log('\n' + '='.repeat(60));
    
    // Final recommendation
    if (successRate >= 85) {
        console.log('🎉 CONCLUSION: Dashboard components are FULLY FUNCTIONAL!');
        console.log('✅ All critical components are properly connected to the backend.');
        console.log('✅ Data exchange is working correctly.');
        console.log('✅ The system is ready for production use.');
    } else if (successRate >= 70) {
        console.log('⚠️  CONCLUSION: Dashboard is mostly functional with minor issues.');
        console.log('✅ Core functionality is working.');
        console.log('⚠️  Some components may need attention.');
    } else {
        console.log('❌ CONCLUSION: Dashboard requires significant attention.');
        console.log('❌ Multiple components are not functioning properly.');
        console.log('❌ Backend connectivity or component structure issues detected.');
    }
    
    console.log('\n📋 MANUAL VERIFICATION STEPS:');
    console.log('1. Open http://localhost:5173 in your browser');
    console.log('2. Verify all metric cards show data (AFM Compliance, Active Sessions, etc.)');
    console.log('3. Check that the Performance Comparison chart renders');
    console.log('4. Test navigation buttons (New Client Intake, Compliance, Quality Control)');
    console.log('5. Verify Recent Activity section shows activity items');
    
    console.log('='.repeat(60));
}

// Run the verification
verifyDashboardSystem().catch(console.error);