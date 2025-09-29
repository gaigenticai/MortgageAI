/**
 * REAL Dashboard Testing - Actually check what loads
 */

const { chromium } = require('playwright');

async function realDashboardTest() {
    console.log('🔍 REAL DASHBOARD TESTING - Let\'s see what actually loads...');
    
    let browser;
    let page;
    
    try {
        browser = await chromium.launch({ 
            headless: false,
            devtools: true
        });
        page = await browser.newPage();
        
        console.log('\n📍 Step 1: Testing if frontend server responds...');
        try {
            await page.goto('http://localhost:5173', { 
                waitUntil: 'domcontentloaded',
                timeout: 10000 
            });
            console.log('✅ Frontend server is responding');
        } catch (error) {
            console.log('❌ Frontend server not responding:', error.message);
            return;
        }
        
        console.log('\n📍 Step 2: Checking what actually renders...');
        await page.waitForTimeout(3000);
        
        // Check if React root exists
        const rootElement = await page.$('#root');
        if (rootElement) {
            console.log('✅ React root element found');
            
            const rootContent = await page.$eval('#root', el => el.innerHTML);
            console.log('📝 Root content length:', rootContent.length);
            
            if (rootContent.length < 10) {
                console.log('❌ Root element is empty - React not rendering');
            }
        } else {
            console.log('❌ React root element not found');
        }
        
        console.log('\n📍 Step 3: Taking diagnostic screenshot...');
        await page.screenshot({ 
            path: './testing/screenshots/actual-dashboard.png',
            fullPage: true 
        });
        console.log('📸 Screenshot saved to ./testing/screenshots/actual-dashboard.png');
        
        console.log('\n📍 Step 4: Checking console errors...');
        const consoleMessages = [];
        page.on('console', msg => {
            consoleMessages.push(`${msg.type()}: ${msg.text()}`);
            console.log(`🖥️  Console ${msg.type()}: ${msg.text()}`);
        });
        
        page.on('pageerror', error => {
            console.log(`❌ Page Error: ${error.message}`);
        });
        
        // Refresh to catch any new errors
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForTimeout(2000);
        
        console.log('\n📍 Step 5: Checking page content...');
        const pageText = await page.textContent('body');
        const title = await page.title();
        
        console.log('📄 Page title:', title);
        console.log('📝 Body text preview:', pageText.substring(0, 200));
        
        if (pageText.includes('MortgageAI') || pageText.includes('Dashboard')) {
            console.log('✅ Dashboard content found');
        } else {
            console.log('❌ No dashboard content visible');
        }
        
        console.log('\n📍 Step 6: Manual inspection time...');
        console.log('Browser window is open for manual inspection.');
        console.log('Press Enter to continue or Ctrl+C to exit...');
        
        // Keep browser open for manual inspection
        await new Promise(resolve => {
            process.stdin.once('data', resolve);
        });
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

realDashboardTest().catch(console.error);