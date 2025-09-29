/**
 * Debug Dashboard Loading
 */

const { chromium } = require('playwright');

async function debugDashboard() {
    let browser;
    let page;
    
    try {
        browser = await chromium.launch({ headless: false });
        page = await browser.newPage();
        
        console.log('🌐 Loading dashboard...');
        await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
        
        // Wait for potential loading
        await page.waitForTimeout(5000);
        
        // Take screenshot
        await page.screenshot({ path: './testing/screenshots/dashboard-debug.png', fullPage: true });
        console.log('📸 Screenshot saved');
        
        // Get page title
        const title = await page.title();
        console.log('📄 Page title:', title);
        
        // Get page HTML to see what's actually loaded
        const html = await page.content();
        console.log('📝 Page HTML length:', html.length);
        
        // Look for specific elements
        const body = await page.textContent('body');
        console.log('📝 Body text preview:', body.substring(0, 500));
        
        // Check if React has rendered
        const reactElements = await page.$$('[data-reactroot], .mantine-Container-root, .mantine-Card-root');
        console.log('⚛️  React elements found:', reactElements.length);
        
        // Check for any errors in console
        page.on('console', msg => console.log('🖥️  Console:', msg.text()));
        page.on('pageerror', error => console.log('❌ Page error:', error.message));
        
        // Wait a bit more and try again
        await page.waitForTimeout(3000);
        
        const finalHTML = await page.textContent('body');
        console.log('📝 Final body text preview:', finalHTML.substring(0, 500));
        
    } catch (error) {
        console.error('❌ Debug failed:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

debugDashboard().catch(console.error);