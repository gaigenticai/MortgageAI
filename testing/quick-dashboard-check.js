/**
 * Quick Dashboard Loading Check
 */

const { chromium } = require('playwright');

async function quickCheck() {
    console.log('🔍 QUICK DASHBOARD CHECK - Is it actually working now?');
    
    let browser;
    try {
        browser = await chromium.launch({ headless: false });
        const page = await browser.newPage();
        
        console.log('📍 Loading dashboard...');
        await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
        
        // Wait for React to render
        await page.waitForTimeout(3000);
        
        // Check if React content is there
        const rootContent = await page.$eval('#root', el => el.innerHTML);
        console.log('📝 Root content length:', rootContent.length);
        
        if (rootContent.length > 100) {
            console.log('✅ React is rendering content!');
            
            // Check for specific dashboard elements
            const title = await page.title();
            console.log('📄 Page title:', title);
            
            const bodyText = await page.textContent('body');
            const hasContent = bodyText.includes('Dutch Mortgage Dashboard') || 
                             bodyText.includes('MortgageAI') ||
                             bodyText.includes('AFM Compliance');
            
            console.log('📝 Body text preview:', bodyText.substring(0, 300));
            
            if (hasContent) {
                console.log('✅ Dashboard content found!');
                
                // Take a screenshot
                await page.screenshot({ 
                    path: './testing/screenshots/working-dashboard.png',
                    fullPage: true 
                });
                console.log('📸 Screenshot saved');
                
                // Check for specific components
                const checks = [
                    { name: 'AFM Compliance', text: 'AFM Compliance' },
                    { name: 'Active Sessions', text: 'Active Sessions' },
                    { name: 'Quality Score', text: 'Quality' },
                    { name: 'Recent Activity', text: 'Recent Activity' }
                ];
                
                for (const check of checks) {
                    const found = bodyText.includes(check.text);
                    console.log(`${found ? '✅' : '❌'} ${check.name}: ${found ? 'Found' : 'Not found'}`);
                }
                
            } else {
                console.log('❌ No dashboard content visible');
            }
        } else {
            console.log('❌ React still not rendering properly');
        }
        
        console.log('\n🖥️  Browser open for manual inspection. Press Enter to close...');
        await new Promise(resolve => {
            process.stdin.once('data', resolve);
        });
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

quickCheck().catch(console.error);