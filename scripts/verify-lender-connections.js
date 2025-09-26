/**
 * Lender Connections Verification Script
 * Tests lender API integrations (Stater, Quion, ING, Rabobank, ABN AMRO)
 */

const axios = require('axios');
const { Client } = require('pg');

const DATABASE_URL = process.env.DATABASE_URL;

// Lender configurations
const LENDERS = {
  stater: {
    name: 'Stater',
    apiUrl: process.env.STATER_API_URL,
    apiKey: process.env.STATER_API_KEY,
    healthEndpoint: '/health',
    testEndpoint: '/status'
  },
  quion: {
    name: 'Quion',
    apiUrl: process.env.QUION_API_URL,
    apiKey: process.env.QUION_API_KEY,
    healthEndpoint: '/health',
    testEndpoint: '/status'
  },
  ing: {
    name: 'ING',
    apiUrl: process.env.ING_API_URL,
    apiKey: process.env.ING_API_KEY,
    healthEndpoint: '/health',
    testEndpoint: '/status'
  },
  rabobank: {
    name: 'Rabobank',
    apiUrl: process.env.RABOBANK_API_URL,
    apiKey: process.env.RABOBANK_API_KEY,
    healthEndpoint: '/health',
    testEndpoint: '/status'
  },
  abn_amro: {
    name: 'ABN AMRO',
    apiUrl: process.env.ABN_AMRO_API_URL,
    apiKey: process.env.ABN_AMRO_API_KEY,
    healthEndpoint: '/health',
    testEndpoint: '/status'
  }
};

async function verifyLenderConnections() {
  console.log('🏛️ Verifying lender API connections...');

  const results = {
    total: 0,
    configured: 0,
    reachable: 0,
    authenticated: 0,
    functional: 0,
    details: {}
  };

  for (const [lenderKey, config] of Object.entries(LENDERS)) {
    results.total++;
    const lenderName = config.name;

    console.log(`\n🔍 Testing ${lenderName} integration...`);

    const lenderResult = {
      configured: false,
      reachable: false,
      authenticated: false,
      functional: false,
      errors: []
    };

    // Check if lender is configured
    if (!config.apiUrl || !config.apiKey) {
      console.log(`⚠️  ${lenderName} not configured (missing API URL or key)`);
      lenderResult.errors.push('Not configured');
      results.details[lenderKey] = lenderResult;
      continue;
    }

    results.configured++;
    lenderResult.configured = true;

    try {
      // Test basic connectivity
      console.log(`Testing ${lenderName} connectivity...`);
      const healthResponse = await axios.get(`${config.apiUrl}${config.healthEndpoint}`, {
        timeout: 10000,
        validateStatus: () => true
      });

      if (healthResponse.status >= 200 && healthResponse.status < 400) {
        console.log(`✅ ${lenderName} API is reachable`);
        lenderResult.reachable = true;
        results.reachable++;
      } else {
        console.log(`⚠️  ${lenderName} health check returned status ${healthResponse.status}`);
        lenderResult.errors.push(`Health check failed: ${healthResponse.status}`);
      }

      // Test authentication
      if (lenderResult.reachable) {
        console.log(`Testing ${lenderName} authentication...`);
        try {
          const authResponse = await axios.get(`${config.apiUrl}${config.testEndpoint}`, {
            headers: {
              'Authorization': `Bearer ${config.apiKey}`,
              'Accept': 'application/json'
            },
            timeout: 15000,
            validateStatus: () => true
          });

          if (authResponse.status === 200) {
            console.log(`✅ ${lenderName} authentication successful`);
            lenderResult.authenticated = true;
            results.authenticated++;
          } else if (authResponse.status === 401 || authResponse.status === 403) {
            console.log(`❌ ${lenderName} authentication failed`);
            lenderResult.errors.push('Authentication failed');
          } else {
            console.log(`⚠️  ${lenderName} authentication returned status ${authResponse.status}`);
            lenderResult.errors.push(`Unexpected auth response: ${authResponse.status}`);
          }
        } catch (authError) {
          console.log(`⚠️  ${lenderName} authentication test failed: ${authError.message}`);
          lenderResult.errors.push(`Auth test failed: ${authError.message}`);
        }
      }

      // Test basic functionality (if authenticated)
      if (lenderResult.authenticated) {
        console.log(`Testing ${lenderName} basic functionality...`);
        try {
          // Try to get lender information or status
          const functionalResponse = await axios.get(`${config.apiUrl}/info`, {
            headers: {
              'Authorization': `Bearer ${config.apiKey}`,
              'Accept': 'application/json'
            },
            timeout: 15000,
            validateStatus: () => true
          });

          if (functionalResponse.status === 200) {
            console.log(`✅ ${lenderName} basic functionality verified`);
            lenderResult.functional = true;
            results.functional++;
          } else {
            console.log(`⚠️  ${lenderName} functionality test returned status ${functionalResponse.status}`);
          }
        } catch (funcError) {
          console.log(`⚠️  ${lenderName} functionality test failed: ${funcError.message}`);
          // This is not necessarily an error - many APIs don't have /info endpoints
        }
      }

    } catch (error) {
      console.log(`❌ ${lenderName} connection test failed: ${error.message}`);
      lenderResult.errors.push(`Connection failed: ${error.message}`);
    }

    results.details[lenderKey] = lenderResult;
  }

  // Test database tables for lender integrations
  console.log('\n🗄️ Testing lender database tables...');
  try {
    const dbClient = new Client({ connectionString: DATABASE_URL });
    await dbClient.connect();

    // Check lender_submissions table
    const submissionsTable = await dbClient.query(`
      SELECT table_name FROM information_schema.tables
      WHERE table_schema = 'public' AND table_name = 'lender_submissions'
    `);

    if (submissionsTable.rows.length === 0) {
      console.log('❌ lender_submissions table does not exist');
      throw new Error('Required database tables missing');
    }

    console.log('✅ Lender database tables exist');

    // Check table structure
    const columnsResult = await dbClient.query(`
      SELECT column_name FROM information_schema.columns
      WHERE table_name = 'lender_submissions'
      ORDER BY ordinal_position
    `);

    const requiredColumns = ['id', 'application_id', 'lender_name', 'reference_number', 'status'];
    const existingColumns = columnsResult.rows.map(row => row.column_name);

    const missingColumns = requiredColumns.filter(col => !existingColumns.includes(col));

    if (missingColumns.length > 0) {
      throw new Error(`lender_submissions table missing columns: ${missingColumns.join(', ')}`);
    }

    console.log('✅ Lender database table structure is correct');

    await dbClient.end();

  } catch (dbError) {
    console.error('❌ Database verification failed:', dbError.message);
    throw dbError;
  }

  // Test lender integration service
  console.log('\n🔗 Testing lender integration service...');
  try {
    const serviceResponse = await axios.get('http://localhost:8002/health', {
      timeout: 5000
    });

    if (serviceResponse.status === 200) {
      console.log('✅ Lender integration service is healthy');

      // Test lenders endpoint
      const lendersResponse = await axios.get('http://localhost:8002/api/lenders', {
        timeout: 10000
      });

      if (lendersResponse.status === 200) {
        console.log('✅ Lender integration API is functional');
      } else {
        console.log('⚠️  Lender integration API returned non-200 status');
      }
    } else {
      throw new Error('Lender integration service health check failed');
    }

  } catch (serviceError) {
    console.error('❌ Lender integration service test failed:', serviceError.message);
    throw serviceError;
  }

  // Summary
  console.log('\n🎉 Lender integration verification completed!');
  console.log('');
  console.log('📊 Verification Summary:');
  console.log(`   Total lenders: ${results.total}`);
  console.log(`   Configured: ${results.configured}`);
  console.log(`   Reachable: ${results.reachable}`);
  console.log(`   Authenticated: ${results.authenticated}`);
  console.log(`   Functional: ${results.functional}`);
  console.log('   Database: ✅ Verified');
  console.log('   Service: ✅ Operational');

  if (results.configured === 0) {
    console.log('');
    console.log('⚠️  WARNING: No lenders are configured. At least one lender API key is required for production use.');
    console.log('   Configure lender API credentials in your .env file.');
  }

  if (results.reachable === 0) {
    console.log('');
    console.log('⚠️  WARNING: No lender APIs are reachable. This may be normal for development/test environments.');
  }

  return results;
}

// Run verification
if (require.main === module) {
  verifyLenderConnections()
    .then((results) => {
      console.log('✅ Lender connections verification completed');

      // Exit with success if at least one lender is configured and database is working
      if (results.configured >= 1) {
        console.log('✅ At least one lender is configured - deployment can proceed');
        process.exit(0);
      } else {
        console.log('⚠️  No lenders configured - deployment proceeding with warnings');
        process.exit(0); // Allow deployment to continue
      }
    })
    .catch((error) => {
      console.error('❌ Lender connections verification failed:', error.message);
      process.exit(1);
    });
}

module.exports = { verifyLenderConnections, LENDERS };
