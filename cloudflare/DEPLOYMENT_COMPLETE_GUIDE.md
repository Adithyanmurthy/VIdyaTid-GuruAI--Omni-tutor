# VidyaTid - Complete Cloudflare Deployment Guide

## Overview

This guide covers the complete deployment of VidyaTid to Cloudflare's edge platform, including Workers, Pages, D1, R2, KV, and Workers AI.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Cloudflare Edge Network                 │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Cloudflare │  │  Cloudflare  │  │  Cloudflare  │ │
│  │    Pages     │  │   Workers    │  │  Workers AI  │ │
│  │  (Frontend)  │  │  (Backend)   │  │   (LLM/AI)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │          │
│         └──────────────────┼──────────────────┘          │
│                            │                             │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐ │
│  │  Cloudflare  │  │  Cloudflare  │  │  Cloudflare  │ │
│  │      D1      │  │      R2      │  │     KV       │ │
│  │  (Database)  │  │  (Storage)   │  │   (Cache)    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Cloudflare account (free tier is sufficient to start)
- Node.js and npm installed
- Python 3.8+ installed
- Git installed
- Domain name (optional, but recommended)

## Phase 1: Database Migration (D1)

### 1.1 Export Current Database

```bash
python cloudflare/export_sqlite_data.py
```

This exports all data to `cloudflare/export_data/`

### 1.2 Create D1 Database

```bash
# Install Wrangler
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create D1 database
wrangler d1 create guruai-db
```

Save the database ID from the output.

### 1.3 Apply Schema

```bash
wrangler d1 execute guruai-db --file=cloudflare/d1_schema.sql
```

### 1.4 Import Data

```bash
# Generate import SQL
python cloudflare/import_to_d1.py

# Import to D1
wrangler d1 execute guruai-db --file=cloudflare/d1_import.sql
```

### 1.5 Verify Migration

```bash
python cloudflare/test_d1_migration.py
```

**Status**: ✅ Database migrated to D1

## Phase 2: Workers AI Configuration

### 2.1 Configure Credentials

Add to `.env`:

```env
USE_CLOUDFLARE_AI=true
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token
```

### 2.2 Test AI Integration

```bash
python cloudflare/test_cloudflare_production.py
```

**Status**: ✅ Workers AI configured and tested

## Phase 3: R2 Storage Setup

### 3.1 Create R2 Bucket

```bash
wrangler r2 bucket create guruai-storage
```

### 3.2 Generate API Tokens

1. Go to R2 dashboard
2. Click "Manage R2 API Tokens"
3. Create token with Read & Write permissions
4. Save Access Key ID and Secret Access Key

### 3.3 Upload Files

```bash
# Install boto3
pip install boto3

# Upload files
python cloudflare/upload_to_r2.py
```

### 3.4 Configure Custom Domain (Optional)

1. In R2 dashboard, select bucket
2. Go to Settings → Custom Domains
3. Add domain: `storage.vidyatid.com`
4. Configure DNS as instructed

**Status**: ✅ R2 storage configured

## Phase 4: KV Caching Setup

### 4.1 Create KV Namespace

```bash
wrangler kv:namespace create "guruai-cache"
wrangler kv:namespace create "guruai-cache" --preview
```

Save the namespace IDs.

### 4.2 Test KV Integration

```bash
python cloudflare/test_kv.py
```

**Status**: ✅ KV caching configured

## Phase 5: Workers Deployment

### 5.1 Create Workers Project

```bash
# Create project directory
mkdir guruai-workers
cd guruai-workers

# Initialize project
wrangler init
```

### 5.2 Configure wrangler.toml

Copy `cloudflare/wrangler.toml.template` and update with your IDs:

```toml
name = "guruai-api"
main = "src/index.js"
compatibility_date = "2024-01-01"

account_id = "YOUR_ACCOUNT_ID"

[[d1_databases]]
binding = "DB"
database_name = "guruai-db"
database_id = "YOUR_D1_DATABASE_ID"

[ai]
binding = "AI"

[[kv_namespaces]]
binding = "KV"
id = "YOUR_KV_NAMESPACE_ID"

[[r2_buckets]]
binding = "R2"
bucket_name = "guruai-storage"
```

### 5.3 Create Worker Code

Create `src/index.js`:

```javascript
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }
    
    // Route handling
    if (url.pathname === '/api/chat' && request.method === 'POST') {
      return handleChat(request, env, corsHeaders);
    }
    
    if (url.pathname === '/api/subscription/current' && request.method === 'GET') {
      return handleGetSubscription(request, env, corsHeaders);
    }
    
    return new Response('VidyaTid API', { 
      status: 200,
      headers: corsHeaders
    });
  },
  
  // Scheduled tasks
  async scheduled(event, env, ctx) {
    // Daily reset at midnight UTC
    if (event.cron === '0 0 * * *') {
      await resetDailyUsage(env);
    }
    
    // Hourly subscription expiration check
    if (event.cron === '0 * * * *') {
      await checkExpiredSubscriptions(env);
    }
  }
};

async function handleChat(request, env, corsHeaders) {
  try {
    const { query, userId } = await request.json();
    
    // Check subscription and usage
    const canQuery = await checkQueryLimit(userId, env);
    if (!canQuery) {
      return new Response(
        JSON.stringify({ error: 'Query limit reached' }),
        { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    
    // Generate response with Workers AI
    const response = await env.AI.run('@cf/meta/llama-3.1-8b-instruct', {
      messages: [
        { role: 'system', content: 'You are an AI tutor for JEE and NEET students.' },
        { role: 'user', content: query }
      ]
    });
    
    // Increment usage counter
    await incrementUsage(userId, env);
    
    return new Response(
      JSON.stringify({ answer: response.response }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

async function handleGetSubscription(request, env, corsHeaders) {
  try {
    const userId = getUserIdFromRequest(request);
    
    // Try cache first
    const cacheKey = `subscription:${userId}`;
    let subscription = await env.KV.get(cacheKey, 'json');
    
    if (!subscription) {
      // Query D1
      const result = await env.DB.prepare(
        'SELECT * FROM subscriptions WHERE user_id = ? AND status = ?'
      ).bind(userId, 'active').first();
      
      subscription = result;
      
      // Cache for 5 minutes
      if (subscription) {
        await env.KV.put(cacheKey, JSON.stringify(subscription), {
          expirationTtl: 300
        });
      }
    }
    
    return new Response(
      JSON.stringify(subscription || {}),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

async function checkQueryLimit(userId, env) {
  // Get usage from KV
  const usageKey = `usage:${userId}:${new Date().toISOString().split('T')[0]}`;
  const usage = await env.KV.get(usageKey, 'json');
  
  if (!usage) return true;
  
  return usage.query_count < usage.queries_limit;
}

async function incrementUsage(userId, env) {
  const usageKey = `usage:${userId}:${new Date().toISOString().split('T')[0]}`;
  const usage = await env.KV.get(usageKey, 'json') || { query_count: 0 };
  
  usage.query_count++;
  
  await env.KV.put(usageKey, JSON.stringify(usage), {
    expirationTtl: 86400 // 24 hours
  });
}

async function resetDailyUsage(env) {
  // Reset all users' daily counters
  const users = await env.DB.prepare('SELECT user_id FROM users').all();
  
  for (const user of users.results) {
    const usageKey = `usage:${user.user_id}:${new Date().toISOString().split('T')[0]}`;
    await env.KV.delete(usageKey);
  }
}

async function checkExpiredSubscriptions(env) {
  // Find expired subscriptions
  const now = new Date().toISOString();
  const expired = await env.DB.prepare(
    'SELECT * FROM subscriptions WHERE end_date < ? AND status = ?'
  ).bind(now, 'active').all();
  
  // Update status
  for (const sub of expired.results) {
    await env.DB.prepare(
      'UPDATE subscriptions SET status = ? WHERE subscription_id = ?'
    ).bind('expired', sub.subscription_id).run();
    
    // Invalidate cache
    await env.KV.delete(`subscription:${sub.user_id}`);
  }
}

function getUserIdFromRequest(request) {
  // Extract user ID from Authorization header or session
  const auth = request.headers.get('Authorization');
  // Implement your auth logic here
  return 'user-id-from-token';
}
```

### 5.4 Deploy Worker

```bash
wrangler deploy
```

Your API will be available at: `https://guruai-api.your-subdomain.workers.dev`

### 5.5 Configure Scheduled Tasks

Add to `wrangler.toml`:

```toml
[triggers]
crons = [
  "0 0 * * *",    # Daily reset at midnight UTC
  "0 * * * *",    # Hourly subscription check
  "0 0 1 * *"     # Monthly prediction reset
]
```

Redeploy:

```bash
wrangler deploy
```

**Status**: ✅ Workers deployed

## Phase 6: Pages Deployment (Frontend)

### 6.1 Prepare Frontend

Create `frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VidyaTid - AI Tutor for JEE & NEET</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎓 VidyaTid</h1>
            <p>Your AI Study Partner for JEE & NEET</p>
        </header>
        
        <main>
            <div id="chat-container"></div>
            <div class="input-area">
                <input type="text" id="query-input" placeholder="Ask any question...">
                <button id="send-btn">Send</button>
            </div>
        </main>
    </div>
    
    <script src="app.js"></script>
</body>
</html>
```

Create `frontend/app.js`:

```javascript
const API_URL = 'https://guruai-api.your-subdomain.workers.dev';

async function sendQuery(query) {
    const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, userId: 'demo-user' })
    });
    
    return await response.json();
}

document.getElementById('send-btn').addEventListener('click', async () => {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    
    if (!query) return;
    
    // Add user message
    addMessage(query, 'user');
    input.value = '';
    
    // Show loading
    const loadingId = addMessage('Thinking...', 'bot', true);
    
    try {
        const result = await sendQuery(query);
        removeMessage(loadingId);
        addMessage(result.answer, 'bot');
    } catch (error) {
        removeMessage(loadingId);
        addMessage('Sorry, something went wrong.', 'bot');
    }
});

function addMessage(text, type, isLoading = false) {
    const container = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    if (isLoading) messageDiv.id = 'loading-message';
    messageDiv.textContent = text;
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    return messageDiv.id;
}

function removeMessage(id) {
    const message = document.getElementById(id);
    if (message) message.remove();
}
```

### 6.2 Deploy to Pages

```bash
# Initialize git
cd frontend
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin https://github.com/yourusername/guruai-frontend.git
git push -u origin main

# Deploy with Wrangler
wrangler pages deploy frontend --project-name=guruai
```

Or use Cloudflare Dashboard:
1. Go to **Workers & Pages**
2. Click **Create application** → **Pages**
3. Connect to GitHub
4. Select repository
5. Configure build settings (if needed)
6. Deploy

Your frontend will be available at: `https://guruai.pages.dev`

### 6.3 Configure Custom Domain (Optional)

1. In Pages dashboard, go to **Custom domains**
2. Click **Set up a custom domain**
3. Enter domain: `vidyatid.com`
4. Follow DNS configuration instructions

**Status**: ✅ Frontend deployed

## Phase 7: Environment Configuration

### 7.1 Set Worker Secrets

```bash
# Set secrets (not in wrangler.toml for security)
wrangler secret put RAZORPAY_KEY_ID
wrangler secret put RAZORPAY_KEY_SECRET
wrangler secret put SENDGRID_API_KEY
wrangler secret put JWT_SECRET
```

### 7.2 Update Environment Variables

Update `.env` for local development:

```env
# Cloudflare Configuration
USE_CLOUDFLARE_D1=true
USE_CLOUDFLARE_AI=true
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token
CLOUDFLARE_D1_DATABASE_ID=your-d1-id
CLOUDFLARE_KV_NAMESPACE_ID=your-kv-id
CLOUDFLARE_R2_BUCKET=guruai-storage
CLOUDFLARE_R2_ACCESS_KEY=your-r2-access-key
CLOUDFLARE_R2_SECRET_KEY=your-r2-secret-key
CLOUDFLARE_R2_ENDPOINT=your-r2-endpoint
CLOUDFLARE_R2_PUBLIC_URL=https://storage.vidyatid.com

# Payment Configuration
RAZORPAY_KEY_ID=your-razorpay-key
RAZORPAY_KEY_SECRET=your-razorpay-secret

# Email Configuration
SENDGRID_API_KEY=your-sendgrid-key
SENDGRID_FROM_EMAIL=noreply@vidyatid.com

# Application
JWT_SECRET=your-jwt-secret
ENVIRONMENT=production
```

**Status**: ✅ Environment configured

## Phase 8: Testing and Verification

### 8.1 Test All Components

```bash
# Test D1
wrangler d1 execute guruai-db --command="SELECT COUNT(*) FROM users"

# Test Workers AI
python cloudflare/test_cloudflare_production.py

# Test R2
python cloudflare/test_r2.py

# Test KV
python cloudflare/test_kv.py

# Test Worker endpoints
curl https://guruai-api.your-subdomain.workers.dev/api/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Newton'\''s first law?","userId":"test"}'
```

### 8.2 Load Testing

```bash
# Install k6
brew install k6  # macOS
# or download from https://k6.io/

# Run load test
k6 run cloudflare/load-test.js
```

**Status**: ✅ All components tested

## Phase 9: Monitoring and Analytics

### 9.1 Enable Workers Analytics

1. Go to Workers dashboard
2. Select your worker
3. View **Metrics** tab
4. Monitor:
   - Requests per second
   - Error rate
   - CPU time
   - Duration

### 9.2 Set Up Alerts

1. Go to **Notifications**
2. Create alerts for:
   - Error rate > 5%
   - CPU time > 50ms
   - Request rate spike

### 9.3 Enable Logs

```bash
# Tail logs in real-time
wrangler tail guruai-api
```

**Status**: ✅ Monitoring configured

## Phase 10: Production Checklist

- [ ] D1 database migrated and verified
- [ ] Workers AI configured and tested
- [ ] R2 storage set up with files uploaded
- [ ] KV caching implemented
- [ ] Workers deployed with all routes
- [ ] Frontend deployed to Pages
- [ ] Custom domains configured
- [ ] SSL certificates active
- [ ] Environment variables set
- [ ] Secrets configured
- [ ] Scheduled tasks running
- [ ] Monitoring enabled
- [ ] Alerts configured
- [ ] Load testing completed
- [ ] Backup strategy implemented
- [ ] Documentation updated

## Cost Summary

### Free Tier Limits

| Service | Free Tier | Estimated Usage | Cost |
|---------|-----------|-----------------|------|
| Workers | 100k req/day | 50k req/day | $0 |
| D1 | 5M reads/day | 100k reads/day | $0 |
| R2 | 10 GB storage | 5 GB | $0 |
| KV | 100k reads/day | 50k reads/day | $0 |
| Workers AI | 10k neurons/day | 5k neurons/day | $0 |
| Pages | Unlimited | N/A | $0 |

**Total Monthly Cost: $0** (within free tier)

## Scaling Beyond Free Tier

When you exceed free tier:

| Service | Paid Pricing |
|---------|--------------|
| Workers | $5/10M requests |
| D1 | $0.001/1M reads |
| R2 | $0.015/GB/month |
| KV | $0.50/1M reads |
| Workers AI | $0.011/1k neurons |

**Example at 1M users/month:**
- Workers: ~$50
- D1: ~$10
- R2: ~$5
- KV: ~$25
- Workers AI: ~$110

**Total: ~$200/month** (vs $5000+ on traditional hosting)

## Troubleshooting

See individual guides for detailed troubleshooting:
- `D1_MIGRATION_GUIDE.md`
- `R2_STORAGE_SETUP.md`
- `KV_CACHING_SETUP.md`
- `CLOUDFLARE_COMPLETE_GUIDE.md`

## Support

- **Cloudflare Docs**: https://developers.cloudflare.com/
- **Discord**: https://discord.gg/cloudflaredev
- **Community**: https://community.cloudflare.com/

## Conclusion

You've successfully deployed VidyaTid to Cloudflare's edge platform! Your application now benefits from:

✅ Global distribution
✅ Low latency
✅ Automatic scaling
✅ Cost-effective hosting
✅ Built-in security
✅ 99.99% uptime

Enjoy building with Cloudflare! 🚀
