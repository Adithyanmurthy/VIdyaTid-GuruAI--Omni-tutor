# 🚀 VidyaTid - Complete Cloudflare Production Setup Guide

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Cloudflare Account Setup](#step-1-cloudflare-account-setup)
4. [Step 2: Database Setup (D1)](#step-2-database-setup-d1)
5. [Step 3: Vector Database Setup (Vectorize)](#step-3-vector-database-setup-vectorize)
6. [Step 4: File Storage Setup (R2)](#step-4-file-storage-setup-r2)
7. [Step 5: Backend Setup (Workers)](#step-5-backend-setup-workers)
8. [Step 6: Frontend Setup (Pages)](#step-6-frontend-setup-pages)
9. [Step 7: AI Models Integration](#step-7-ai-models-integration)
10. [Step 8: Deployment](#step-8-deployment)
11. [Step 9: Testing](#step-9-testing)
12. [Step 10: Monitoring & Scaling](#step-10-monitoring--scaling)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOUDFLARE ECOSYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Frontend   │────────▶│   Backend    │                  │
│  │ Pages (HTML) │         │   Workers    │                  │
│  │   CSS, JS    │         │   (Python)   │                  │
│  └──────────────┘         └──────┬───────┘                  │
│                                   │                           │
│                    ┌──────────────┼──────────────┐           │
│                    │              │              │           │
│           ┌────────▼─────┐  ┌────▼─────┐  ┌────▼─────┐     │
│           │   D1 SQL     │  │ Vectorize│  │  R2 Files│     │
│           │  (Users,     │  │  (NCERT  │  │ (Images, │     │
│           │  Sessions)   │  │Embeddings)│  │  PDFs)   │     │
│           └──────────────┘  └──────────┘  └──────────┘     │
│                                                               │
│           ┌──────────────────────────────────────┐           │
│           │      Cloudflare Workers AI           │           │
│           ├──────────────────────────────────────┤           │
│           │ • Llama 3.1 8B (Chat)               │           │
│           │ • BGE Embeddings (Search)           │           │
│           │ • ResNet-50 (Image Recognition)     │           │
│           └──────────────────────────────────────┘           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

### Required Tools:
```bash
# Install Node.js (for Wrangler CLI)
# Download from: https://nodejs.org/

# Install Wrangler (Cloudflare CLI)
npm install -g wrangler

# Verify installation
wrangler --version
```

### Required Accounts:
- Cloudflare account (free tier works)
- GitHub account (for deployment)
- Domain name (optional, Cloudflare provides free subdomain)

---

## 🔧 Step 1: Cloudflare Account Setup

### 1.1 Create Cloudflare Account
```bash
# Login to Cloudflare
wrangler login
```

### 1.2 Get Account ID
```bash
# Get your account ID
wrangler whoami
```

Save your Account ID - you'll need it!

### 1.3 Create API Token
1. Go to: https://dash.cloudflare.com/profile/api-tokens
2. Click "Create Token"
3. Use template: "Edit Cloudflare Workers"
4. Add permissions:
   - Account > Workers AI > Edit
   - Account > D1 > Edit
   - Account > Vectorize > Edit
   - Account > R2 > Edit
5. Save the token securely

---

## 💾 Step 2: Database Setup (D1)

### 2.1 Create D1 Database
```bash
# Create main database
wrangler d1 create vidyatid-db

# Save the database ID from output
```

### 2.2 Create Database Schema
Create `schema.sql`:

```sql
-- Users table
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    preferences TEXT,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP
);

-- Sessions table
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_token TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP,
    is_valid INTEGER DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Progress tracking
CREATE TABLE user_progress (
    progress_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    chapter TEXT NOT NULL,
    questions_attempted INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Chat history
CREATE TABLE chat_history (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Create indexes
CREATE INDEX idx_sessions_token ON sessions(session_token);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_progress_user ON user_progress(user_id);
CREATE INDEX idx_chat_user ON chat_history(user_id);
```

### 2.3 Initialize Database
```bash
# Apply schema
wrangler d1 execute vidyatid-db --file=schema.sql
```

---

## 🔍 Step 3: Vector Database Setup (Vectorize)

### 3.1 Create Vectorize Index
```bash
# Create vector index for NCERT content
wrangler vectorize create ncert-embeddings \
  --dimensions=768 \
  --metric=cosine

# Save the index name
```

### 3.2 Prepare NCERT Content for Embedding

Create `prepare_embeddings.py`:

```python
import json
import os

def prepare_ncert_chunks():
    """Prepare NCERT content in chunks for embedding"""
    
    chunks = []
    
    # Example structure
    ncert_data = {
        "Biology": {
            "Class 11": {
                "Chapter 1": {
                    "title": "The Living World",
                    "content": "Living organisms exhibit certain characteristics..."
                }
            }
        }
    }
    
    for subject, classes in ncert_data.items():
        for class_name, chapters in classes.items():
            for chapter_num, chapter_data in chapters.items():
                chunk = {
                    "id": f"{subject}_{class_name}_{chapter_num}",
                    "text": f"{chapter_data['title']}: {chapter_data['content']}",
                    "metadata": {
                        "subject": subject,
                        "class": class_name,
                        "chapter": chapter_num,
                        "title": chapter_data['title']
                    }
                }
                chunks.append(chunk)
    
    # Save to JSON
    with open('ncert_chunks.json', 'w') as f:
        json.dump(chunks, f, indent=2)
    
    return chunks

if __name__ == "__main__":
    chunks = prepare_ncert_chunks()
    print(f"Prepared {len(chunks)} chunks for embedding")
```

---

## 📁 Step 4: File Storage Setup (R2)

### 4.1 Create R2 Buckets
```bash
# Create bucket for user uploads
wrangler r2 bucket create vidyatid-uploads

# Create bucket for NCERT PDFs
wrangler r2 bucket create vidyatid-ncert-pdfs

# Create bucket for diagrams
wrangler r2 bucket create vidyatid-diagrams
```

### 4.2 Configure CORS (if needed)
```bash
# Create cors.json
cat > cors.json << EOF
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3000
  }
]
EOF

# Apply CORS
wrangler r2 bucket cors put vidyatid-uploads --cors-file=cors.json
```

---

## ⚙️ Step 5: Backend Setup (Workers)

### 5.1 Create Worker Project
```bash
# Create new worker
mkdir vidyatid-worker
cd vidyatid-worker
npm init -y
npm install wrangler --save-dev
```

### 5.2 Create `wrangler.toml`
```toml
name = "vidyatid-api"
main = "src/index.js"
compatibility_date = "2024-01-01"

# Account settings
account_id = "YOUR_ACCOUNT_ID"

# D1 Database binding
[[d1_databases]]
binding = "DB"
database_name = "vidyatid-db"
database_id = "YOUR_DATABASE_ID"

# Vectorize binding
[[vectorize]]
binding = "VECTORIZE"
index_name = "ncert-embeddings"

# R2 bindings
[[r2_buckets]]
binding = "UPLOADS"
bucket_name = "vidyatid-uploads"

[[r2_buckets]]
binding = "NCERT_PDFS"
bucket_name = "vidyatid-ncert-pdfs"

[[r2_buckets]]
binding = "DIAGRAMS"
bucket_name = "vidyatid-diagrams"

# AI binding
[ai]
binding = "AI"

# Environment variables
[vars]
ENVIRONMENT = "production"
```

### 5.3 Create Main Worker (`src/index.js`)

```javascript
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }
    
    // Route handling
    if (url.pathname === '/api/chat') {
      return handleChat(request, env, corsHeaders);
    }
    
    if (url.pathname === '/api/search') {
      return handleSearch(request, env, corsHeaders);
    }
    
    if (url.pathname === '/api/image-analyze') {
      return handleImageAnalysis(request, env, corsHeaders);
    }
    
    if (url.pathname.startsWith('/api/auth/')) {
      return handleAuth(request, env, corsHeaders);
    }
    
    return new Response('Not Found', { status: 404 });
  }
};

// Chat handler with RAG
async function handleChat(request, env, corsHeaders) {
  try {
    const { question, user_id } = await request.json();
    
    // Step 1: Generate embedding for question
    const embeddingResponse = await env.AI.run(
      '@cf/baai/bge-base-en-v1.5',
      { text: question }
    );
    
    // Step 2: Search vector database
    const searchResults = await env.VECTORIZE.query(
      embeddingResponse.data[0],
      { topK: 3 }
    );
    
    // Step 3: Build context from search results
    const context = searchResults.matches
      .map(match => match.metadata.text)
      .join('\n\n');
    
    // Step 4: Generate answer with Llama 3.1
    const chatResponse = await env.AI.run(
      '@cf/meta/llama-3.1-8b-instruct',
      {
        messages: [
          {
            role: 'system',
            content: 'You are VidyaTid, an AI tutor for JEE & NEET preparation. Answer based on NCERT content.'
          },
          {
            role: 'user',
            content: `Context from NCERT:\n${context}\n\nQuestion: ${question}\n\nProvide a clear, accurate answer.`
          }
        ],
        max_tokens: 1024,
        temperature: 0.7
      }
    );
    
    // Step 5: Save to chat history
    await env.DB.prepare(
      'INSERT INTO chat_history (chat_id, user_id, question, answer) VALUES (?, ?, ?, ?)'
    ).bind(
      crypto.randomUUID(),
      user_id,
      question,
      chatResponse.response
    ).run();
    
    return new Response(
      JSON.stringify({
        answer: chatResponse.response,
        sources: searchResults.matches.map(m => m.metadata)
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

// Search handler
async function handleSearch(request, env, corsHeaders) {
  try {
    const { query } = await request.json();
    
    // Generate embedding
    const embedding = await env.AI.run(
      '@cf/baai/bge-base-en-v1.5',
      { text: query }
    );
    
    // Search vector database
    const results = await env.VECTORIZE.query(
      embedding.data[0],
      { topK: 10 }
    );
    
    return new Response(
      JSON.stringify({ results: results.matches }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

// Image analysis handler
async function handleImageAnalysis(request, env, corsHeaders) {
  try {
    const formData = await request.formData();
    const image = formData.get('image');
    
    // Convert image to array buffer
    const imageBuffer = await image.arrayBuffer();
    
    // Analyze with ResNet-50
    const analysis = await env.AI.run(
      '@cf/microsoft/resnet-50',
      { image: Array.from(new Uint8Array(imageBuffer)) }
    );
    
    return new Response(
      JSON.stringify({ analysis }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

// Auth handler
async function handleAuth(request, env, corsHeaders) {
  const url = new URL(request.url);
  const path = url.pathname;
  
  if (path === '/api/auth/register') {
    return handleRegister(request, env, corsHeaders);
  }
  
  if (path === '/api/auth/login') {
    return handleLogin(request, env, corsHeaders);
  }
  
  if (path === '/api/auth/logout') {
    return handleLogout(request, env, corsHeaders);
  }
  
  return new Response('Not Found', { status: 404 });
}

async function handleRegister(request, env, corsHeaders) {
  try {
    const { username, email, password } = await request.json();
    
    // Hash password (use a proper hashing library in production)
    const passwordHash = await hashPassword(password);
    
    const userId = crypto.randomUUID();
    
    // Insert user
    await env.DB.prepare(
      'INSERT INTO users (user_id, username, email, password_hash) VALUES (?, ?, ?, ?)'
    ).bind(userId, username, email, passwordHash).run();
    
    // Create session
    const sessionToken = crypto.randomUUID();
    const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
    
    await env.DB.prepare(
      'INSERT INTO sessions (session_id, user_id, session_token, expires_at) VALUES (?, ?, ?, ?)'
    ).bind(crypto.randomUUID(), userId, sessionToken, expiresAt.toISOString()).run();
    
    return new Response(
      JSON.stringify({
        status: 'success',
        user: { user_id: userId, username, email },
        session: { session_token: sessionToken, expires_at: expiresAt }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

async function handleLogin(request, env, corsHeaders) {
  // Similar to register but verify password
  // Implementation details...
}

async function handleLogout(request, env, corsHeaders) {
  // Invalidate session
  // Implementation details...
}

async function hashPassword(password) {
  const encoder = new TextEncoder();
  const data = encoder.encode(password);
  const hash = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(hash))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}
```

---

## 🎨 Step 6: Frontend Setup (Pages)

### 6.1 Prepare Frontend Files
```bash
# Create frontend directory
mkdir vidyatid-frontend
cd vidyatid-frontend

# Copy your existing files
cp -r ../templates ./
cp -r ../static ./
```

### 6.2 Update API Endpoints

Update `static/js/script.js` to use Worker API:

```javascript
// Update API base URL
const API_BASE = 'https://vidyatid-api.YOUR_SUBDOMAIN.workers.dev';

// Update fetch calls
async function sendMessage(message) {
    const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('session_token')}`
        },
        body: JSON.stringify({
            question: message,
            user_id: getUserId()
        })
    });
    
    return await response.json();
}
```

### 6.3 Create `wrangler.toml` for Pages
```toml
name = "vidyatid-frontend"
pages_build_output_dir = "./"
compatibility_date = "2024-01-01"
```

---

## 🤖 Step 7: AI Models Integration

### 7.1 Upload NCERT Embeddings

Create `upload_embeddings.js`:

```javascript
import { readFileSync } from 'fs';

async function uploadEmbeddings() {
    const chunks = JSON.parse(readFileSync('ncert_chunks.json', 'utf8'));
    
    for (const chunk of chunks) {
        // Generate embedding
        const embedding = await fetch(
            `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/@cf/baai/bge-base-en-v1.5`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_TOKEN}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: chunk.text })
            }
        );
        
        const embeddingData = await embedding.json();
        
        // Upload to Vectorize
        await fetch(
            `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/vectorize/indexes/ncert-embeddings/insert`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_TOKEN}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    id: chunk.id,
                    values: embeddingData.result.data[0],
                    metadata: {
                        text: chunk.text,
                        ...chunk.metadata
                    }
                })
            }
        );
        
        console.log(`Uploaded: ${chunk.id}`);
    }
}

uploadEmbeddings();
```

Run:
```bash
node upload_embeddings.js
```

---

## 🚀 Step 8: Deployment

### 8.1 Deploy Backend Worker
```bash
cd vidyatid-worker
wrangler deploy
```

### 8.2 Deploy Frontend to Pages
```bash
cd vidyatid-frontend
wrangler pages deploy ./ --project-name=vidyatid
```

### 8.3 Configure Custom Domain (Optional)
```bash
# Add custom domain
wrangler pages domains add vidyatid.com --project-name=vidyatid
```

---

## ✅ Step 9: Testing

### 9.1 Test Chat API
```bash
curl -X POST https://vidyatid-api.YOUR_SUBDOMAIN.workers.dev/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain photosynthesis from Class 11 Biology",
    "user_id": "test-user"
  }'
```

### 9.2 Test Search API
```bash
curl -X POST https://vidyatid-api.YOUR_SUBDOMAIN.workers.dev/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Newton laws of motion"
  }'
```

### 9.3 Test Frontend
Visit: `https://vidyatid.pages.dev`

---

## 📊 Step 10: Monitoring & Scaling

### 10.1 Enable Analytics
```bash
# View analytics
wrangler tail vidyatid-api
```

### 10.2 Set Up Alerts
1. Go to Cloudflare Dashboard
2. Navigate to Workers > vidyatid-api
3. Set up alerts for:
   - Error rate > 5%
   - Response time > 3s
   - Request volume spikes

### 10.3 Performance Optimization
```javascript
// Add caching
const cache = caches.default;

async function handleCachedRequest(request, env) {
    const cacheKey = new Request(request.url, request);
    let response = await cache.match(cacheKey);
    
    if (!response) {
        response = await handleChat(request, env);
        ctx.waitUntil(cache.put(cacheKey, response.clone()));
    }
    
    return response;
}
```

---

## 💰 Cost Estimation

### Free Tier Limits:
- Workers: 100,000 requests/day
- D1: 5 GB storage, 5M reads/day
- R2: 10 GB storage, 1M reads/month
- Workers AI: 10,000 requests/day per model
- Pages: Unlimited requests

### Paid Tier (if needed):
- Workers: $5/month + $0.50 per million requests
- D1: $0.75 per million reads
- R2: $0.015 per GB/month
- Workers AI: $0.011 per 1,000 requests

**Estimated cost for 10,000 users/day: $20-50/month**

---

## 🎉 Congratulations!

Your VidyaTid platform is now live on Cloudflare with:
- ✅ Llama 3.1 8B for intelligent chat
- ✅ BGE embeddings for NCERT search
- ✅ ResNet-50 for image recognition
- ✅ D1 database for user data
- ✅ Vectorize for semantic search
- ✅ R2 for file storage
- ✅ Global CDN for fast delivery

**Your production URL:** `https://vidyatid.pages.dev`

---

## 📚 Next Steps

1. Add more NCERT content to vector database
2. Implement user analytics
3. Add payment integration (if needed)
4. Set up automated backups
5. Configure monitoring and alerts
6. Optimize for mobile devices
7. Add PWA support for offline access

---

## 🆘 Troubleshooting

### Common Issues:

**Issue: Worker not deploying**
```bash
# Check wrangler.toml syntax
wrangler deploy --dry-run
```

**Issue: Database connection failed**
```bash
# Verify database binding
wrangler d1 list
```

**Issue: AI model timeout**
```javascript
// Increase timeout
const response = await env.AI.run(model, input, {
    timeout: 30000 // 30 seconds
});
```

---

## 📞 Support

- Cloudflare Docs: https://developers.cloudflare.com
- Workers AI: https://developers.cloudflare.com/workers-ai
- Community: https://community.cloudflare.com

---

**Built with ❤️ for VidyaTid - Knowledge Re-Envisioned**
