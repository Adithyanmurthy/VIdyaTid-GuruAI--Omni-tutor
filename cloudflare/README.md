# Cloudflare Deployment Resources

This directory contains all resources needed to deploy VidyaTid to Cloudflare's edge platform.

## 📁 Directory Structure

```
cloudflare/
├── README.md                           # This file
├── DEPLOYMENT_COMPLETE_GUIDE.md        # Complete deployment guide
├── D1_MIGRATION_GUIDE.md               # Database migration guide
├── R2_STORAGE_SETUP.md                 # R2 storage setup guide
├── KV_CACHING_SETUP.md                 # KV caching setup guide
├── d1_schema.sql                       # D1 database schema
├── export_sqlite_data.py               # Export local database to JSON
├── import_to_d1.py                     # Generate D1 import SQL
├── test_d1_migration.py                # Test database migration
├── test_cloudflare_production.py       # Test Cloudflare AI integration
├── wrangler.toml.template              # Wrangler configuration template
└── export_data/                        # Exported database data (generated)
```

## 🚀 Quick Start

### 1. Database Migration (D1)

```bash
# Export current database
python cloudflare/export_sqlite_data.py

# Create D1 database
wrangler d1 create guruai-db

# Apply schema
wrangler d1 execute guruai-db --file=cloudflare/d1_schema.sql

# Generate and import data
python cloudflare/import_to_d1.py
wrangler d1 execute guruai-db --file=cloudflare/d1_import.sql

# Verify migration
python cloudflare/test_d1_migration.py
```

**Documentation**: `D1_MIGRATION_GUIDE.md`

### 2. Workers AI Configuration

```bash
# Configure credentials in .env
USE_CLOUDFLARE_AI=true
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token

# Test integration
python cloudflare/test_cloudflare_production.py
```

**Features**:
- ✅ Automatic retry with exponential backoff
- ✅ Fallback to local models
- ✅ Request timeout handling
- ✅ Performance monitoring

### 3. R2 Storage Setup

```bash
# Create bucket
wrangler r2 bucket create guruai-storage

# Generate API tokens (via dashboard)
# Then configure in .env

# Upload files
pip install boto3
python cloudflare/upload_to_r2.py
```

**Documentation**: `R2_STORAGE_SETUP.md`

### 4. KV Caching Setup

```bash
# Create namespace
wrangler kv:namespace create "guruai-cache"

# Test integration
python cloudflare/test_kv.py
```

**Documentation**: `KV_CACHING_SETUP.md`

### 5. Complete Deployment

Follow the comprehensive guide:

```bash
# Read the complete deployment guide
cat cloudflare/DEPLOYMENT_COMPLETE_GUIDE.md
```

## 📚 Documentation

### Main Guides

1. **DEPLOYMENT_COMPLETE_GUIDE.md** - Complete end-to-end deployment
   - All phases from database to frontend
   - Configuration examples
   - Testing procedures
   - Cost estimation

2. **D1_MIGRATION_GUIDE.md** - Database migration to D1
   - Export/import procedures
   - Schema management
   - Verification steps
   - Troubleshooting

3. **R2_STORAGE_SETUP.md** - R2 object storage setup
   - Bucket creation
   - File uploads
   - CDN configuration
   - Cost optimization

4. **KV_CACHING_SETUP.md** - KV caching implementation
   - Namespace creation
   - Caching strategies
   - Cache invalidation
   - Performance benefits

## 🛠️ Scripts

### Database Scripts

- **export_sqlite_data.py** - Export local SQLite to JSON
  ```bash
  python cloudflare/export_sqlite_data.py
  ```

- **import_to_d1.py** - Generate D1 import SQL
  ```bash
  python cloudflare/import_to_d1.py
  ```

- **test_d1_migration.py** - Verify migration
  ```bash
  python cloudflare/test_d1_migration.py
  ```

### Testing Scripts

- **test_cloudflare_production.py** - Comprehensive AI tests
  ```bash
  python cloudflare/test_cloudflare_production.py
  ```
  Tests:
  - Configuration
  - Basic chat
  - Embeddings
  - RAG
  - Performance
  - Fallback

## 🏗️ Architecture

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

## 💰 Cost Estimation

### Free Tier (Perfect for Starting)

| Service | Free Tier | Typical Usage | Cost |
|---------|-----------|---------------|------|
| Workers | 100k req/day | 50k req/day | $0 |
| D1 | 5M reads/day | 100k reads/day | $0 |
| R2 | 10 GB storage | 5 GB | $0 |
| KV | 100k reads/day | 50k reads/day | $0 |
| Workers AI | 10k neurons/day | 5k neurons/day | $0 |
| Pages | Unlimited | N/A | $0 |

**Total: $0/month** (within free tier)

### Scaling (When You Grow)

At 1M users/month:
- Workers: ~$50
- D1: ~$10
- R2: ~$5
- KV: ~$25
- Workers AI: ~$110

**Total: ~$200/month** (vs $5000+ traditional hosting)

## ✅ Deployment Checklist

### Phase 1: Database
- [ ] Export local database
- [ ] Create D1 database
- [ ] Apply schema
- [ ] Import data
- [ ] Verify migration

### Phase 2: AI
- [ ] Configure credentials
- [ ] Test chat functionality
- [ ] Test embeddings
- [ ] Test RAG
- [ ] Verify fallback

### Phase 3: Storage
- [ ] Create R2 bucket
- [ ] Generate API tokens
- [ ] Upload files
- [ ] Configure CDN
- [ ] Test access

### Phase 4: Caching
- [ ] Create KV namespace
- [ ] Implement caching
- [ ] Test cache operations
- [ ] Configure TTLs
- [ ] Monitor hit rates

### Phase 5: Workers
- [ ] Create Workers project
- [ ] Configure wrangler.toml
- [ ] Implement routes
- [ ] Set up scheduled tasks
- [ ] Deploy

### Phase 6: Frontend
- [ ] Prepare frontend code
- [ ] Deploy to Pages
- [ ] Configure custom domain
- [ ] Test end-to-end
- [ ] Enable analytics

### Phase 7: Production
- [ ] Set secrets
- [ ] Configure monitoring
- [ ] Set up alerts
- [ ] Load testing
- [ ] Documentation

## 🔧 Troubleshooting

### Common Issues

**Issue**: "Database not found"
```bash
# Solution: Verify database ID
wrangler d1 list
```

**Issue**: "Cloudflare AI request failed"
```bash
# Solution: Check credentials
python cloudflare/test_cloudflare_production.py
```

**Issue**: "R2 access denied"
```bash
# Solution: Verify API token permissions
wrangler r2 object list guruai-storage
```

**Issue**: "KV namespace not found"
```bash
# Solution: Check namespace ID
wrangler kv:namespace list
```

## 📖 Resources

### Official Documentation
- **Cloudflare Docs**: https://developers.cloudflare.com/
- **Workers**: https://developers.cloudflare.com/workers/
- **D1**: https://developers.cloudflare.com/d1/
- **R2**: https://developers.cloudflare.com/r2/
- **KV**: https://developers.cloudflare.com/kv/
- **Workers AI**: https://developers.cloudflare.com/workers-ai/
- **Pages**: https://developers.cloudflare.com/pages/

### Community
- **Discord**: https://discord.gg/cloudflaredev
- **Community Forum**: https://community.cloudflare.com/
- **GitHub**: https://github.com/cloudflare

### Tools
- **Wrangler CLI**: https://developers.cloudflare.com/workers/wrangler/
- **Dashboard**: https://dash.cloudflare.com/

## 🎯 Next Steps

1. **Start with Database Migration**
   ```bash
   python cloudflare/export_sqlite_data.py
   ```

2. **Test AI Integration**
   ```bash
   python cloudflare/test_cloudflare_production.py
   ```

3. **Follow Complete Guide**
   ```bash
   cat cloudflare/DEPLOYMENT_COMPLETE_GUIDE.md
   ```

4. **Deploy to Production**
   - Follow phase-by-phase deployment
   - Test each component
   - Monitor performance
   - Scale as needed

## 🚀 Benefits

✅ **Global Distribution** - Serve users worldwide with low latency
✅ **Auto-Scaling** - Handle traffic spikes automatically
✅ **Cost-Effective** - Start free, scale affordably
✅ **High Availability** - 99.99% uptime SLA
✅ **Built-in Security** - DDoS protection, SSL/TLS
✅ **Easy Deployment** - One command to deploy
✅ **Integrated AI** - Workers AI included
✅ **No Egress Fees** - R2 storage with free bandwidth

## 📞 Support

Need help? Check:
1. Documentation in this directory
2. Cloudflare documentation
3. Community forums
4. Discord server

Happy deploying! 🎉
