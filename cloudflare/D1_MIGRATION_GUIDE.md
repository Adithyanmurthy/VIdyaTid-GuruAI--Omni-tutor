# Cloudflare D1 Migration Guide

This guide walks you through migrating your VidyaTid database from local SQLite to Cloudflare D1.

## Overview

Cloudflare D1 is a serverless SQL database built on SQLite, running at the edge. It provides:
- **Global distribution**: Data replicated across Cloudflare's network
- **Low latency**: Queries execute close to users
- **Serverless**: No infrastructure to manage
- **Free tier**: 5GB storage, 5 million reads/day, 100k writes/day

## Prerequisites

1. **Cloudflare Account**: Sign up at https://dash.cloudflare.com
2. **Wrangler CLI**: Install with `npm install -g wrangler`
3. **Wrangler Login**: Run `wrangler login`

## Migration Steps

### Step 1: Export Current Database

Export your current SQLite database to JSON format:

```bash
python cloudflare/export_sqlite_data.py
```

This will:
- Read data from `guruai.db`
- Export each table to JSON files in `cloudflare/export_data/`
- Create metadata file with export information

**Output:**
```
cloudflare/export_data/
├── users.json
├── subscriptions.json
├── payments.json
├── usage.json
├── progress.json
├── sessions.json
├── questions.json
└── export_metadata.json
```

### Step 2: Create D1 Database

Create a new D1 database in Cloudflare:

```bash
wrangler d1 create guruai-db
```

**Output:**
```
✅ Successfully created DB 'guruai-db'

[[d1_databases]]
binding = "DB"
database_name = "guruai-db"
database_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

**Important**: Copy the `database_id` - you'll need it for configuration.

### Step 3: Apply Database Schema

Apply the D1 schema to create all tables:

```bash
wrangler d1 execute guruai-db --file=cloudflare/d1_schema.sql
```

This creates:
- Users table
- Subscriptions table
- Payments table
- Usage table
- Progress table
- Sessions table
- Questions table
- All indexes and constraints

### Step 4: Generate Import SQL

Generate SQL INSERT statements from the exported JSON:

```bash
python cloudflare/import_to_d1.py
```

This creates `cloudflare/d1_import.sql` with all INSERT statements.

### Step 5: Import Data to D1

Import the data into your D1 database:

```bash
wrangler d1 execute guruai-db --file=cloudflare/d1_import.sql
```

**Note**: For large datasets, you may need to split the import file into smaller chunks.

### Step 6: Verify Migration

Verify the data was imported correctly:

```bash
# Check user count
wrangler d1 execute guruai-db --command="SELECT COUNT(*) as count FROM users"

# Check subscription count
wrangler d1 execute guruai-db --command="SELECT COUNT(*) as count FROM subscriptions"

# Check payment count
wrangler d1 execute guruai-db --command="SELECT COUNT(*) as count FROM payments"

# Check usage count
wrangler d1 execute guruai-db --command="SELECT COUNT(*) as count FROM usage"

# Sample user data
wrangler d1 execute guruai-db --command="SELECT user_id, username, created_at FROM users LIMIT 5"
```

### Step 7: Update Application Configuration

Update your `.env` file with D1 configuration:

```env
# Enable Cloudflare D1
USE_CLOUDFLARE_D1=true
CLOUDFLARE_D1_DATABASE_ID=your-database-id-here

# Cloudflare Account (if not already set)
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token
```

Update `wrangler.toml` in your Workers project:

```toml
name = "guruai-api"
main = "src/index.js"
compatibility_date = "2024-01-01"

# Bind D1 database
[[d1_databases]]
binding = "DB"
database_name = "guruai-db"
database_id = "your-database-id-here"

# Bind Workers AI
[ai]
binding = "AI"
```

## Testing the Migration

### Local Testing

Test D1 queries locally using Wrangler:

```bash
# Start local development with D1
wrangler dev --local

# Or test specific queries
wrangler d1 execute guruai-db --local --command="SELECT * FROM users LIMIT 1"
```

### Staging Environment

Create a staging D1 database for testing:

```bash
# Create staging database
wrangler d1 create guruai-db-staging

# Apply schema
wrangler d1 execute guruai-db-staging --file=cloudflare/d1_schema.sql

# Import test data
wrangler d1 execute guruai-db-staging --file=cloudflare/d1_import.sql
```

## Rollback Plan

If you need to rollback to local SQLite:

1. **Keep backup**: Don't delete `guruai.db` until migration is verified
2. **Update .env**: Set `USE_CLOUDFLARE_D1=false`
3. **Restart application**: Application will use local SQLite

## Performance Considerations

### Query Optimization

D1 is optimized for read-heavy workloads:

```sql
-- Good: Use indexes
SELECT * FROM users WHERE username = ?

-- Good: Limit results
SELECT * FROM usage WHERE user_id = ? ORDER BY date DESC LIMIT 30

-- Avoid: Full table scans
SELECT * FROM users WHERE preferences LIKE '%something%'
```

### Caching Strategy

Implement caching for frequently accessed data:

```javascript
// Cache subscription data in KV
const cacheKey = `subscription:${userId}`;
let subscription = await env.KV.get(cacheKey, 'json');

if (!subscription) {
  // Query D1
  const result = await env.DB.prepare(
    'SELECT * FROM subscriptions WHERE user_id = ?'
  ).bind(userId).first();
  
  subscription = result;
  
  // Cache for 5 minutes
  await env.KV.put(cacheKey, JSON.stringify(subscription), {
    expirationTtl: 300
  });
}
```

### Batch Operations

Batch multiple operations for better performance:

```javascript
// Batch inserts
const stmt = env.DB.prepare(
  'INSERT INTO usage (usage_id, user_id, date, query_count, queries_limit) VALUES (?, ?, ?, ?, ?)'
);

const batch = [
  stmt.bind(id1, userId1, date1, 0, 10),
  stmt.bind(id2, userId2, date2, 0, 50),
  stmt.bind(id3, userId3, date3, 0, 200)
];

await env.DB.batch(batch);
```

## Monitoring

### D1 Analytics

Monitor D1 usage in Cloudflare Dashboard:
- Go to **Workers & Pages** → **D1**
- Select your database
- View metrics: queries/day, storage used, read/write operations

### Query Logging

Enable query logging for debugging:

```javascript
// Log slow queries
const start = Date.now();
const result = await env.DB.prepare(query).bind(...params).all();
const duration = Date.now() - start;

if (duration > 100) {
  console.log(`Slow query (${duration}ms):`, query);
}
```

## Troubleshooting

### Common Issues

**Issue**: "Database not found"
```bash
# Solution: Verify database ID in wrangler.toml
wrangler d1 list
```

**Issue**: "Foreign key constraint failed"
```bash
# Solution: Import tables in correct order (users first, then subscriptions, etc.)
# Or disable foreign keys temporarily:
PRAGMA foreign_keys = OFF;
-- import data
PRAGMA foreign_keys = ON;
```

**Issue**: "Too many SQL variables"
```bash
# Solution: Reduce batch size in import script
# SQLite has a limit of 999 variables per query
```

**Issue**: "Query timeout"
```bash
# Solution: Optimize query with indexes or reduce result set
# D1 has a 30-second timeout per query
```

### Getting Help

- **Cloudflare Docs**: https://developers.cloudflare.com/d1/
- **Discord**: https://discord.gg/cloudflaredev
- **Community Forum**: https://community.cloudflare.com/

## Cost Estimation

### Free Tier Limits
- **Storage**: 5 GB
- **Reads**: 5 million/day
- **Writes**: 100,000/day

### Paid Tier (if needed)
- **Storage**: $0.75/GB/month
- **Reads**: $0.001 per million
- **Writes**: $1.00 per million

### Example Costs

For 10,000 active users:
- **Queries**: ~100,000/day = 3M/month = FREE
- **Storage**: ~500 MB = FREE
- **Writes**: ~10,000/day = 300k/month = FREE

**Total: $0/month** (within free tier)

## Next Steps

After successful migration:

1. ✅ **Update application code** to use D1 connection
2. ✅ **Test all features** thoroughly
3. ✅ **Monitor performance** for first week
4. ✅ **Set up automated backups** (export to R2)
5. ✅ **Update documentation** with new architecture

## Backup Strategy

### Automated Backups

Create a scheduled Worker to backup D1 to R2:

```javascript
// Scheduled backup worker
export default {
  async scheduled(event, env, ctx) {
    // Export all tables
    const tables = ['users', 'subscriptions', 'payments', 'usage'];
    
    for (const table of tables) {
      const data = await env.DB.prepare(`SELECT * FROM ${table}`).all();
      
      // Upload to R2
      const key = `backups/${table}-${Date.now()}.json`;
      await env.R2.put(key, JSON.stringify(data.results));
    }
    
    console.log('Backup complete');
  }
};
```

Schedule in `wrangler.toml`:
```toml
[triggers]
crons = ["0 2 * * *"]  # Daily at 2 AM UTC
```

## Conclusion

You've successfully migrated to Cloudflare D1! Your database is now:
- ✅ Globally distributed
- ✅ Highly available
- ✅ Automatically scaled
- ✅ Cost-effective

Enjoy the benefits of edge computing! 🚀
