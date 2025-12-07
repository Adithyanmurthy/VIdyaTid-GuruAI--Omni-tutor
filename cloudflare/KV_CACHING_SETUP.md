# Cloudflare KV Caching Setup Guide

## Overview

Cloudflare Workers KV is a global, low-latency key-value data store. Perfect for caching subscription data, tier configurations, and session information.

## What to Cache in KV

- **Subscription Data**: User subscription status and tier (5 min TTL)
- **Tier Configuration**: Pricing and feature limits (1 hour TTL)
- **Session Data**: User sessions and authentication tokens
- **Rate Limiting**: API rate limit counters
- **Feature Flags**: Dynamic feature toggles

## Step 1: Create KV Namespace

### Using Wrangler CLI

```bash
# Create production namespace
wrangler kv:namespace create "guruai-cache"

# Create preview namespace for development
wrangler kv:namespace create "guruai-cache" --preview

# Output will show:
# [[kv_namespaces]]
# binding = "KV"
# id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Using Cloudflare Dashboard

1. Go to https://dash.cloudflare.com
2. Navigate to **Workers & Pages** → **KV**
3. Click **Create namespace**
4. Enter name: `guruai-cache`
5. Click **Add**

## Step 2: Configure wrangler.toml

Add KV namespace binding to `wrangler.toml`:

```toml
# Production KV
[[kv_namespaces]]
binding = "KV"
id = "your-production-kv-id-here"

# Preview KV (for development)
[[kv_namespaces]]
binding = "KV"
preview_id = "your-preview-kv-id-here"
```

## Step 3: Implement KV Caching Service

Create `services/kv_cache.py`:

```python
"""
Cloudflare KV caching service for subscription data
"""
import json
import logging
from typing import Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class KVCache:
    """
    Cloudflare KV cache wrapper for Python.
    
    Note: In production Workers, use native KV bindings.
    This is for local development and testing.
    """
    
    def __init__(self, namespace_id: str, api_token: str, account_id: str):
        self.namespace_id = namespace_id
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from KV"""
        import requests
        
        url = f"{self.base_url}/values/{key}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            
            # Try to parse as JSON
            try:
                return response.json()
            except:
                return response.text
        except Exception as e:
            logger.error(f"KV get failed for {key}: {e}")
            return None
    
    def put(self, key: str, value: Any, ttl: int = None):
        """Put value in KV with optional TTL"""
        import requests
        
        url = f"{self.base_url}/values/{key}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Prepare data
        if isinstance(value, (dict, list)):
            data = json.dumps(value)
            headers["Content-Type"] = "application/json"
        else:
            data = str(value)
            headers["Content-Type"] = "text/plain"
        
        # Add TTL if specified
        params = {}
        if ttl:
            params["expiration_ttl"] = ttl
        
        try:
            response = requests.put(url, headers=headers, data=data, params=params)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"KV put failed for {key}: {e}")
            raise
    
    def delete(self, key: str):
        """Delete value from KV"""
        import requests
        
        url = f"{self.base_url}/values/{key}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.delete(url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"KV delete failed for {key}: {e}")
            raise
    
    def list_keys(self, prefix: str = None, limit: int = 1000) -> list:
        """List keys in KV"""
        import requests
        
        url = f"{self.base_url}/keys"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        params = {"limit": limit}
        
        if prefix:
            params["prefix"] = prefix
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("result", [])
        except Exception as e:
            logger.error(f"KV list failed: {e}")
            return []


# Cache key generators
def subscription_cache_key(user_id: str) -> str:
    """Generate cache key for subscription data"""
    return f"subscription:{user_id}"

def tier_config_cache_key() -> str:
    """Generate cache key for tier configuration"""
    return "config:tiers"

def usage_cache_key(user_id: str, date: str) -> str:
    """Generate cache key for usage data"""
    return f"usage:{user_id}:{date}"

def rate_limit_cache_key(user_id: str) -> str:
    """Generate cache key for rate limiting"""
    return f"ratelimit:{user_id}"


# Cache TTL constants (in seconds)
SUBSCRIPTION_TTL = 300  # 5 minutes
TIER_CONFIG_TTL = 3600  # 1 hour
USAGE_TTL = 60  # 1 minute
RATE_LIMIT_TTL = 86400  # 24 hours
```

## Step 4: Implement Caching in Services

### Subscription Service with KV Cache

```python
# services/subscription_service.py

from services.kv_cache import (
    KVCache,
    subscription_cache_key,
    SUBSCRIPTION_TTL
)

class SubscriptionService:
    def __init__(self, db_session, kv_cache: KVCache = None):
        self.db = db_session
        self.kv = kv_cache
    
    def get_user_subscription(self, user_id: str):
        """Get subscription with KV caching"""
        
        # Try cache first
        if self.kv:
            cache_key = subscription_cache_key(user_id)
            cached = self.kv.get(cache_key)
            if cached:
                logger.debug(f"Subscription cache hit for {user_id}")
                return cached
        
        # Cache miss - query database
        subscription = self.db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == 'active'
        ).first()
        
        if not subscription:
            return None
        
        subscription_data = subscription.to_dict()
        
        # Store in cache
        if self.kv:
            cache_key = subscription_cache_key(user_id)
            self.kv.put(cache_key, subscription_data, ttl=SUBSCRIPTION_TTL)
            logger.debug(f"Cached subscription for {user_id}")
        
        return subscription_data
    
    def invalidate_subscription_cache(self, user_id: str):
        """Invalidate subscription cache after updates"""
        if self.kv:
            cache_key = subscription_cache_key(user_id)
            self.kv.delete(cache_key)
            logger.debug(f"Invalidated subscription cache for {user_id}")
```

### Tier Configuration Caching

```python
# services/tier_config.py

from services.kv_cache import (
    KVCache,
    tier_config_cache_key,
    TIER_CONFIG_TTL
)

def get_tier_config(kv_cache: KVCache = None):
    """Get tier configuration with caching"""
    
    # Try cache first
    if kv_cache:
        cache_key = tier_config_cache_key()
        cached = kv_cache.get(cache_key)
        if cached:
            logger.debug("Tier config cache hit")
            return cached
    
    # Cache miss - load from config
    from services.tier_config import TIER_CONFIG
    
    # Store in cache
    if kv_cache:
        cache_key = tier_config_cache_key()
        kv_cache.put(cache_key, TIER_CONFIG, ttl=TIER_CONFIG_TTL)
        logger.debug("Cached tier config")
    
    return TIER_CONFIG
```

## Step 5: Implement in Cloudflare Workers

For production Workers, use native KV bindings:

```javascript
// src/index.js

export default {
  async fetch(request, env) {
    // Get subscription from KV
    const userId = getUserIdFromRequest(request);
    const cacheKey = `subscription:${userId}`;
    
    // Try cache first
    let subscription = await env.KV.get(cacheKey, 'json');
    
    if (!subscription) {
      // Cache miss - query D1
      const result = await env.DB.prepare(
        'SELECT * FROM subscriptions WHERE user_id = ? AND status = ?'
      ).bind(userId, 'active').first();
      
      subscription = result;
      
      // Cache for 5 minutes
      await env.KV.put(cacheKey, JSON.stringify(subscription), {
        expirationTtl: 300
      });
    }
    
    return new Response(JSON.stringify(subscription), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
```

## Step 6: Implement Rate Limiting

```javascript
// Rate limiting with KV

async function checkRateLimit(userId, env) {
  const key = `ratelimit:${userId}`;
  const limit = 100; // requests per day
  
  // Get current count
  const count = await env.KV.get(key);
  const currentCount = count ? parseInt(count) : 0;
  
  if (currentCount >= limit) {
    return { allowed: false, remaining: 0 };
  }
  
  // Increment count
  await env.KV.put(key, (currentCount + 1).toString(), {
    expirationTtl: 86400 // 24 hours
  });
  
  return { allowed: true, remaining: limit - currentCount - 1 };
}
```

## Step 7: Cache Invalidation Strategy

### Invalidate on Updates

```javascript
// When subscription is updated
async function updateSubscription(userId, newTier, env) {
  // Update database
  await env.DB.prepare(
    'UPDATE subscriptions SET tier = ? WHERE user_id = ?'
  ).bind(newTier, userId).run();
  
  // Invalidate cache
  await env.KV.delete(`subscription:${userId}`);
}
```

### Invalidate on Schedule

```javascript
// Scheduled cache cleanup
export default {
  async scheduled(event, env, ctx) {
    // Clear expired rate limits
    const keys = await env.KV.list({ prefix: 'ratelimit:' });
    
    for (const key of keys.keys) {
      const value = await env.KV.get(key.name);
      if (!value) {
        await env.KV.delete(key.name);
      }
    }
  }
};
```

## Step 8: Test KV Integration

Create `cloudflare/test_kv.py`:

```python
"""
Test KV caching integration
"""
import os
from dotenv import load_dotenv
from services.kv_cache import KVCache

load_dotenv()

def test_kv():
    print("Testing Cloudflare KV...")
    
    kv = KVCache(
        namespace_id=os.getenv('CLOUDFLARE_KV_NAMESPACE_ID'),
        api_token=os.getenv('CLOUDFLARE_API_TOKEN'),
        account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID')
    )
    
    # Test put
    print("\n1. Testing put...")
    kv.put('test-key', {'message': 'Hello KV!'}, ttl=60)
    print("   ✓ Put successful")
    
    # Test get
    print("\n2. Testing get...")
    value = kv.get('test-key')
    print(f"   ✓ Got value: {value}")
    
    # Test list
    print("\n3. Testing list...")
    keys = kv.list_keys(prefix='test-')
    print(f"   ✓ Found {len(keys)} keys")
    
    # Test delete
    print("\n4. Testing delete...")
    kv.delete('test-key')
    print("   ✓ Deleted")
    
    print("\n✅ All KV tests passed!")

if __name__ == '__main__':
    test_kv()
```

## Performance Benefits

### Without KV Cache
```
User Request → Worker → D1 Query (50-100ms) → Response
Total: ~100ms per request
```

### With KV Cache
```
User Request → Worker → KV Get (1-5ms) → Response
Total: ~10ms per request (10x faster!)
```

## Cost Estimation

### Free Tier
- **Reads**: 100,000/day
- **Writes**: 1,000/day
- **Storage**: 1 GB
- **List operations**: 1,000/day

### Paid Tier (if needed)
- **Reads**: $0.50 per million
- **Writes**: $5.00 per million
- **Storage**: $0.50/GB/month

### Example for VidyaTid

Assuming:
- 10,000 active users
- 10 subscription checks per user per day = 100k reads/day
- 100 subscription updates per day = 100 writes/day
- 1 MB storage

**Monthly Cost: $0** (within free tier)

## Best Practices

### 1. Use Appropriate TTLs

```javascript
// Short TTL for frequently changing data
await env.KV.put(key, value, { expirationTtl: 60 }); // 1 minute

// Medium TTL for subscription data
await env.KV.put(key, value, { expirationTtl: 300 }); // 5 minutes

// Long TTL for static config
await env.KV.put(key, value, { expirationTtl: 3600 }); // 1 hour
```

### 2. Implement Cache-Aside Pattern

```javascript
async function getData(key, fetchFunction, ttl) {
  // Try cache
  let data = await env.KV.get(key, 'json');
  
  if (!data) {
    // Cache miss - fetch from source
    data = await fetchFunction();
    
    // Store in cache
    await env.KV.put(key, JSON.stringify(data), {
      expirationTtl: ttl
    });
  }
  
  return data;
}
```

### 3. Handle Cache Failures Gracefully

```javascript
async function getWithFallback(key, fetchFunction) {
  try {
    // Try cache
    const cached = await env.KV.get(key, 'json');
    if (cached) return cached;
  } catch (error) {
    console.error('KV get failed:', error);
  }
  
  // Fallback to source
  return await fetchFunction();
}
```

### 4. Monitor Cache Hit Rate

```javascript
let cacheHits = 0;
let cacheMisses = 0;

async function getWithMetrics(key, fetchFunction) {
  const cached = await env.KV.get(key, 'json');
  
  if (cached) {
    cacheHits++;
    return cached;
  }
  
  cacheMisses++;
  const data = await fetchFunction();
  await env.KV.put(key, JSON.stringify(data));
  return data;
}

// Log metrics periodically
console.log(`Cache hit rate: ${(cacheHits / (cacheHits + cacheMisses) * 100).toFixed(2)}%`);
```

## Troubleshooting

### Issue: "Namespace not found"

**Solution**: Verify namespace ID in wrangler.toml
```bash
wrangler kv:namespace list
```

### Issue: "Key not found" (404)

**Solution**: This is normal for cache misses. Handle gracefully:
```javascript
const value = await env.KV.get(key);
if (!value) {
  // Cache miss - fetch from source
}
```

### Issue: "Value too large"

**Solution**: KV has a 25 MB limit per value. For larger data:
- Split into multiple keys
- Store in R2 instead
- Compress before storing

## Next Steps

After KV setup:
1. ✅ Implement caching in services
2. ✅ Test cache hit rates
3. ✅ Monitor performance improvements
4. ✅ Set up cache invalidation
5. ✅ Configure TTLs appropriately

## Resources

- **KV Documentation**: https://developers.cloudflare.com/kv/
- **API Reference**: https://developers.cloudflare.com/api/operations/workers-kv-namespace-list-namespaces
- **Pricing**: https://developers.cloudflare.com/kv/platform/pricing/
- **Best Practices**: https://developers.cloudflare.com/kv/learning/kv-best-practices/
