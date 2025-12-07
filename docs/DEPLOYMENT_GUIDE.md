# VidyaTid Subscription System - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the VidyaTid subscription pricing system to production. It covers environment configuration, database migrations, Cloudflare setup, and deployment procedures.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Database Migration](#database-migration)
4. [Cloudflare Setup](#cloudflare-setup)
5. [Service Configuration](#service-configuration)
6. [Deployment Steps](#deployment-steps)
7. [Post-Deployment Verification](#post-deployment-verification)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting](#troubleshooting)
10. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Prerequisites

### Required Accounts and Services

- **Cloudflare Account**: Free or paid tier
- **Razorpay Account**: For payment processing
- **SendGrid Account**: For email notifications
- **Domain Name**: Optional but recommended

### Required Tools

```bash
# Node.js and npm (for Wrangler CLI)
node --version  # v16.0.0 or higher
npm --version   # v8.0.0 or higher

# Python (for backend)
python --version  # 3.8 or higher

# Wrangler CLI (Cloudflare)
npm install -g wrangler
wrangler --version

# Git
git --version
```

### Required Credentials

Before starting, gather these credentials:

1. **Cloudflare**:
   - Account ID
   - API Token (with Workers, D1, R2, KV, and Workers AI permissions)

2. **Razorpay**:
   - Key ID (Test and Live)
   - Key Secret (Test and Live)
   - Webhook Secret

3. **SendGrid**:
   - API Key
   - Verified sender email

---

## Environment Configuration

### 1. Create Environment Files

Create `.env` file in the project root:

```env
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-change-this
DEBUG=False

# Database Configuration
DATABASE_URL=sqlite:///guruai.db  # Local development
# For production, this will be replaced by Cloudflare D1

# Cloudflare Configuration
USE_CLOUDFLARE_AI=true
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
CLOUDFLARE_API_TOKEN=your-cloudflare-api-token
CLOUDFLARE_D1_DATABASE_ID=your-d1-database-id
CLOUDFLARE_R2_BUCKET_NAME=vidyatid-storage
CLOUDFLARE_KV_NAMESPACE_ID=your-kv-namespace-id

# Razorpay Configuration (Production)
RAZORPAY_KEY_ID=rzp_live_xxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=your-razorpay-secret
RAZORPAY_WEBHOOK_SECRET=your-webhook-secret

# Razorpay Configuration (Test)
RAZORPAY_TEST_KEY_ID=rzp_test_xxxxxxxxxxxxx
RAZORPAY_TEST_KEY_SECRET=your-test-secret
RAZORPAY_TEST_WEBHOOK_SECRET=your-test-webhook-secret

# SendGrid Configuration
SENDGRID_API_KEY=SG.xxxxxxxxxxxxxxxxxxxxx
SENDGRID_FROM_EMAIL=noreply@vidyatid.com
SENDGRID_FROM_NAME=VidyaTid

# Application URLs
APP_URL=https://vidyatid.com
API_URL=https://api.vidyatid.com

# Subscription Configuration
DEFAULT_TIER=free
DAILY_RESET_TIME=00:00  # UTC
MONTHLY_RESET_DAY=1

# Feature Flags
ENABLE_SUBSCRIPTIONS=true
ENABLE_PREDICTIONS=true
ENABLE_WEBHOOKS=true
```

### 2. Validate Environment Variables

Run the validation script:

```bash
python scripts/validate_env.py
```

This checks that all required environment variables are set and valid.

---

## Database Migration

### 1. Backup Current Database

**CRITICAL**: Always backup before migration!

```bash
# Create backup directory
mkdir -p backups

# Backup SQLite database
cp guruai.db backups/guruai_backup_$(date +%Y%m%d_%H%M%S).db

# Verify backup
ls -lh backups/
```

### 2. Run Migration Script

The migration adds new tables and columns for the subscription system:

```bash
# Run migration
python migrate_subscription_system.py

# Expected output:
# ✓ Created usage table
# ✓ Added scheduled_tier_change to subscriptions
# ✓ Added scheduled_change_date to subscriptions
# ✓ Added payment_type to payments
# ✓ Added metadata to payments
# ✓ Added subscription_id to payments
# ✓ Created indexes
# Migration completed successfully!
```

### 3. Verify Migration

```bash
# Check database schema
python scripts/verify_schema.py

# Expected output:
# ✓ usage table exists
# ✓ subscriptions table has new columns
# ✓ payments table has new columns
# ✓ All indexes created
# Schema verification passed!
```

### 4. Initialize Default Data

```bash
# Create free tier subscriptions for existing users
python scripts/initialize_subscriptions.py

# Expected output:
# Processing 150 users...
# ✓ Created 150 free tier subscriptions
# ✓ Initialized 150 usage records
# Initialization completed!
```

---

## Cloudflare Setup

### Step 1: Login to Cloudflare

```bash
wrangler login
```

This opens a browser window for authentication.

### Step 2: Create D1 Database

```bash
# Create database
wrangler d1 create vidyatid-production

# Save the database ID from output:
# database_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

Update `.env` with the database ID:
```env
CLOUDFLARE_D1_DATABASE_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Step 3: Apply Database Schema

```bash
# Apply schema to D1
wrangler d1 execute vidyatid-production --file=cloudflare/d1_schema.sql

# Expected output:
# ✓ Executed 15 statements
```

### Step 4: Export and Import Data

```bash
# Export current SQLite data
python cloudflare/export_sqlite_data.py

# Generate D1 import SQL
python cloudflare/import_to_d1.py

# Import to D1
wrangler d1 execute vidyatid-production --file=cloudflare/d1_import.sql

# Expected output:
# ✓ Imported 1,250 rows
```

### Step 5: Create R2 Bucket

```bash
# Create R2 bucket for file storage
wrangler r2 bucket create vidyatid-storage

# Expected output:
# ✓ Created bucket vidyatid-storage
```

### Step 6: Create KV Namespace

```bash
# Create KV namespace for caching
wrangler kv:namespace create "SUBSCRIPTION_CACHE"

# Save the namespace ID from output:
# id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Update `.env`:
```env
CLOUDFLARE_KV_NAMESPACE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 7: Configure Workers AI

Workers AI is automatically available in your Cloudflare account. No additional setup required.

Verify access:

```bash
python test_cloudflare_ai.py

# Expected output:
# ✓ LLM inference working
# ✓ Embedding generation working
# ✓ Workers AI configured correctly
```

---

## Service Configuration

### 1. Configure Razorpay

#### Create Subscription Plans

Log into Razorpay Dashboard and create plans:

**Starter Plan (Monthly)**:
- Plan ID: `plan_starter_monthly`
- Amount: ₹99 (9900 paise)
- Interval: 1 month
- Description: "Starter Plan - Monthly"

**Starter Plan (Yearly)**:
- Plan ID: `plan_starter_yearly`
- Amount: ₹990 (99000 paise)
- Interval: 1 year
- Description: "Starter Plan - Yearly"

**Premium Plan (Monthly)**:
- Plan ID: `plan_premium_monthly`
- Amount: ₹299 (29900 paise)
- Interval: 1 month
- Description: "Premium Plan - Monthly"

**Premium Plan (Yearly)**:
- Plan ID: `plan_premium_yearly`
- Amount: ₹2990 (299000 paise)
- Interval: 1 year
- Description: "Premium Plan - Yearly"

**Ultimate Plan (Monthly)**:
- Plan ID: `plan_ultimate_monthly`
- Amount: ₹499 (49900 paise)
- Interval: 1 month
- Description: "Ultimate Plan - Monthly"

**Ultimate Plan (Yearly)**:
- Plan ID: `plan_ultimate_yearly`
- Amount: ₹4990 (499000 paise)
- Interval: 1 year
- Description: "Ultimate Plan - Yearly"

#### Configure Webhook

1. Go to Razorpay Dashboard → Settings → Webhooks
2. Add webhook URL: `https://your-domain.com/api/payment/webhook`
3. Select events:
   - `payment.captured`
   - `payment.failed`
   - `subscription.activated`
   - `subscription.charged`
   - `subscription.cancelled`
   - `subscription.completed`
4. Save webhook secret to `.env`

### 2. Configure SendGrid

#### Create Email Templates

Create these templates in SendGrid:

1. **Subscription Welcome** (ID: `d-welcome-xxxxx`)
2. **Subscription Renewal Reminder** (ID: `d-renewal-xxxxx`)
3. **Subscription Expired** (ID: `d-expired-xxxxx`)
4. **Subscription Cancelled** (ID: `d-cancelled-xxxxx`)
5. **Payment Success** (ID: `d-payment-success-xxxxx`)
6. **Payment Failed** (ID: `d-payment-failed-xxxxx`)
7. **Usage Limit Warning** (ID: `d-usage-warning-xxxxx`)

Update `services/email_service.py` with template IDs.

#### Verify Sender

1. Go to SendGrid → Settings → Sender Authentication
2. Verify your domain or single sender email
3. Update `.env` with verified email

### 3. Configure Scheduled Tasks

#### Daily Reset Task

Create cron job or use Cloudflare Cron Triggers:

```toml
# wrangler.toml
[triggers]
crons = ["0 0 * * *"]  # Daily at midnight UTC
```

Or use system cron:

```bash
# Edit crontab
crontab -e

# Add daily reset at midnight UTC
0 0 * * * cd /path/to/vidyatid && python -c "from services.scheduled_tasks import run_daily_reset; run_daily_reset()"
```

#### Hourly Expiration Check

```bash
# Add hourly expiration check
0 * * * * cd /path/to/vidyatid && python -c "from services.scheduled_tasks import check_expirations; check_expirations()"
```

#### Monthly Prediction Reset

```bash
# Add monthly reset on 1st of each month
0 0 1 * * cd /path/to/vidyatid && python -c "from services.scheduled_tasks import reset_monthly_predictions; reset_monthly_predictions()"
```

---

## Deployment Steps

### Phase 1: Staging Deployment

#### 1. Deploy to Staging Environment

```bash
# Set staging environment
export FLASK_ENV=staging
export APP_URL=https://staging.vidyatid.com

# Run database migrations
python migrate_subscription_system.py

# Start application
python start_app.py
```

#### 2. Run Integration Tests

```bash
# Run full test suite
pytest tests/ -v

# Run subscription-specific tests
pytest tests/test_subscription_service.py -v
pytest tests/test_payment_service.py -v
pytest tests/test_usage_tracker.py -v

# Expected output:
# ✓ 45 tests passed
```

#### 3. Test Payment Flow

```bash
# Use Razorpay test mode
export RAZORPAY_KEY_ID=$RAZORPAY_TEST_KEY_ID
export RAZORPAY_KEY_SECRET=$RAZORPAY_TEST_KEY_SECRET

# Run payment integration test
python test_integration_subscription.py

# Expected output:
# ✓ Order creation successful
# ✓ Payment verification successful
# ✓ Subscription activated
# ✓ Email sent
```

#### 4. Test Webhook Processing

```bash
# Send test webhook
python scripts/test_webhook.py

# Expected output:
# ✓ Webhook signature verified
# ✓ Subscription updated
# ✓ Email notification sent
```

### Phase 2: Production Deployment

#### 1. Switch to Production Credentials

Update `.env` with production credentials:

```env
FLASK_ENV=production
RAZORPAY_KEY_ID=rzp_live_xxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=your-live-secret
APP_URL=https://vidyatid.com
```

#### 2. Deploy Backend to Cloudflare Workers

```bash
# Configure wrangler.toml
cat > wrangler.toml << EOF
name = "vidyatid-api"
main = "app.py"
compatibility_date = "2024-01-15"

[env.production]
workers_dev = false
route = "api.vidyatid.com/*"

[[d1_databases]]
binding = "DB"
database_name = "vidyatid-production"
database_id = "$CLOUDFLARE_D1_DATABASE_ID"

[[r2_buckets]]
binding = "STORAGE"
bucket_name = "vidyatid-storage"

[[kv_namespaces]]
binding = "CACHE"
id = "$CLOUDFLARE_KV_NAMESPACE_ID"

[ai]
binding = "AI"
EOF

# Deploy to Workers
wrangler deploy --env production

# Expected output:
# ✓ Built successfully
# ✓ Deployed to https://api.vidyatid.com
```

#### 3. Deploy Frontend to Cloudflare Pages

```bash
# Build frontend
npm run build

# Deploy to Pages
wrangler pages deploy dist --project-name=vidyatid

# Expected output:
# ✓ Deployed to https://vidyatid.pages.dev
```

#### 4. Configure Custom Domain

```bash
# Add custom domain to Pages
wrangler pages domain add vidyatid.com --project-name=vidyatid

# Expected output:
# ✓ Domain added: vidyatid.com
# ✓ SSL certificate provisioned
```

#### 5. Enable Production Webhooks

Update Razorpay webhook URL to production:
```
https://api.vidyatid.com/api/payment/webhook
```

---

## Post-Deployment Verification

### 1. Health Check

```bash
# Check API health
curl https://api.vidyatid.com/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}
```

### 2. Test Subscription Flow

1. **Create Account**: Register new user
2. **View Pricing**: Navigate to `/pricing`
3. **Select Plan**: Choose Starter plan
4. **Complete Payment**: Use real payment method
5. **Verify Activation**: Check subscription status
6. **Test Features**: Access premium features
7. **Check Usage**: View usage dashboard

### 3. Test Webhook Delivery

```bash
# Check webhook logs
wrangler tail --env production

# Submit test payment and verify webhook received
```

### 4. Verify Email Delivery

```bash
# Check SendGrid activity
# Go to SendGrid Dashboard → Activity

# Verify emails sent:
# - Welcome email
# - Payment confirmation
```

### 5. Monitor Error Rates

```bash
# Check Cloudflare Analytics
# Go to Cloudflare Dashboard → Analytics

# Verify:
# - 0% error rate
# - < 500ms response time
# - No failed requests
```

---

## Rollback Procedures

### Emergency Rollback

If critical issues are detected after deployment:

#### 1. Rollback Workers Deployment

```bash
# List deployments
wrangler deployments list --env production

# Rollback to previous version
wrangler rollback --env production --deployment-id <previous-deployment-id>

# Expected output:
# ✓ Rolled back to deployment <id>
```

#### 2. Rollback Database Changes

```bash
# Restore from backup
cp backups/guruai_backup_YYYYMMDD_HHMMSS.db guruai.db

# For D1, restore from snapshot
wrangler d1 restore vidyatid-production --snapshot-id <snapshot-id>
```

#### 3. Disable New Features

```bash
# Update environment variables
wrangler secret put ENABLE_SUBSCRIPTIONS --env production
# Enter: false

wrangler secret put ENABLE_PREDICTIONS --env production
# Enter: false
```

#### 4. Notify Users

Send notification email about temporary service disruption:

```bash
python scripts/send_maintenance_notification.py
```

### Gradual Rollback

For non-critical issues, use gradual rollback:

#### 1. Route Traffic to Old Version

```bash
# Update route to split traffic
# 90% to old version, 10% to new version
wrangler route update --env production --weight 10
```

#### 2. Monitor Metrics

```bash
# Watch error rates
wrangler tail --env production | grep ERROR
```

#### 3. Adjust Traffic Split

```bash
# If stable, increase to 50/50
wrangler route update --env production --weight 50

# If issues persist, rollback completely
wrangler route update --env production --weight 0
```

---

## Troubleshooting

### Common Issues

#### Issue: Database Connection Errors

**Symptoms**:
```
Error: Unable to connect to D1 database
```

**Solution**:
```bash
# Verify D1 database ID
wrangler d1 list

# Check binding in wrangler.toml
cat wrangler.toml | grep d1_databases

# Test connection
wrangler d1 execute vidyatid-production --command "SELECT 1"
```

#### Issue: Webhook Signature Verification Failed

**Symptoms**:
```
Error: Invalid webhook signature
```

**Solution**:
```bash
# Verify webhook secret
echo $RAZORPAY_WEBHOOK_SECRET

# Check Razorpay dashboard for correct secret
# Update environment variable
wrangler secret put RAZORPAY_WEBHOOK_SECRET --env production
```

#### Issue: Payment Order Creation Failed

**Symptoms**:
```
Error: Unable to create Razorpay order
```

**Solution**:
```bash
# Verify Razorpay credentials
python scripts/test_razorpay.py

# Check API key permissions
# Ensure key has "Create Order" permission

# Verify account is activated
# Check Razorpay dashboard for account status
```

#### Issue: Email Delivery Failed

**Symptoms**:
```
Error: SendGrid API error 401
```

**Solution**:
```bash
# Verify SendGrid API key
python scripts/test_sendgrid.py

# Check sender verification
# Ensure sender email is verified in SendGrid

# Check API key permissions
# Ensure key has "Mail Send" permission
```

#### Issue: Daily Reset Not Running

**Symptoms**:
- Usage counters not resetting at midnight
- Users reporting limit reached after reset time

**Solution**:
```bash
# Check cron job status
crontab -l

# Manually trigger reset
python -c "from services.scheduled_tasks import run_daily_reset; run_daily_reset()"

# Check logs
tail -f logs/scheduled_tasks.log

# Verify timezone
date -u  # Should show UTC time
```

#### Issue: High Response Times

**Symptoms**:
- API responses > 2 seconds
- Timeout errors

**Solution**:
```bash
# Check database query performance
wrangler d1 execute vidyatid-production --command "EXPLAIN QUERY PLAN SELECT * FROM subscriptions WHERE user_id = 'xxx'"

# Add missing indexes
wrangler d1 execute vidyatid-production --file=scripts/add_indexes.sql

# Enable KV caching
python scripts/enable_caching.py

# Monitor improvement
wrangler tail --env production | grep "response_time"
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug environment variable
wrangler secret put DEBUG --env production
# Enter: true

# View detailed logs
wrangler tail --env production --format pretty

# Disable debug mode after troubleshooting
wrangler secret put DEBUG --env production
# Enter: false
```

### Support Escalation

If issues persist:

1. **Check Status Pages**:
   - Cloudflare: https://www.cloudflarestatus.com/
   - Razorpay: https://status.razorpay.com/
   - SendGrid: https://status.sendgrid.com/

2. **Contact Support**:
   - Cloudflare: https://dash.cloudflare.com/support
   - Razorpay: support@razorpay.com
   - SendGrid: https://support.sendgrid.com/

3. **Community Forums**:
   - Cloudflare: https://community.cloudflare.com/
   - Razorpay: https://razorpay.com/support/

---

## Monitoring and Maintenance

### Key Metrics to Monitor

#### 1. Subscription Metrics

```sql
-- Active subscriptions by tier
SELECT tier, COUNT(*) as count
FROM subscriptions
WHERE status = 'active'
GROUP BY tier;

-- Subscription churn rate
SELECT
  DATE(cancelled_at) as date,
  COUNT(*) as cancellations
FROM subscriptions
WHERE cancelled_at IS NOT NULL
GROUP BY DATE(cancelled_at)
ORDER BY date DESC
LIMIT 30;

-- Revenue by tier (last 30 days)
SELECT
  s.tier,
  SUM(p.amount) / 100 as revenue_inr
FROM payments p
JOIN subscriptions s ON p.subscription_id = s.subscription_id
WHERE p.created_at >= DATE('now', '-30 days')
  AND p.status = 'captured'
GROUP BY s.tier;
```

#### 2. Usage Metrics

```sql
-- Daily active users
SELECT
  date,
  COUNT(DISTINCT user_id) as dau
FROM usage
WHERE date >= DATE('now', '-7 days')
GROUP BY date
ORDER BY date DESC;

-- Average queries per user by tier
SELECT
  s.tier,
  AVG(u.query_count) as avg_queries
FROM usage u
JOIN subscriptions s ON u.user_id = s.user_id
WHERE u.date >= DATE('now', '-30 days')
GROUP BY s.tier;

-- Users approaching limits
SELECT
  u.user_id,
  u.query_count,
  u.queries_limit,
  (u.query_count * 100.0 / u.queries_limit) as usage_percent
FROM usage u
WHERE u.date = DATE('now')
  AND u.queries_limit > 0
  AND (u.query_count * 100.0 / u.queries_limit) > 80
ORDER BY usage_percent DESC;
```

#### 3. Payment Metrics

```sql
-- Payment success rate (last 7 days)
SELECT
  DATE(created_at) as date,
  COUNT(*) as total_payments,
  SUM(CASE WHEN status = 'captured' THEN 1 ELSE 0 END) as successful,
  (SUM(CASE WHEN status = 'captured' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
FROM payments
WHERE created_at >= DATE('now', '-7 days')
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Failed payments requiring attention
SELECT
  p.payment_id,
  p.user_id,
  p.amount / 100 as amount_inr,
  p.status,
  p.created_at
FROM payments p
WHERE p.status = 'failed'
  AND p.created_at >= DATE('now', '-7 days')
ORDER BY p.created_at DESC;
```

### Automated Alerts

Set up alerts for critical events:

#### 1. Cloudflare Workers Alert

```bash
# Configure alert in Cloudflare Dashboard
# Go to Workers → vidyatid-api → Triggers → Alerts

# Alert conditions:
# - Error rate > 5%
# - Response time > 2 seconds
# - CPU time > 50ms
```

#### 2. Payment Failure Alert

```python
# Add to services/scheduled_tasks.py

def check_payment_failures():
    """Alert if payment failure rate exceeds threshold"""
    from models.database import SessionLocal
    from models.payment import Payment
    from datetime import datetime, timedelta
    
    db = SessionLocal()
    try:
        # Get payments from last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_payments = db.query(Payment).filter(
            Payment.created_at >= one_hour_ago
        ).all()
        
        if len(recent_payments) == 0:
            return
        
        failed = sum(1 for p in recent_payments if p.status == 'failed')
        failure_rate = (failed / len(recent_payments)) * 100
        
        if failure_rate > 10:  # Alert if > 10% failure rate
            send_alert_email(
                subject=f"High Payment Failure Rate: {failure_rate:.1f}%",
                message=f"{failed} of {len(recent_payments)} payments failed in the last hour"
            )
    finally:
        db.close()
```

#### 3. Database Size Alert

```bash
# Add to crontab - check daily
0 2 * * * cd /path/to/vidyatid && python scripts/check_db_size.py
```

```python
# scripts/check_db_size.py

import os
from services.email_service import send_alert_email

# Check D1 database size
db_size_mb = get_d1_size()  # Implement based on Cloudflare API

if db_size_mb > 900:  # Alert at 90% of 1GB limit
    send_alert_email(
        subject=f"Database Size Warning: {db_size_mb}MB",
        message=f"D1 database is at {db_size_mb}MB. Consider archiving old data."
    )
```

### Maintenance Tasks

#### Weekly Tasks

```bash
# 1. Review error logs
wrangler tail --env production | grep ERROR > weekly_errors.log

# 2. Check subscription expirations
python scripts/check_upcoming_expirations.py

# 3. Verify webhook delivery
python scripts/verify_webhooks.py

# 4. Review usage patterns
python scripts/generate_usage_report.py
```

#### Monthly Tasks

```bash
# 1. Database optimization
wrangler d1 execute vidyatid-production --command "VACUUM"

# 2. Archive old data
python scripts/archive_old_data.py

# 3. Review and update pricing
python scripts/review_pricing.py

# 4. Generate financial report
python scripts/generate_financial_report.py

# 5. Update dependencies
pip list --outdated
npm outdated
```

#### Quarterly Tasks

```bash
# 1. Security audit
python scripts/security_audit.py

# 2. Performance review
python scripts/performance_review.py

# 3. Cost optimization
python scripts/analyze_cloudflare_costs.py

# 4. Feature usage analysis
python scripts/analyze_feature_usage.py
```

---

## Performance Optimization

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_status
ON subscriptions(user_id, status);

CREATE INDEX IF NOT EXISTS idx_usage_user_date
ON usage(user_id, date);

CREATE INDEX IF NOT EXISTS idx_payments_user_status
ON payments(user_id, status);

-- Analyze query performance
EXPLAIN QUERY PLAN
SELECT * FROM subscriptions
WHERE user_id = ? AND status = 'active';
```

### Caching Strategy

```python
# Implement in services/subscription_service.py

from functools import lru_cache
from datetime import datetime, timedelta

# Cache subscription data for 5 minutes
@lru_cache(maxsize=1000)
def get_cached_subscription(user_id: str, cache_time: int):
    """Cache subscription data with time-based invalidation"""
    return get_user_subscription(user_id)

def get_subscription_with_cache(user_id: str):
    """Get subscription with caching"""
    # Use current minute as cache key
    cache_time = int(datetime.utcnow().timestamp() / 300)  # 5-minute buckets
    return get_cached_subscription(user_id, cache_time)
```

### CDN Configuration

```bash
# Configure Cloudflare caching rules
# Go to Cloudflare Dashboard → Caching → Cache Rules

# Rule 1: Cache static assets
# URL: *.css, *.js, *.png, *.jpg
# Cache TTL: 1 year

# Rule 2: Cache API responses
# URL: /api/pricing
# Cache TTL: 1 hour

# Rule 3: Bypass cache for user-specific data
# URL: /api/subscription/current, /api/usage/*
# Cache: Bypass
```

---

## Security Checklist

- [ ] All environment variables stored securely
- [ ] Webhook signature verification enabled
- [ ] HTTPS enforced on all endpoints
- [ ] API rate limiting configured
- [ ] SQL injection prevention verified
- [ ] XSS protection enabled
- [ ] CSRF tokens implemented
- [ ] Sensitive data encrypted at rest
- [ ] Access logs enabled
- [ ] Regular security audits scheduled
- [ ] Dependency vulnerabilities checked
- [ ] Backup and recovery tested

---

## Conclusion

This deployment guide covers the complete process of deploying the VidyaTid subscription system to production. Follow each step carefully and verify at each stage.

For additional support:
- Documentation: https://docs.vidyatid.com
- Support Email: support@vidyatid.com
- Status Page: https://status.vidyatid.com

---

*Last Updated: 2024-01-15*
*Version: 1.0*
