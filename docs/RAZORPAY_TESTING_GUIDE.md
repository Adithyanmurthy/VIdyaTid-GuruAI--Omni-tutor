## Razorpay Integration - Complete Testing Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install razorpay python-dotenv
```

### 2. Run Setup Script
```bash
python setup_razorpay.py
```

This will guide you through:
- Adding API keys
- Configuring webhook secret
- Setting up plan IDs

### 3. Start the App
```bash
python app.py
```

### 4. Visit Pricing Page
```
http://127.0.0.1:5001/pricing
```

## Test Mode Credentials

### Test Cards (Always Successful)
```
Card Number: 4111 1111 1111 1111
CVV: Any 3 digits
Expiry: Any future date
Name: Any name
```

### Test Cards (Always Fail)
```
Card Number: 4000 0000 0000 0002
```

### Test UPI
```
Success: success@razorpay
Failure: failure@razorpay
```

### Test Net Banking
- Select any bank
- Use credentials: razorpay / razorpay

## Testing Checklist

### Basic Flow
- [ ] Visit /pricing page
- [ ] See 3 plans (Free, Basic, Premium)
- [ ] Toggle between Monthly/Yearly
- [ ] Prices update correctly
- [ ] Click "Subscribe Now" on Basic Monthly

### Payment Flow
- [ ] Login/Register if not authenticated
- [ ] Razorpay checkout opens
- [ ] Enter test card details
- [ ] Payment succeeds
- [ ] Success message shows
- [ ] Redirected to /chat
- [ ] Subscription shows as "active" in database

### Subscription Management
- [ ] Visit /pricing again
- [ ] Current plan shows "Current Plan" button
- [ ] Can upgrade to Premium
- [ ] Can cancel subscription
- [ ] Cancellation reflects in database

### Webhook Testing
- [ ] Use ngrok to expose local server
- [ ] Configure webhook in Razorpay dashboard
- [ ] Make a test payment
- [ ] Check webhook is received
- [ ] Verify signature validation
- [ ] Check database updates

## Database Verification

### Check Subscription
```bash
python -c "from models.database import SessionLocal; from models.subscription import Subscription; db = SessionLocal(); subs = db.query(Subscription).all(); print([s.to_dict() for s in subs])"
```

### Check Payments
```bash
python -c "from models.database import SessionLocal; from models.payment import Payment; db = SessionLocal(); payments = db.query(Payment).all(); print([p.to_dict() for p in payments])"
```

## Common Issues & Solutions

### Issue: "Razorpay not configured"
**Solution:** Run `python setup_razorpay.py` and add API keys

### Issue: "Plan not found"
**Solution:** Create plans in Razorpay dashboard and add Plan IDs to .env

### Issue: Webhook not receiving events
**Solution:** 
1. Use ngrok: `ngrok http 5001`
2. Update webhook URL in Razorpay dashboard
3. Verify webhook secret matches

### Issue: Payment succeeds but subscription not activated
**Solution:** Check webhook logs and verify signature validation

### Issue: "Invalid signature" error
**Solution:** Verify Key Secret is correct in .env

## Production Checklist

Before going live:

- [ ] Complete KYC in Razorpay dashboard
- [ ] Add bank account for settlements
- [ ] Switch to Live API keys (rzp_live_)
- [ ] Update webhook URL to production domain
- [ ] Test with real small payment (₹1)
- [ ] Set up GST invoicing
- [ ] Add Terms & Conditions page
- [ ] Add Refund Policy page
- [ ] Set up email notifications
- [ ] Configure SSL/HTTPS
- [ ] Test all payment methods (UPI, Cards, Net Banking)
- [ ] Set up monitoring and alerts

## API Endpoints

### Get Pricing
```
GET /api/payment/pricing
```

### Create Subscription
```
POST /api/payment/subscription/create
Body: { "tier": "basic", "duration": "monthly" }
Requires: Authentication
```

### Verify Payment
```
POST /api/payment/verify
Body: { 
  "razorpay_order_id": "...",
  "razorpay_payment_id": "...",
  "razorpay_signature": "..."
}
Requires: Authentication
```

### Get User Subscription
```
GET /api/payment/subscription
Requires: Authentication
```

### Cancel Subscription
```
POST /api/payment/subscription/cancel
Requires: Authentication
```

### Payment History
```
GET /api/payment/history?limit=10
Requires: Authentication
```

### Webhook
```
POST /api/payment/webhook
Headers: X-Razorpay-Signature
```

## Monitoring

### Check Razorpay Dashboard
- Payments: https://dashboard.razorpay.com/app/payments
- Subscriptions: https://dashboard.razorpay.com/app/subscriptions
- Webhooks: https://dashboard.razorpay.com/app/webhooks

### Application Logs
```bash
tail -f logs/guruai.log | grep payment
```

### Database Queries
```sql
-- Active subscriptions
SELECT * FROM subscriptions WHERE status = 'active';

-- Recent payments
SELECT * FROM payments ORDER BY created_at DESC LIMIT 10;

-- Revenue by tier
SELECT tier, COUNT(*), SUM(amount) FROM subscriptions 
WHERE status = 'active' GROUP BY tier;
```

## Support

- Razorpay Docs: https://razorpay.com/docs/
- Support: support@razorpay.com
- Dashboard: https://dashboard.razorpay.com/

## Next Steps

1. Test the complete flow in Test Mode
2. Create subscription plans in Razorpay dashboard
3. Set up webhook for production
4. Complete KYC for Live Mode
5. Launch! 🚀
