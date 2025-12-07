# Razorpay Quick Reference Card

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install
pip install razorpay

# 2. Configure
python setup_razorpay.py

# 3. Start
python app.py

# 4. Test
# Visit: http://127.0.0.1:5001/pricing
# Card: 4111 1111 1111 1111
```

## 📋 Test Credentials

| Method | Success | Failure |
|--------|---------|---------|
| **Card** | 4111 1111 1111 1111 | 4000 0000 0000 0002 |
| **UPI** | success@razorpay | failure@razorpay |
| **Net Banking** | razorpay / razorpay | - |

## 💰 Pricing Plans

| Plan | Monthly | Yearly | Save |
|------|---------|--------|------|
| Free | ₹0 | ₹0 | - |
| Basic | ₹199 | ₹1,999 | 17% |
| Premium | ₹499 | ₹4,999 | 17% |

## 🔑 Environment Variables

```env
RAZORPAY_KEY_ID=rzp_test_xxx
RAZORPAY_KEY_SECRET=your_secret
RAZORPAY_WEBHOOK_SECRET=webhook_secret
RAZORPAY_PLAN_BASIC_MONTHLY=plan_xxx
RAZORPAY_PLAN_BASIC_YEARLY=plan_xxx
RAZORPAY_PLAN_PREMIUM_MONTHLY=plan_xxx
RAZORPAY_PLAN_PREMIUM_YEARLY=plan_xxx
```

## 🌐 API Endpoints

```
GET  /pricing                          # Pricing page
GET  /api/payment/pricing              # Get pricing
POST /api/payment/subscription/create  # Create subscription
POST /api/payment/verify               # Verify payment
POST /api/payment/webhook              # Webhook handler
GET  /api/payment/subscription         # Get subscription
POST /api/payment/subscription/cancel  # Cancel subscription
GET  /api/payment/history              # Payment history
```

## 🔍 Quick Checks

```bash
# Check subscription
python -c "from models.database import SessionLocal; from models.subscription import Subscription; db = SessionLocal(); print(db.query(Subscription).count())"

# Check payments
python -c "from models.database import SessionLocal; from models.payment import Payment; db = SessionLocal(); print(db.query(Payment).count())"

# Test Razorpay connection
python -c "from services.payment_service import get_payment_service; ps = get_payment_service(); print('Connected!' if ps.client else 'Not configured')"
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Razorpay not configured" | Run `python setup_razorpay.py` |
| "Plan not found" | Add Plan IDs to `.env` |
| Webhook not working | Use ngrok, update URL |
| Payment not activating | Check webhook logs |

## 📊 Dashboard Links

- **Payments:** https://dashboard.razorpay.com/app/payments
- **Subscriptions:** https://dashboard.razorpay.com/app/subscriptions
- **Plans:** https://dashboard.razorpay.com/app/subscriptions/plans
- **Webhooks:** https://dashboard.razorpay.com/app/webhooks
- **API Keys:** https://dashboard.razorpay.com/app/keys

## 💡 Pro Tips

1. **Test Mode First:** Always test in Test Mode before going live
2. **Webhook Testing:** Use ngrok for local webhook testing
3. **Plan IDs:** Create plans in dashboard before testing
4. **Signature Verification:** Always verify payment signatures
5. **Error Handling:** Log all payment errors for debugging

## 📞 Support

- **Docs:** https://razorpay.com/docs/
- **Email:** support@razorpay.com
- **Phone:** 1800-102-0555

## ✅ Go-Live Checklist

- [ ] Complete KYC
- [ ] Add bank account
- [ ] Test all payment methods
- [ ] Switch to Live keys
- [ ] Update webhook URL
- [ ] Test with ₹1 payment
- [ ] Set up GST invoicing
- [ ] Add T&C and Refund Policy
- [ ] Monitor first 10 payments
- [ ] Set up alerts

## 🎯 Success Metrics

Track these in your dashboard:
- Conversion rate
- MRR (Monthly Recurring Revenue)
- Churn rate
- Payment success rate
- Popular payment methods

---

**Need detailed help?** See `RAZORPAY_SETUP_GUIDE.md` and `RAZORPAY_TESTING_GUIDE.md`
