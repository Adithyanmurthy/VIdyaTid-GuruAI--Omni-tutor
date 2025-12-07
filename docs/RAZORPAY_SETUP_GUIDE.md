# Razorpay Integration Guide for VidyaTid

## Overview

Complete payment integration for VidyaTid with:
- Subscription plans (Free, Basic, Premium)
- One-time payments
- UPI, Cards, Net Banking, Wallets
- Automatic subscription management
- Payment verification and webhooks

## Step 1: Create Razorpay Account

### 1.1 Sign Up
1. Go to https://razorpay.com/
2. Click "Sign Up" (top right)
3. Enter business details:
   - Business Name: VidyaTid
   - Category: Education
   - Website: (your domain or leave blank for now)
4. Verify email and phone

### 1.2 Complete KYC (For Live Mode)
- PAN Card
- Business registration (if company)
- Bank account details
- GST number (if applicable)

**Note:** You can start with Test Mode without KYC

### 1.3 Get API Keys

1. Login to Razorpay Dashboard
2. Go to **Settings** → **API Keys**
3. Click **Generate Test Key** (for testing)
4. Copy both:
   - **Key ID** (starts with `rzp_test_`)
   - **Key Secret** (keep this secret!)

## Step 2: Install Dependencies

```bash
pip install razorpay
```

Add to `requirements.txt`:
```
razorpay==1.4.1
```

## Step 3: Environment Configuration

Add to your `.env` file:

```env
# Razorpay Configuration
RAZORPAY_KEY_ID=rzp_test_your_key_id_here
RAZORPAY_KEY_SECRET=your_key_secret_here
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret_here

# Subscription Plans (will be created in Razorpay dashboard)
RAZORPAY_PLAN_BASIC_MONTHLY=plan_basic_monthly_id
RAZORPAY_PLAN_BASIC_YEARLY=plan_basic_yearly_id
RAZORPAY_PLAN_PREMIUM_MONTHLY=plan_premium_monthly_id
RAZORPAY_PLAN_PREMIUM_YEARLY=plan_premium_yearly_id
```

## Step 4: Subscription Plans

### Recommended Pricing for JEE/NEET Students

| Plan | Monthly | Yearly | Features |
|------|---------|--------|----------|
| **Free** | ₹0 | ₹0 | 10 questions/day, Basic explanations |
| **Basic** | ₹199 | ₹1,999 (₹166/mo) | 100 questions/day, Detailed explanations, Previous papers |
| **Premium** | ₹499 | ₹4,999 (₹416/mo) | Unlimited questions, AI tutor, Mock tests, Progress tracking |

### Create Plans in Razorpay Dashboard

1. Go to **Subscriptions** → **Plans**
2. Click **Create Plan**
3. Create each plan:

**Basic Monthly:**
- Plan Name: VidyaTid Basic Monthly
- Amount: ₹199
- Billing Cycle: Monthly
- Description: 100 questions/day with detailed explanations

**Basic Yearly:**
- Plan Name: VidyaTid Basic Yearly
- Amount: ₹1,999
- Billing Cycle: Yearly
- Description: 100 questions/day with detailed explanations (Save 17%)

**Premium Monthly:**
- Plan Name: VidyaTid Premium Monthly
- Amount: ₹499
- Billing Cycle: Monthly
- Description: Unlimited AI tutor access with all features

**Premium Yearly:**
- Plan Name: VidyaTid Premium Yearly
- Amount: ₹4,999
- Billing Cycle: Yearly
- Description: Unlimited AI tutor access with all features (Save 17%)

4. Copy each Plan ID and add to `.env`

## Step 5: Database Schema

The payment tables are already created. Verify with:

```bash
python -c "from models.database import engine; from models.subscription import Subscription; from models.payment import Payment; print('Tables exist!')"
```

## Step 6: Test the Integration

### Test Mode Credentials

Use these test cards in Test Mode:

**Success:**
- Card: 4111 1111 1111 1111
- CVV: Any 3 digits
- Expiry: Any future date

**Failure:**
- Card: 4000 0000 0000 0002

**UPI (Test):**
- UPI ID: success@razorpay
- UPI ID (fail): failure@razorpay

## Step 7: Webhook Setup

### 7.1 Create Webhook in Razorpay

1. Go to **Settings** → **Webhooks**
2. Click **Create Webhook**
3. Enter:
   - URL: `https://yourdomain.com/api/payment/webhook`
   - Secret: Generate a random string (save to `.env`)
   - Events: Select all subscription and payment events

### 7.2 Important Events

- `subscription.activated` - Subscription started
- `subscription.charged` - Recurring payment successful
- `subscription.cancelled` - User cancelled
- `subscription.paused` - Subscription paused
- `subscription.completed` - Subscription ended
- `payment.captured` - One-time payment successful
- `payment.failed` - Payment failed

## Step 8: Go Live Checklist

Before switching to Live Mode:

- [ ] Complete KYC verification
- [ ] Add bank account for settlements
- [ ] Test all payment flows in Test Mode
- [ ] Set up webhook in production
- [ ] Update `.env` with live keys (rzp_live_)
- [ ] Test with small real payment
- [ ] Set up GST invoicing
- [ ] Add Terms & Conditions
- [ ] Add Refund Policy

## Step 9: Testing Checklist

Test these scenarios:

- [ ] Free plan signup (no payment)
- [ ] Basic monthly subscription
- [ ] Basic yearly subscription
- [ ] Premium monthly subscription
- [ ] Premium yearly subscription
- [ ] Payment success
- [ ] Payment failure
- [ ] Subscription cancellation
- [ ] Subscription renewal
- [ ] Webhook processing
- [ ] User upgrade (Basic → Premium)
- [ ] User downgrade (Premium → Basic)

## Pricing Strategy Tips

### For Students:

1. **Free Trial**: 7-day free trial of Premium
2. **Student Discount**: 20% off with .edu email
3. **Referral Program**: ₹100 credit for each referral
4. **Exam Season Offers**: 30% off during JEE/NEET months
5. **Bundle Deals**: Premium + Mock Tests at ₹699/month

### Conversion Tactics:

- Show "Most Popular" badge on Premium
- Highlight yearly savings (17% off)
- Limited time offers during exam season
- Money-back guarantee (7 days)
- Free trial for Premium features

## Support & Resources

- **Razorpay Docs**: https://razorpay.com/docs/
- **API Reference**: https://razorpay.com/docs/api/
- **Test Cards**: https://razorpay.com/docs/payments/payments/test-card-details/
- **Support**: support@razorpay.com
- **Dashboard**: https://dashboard.razorpay.com/

## Security Best Practices

1. **Never expose Key Secret** in frontend code
2. **Always verify payments** on server-side
3. **Use webhooks** for subscription updates
4. **Validate webhook signatures** to prevent fraud
5. **Store minimal payment data** (use Razorpay's vault)
6. **Use HTTPS** in production
7. **Log all transactions** for audit trail

## Next Steps

1. Run the setup script: `python setup_razorpay.py`
2. Test payment flow: Visit `/pricing` page
3. Make a test payment
4. Check webhook logs
5. Verify subscription in database

## Troubleshooting

### Payment fails immediately
- Check API keys are correct
- Verify test mode vs live mode
- Check browser console for errors

### Webhook not receiving events
- Verify webhook URL is accessible
- Check webhook secret matches
- Look at Razorpay webhook logs

### Subscription not activating
- Check webhook is processing correctly
- Verify database updates
- Check application logs

## Cost Estimation

For 1000 paying users:

**Monthly Revenue:**
- 500 Basic (₹199): ₹99,500
- 500 Premium (₹499): ₹2,49,500
- **Total**: ₹3,49,000

**Razorpay Fees (2%):**
- ₹6,980/month

**Net Revenue:**
- ₹3,42,020/month

**Yearly (with 30% on yearly plans):**
- ~₹40 lakhs/year net revenue
