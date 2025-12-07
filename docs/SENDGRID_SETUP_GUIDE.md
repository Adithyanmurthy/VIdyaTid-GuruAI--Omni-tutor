# SendGrid Email Integration Guide

## Overview

SendGrid integration for VidyaTid with:
- **100 FREE emails/day** (no credit card required)
- Payment confirmations
- Subscription notifications
- Welcome emails
- Password reset emails
- Payment receipts

## Step 1: Create SendGrid Account

### 1.1 Sign Up
1. Go to https://signup.sendgrid.com/
2. Click "Start for Free"
3. Fill in details:
   - Email: your email
   - Password: create password
   - Company: VidyaTid
   - Website: (optional)
4. Verify email

### 1.2 Complete Setup
1. Login to SendGrid dashboard
2. Complete the setup wizard:
   - Choose "Web App" as integration
   - Skip sender verification for now

## Step 2: Get API Key

### 2.1 Create API Key
1. Go to **Settings** → **API Keys**
2. Click "Create API Key"
3. Name: `VidyaTid Production`
4. Permissions: **Full Access** (or Mail Send only)
5. Click "Create & View"
6. **Copy the API key** (you won't see it again!)

### 2.2 Save to Environment
Add to your `.env` file:
```env
# SendGrid Configuration
SENDGRID_API_KEY=SG.your_api_key_here
SENDGRID_FROM_EMAIL=noreply@vidyatid.com
SENDGRID_FROM_NAME=VidyaTid
```

## Step 3: Verify Sender Email

### 3.1 Single Sender Verification (Free)
1. Go to **Settings** → **Sender Authentication**
2. Click "Verify a Single Sender"
3. Enter:
   - From Name: VidyaTid
   - From Email: noreply@yourdomain.com (or your Gmail)
   - Reply To: support@yourdomain.com
   - Company: VidyaTid
   - Address: Your address
4. Click "Create"
5. Check email and verify

**Note:** For free tier, use Single Sender Verification. Domain authentication requires DNS access.

### 3.2 Using Gmail (Quick Start)
If you don't have a domain yet:
1. Use your Gmail: `yourname@gmail.com`
2. Verify it as sender
3. Users will see emails from your Gmail
4. Later, switch to custom domain

## Step 4: Install Dependencies

```bash
pip install sendgrid
```

Add to `requirements.txt`:
```
sendgrid==6.11.0
```

## Step 5: Email Templates

SendGrid provides:
- Dynamic templates (visual editor)
- Transactional templates (code-based)

We'll use **code-based templates** for flexibility.

## Email Types to Implement

### 1. Welcome Email
- Sent on registration
- Introduces VidyaTid
- Quick start guide

### 2. Payment Confirmation
- Sent after successful payment
- Payment details
- Receipt/Invoice

### 3. Subscription Activated
- Sent when subscription starts
- Plan details
- Features unlocked

### 4. Subscription Renewal
- Sent before renewal
- Renewal date
- Amount to be charged

### 5. Subscription Cancelled
- Sent on cancellation
- Cancellation confirmation
- Feedback request

### 6. Payment Failed
- Sent on payment failure
- Retry instructions
- Support contact

### 7. Password Reset
- Sent on password reset request
- Reset link (valid 1 hour)
- Security notice

## SendGrid Limits

### Free Tier (Forever Free)
- **100 emails/day**
- 2,000 contacts
- Email API
- Email validation (1,000/month)
- Single sender verification

### When to Upgrade
- Need more than 100 emails/day
- Want custom domain authentication
- Need dedicated IP
- Want advanced analytics

### Pricing (if needed)
- **Essentials:** $19.95/mo (50,000 emails)
- **Pro:** $89.95/mo (100,000 emails)

## Best Practices

### 1. Email Sending Strategy
For 100 emails/day limit:
- **Priority 1:** Payment confirmations (critical)
- **Priority 2:** Password resets (critical)
- **Priority 3:** Welcome emails (important)
- **Priority 4:** Marketing emails (optional)

### 2. Optimize Email Usage
- Batch notifications (daily digest)
- Only send critical emails
- Use in-app notifications for non-critical
- Upgrade when you hit limits

### 3. Email Design
- Mobile-responsive
- Clear call-to-action
- Branded (VidyaTid logo)
- Professional footer

### 4. Compliance
- Include unsubscribe link
- Add physical address
- Follow CAN-SPAM Act
- GDPR compliance (if EU users)

## Testing

### Test Email Sending
```bash
python test_sendgrid.py
```

### Test Emails
SendGrid provides test mode:
- Emails sent to "sink" (not delivered)
- Check API response
- Verify template rendering

## Monitoring

### SendGrid Dashboard
- **Activity:** https://app.sendgrid.com/email_activity
- **Stats:** https://app.sendgrid.com/statistics
- **Suppressions:** https://app.sendgrid.com/suppressions

### Track Metrics
- Delivery rate
- Open rate
- Click rate
- Bounce rate
- Spam reports

## Troubleshooting

### Issue: "Forbidden" error
**Solution:** Check API key has correct permissions

### Issue: Emails not delivered
**Solution:** 
1. Check sender is verified
2. Check spam folder
3. Verify email address is valid
4. Check SendGrid activity feed

### Issue: "Daily limit exceeded"
**Solution:** 
1. Optimize email sending
2. Batch notifications
3. Upgrade plan if needed

### Issue: Emails going to spam
**Solution:**
1. Verify domain authentication
2. Warm up IP (send gradually)
3. Avoid spam trigger words
4. Include unsubscribe link

## Alternative: Gmail SMTP (Backup)

If SendGrid has issues, use Gmail SMTP:

```env
# Gmail SMTP (Backup)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

**Note:** Gmail allows 500 emails/day

## Email Templates Preview

### Payment Confirmation
```
Subject: Payment Successful - VidyaTid Premium

Hi [Name],

Thank you for subscribing to VidyaTid Premium!

Payment Details:
- Plan: Premium Monthly
- Amount: ₹499
- Payment ID: pay_xxx
- Date: Nov 24, 2025

Your subscription is now active. Enjoy unlimited access to:
✓ Unlimited AI tutor
✓ All previous papers
✓ Mock tests
✓ Advanced analytics

Start Learning: https://vidyatid.com/chat

Questions? Reply to this email or contact support@vidyatid.com

Best regards,
VidyaTid Team
```

### Welcome Email
```
Subject: Welcome to VidyaTid! 🎓

Hi [Name],

Welcome to VidyaTid - Your AI-powered JEE & NEET preparation partner!

Quick Start:
1. Ask any question from NCERT
2. Upload diagrams for explanations
3. Practice with previous papers
4. Track your progress

Get Started: https://vidyatid.com/chat

Need help? Check our guide: https://vidyatid.com/help

Happy Learning!
VidyaTid Team
```

## Security

### API Key Security
- Never commit API key to Git
- Use environment variables
- Rotate keys periodically
- Use different keys for dev/prod

### Email Security
- Validate email addresses
- Rate limit password resets
- Use secure reset tokens
- Log all email sends

## Cost Estimation

### For 1000 Users
- Daily emails: ~50-100 (within free tier)
- Monthly emails: ~1,500-3,000
- Cost: **FREE** ✓

### For 5000 Users
- Daily emails: ~200-300 (need upgrade)
- Monthly emails: ~6,000-9,000
- Cost: **$19.95/mo** (Essentials plan)

### For 10,000 Users
- Daily emails: ~400-600
- Monthly emails: ~12,000-18,000
- Cost: **$19.95/mo** (Essentials plan)

## Next Steps

1. Create SendGrid account
2. Get API key
3. Verify sender email
4. Run setup script: `python setup_sendgrid.py`
5. Test email sending
6. Integrate with payment flow
7. Monitor delivery rates

## Support

- **SendGrid Docs:** https://docs.sendgrid.com/
- **Support:** https://support.sendgrid.com/
- **Status:** https://status.sendgrid.com/
- **Community:** https://community.sendgrid.com/

## Go-Live Checklist

- [ ] Create SendGrid account
- [ ] Get API key
- [ ] Verify sender email
- [ ] Install sendgrid package
- [ ] Configure environment variables
- [ ] Test email sending
- [ ] Create email templates
- [ ] Integrate with payment flow
- [ ] Test all email types
- [ ] Monitor delivery rates
- [ ] Set up alerts for failures

---

**Ready to send emails!** Follow the setup steps and start engaging with your users.
