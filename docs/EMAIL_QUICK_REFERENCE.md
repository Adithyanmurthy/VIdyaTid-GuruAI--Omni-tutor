# SendGrid Email - Quick Reference

## 🚀 Quick Start (3 Minutes)

```bash
# 1. Install
pip install sendgrid

# 2. Configure
python setup_sendgrid.py

# 3. Test
python test_sendgrid.py

# 4. Done!
```

## 🔑 Get API Key

1. Go to: https://app.sendgrid.com/settings/api_keys
2. Click "Create API Key"
3. Name: VidyaTid
4. Permission: Full Access
5. Copy key (starts with `SG.`)

## ✉️ Email Types

| Email | Trigger | Template |
|-------|---------|----------|
| Welcome | User registers | `send_welcome_email()` |
| Payment Confirmation | Payment success | `send_payment_confirmation()` |
| Subscription Active | Subscription starts | `send_subscription_activated()` |
| Subscription Cancelled | User cancels | `send_subscription_cancelled()` |
| Payment Failed | Payment fails | `send_payment_failed()` |
| Password Reset | Reset request | `send_password_reset()` |

## 📊 Free Tier Limits

- **100 emails/day** (FREE forever)
- 2,000 contacts
- Email API
- No credit card required

## ⚙️ Environment Variables

```env
SENDGRID_API_KEY=SG.your_key_here
SENDGRID_FROM_EMAIL=noreply@vidyatid.com
SENDGRID_FROM_NAME=VidyaTid
```

## 🧪 Test Commands

```bash
# Test all emails
python test_sendgrid.py

# Test specific email
python -c "from services.email_service import get_email_service; get_email_service().send_welcome_email('test@example.com', 'Test')"
```

## 📈 Monitor Usage

- **Activity:** https://app.sendgrid.com/email_activity
- **Stats:** https://app.sendgrid.com/statistics
- **Daily limit:** Check dashboard

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| "Not configured" | Run `python setup_sendgrid.py` |
| Emails not delivered | Verify sender email |
| "Forbidden" error | Check API key permissions |
| Going to spam | Verify domain authentication |

## 💡 Pro Tips

1. **Verify sender first** - Required before sending
2. **Test in spam folder** - Check deliverability
3. **Monitor daily usage** - Stay under 100/day
4. **Use HTML templates** - Better engagement
5. **Track metrics** - Optimize open rates

## 📞 Support

- **Docs:** https://docs.sendgrid.com/
- **Support:** support@sendgrid.com
- **Status:** https://status.sendgrid.com/

## ✅ Setup Checklist

- [ ] Create SendGrid account
- [ ] Get API key
- [ ] Run `python setup_sendgrid.py`
- [ ] Verify sender email
- [ ] Test with `python test_sendgrid.py`
- [ ] Check email delivery
- [ ] Monitor dashboard

---

**Need detailed help?** See `SENDGRID_SETUP_GUIDE.md`
