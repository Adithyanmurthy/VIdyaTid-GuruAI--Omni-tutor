# VidyaTid - Final Pricing Structure

## 🎯 Complete Monetization Model

### **Tier 0: FREE (Before Login)**
**Price:** ₹0
**Limit:** 3 questions (one-time trial)
**Purpose:** Hook users, demonstrate value
**Action:** After 3 questions → Prompt to Sign Up or View Plans

---

### **Tier 1: FREE (After Login)**
**Price:** ₹0
**Limit:** 10 questions per day (resets daily)
**Features:**
- ✅ 10 questions per day
- ✅ Basic NCERT content
- ✅ AI explanations
- ✅ Quiz generation (2 questions)
- ❌ No diagrams access
- ❌ No previous year papers
- ❌ No image doubt solving
- ❌ No progress tracking

**Purpose:** Keep users engaged, build habit
**Action:** After 10 questions → Prompt to Upgrade

---

### **Tier 2: STARTER (₹199/month)** 🔥 MOST POPULAR
**Price:** ₹199/month or ₹1,999/year (Save ₹389)
**Limit:** 100 questions per day
**Features:**
- ✅ **100 questions per day**
- ✅ **Full NCERT content**
- ✅ **AI explanations with diagrams**
- ✅ Quiz generation (5 questions)
- ✅ **Previous year papers (2015-2024)**
- ✅ Basic progress tracking
- ✅ Email support
- ❌ No image doubt solving
- ❌ No mock tests

**Target:** Serious JEE/NEET aspirants
**Expected Conversion:** 50-60% of paid users

---

### **Tier 3: PREMIUM (₹499/month)** ⭐ BEST VALUE
**Price:** ₹499/month or ₹4,999/year (Save ₹989)
**Limit:** UNLIMITED questions
**Features:**
- ✅ **UNLIMITED questions per day**
- ✅ Full NCERT detailed content
- ✅ AI explanations with diagrams
- ✅ Advanced quiz (7 questions)
- ✅ **Previous year papers (2010-2024)**
- ✅ **Image-based doubt solving**
- ✅ **Mock tests (JEE/NEET pattern)**
- ✅ Detailed progress analytics
- ✅ Chapter-wise reports
- ✅ Priority support
- ✅ Offline mode

**Target:** Top performers, serious students
**Expected Conversion:** 30-40% of paid users

---

### **Tier 4: ULTIMATE (₹999/month)** 👑 ALL-INCLUSIVE
**Price:** ₹999/month or ₹9,999/year (Save ₹1,989)
**Limit:** UNLIMITED everything
**Features:**
- ✅ **Everything in Premium Plan**
- ✅ **Personalized AI study plans**
- ✅ **Live doubt clearing sessions**
- ✅ Advanced analytics & rank predictions
- ✅ Video explanations for all topics
- ✅ Performance comparison with peers
- ✅ Revision notes & formula sheets
- ✅ Weekly mock tests
- ✅ Early access to new features
- ✅ 24/7 priority support
- ✅ Ad-free experience
- ✅ **1-on-1 mentorship (monthly)**

**Target:** Premium students, coaching institute students
**Expected Conversion:** 10-15% of paid users

---

## 📊 Revenue Projections (Conservative)

### Year 1 Targets:
| Tier | Users | Monthly Revenue | Annual Revenue |
|------|-------|----------------|----------------|
| Free (Before Login) | 10,000 | ₹0 | ₹0 |
| Free (After Login) | 5,000 | ₹0 | ₹0 |
| Starter (₹199) | 2,000 | ₹3,98,000 | ₹47,76,000 |
| Premium (₹499) | 1,000 | ₹4,99,000 | ₹59,88,000 |
| Ultimate (₹999) | 200 | ₹1,99,800 | ₹23,97,600 |
| **TOTAL** | **18,200** | **₹10,96,800** | **₹1,31,61,600** |

### Year 2 Targets (Growth):
| Tier | Users | Monthly Revenue | Annual Revenue |
|------|-------|----------------|----------------|
| Free (Before Login) | 50,000 | ₹0 | ₹0 |
| Free (After Login) | 25,000 | ₹0 | ₹0 |
| Starter (₹199) | 10,000 | ₹19,90,000 | ₹2,38,80,000 |
| Premium (₹499) | 5,000 | ₹24,95,000 | ₹2,99,40,000 |
| Ultimate (₹999) | 1,000 | ₹9,99,000 | ₹1,19,88,000 |
| **TOTAL** | **91,000** | **₹54,84,000** | **₹6,58,08,000** |

---

## 🎯 Conversion Funnel

```
100,000 Website Visitors
    ↓ (10% try)
10,000 Trial Users (3 questions)
    ↓ (50% sign up)
5,000 Free Users (10 questions/day)
    ↓ (20% upgrade)
1,000 Paid Users
    ↓
    ├─ 500 Starter (₹199) - 50%
    ├─ 300 Premium (₹499) - 30%
    └─ 200 Ultimate (₹999) - 20%

Monthly Revenue: ₹3,98,000
```

---

## 💡 Key Features by Tier

| Feature | Free | Starter | Premium | Ultimate |
|---------|------|---------|---------|----------|
| Questions/Day | 10 | 100 | ∞ | ∞ |
| NCERT Content | Basic | Full | Full+ | Full+ |
| Diagrams | ❌ | ✅ | ✅ | ✅ |
| Quiz Questions | 2 | 5 | 7 | 10 |
| Previous Papers | ❌ | 2015-24 | 2010-24 | 2005-24 |
| Image Solving | ❌ | ❌ | ✅ | ✅ |
| Mock Tests | ❌ | ❌ | ✅ | ✅ |
| Progress Tracking | ❌ | Basic | Advanced | AI-Powered |
| Study Plans | ❌ | ❌ | ❌ | ✅ |
| Live Sessions | ❌ | ❌ | ❌ | ✅ |
| Mentorship | ❌ | ❌ | ❌ | ✅ |
| Support | ❌ | Email | Priority | 24/7 |

---

## 🚀 Implementation Checklist

### Phase 1: Core Setup ✅
- [x] Trial limit (3 questions before login)
- [x] Free tier (10 questions/day after login)
- [x] Pricing page with 4 tiers
- [x] Razorpay integration
- [x] Upgrade modals

### Phase 2: Backend Integration
- [ ] User subscription model in database
- [ ] Plan-based feature gates
- [ ] Daily question counter (server-side)
- [ ] Payment verification webhook
- [ ] Subscription management API

### Phase 3: Feature Gates
- [ ] Limit questions based on plan
- [ ] Enable/disable diagrams by plan
- [ ] Enable/disable image solving by plan
- [ ] Enable/disable mock tests by plan
- [ ] Track usage analytics

### Phase 4: Marketing
- [ ] Landing page optimization
- [ ] Social proof (testimonials)
- [ ] Referral program
- [ ] Email campaigns
- [ ] Social media ads

---

## 💰 Pricing Psychology

### Why This Works:

1. **Free Tier Hook:** 3 questions → Sign up → 10/day builds habit
2. **Starter Sweet Spot:** ₹199 = 1 movie ticket, affordable for students
3. **Premium Value:** ₹499 = Less than 1 coaching class, huge value
4. **Ultimate Prestige:** ₹999 = For serious aspirants, includes mentorship
5. **Annual Discount:** 17% savings encourages commitment

### Competitive Positioning:
- **Unacademy:** ₹15,000-50,000/year → We're 10x cheaper
- **Physics Wallah:** ₹3,000-15,000/year → We're comparable
- **Offline Coaching:** ₹50,000-2,00,000/year → We're 50x cheaper

---

## 📈 Growth Strategy

### Month 1-3: Launch
- Focus: Product quality + word of mouth
- Target: 1,000 paid users
- Revenue: ₹3L/month

### Month 4-6: Marketing
- Focus: Paid ads + influencers
- Target: 3,000 paid users
- Revenue: ₹10L/month

### Month 7-12: Scale
- Focus: Features + partnerships
- Target: 10,000 paid users
- Revenue: ₹30L/month

---

## ✅ Success Metrics

### User Metrics:
- **Trial → Signup:** 50%
- **Free → Paid:** 20%
- **Monthly Churn:** <15%
- **LTV:CAC Ratio:** >10:1

### Financial Metrics:
- **MRR Growth:** 20% month-over-month
- **Gross Margin:** >80%
- **Break-even:** Month 6-8
- **Profitability:** Month 10-12

---

## 🎁 Special Offers

### Launch Offers:
1. **First 1,000 users:** 50% off lifetime
2. **Annual plans:** Extra 17% off
3. **Referral bonus:** ₹100 credit per referral
4. **Student discount:** 20% off with .edu email

### Seasonal Offers:
1. **Exam season:** ₹299 for 2 months (Premium)
2. **Diwali/New Year:** 40% off annual plans
3. **Independence Day:** ₹99 for 3 months (Starter)

---

## 🎯 Recommended Focus

**PRIMARY FOCUS: STARTER PLAN (₹199/month)**

**Why?**
- Affordable for most students
- Includes key features (diagrams, papers)
- Easy upsell to Premium
- High conversion rate
- Best revenue/user ratio

**Marketing Message:**
> "Less than ₹7/day for unlimited JEE/NEET preparation. That's cheaper than a samosa! 🥟"

---

**Next Steps:**
1. Test the complete flow (trial → free → paid)
2. Implement backend subscription logic
3. Add feature gates based on plans
4. Launch marketing campaign
5. Monitor and optimize conversions

**Goal:** ₹10L MRR by Month 6! 🚀
