# Design Document

## Overview

The subscription pricing system for VidyaTid provides a tiered access model with four subscription levels: Free (10 queries/day), Starter (₹99/month), Premium (₹299/month), and Ultimate (₹499/month). The system integrates with Razorpay for payment processing, implements feature gates to control access based on subscription tier, tracks daily usage with automatic resets, and manages subscription lifecycle including upgrades, downgrades, and renewals.

The design follows a service-oriented architecture with clear separation between payment processing, subscription management, usage tracking, and feature gating. The system is built on the existing Flask application with SQLAlchemy models and integrates with the current authentication system.

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Frontend UI   │
│  (Templates)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask Routes   │
│  (Blueprints)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Service Layer                    │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │  Payment     │  │  Subscription   │ │
│  │  Service     │  │  Service        │ │
│  └──────────────┘  └─────────────────┘ │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │  Usage       │  │  Feature Gate   │ │
│  │  Tracker     │  │  Service        │ │
│  └──────────────┘  └─────────────────┘ │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Data Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │  User    │  │Subscription│ │Payment ││
│  │  Model   │  │  Model     │ │ Model  ││
│  └──────────┘  └──────────┘  └────────┘│
│  ┌──────────┐                           │
│  │  Usage   │                           │
│  │  Model   │                           │
│  └──────────┘                           │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  SQLite/Postgres│
│    Database     │
└─────────────────┘

External:
┌─────────────────┐
│   Razorpay API  │
│  (Payments &    │
│   Webhooks)     │
└─────────────────┘
```

### Component Interaction Flow

1. **Subscription Purchase Flow**:
   - User selects plan → Frontend sends request to `/api/payment/create-order`
   - PaymentService creates Razorpay order → Returns order details to frontend
   - Frontend displays Razorpay checkout → User completes payment
   - Razorpay sends webhook → PaymentService verifies signature
   - SubscriptionService activates subscription → UsageTracker initializes counters
   - EmailService sends confirmation

2. **Query Submission Flow**:
   - User submits query → FeatureGateService checks subscription tier
   - UsageTracker checks daily limit → Decrements counter if allowed
   - QueryHandler processes query → Returns response
   - UsageTracker logs usage

3. **Daily Reset Flow**:
   - Scheduled task runs at midnight UTC
   - UsageTracker resets all counters based on tier limits
   - Logs reset completion

## Components and Interfaces

### 1. SubscriptionService

**Purpose**: Manages subscription lifecycle, tier changes, and status checks.

**Key Methods**:
```python
class SubscriptionService:
    def get_user_subscription(user_id: str) -> Optional[SubscriptionInfo]
    def create_subscription(user_id: str, tier: str, duration_days: int, razorpay_sub_id: str) -> Subscription
    def upgrade_subscription(user_id: str, new_tier: str) -> UpgradeResult
    def downgrade_subscription(user_id: str, new_tier: str) -> DowngradeResult
    def cancel_subscription(user_id: str) -> bool
    def renew_subscription(user_id: str) -> bool
    def check_and_expire_subscriptions() -> int  # Returns count of expired
    def get_tier_features(tier: str) -> Dict[str, Any]
```

**Dependencies**: Database session, User model, Subscription model, EmailService

### 2. PaymentService (Enhanced)

**Purpose**: Handles Razorpay integration, payment processing, and webhook verification.

**Key Methods** (additions to existing):
```python
class PaymentService:
    # Existing methods remain
    def create_order_for_tier(user_id: str, tier: str, duration: str) -> OrderInfo
    def process_webhook(payload: str, signature: str) -> WebhookResult
    def calculate_prorated_amount(current_tier: str, new_tier: str, days_remaining: int) -> int
    def process_upgrade_payment(user_id: str, new_tier: str) -> PaymentResult
    def issue_refund(payment_id: str, amount: int, reason: str) -> RefundResult
```

**Dependencies**: Razorpay client, Database session, Payment model, SubscriptionService

### 3. UsageTracker

**Purpose**: Tracks daily query usage, enforces limits, and handles resets.

**Key Methods**:
```python
class UsageTracker:
    def get_usage(user_id: str, date: datetime) -> UsageInfo
    def increment_usage(user_id: str, query_type: str) -> bool
    def check_limit(user_id: str) -> LimitCheckResult
    def get_remaining_queries(user_id: str) -> int
    def reset_daily_usage(date: datetime) -> int  # Returns count reset
    def get_usage_history(user_id: str, days: int) -> List[UsageInfo]
    def get_usage_stats(user_id: str) -> UsageStats
```

**Dependencies**: Database session, Usage model, SubscriptionService

### 4. FeatureGateService

**Purpose**: Controls access to features based on subscription tier.

**Key Methods**:
```python
class FeatureGateService:
    def can_access_feature(user_id: str, feature: str) -> AccessResult
    def get_available_features(user_id: str) -> List[str]
    def check_query_limit(user_id: str) -> bool
    def check_diagram_access(user_id: str) -> bool
    def check_image_solving_access(user_id: str) -> bool
    def check_mock_test_access(user_id: str) -> bool
    def check_previous_papers_access(user_id: str, year: int) -> bool
    def get_upgrade_prompt(user_id: str, feature: str) -> UpgradePrompt
```

**Dependencies**: SubscriptionService, UsageTracker

### 5. EmailService (Enhanced)

**Purpose**: Sends subscription-related email notifications.

**Key Methods** (additions to existing):
```python
class EmailService:
    # Existing methods remain
    def send_subscription_welcome(user_email: str, tier: str, end_date: datetime)
    def send_subscription_renewal_reminder(user_email: str, tier: str, days_remaining: int)
    def send_subscription_expired(user_email: str, tier: str)
    def send_subscription_cancelled(user_email: str, tier: str, end_date: datetime)
    def send_upgrade_confirmation(user_email: str, old_tier: str, new_tier: str)
    def send_payment_failed(user_email: str, amount: int, reason: str)
    def send_usage_limit_warning(user_email: str, queries_remaining: int)
```

**Dependencies**: SendGrid API, User model

## Data Models

### Enhanced Subscription Model

```python
class Subscription(Base):
    __tablename__ = 'subscriptions'
    
    subscription_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    tier = Column(String(20), nullable=False)  # free, starter, premium, ultimate
    status = Column(String(20), nullable=False)  # active, cancelled, expired, pending
    razorpay_subscription_id = Column(String(100), nullable=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    auto_renew = Column(Boolean, default=True)
    cancelled_at = Column(DateTime, nullable=True)
    scheduled_tier_change = Column(String(20), nullable=True)  # For downgrades
    scheduled_change_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="subscription")
    usage_records = relationship("Usage", back_populates="subscription")
```

### New Usage Model

```python
class Usage(Base):
    __tablename__ = 'usage'
    
    usage_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    subscription_id = Column(String(36), ForeignKey('subscriptions.subscription_id'), nullable=True)
    date = Column(Date, nullable=False, index=True)
    query_count = Column(Integer, default=0)
    queries_limit = Column(Integer, nullable=False)
    feature_usage = Column(JSON, default=dict)  # Track specific feature usage
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Composite index for efficient lookups
    __table_args__ = (
        Index('idx_user_date', 'user_id', 'date'),
    )
    
    # Relationships
    user = relationship("User", back_populates="usage_records")
    subscription = relationship("Subscription", back_populates="usage_records")
```

### Enhanced Payment Model

```python
class Payment(Base):
    __tablename__ = 'payments'
    
    payment_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    subscription_id = Column(String(36), ForeignKey('subscriptions.subscription_id'), nullable=True)
    razorpay_payment_id = Column(String(100), nullable=False, unique=True)
    razorpay_order_id = Column(String(100), nullable=True)
    amount = Column(Integer, nullable=False)  # In paise
    currency = Column(String(3), default='INR')
    status = Column(String(20), nullable=False)  # pending, captured, failed, refunded
    payment_method = Column(String(50), nullable=True)
    payment_type = Column(String(20), nullable=False)  # subscription, upgrade, one_time
    metadata = Column(JSON, default=dict)  # Store additional payment info
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription")
```

### Tier Configuration

```python
TIER_CONFIG = {
    'free': {
        'name': 'Free',
        'price_monthly': 0,
        'price_yearly': 0,
        'queries_per_day': 10,
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': False,
            'previous_papers': False,
            'image_solving': False,
            'mock_tests': False,
            'progress_tracking': False,
            'advanced_analytics': False,
            'quiz_questions': 2
        }
    },
    'starter': {
        'name': 'Starter',
        'price_monthly': 9900,  # ₹99 in paise
        'price_yearly': 99000,  # ₹990 (17% discount)
        'queries_per_day': 50,
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': True,
            'previous_papers': True,
            'previous_papers_years': range(2015, 2025),
            'image_solving': False,
            'mock_tests': False,
            'progress_tracking': True,
            'advanced_analytics': False,
            'quiz_questions': 5
        }
    },
    'premium': {
        'name': 'Premium',
        'price_monthly': 29900,  # ₹299 in paise
        'price_yearly': 299000,  # ₹2990 (17% discount)
        'queries_per_day': 200,
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': True,
            'previous_papers': True,
            'previous_papers_years': range(2010, 2025),
            'image_solving': True,
            'mock_tests': True,
            'progress_tracking': True,
            'advanced_analytics': True,
            'quiz_questions': 7
        }
    },
    'ultimate': {
        'name': 'Ultimate',
        'price_monthly': 49900,  # ₹499 in paise
        'price_yearly': 499000,  # ₹4990 (17% discount)
        'queries_per_day': -1,  # Unlimited
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': True,
            'previous_papers': True,
            'previous_papers_years': range(2005, 2025),
            'image_solving': True,
            'mock_tests': True,
            'progress_tracking': True,
            'advanced_analytics': True,
            'personalized_study_plans': True,
            'priority_support': True,
            'quiz_questions': 10
        }
    }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Subscription Creation and Initialization

Property 1: New user free tier initialization
*For any* new user registration, the system should create a free tier subscription with exactly 10 queries per day limit
**Validates: Requirements 1.1**

Property 2: Query counter decrement consistency
*For any* user with an active subscription, submitting a query should decrement their daily counter by exactly 1
**Validates: Requirements 1.2, 2.3, 3.3**

Property 3: Query limit enforcement
*For any* user who has reached their daily query limit, attempting additional queries should be rejected until the next daily reset
**Validates: Requirements 1.3**

### Payment and Subscription Activation

Property 4: Payment amount correctness
*For any* subscription tier selection, the Razorpay payment interface should display the exact amount corresponding to that tier (₹99 for Starter, ₹299 for Premium, ₹499 for Ultimate)
**Validates: Requirements 2.1, 3.1, 4.1**

Property 5: Subscription activation on payment
*For any* successful payment, the system should create an active subscription with status 'active' and end_date exactly 30 days from the start_date
**Validates: Requirements 2.2, 3.2, 4.2**

Property 6: Payment record creation
*For any* payment initiation, the system should create a payment record with status 'pending' before processing
**Validates: Requirements 10.1**

Property 7: Payment status transitions
*For any* payment record, status transitions should follow the valid state machine: pending → captured/failed, and captured → refunded
**Validates: Requirements 10.2, 10.3, 10.4**

### Feature Access Control

Property 8: Free tier feature restrictions
*For any* free tier user, attempting to access premium features (diagrams, previous papers, image solving, mock tests) should result in access denial and an upgrade prompt
**Validates: Requirements 1.5, 8.1**

Property 9: Starter tier feature access
*For any* Starter tier user, accessing diagrams and previous papers from 2015-2024 should be granted, while image solving and mock tests should be denied
**Validates: Requirements 2.4, 2.5, 8.2, 8.3**

Property 10: Premium tier feature access
*For any* Premium tier user, accessing all Premium and lower tier features (including image solving and mock tests) should be granted
**Validates: Requirements 3.4, 3.5, 3.6, 8.4**

Property 11: Ultimate tier unrestricted access
*For any* Ultimate tier user, accessing any feature should be granted without restrictions, and query submissions should not decrement any counter
**Validates: Requirements 4.3, 4.4, 4.5, 8.5**

### Usage Tracking and Resets

Property 12: Daily reset completeness
*For any* daily reset operation at midnight UTC, all users' query counters should be reset to their tier's daily limit
**Validates: Requirements 1.4, 11.1, 11.2**

Property 13: Post-reset query availability
*For any* user after a daily reset, the first query submission should use the new daily quota and not be rejected
**Validates: Requirements 11.3, 11.5**

Property 14: Usage limit warning threshold
*For any* user who reaches 80% of their daily query limit, a warning notification should be displayed
**Validates: Requirements 6.4**

### Subscription Lifecycle Management

Property 15: Subscription expiration
*For any* subscription where the current time exceeds the end_date, the subscription status should be updated to 'expired'
**Validates: Requirements 5.1**

Property 16: Expired subscription access control
*For any* user with an expired subscription, attempting to access premium features should result in a renewal prompt
**Validates: Requirements 5.2**

Property 17: Cancellation state consistency
*For any* subscription cancellation, the system should set auto_renew to false and record a non-null cancelled_at timestamp
**Validates: Requirements 5.3**

Property 18: Cancelled subscription downgrade
*For any* cancelled subscription that reaches its end_date, the user should be downgraded to free tier
**Validates: Requirements 5.4**

### Upgrade and Downgrade Operations

Property 19: Immediate upgrade activation
*For any* user upgrade to a higher tier, the new tier should be activated immediately and the end_date should be adjusted proportionally based on remaining days
**Validates: Requirements 5.5, 9.2**

Property 20: Prorated upgrade calculation
*For any* upgrade operation, the charged amount should equal the difference between the new tier price and the prorated credit from the remaining days of the current subscription
**Validates: Requirements 9.1**

Property 21: Scheduled downgrade preservation
*For any* downgrade request, the system should schedule the downgrade for the next billing cycle and maintain current tier access until end_date
**Validates: Requirements 9.3, 9.4**

Property 22: Scheduled downgrade execution
*For any* subscription with a scheduled downgrade, when the end_date is reached, the system should activate the lower tier
**Validates: Requirements 9.5**

### Webhook Processing and Security

Property 23: Webhook signature verification
*For any* incoming Razorpay webhook, the system should verify the signature before processing the payload
**Validates: Requirements 7.1**

Property 24: Valid webhook processing
*For any* webhook with a valid signature, the system should create or update the user's subscription record accordingly
**Validates: Requirements 7.2**

Property 25: Invalid webhook rejection
*For any* webhook with an invalid signature, the system should reject the webhook and log a security event without processing
**Validates: Requirements 7.3**

Property 26: Payment failure handling
*For any* failed payment, the system should record the failure status and send a notification email to the user
**Validates: Requirements 7.4, 12.4**

### User Interface and Notifications

Property 27: Account page information accuracy
*For any* user viewing their account page, the displayed tier, queries remaining, and subscription end date should match the current database values
**Validates: Requirements 6.1**

Property 28: Usage history completeness
*For any* user viewing their usage history, the system should display daily query usage for exactly the past 30 days
**Validates: Requirements 6.2**

Property 29: Renewal reminder timing
*For any* subscription that is exactly 7 days from expiration, the system should send a renewal reminder email
**Validates: Requirements 6.3, 12.2**

Property 30: Current plan highlighting
*For any* user viewing available plans, their current tier should be highlighted and only higher tiers should be shown as upgrade options
**Validates: Requirements 6.5**

### Email Notification Consistency

Property 31: Subscription lifecycle emails
*For any* subscription state change (activation, expiration, cancellation), the system should send the corresponding email notification
**Validates: Requirements 7.5, 12.1, 12.3, 12.5**

## Error Handling

### Payment Errors

1. **Razorpay API Failures**
   - Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
   - Fallback: Store payment intent in database for manual processing
   - User notification: Display error message with support contact

2. **Webhook Signature Verification Failures**
   - Log security event with full payload details
   - Alert administrators via email
   - Return 401 Unauthorized to Razorpay
   - Do not process the webhook

3. **Payment Capture Failures**
   - Record failed payment in database
   - Send failure notification email to user
   - Provide retry option in user dashboard
   - Maintain subscription in pending state

### Subscription Errors

1. **Subscription Creation Failures**
   - Rollback database transaction
   - Refund payment if already captured
   - Log error with full context
   - Display user-friendly error message

2. **Upgrade/Downgrade Failures**
   - Maintain current subscription state
   - Log error details
   - Notify user of failure
   - Provide manual support option

3. **Expiration Check Failures**
   - Log error and continue with other subscriptions
   - Retry failed subscriptions in next cycle
   - Alert administrators if failure rate exceeds threshold

### Usage Tracking Errors

1. **Counter Increment Failures**
   - Use database transactions for atomicity
   - Retry once on failure
   - If retry fails, allow query but log error
   - Prevent negative counters with database constraints

2. **Daily Reset Failures**
   - Log failed user IDs
   - Retry failed resets individually
   - Alert administrators if failures exceed 1%
   - Provide manual reset tool for administrators

3. **Limit Check Failures**
   - Default to allowing query (fail open for user experience)
   - Log error for investigation
   - Alert if failure rate exceeds threshold

### Feature Gate Errors

1. **Subscription Lookup Failures**
   - Cache subscription data to reduce database dependency
   - Default to free tier if lookup fails
   - Log error and alert administrators
   - Retry lookup on next request

2. **Feature Configuration Errors**
   - Use hardcoded fallback configuration
   - Log error with configuration details
   - Alert administrators immediately
   - Prevent application crash

## Testing Strategy

### Unit Testing

The system will use pytest for unit testing with the following focus areas:

1. **Service Layer Tests**
   - SubscriptionService: Test tier changes, status checks, expiration logic
   - PaymentService: Test order creation, webhook verification, refund processing
   - UsageTracker: Test counter operations, reset logic, limit checks
   - FeatureGateService: Test access control for all tier/feature combinations

2. **Model Tests**
   - Subscription model: Test status transitions, date calculations, relationships
   - Usage model: Test counter operations, date indexing, aggregations
   - Payment model: Test status transitions, amount calculations

3. **Integration Tests**
   - End-to-end subscription purchase flow
   - Webhook processing with mock Razorpay responses
   - Daily reset with multiple users and tiers
   - Upgrade/downgrade scenarios with payment calculations

### Property-Based Testing

The system will use Hypothesis (Python) for property-based testing. Each property-based test will run a minimum of 100 iterations to ensure comprehensive coverage across random inputs.

**Property-Based Testing Library**: Hypothesis (https://hypothesis.readthedocs.io/)

**Test Configuration**:
```python
from hypothesis import given, settings, strategies as st

@settings(max_examples=100)
@given(...)
def test_property_name(...):
    # Test implementation
```

**Tagging Convention**: Each property-based test must include a comment with the format:
```python
# Feature: subscription-pricing-system, Property X: [property description]
```

**Property Test Coverage**:
- All 31 correctness properties will be implemented as property-based tests
- Tests will generate random users, subscriptions, payments, and usage data
- Tests will verify invariants hold across all generated inputs
- Tests will use Hypothesis strategies for realistic data generation

**Example Property Test Structure**:
```python
@given(
    user_id=st.uuids(),
    tier=st.sampled_from(['free', 'starter', 'premium', 'ultimate']),
    query_count=st.integers(min_value=0, max_value=500)
)
@settings(max_examples=100)
def test_query_counter_decrement(user_id, tier, query_count):
    """
    Feature: subscription-pricing-system, Property 2: Query counter decrement consistency
    
    For any user with an active subscription, submitting a query should 
    decrement their daily counter by exactly 1.
    """
    # Test implementation
```

### Manual Testing Checklist

1. **Payment Flow**
   - [ ] Test Razorpay checkout for each tier
   - [ ] Verify payment success webhook processing
   - [ ] Test payment failure scenarios
   - [ ] Verify email notifications

2. **Feature Access**
   - [ ] Test feature gates for each tier
   - [ ] Verify upgrade prompts display correctly
   - [ ] Test feature access after tier changes

3. **Usage Tracking**
   - [ ] Submit queries and verify counter decrements
   - [ ] Test limit enforcement at boundary
   - [ ] Verify daily reset at midnight
   - [ ] Test usage history display

4. **Subscription Management**
   - [ ] Test upgrade flow with proration
   - [ ] Test downgrade scheduling
   - [ ] Test cancellation flow
   - [ ] Verify expiration handling

## Performance Considerations

### Database Optimization

1. **Indexing Strategy**
   - Composite index on (user_id, date) for usage table
   - Index on user_id for subscriptions table
   - Index on razorpay_payment_id for payments table
   - Index on status and end_date for subscription queries

2. **Query Optimization**
   - Use database-level counter increments for atomicity
   - Batch daily reset operations
   - Cache subscription data in application memory
   - Use connection pooling for concurrent requests

3. **Caching Strategy**
   - Cache user subscription data for 5 minutes
   - Cache tier configuration permanently
   - Invalidate cache on subscription changes
   - Use Redis for distributed caching in production

### Scalability

1. **Horizontal Scaling**
   - Stateless service design allows multiple instances
   - Database connection pooling per instance
   - Shared cache layer (Redis) for consistency

2. **Background Jobs**
   - Daily reset runs as scheduled task (cron/celery)
   - Expiration checks run every hour
   - Email notifications queued for async processing
   - Webhook processing uses job queue for reliability

3. **Rate Limiting**
   - Implement rate limiting on API endpoints
   - Separate rate limits for authenticated vs unauthenticated
   - Use Redis for distributed rate limiting

### Monitoring and Alerts

1. **Key Metrics**
   - Subscription creation rate
   - Payment success/failure rate
   - Daily active users by tier
   - Query usage by tier
   - Webhook processing latency
   - Daily reset completion time

2. **Alerts**
   - Payment failure rate > 5%
   - Webhook signature verification failures
   - Daily reset failures > 1%
   - Database connection pool exhaustion
   - API response time > 2 seconds

## Security Considerations

### Payment Security

1. **Webhook Verification**
   - Always verify Razorpay webhook signatures
   - Use constant-time comparison for signatures
   - Log all verification failures
   - Rate limit webhook endpoint

2. **Payment Data**
   - Never store credit card details
   - Store only Razorpay payment IDs
   - Encrypt sensitive payment metadata
   - Comply with PCI DSS requirements

### Access Control

1. **Feature Gates**
   - Server-side enforcement only (never trust client)
   - Check subscription status on every request
   - Validate tier permissions against configuration
   - Log unauthorized access attempts

2. **API Security**
   - Require authentication for all subscription endpoints
   - Validate user owns the subscription being modified
   - Prevent privilege escalation attacks
   - Rate limit API requests

### Data Privacy

1. **User Data**
   - Hash sensitive user information
   - Implement data retention policies
   - Provide data export functionality
   - Support account deletion with data cleanup

2. **Audit Logging**
   - Log all subscription changes
   - Log all payment transactions
   - Log feature access denials
   - Retain logs for compliance requirements

## Deployment Strategy

### Database Migrations

1. **Migration Order**
   - Create usage table
   - Add new columns to subscription table
   - Add new columns to payment table
   - Create indexes
   - Migrate existing data

2. **Data Migration**
   - Create free tier subscriptions for existing users
   - Initialize usage records for all users
   - Backfill payment records if needed
   - Verify data integrity after migration

### Feature Rollout

1. **Phase 1: Backend Infrastructure**
   - Deploy new models and migrations
   - Deploy service layer components
   - Deploy API endpoints
   - Test with internal users

2. **Phase 2: Payment Integration**
   - Configure Razorpay plans
   - Deploy payment routes
   - Test payment flow end-to-end
   - Enable for beta users

3. **Phase 3: Feature Gates**
   - Deploy feature gate service
   - Enable gates for new features
   - Test access control
   - Monitor for issues

4. **Phase 4: Full Launch**
   - Enable for all users
   - Monitor metrics closely
   - Gather user feedback
   - Iterate based on data

### Rollback Plan

1. **Database Rollback**
   - Keep old schema columns during transition
   - Maintain backward compatibility
   - Test rollback procedure in staging
   - Document rollback steps

2. **Feature Flags**
   - Use feature flags for gradual rollout
   - Ability to disable features without deployment
   - Per-tier feature toggles
   - Emergency kill switch for critical issues

## Question Paper Prediction Feature

### Overview

The question paper prediction feature uses AI and pattern analysis to predict future NEET/JEE questions based on historical data. This feature will be integrated into the subscription tiers with different levels of access.

### Tier-Based Access

```python
PREDICTION_FEATURES = {
    'free': {
        'chapter_analysis': False,
        'prediction_insights': False,
        'smart_paper_generation': False,
        'complete_paper_prediction': False,
        'predictions_per_month': 0
    },
    'starter': {
        'chapter_analysis': True,
        'prediction_insights': True,
        'smart_paper_generation': False,
        'complete_paper_prediction': False,
        'predictions_per_month': 2
    },
    'premium': {
        'chapter_analysis': True,
        'prediction_insights': True,
        'smart_paper_generation': True,
        'complete_paper_prediction': True,
        'predictions_per_month': 10
    },
    'ultimate': {
        'chapter_analysis': True,
        'prediction_insights': True,
        'smart_paper_generation': True,
        'complete_paper_prediction': True,
        'predictions_per_month': -1  # Unlimited
    }
}
```

### Components

1. **QuestionPredictor Service** (Enhanced)
   - Analyze historical question patterns
   - Predict high-probability chapters
   - Generate smart practice papers based on weak areas
   - Create complete predicted NEET/JEE papers
   - Provide confidence scores and insights

2. **Prediction Usage Tracking**
   - Track predictions used per month
   - Reset monthly prediction counter
   - Enforce tier-based limits
   - Log prediction accuracy for improvement

3. **API Endpoints** (Existing, to be enhanced)
   - `/api/prediction/predict-paper` - Generate predicted paper
   - `/api/prediction/insights/<subject>` - Get prediction insights
   - `/api/prediction/smart-paper` - Generate personalized practice paper
   - `/api/prediction/chapter-analysis/<subject>` - Chapter-wise analysis
   - `/api/prediction/complete-neet/<year>` - Full NEET prediction

### Implementation Requirements

1. **Feature Gate Integration**
   - Add prediction feature checks to FeatureGateService
   - Enforce monthly prediction limits
   - Display upgrade prompts for restricted features

2. **Usage Tracking Enhancement**
   - Add prediction_count field to Usage model
   - Track predictions separately from queries
   - Monthly reset for prediction counters

3. **UI/UX Components**
   - Prediction dashboard showing available predictions
   - Chapter analysis visualization
   - Confidence score indicators
   - Upgrade prompts for premium predictions

## Cloudflare Deployment Architecture

### Overview

The application is deployed on Cloudflare infrastructure, leveraging Cloudflare Workers, Pages, D1 (SQLite), R2 (Storage), and Workers AI.

### Architecture Components

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
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                   ┌────────┴────────┐
                   │   Razorpay API  │
                   │   (Payments)    │
                   └─────────────────┘
```

### Cloudflare Services Integration

1. **Cloudflare Workers (Backend API)**
   - Handle all API requests
   - Process webhooks from Razorpay
   - Execute business logic
   - Manage authentication and sessions

2. **Cloudflare D1 (Database)**
   - SQLite-compatible database at the edge
   - Store users, subscriptions, payments, usage
   - Automatic replication across regions
   - Low-latency queries

3. **Cloudflare Workers AI**
   - LLM inference for query handling
   - Embedding generation for RAG
   - Image processing for doubt solving
   - Question prediction AI models

4. **Cloudflare R2 (Object Storage)**
   - Store NCERT PDFs and diagrams
   - Store previous year papers
   - Store user-uploaded images
   - CDN-backed for fast delivery

5. **Cloudflare KV (Key-Value Store)**
   - Cache subscription data
   - Cache tier configurations
   - Store session data
   - Rate limiting counters

6. **Cloudflare Pages (Frontend)**
   - Static site hosting
   - Automatic deployments from Git
   - Edge-optimized delivery
   - Built-in SSL/TLS

### Production Configuration

```python
# config.py additions for Cloudflare
class ProductionConfig(Config):
    # Cloudflare D1 Database
    USE_CLOUDFLARE_D1 = True
    CLOUDFLARE_D1_DATABASE_ID = os.getenv('CLOUDFLARE_D1_DATABASE_ID')
    
    # Cloudflare Workers AI
    USE_CLOUDFLARE_AI = True
    CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
    CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
    
    # Cloudflare R2 Storage
    CLOUDFLARE_R2_BUCKET = os.getenv('CLOUDFLARE_R2_BUCKET')
    CLOUDFLARE_R2_ACCESS_KEY = os.getenv('CLOUDFLARE_R2_ACCESS_KEY')
    CLOUDFLARE_R2_SECRET_KEY = os.getenv('CLOUDFLARE_R2_SECRET_KEY')
    
    # Cloudflare KV
    CLOUDFLARE_KV_NAMESPACE = os.getenv('CLOUDFLARE_KV_NAMESPACE')
    
    # Performance
    CACHE_TTL = 300  # 5 minutes
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_SAMESITE = 'Strict'
```

### Deployment Strategy

1. **Database Migration to D1**
   - Export existing SQLite data
   - Create D1 database schema
   - Import data to D1
   - Update connection strings
   - Test thoroughly before cutover

2. **File Migration to R2**
   - Upload NCERT content to R2
   - Upload diagrams to R2
   - Update file paths in code
   - Configure CDN URLs
   - Test file access

3. **Worker Deployment**
   - Package Flask app for Workers
   - Configure environment variables
   - Deploy to Workers
   - Test API endpoints
   - Monitor performance

4. **Frontend Deployment**
   - Build static assets
   - Deploy to Cloudflare Pages
   - Configure custom domain
   - Test user flows
   - Enable analytics

### Performance Optimizations

1. **Edge Caching**
   - Cache static content at edge
   - Cache API responses where appropriate
   - Use KV for frequently accessed data
   - Implement cache invalidation strategy

2. **Database Optimization**
   - Use prepared statements
   - Implement connection pooling
   - Optimize queries for D1
   - Use indexes effectively

3. **AI Model Optimization**
   - Use Cloudflare Workers AI for inference
   - Cache embeddings in KV
   - Batch AI requests where possible
   - Implement request queuing

## UI/UX Implementation

### Design System

1. **Color Palette**
   ```css
   :root {
     --primary: #4F46E5;      /* Indigo - Primary actions */
     --secondary: #10B981;    /* Green - Success states */
     --accent: #F59E0B;       /* Amber - Highlights */
     --danger: #EF4444;       /* Red - Errors/warnings */
     --background: #F9FAFB;   /* Light gray - Background */
     --surface: #FFFFFF;      /* White - Cards/surfaces */
     --text-primary: #111827; /* Dark gray - Primary text */
     --text-secondary: #6B7280; /* Medium gray - Secondary text */
   }
   ```

2. **Typography**
   - Headings: Inter (Bold, 600-700 weight)
   - Body: Inter (Regular, 400 weight)
   - Code/Numbers: JetBrains Mono

3. **Spacing System**
   - Base unit: 4px
   - Scale: 4, 8, 12, 16, 24, 32, 48, 64px

### Key UI Components

1. **Pricing Cards**
   ```
   ┌─────────────────────────────┐
   │  [TIER NAME]                │
   │  ₹XX/month                  │
   │  ─────────────────────      │
   │  ✓ Feature 1                │
   │  ✓ Feature 2                │
   │  ✓ Feature 3                │
   │  ─────────────────────      │
   │  [Subscribe Button]         │
   └─────────────────────────────┘
   ```
   - Highlight current plan
   - Show "Most Popular" badge
   - Animate on hover
   - Responsive grid layout

2. **Usage Dashboard**
   ```
   ┌─────────────────────────────────────┐
   │  Daily Queries                      │
   │  ████████░░ 8/10 used               │
   │                                     │
   │  Predictions This Month             │
   │  ██░░░░░░░░ 2/10 used               │
   │                                     │
   │  Subscription Status                │
   │  Active until: Dec 31, 2024         │
   │  [Manage Subscription]              │
   └─────────────────────────────────────┘
   ```
   - Real-time usage updates
   - Progress bars with animations
   - Clear call-to-action buttons

3. **Feature Gate Modals**
   ```
   ┌─────────────────────────────────────┐
   │  🔒 Premium Feature                 │
   │                                     │
   │  This feature is available in:      │
   │  • Premium Plan (₹299/month)        │
   │  • Ultimate Plan (₹499/month)       │
   │                                     │
   │  [Upgrade Now]  [Learn More]        │
   └─────────────────────────────────────┘
   ```
   - Clear feature explanation
   - Show which tiers include feature
   - Easy upgrade path
   - Dismissible

4. **Payment Flow**
   - Step indicator (1. Select Plan → 2. Payment → 3. Confirmation)
   - Razorpay embedded checkout
   - Loading states during processing
   - Success/failure animations
   - Email confirmation display

5. **Subscription Management**
   ```
   ┌─────────────────────────────────────┐
   │  Current Plan: Premium              │
   │  Status: Active                     │
   │  Renews: Jan 15, 2025               │
   │                                     │
   │  [Upgrade to Ultimate]              │
   │  [Cancel Subscription]              │
   │                                     │
   │  Payment History                    │
   │  • Dec 15, 2024 - ₹299 (Success)   │
   │  • Nov 15, 2024 - ₹299 (Success)   │
   └─────────────────────────────────────┘
   ```
   - Clear status indicators
   - Easy upgrade/downgrade options
   - Payment history table
   - Download invoice buttons

### Responsive Design

1. **Mobile-First Approach**
   - Stack pricing cards vertically on mobile
   - Collapsible navigation menu
   - Touch-friendly button sizes (min 44x44px)
   - Optimized forms for mobile input

2. **Breakpoints**
   - Mobile: < 640px
   - Tablet: 640px - 1024px
   - Desktop: > 1024px

3. **Progressive Enhancement**
   - Core functionality works without JavaScript
   - Enhanced interactions with JavaScript
   - Graceful degradation for older browsers

### Accessibility

1. **WCAG 2.1 AA Compliance**
   - Color contrast ratio ≥ 4.5:1
   - Keyboard navigation support
   - Screen reader compatibility
   - Focus indicators on all interactive elements

2. **Semantic HTML**
   - Proper heading hierarchy
   - ARIA labels where needed
   - Form labels and error messages
   - Alt text for images

3. **Loading States**
   - Skeleton screens for content loading
   - Spinner animations for actions
   - Progress indicators for long operations
   - Error states with retry options

### User Flows

1. **New User Onboarding**
   - Welcome screen explaining tiers
   - Quick tour of features
   - Prompt to try free tier
   - Gentle upgrade nudges

2. **Upgrade Flow**
   - Compare current vs target tier
   - Show prorated pricing
   - One-click upgrade
   - Immediate feature access

3. **Usage Limit Reached**
   - Clear notification
   - Show remaining time until reset
   - Upgrade option
   - Alternative: wait for reset

4. **Subscription Expiring**
   - 7-day advance notice
   - One-click renewal
   - Show what will be lost
   - Grace period explanation

## Future Enhancements

1. **Annual Subscriptions**
   - Add yearly billing option with 17% discount
   - Proration logic for annual plans
   - Annual renewal reminders

2. **Referral Program**
   - Give referrer and referee bonus queries
   - Track referral conversions
   - Reward top referrers

3. **Usage Analytics Dashboard**
   - Detailed usage charts
   - Comparison with peer groups
   - Personalized recommendations

4. **Dynamic Pricing**
   - Regional pricing based on location
   - Student discounts with verification
   - Promotional pricing campaigns

5. **Team/Institution Plans**
   - Multi-user subscriptions
   - Centralized billing
   - Admin dashboard for institutions
   - Bulk discounts

6. **Advanced Prediction Features**
   - AI-powered weak area detection
   - Personalized study plans based on predictions
   - Real-time prediction accuracy tracking
   - Collaborative prediction insights
