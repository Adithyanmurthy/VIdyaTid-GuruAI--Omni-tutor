# VidyaTid API Documentation

## Overview

This document provides comprehensive documentation for the VidyaTid subscription pricing system API. The API follows RESTful principles and uses JSON for request and response payloads.

**Base URL**: `https://your-domain.com`

**API Version**: 1.0

## Authentication

All authenticated endpoints require a valid authentication token. The token can be provided in two ways:

1. **Authorization Header** (Recommended):
   ```
   Authorization: Bearer <your_token>
   ```

2. **Cookie**:
   ```
   token=<your_token>
   ```

### Authentication Errors

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 401 | `UNAUTHORIZED` | Missing or invalid authentication token |
| 403 | `FORBIDDEN` | Valid token but insufficient permissions |

---

## Subscription Endpoints

### Get Current Subscription

Retrieve the authenticated user's current subscription details.

**Endpoint**: `GET /api/subscription/current`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "subscription": {
    "subscription_id": "sub_abc123",
    "user_id": "user_xyz789",
    "tier": "premium",
    "status": "active",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "auto_renew": true,
    "is_active": true,
    "days_remaining": 15,
    "cancelled_at": null,
    "scheduled_tier_change": null,
    "scheduled_change_date": null
  }
}
```

**Response** (200 OK - No Subscription):
```json
{
  "subscription": null,
  "message": "No subscription found"
}
```

**Response Fields**:
- `subscription_id` (string): Unique subscription identifier
- `user_id` (string): User identifier
- `tier` (string): Subscription tier - `free`, `starter`, `premium`, or `ultimate`
- `status` (string): Subscription status - `active`, `cancelled`, `expired`, or `pending`
- `start_date` (string): ISO 8601 timestamp of subscription start
- `end_date` (string): ISO 8601 timestamp of subscription end
- `auto_renew` (boolean): Whether subscription will auto-renew
- `is_active` (boolean): Whether subscription is currently active
- `days_remaining` (integer): Days until subscription expires
- `cancelled_at` (string|null): ISO 8601 timestamp of cancellation
- `scheduled_tier_change` (string|null): Tier scheduled for next billing cycle
- `scheduled_change_date` (string|null): Date when tier change will occur

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 SUBSCRIPTION_ERROR`: Internal server error

---

### Upgrade Subscription

Upgrade the user's subscription to a higher tier with immediate activation and prorated pricing.

**Endpoint**: `POST /api/subscription/upgrade`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "new_tier": "premium"
}
```

**Request Fields**:
- `new_tier` (string, required): Target tier - `starter`, `premium`, or `ultimate`

**Response** (200 OK):
```json
{
  "status": "success",
  "upgrade": {
    "success": true,
    "message": "Successfully upgraded to premium tier",
    "prorated_amount": 20000,
    "new_end_date": "2024-02-15T23:59:59Z"
  },
  "message": "Successfully upgraded to premium tier"
}
```

**Response Fields**:
- `success` (boolean): Whether upgrade was successful
- `message` (string): Human-readable status message
- `prorated_amount` (integer): Amount charged in paise (₹200.00 = 20000)
- `new_end_date` (string): ISO 8601 timestamp of new subscription end date

**Error Responses**:
- `400 MISSING_FIELDS`: `new_tier` not provided
- `400 INVALID_TIER`: Invalid tier name
- `400 UPGRADE_FAILED`: Upgrade operation failed (e.g., already on higher tier)
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 UPGRADE_ERROR`: Internal server error

---

### Downgrade Subscription

Schedule a downgrade to a lower tier. The downgrade takes effect at the end of the current billing cycle.

**Endpoint**: `POST /api/subscription/downgrade`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "new_tier": "starter"
}
```

**Request Fields**:
- `new_tier` (string, required): Target tier - `free`, `starter`, or `premium`

**Response** (200 OK):
```json
{
  "status": "success",
  "downgrade": {
    "success": true,
    "message": "Downgrade to starter scheduled for 2024-01-31",
    "scheduled_date": "2024-01-31T23:59:59Z"
  },
  "message": "Downgrade to starter scheduled for 2024-01-31"
}
```

**Response Fields**:
- `success` (boolean): Whether downgrade was scheduled successfully
- `message` (string): Human-readable status message
- `scheduled_date` (string): ISO 8601 timestamp when downgrade will occur

**Error Responses**:
- `400 MISSING_FIELDS`: `new_tier` not provided
- `400 INVALID_TIER`: Invalid tier name
- `400 DOWNGRADE_FAILED`: Downgrade operation failed (e.g., already on lower tier)
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 DOWNGRADE_ERROR`: Internal server error

---

### Cancel Subscription

Cancel the user's subscription. The subscription remains active until the end date, but will not auto-renew.

**Endpoint**: `POST /api/subscription/cancel`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Subscription cancelled. You will have access until 2024-01-31",
  "end_date": "2024-01-31T23:59:59Z"
}
```

**Response Fields**:
- `message` (string): Human-readable confirmation message
- `end_date` (string): ISO 8601 timestamp when access will end

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `404 NO_SUBSCRIPTION`: No active subscription found
- `500 CANCEL_FAILED`: Cancellation operation failed
- `500 CANCEL_ERROR`: Internal server error

---

## Payment Endpoints

### Get Pricing Information

Retrieve pricing information for all subscription tiers.

**Endpoint**: `GET /api/payment/pricing`

**Authentication**: Not required

**Response** (200 OK):
```json
{
  "free": {
    "name": "Free",
    "price_monthly": 0,
    "price_yearly": 0,
    "queries_per_day": 10,
    "features": {
      "basic_queries": true,
      "ncert_content": true,
      "diagrams": false,
      "previous_papers": false,
      "image_solving": false,
      "mock_tests": false
    }
  },
  "starter": {
    "name": "Starter",
    "price_monthly": 9900,
    "price_yearly": 99000,
    "queries_per_day": 50,
    "features": {
      "basic_queries": true,
      "ncert_content": true,
      "diagrams": true,
      "previous_papers": true,
      "image_solving": false,
      "mock_tests": false
    }
  },
  "premium": {
    "name": "Premium",
    "price_monthly": 29900,
    "price_yearly": 299000,
    "queries_per_day": 200,
    "features": {
      "basic_queries": true,
      "ncert_content": true,
      "diagrams": true,
      "previous_papers": true,
      "image_solving": true,
      "mock_tests": true
    }
  },
  "ultimate": {
    "name": "Ultimate",
    "price_monthly": 49900,
    "price_yearly": 499000,
    "queries_per_day": -1,
    "features": {
      "basic_queries": true,
      "ncert_content": true,
      "diagrams": true,
      "previous_papers": true,
      "image_solving": true,
      "mock_tests": true,
      "personalized_study_plans": true,
      "priority_support": true
    }
  }
}
```

**Note**: Prices are in paise (₹99 = 9900 paise). `queries_per_day: -1` indicates unlimited queries.

**Error Responses**:
- `500 PRICING_ERROR`: Internal server error

---

### Create Payment Order

Create a Razorpay payment order for subscription purchase or upgrade.

**Endpoint**: `POST /api/payment/order/create`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body** (Tier-based):
```json
{
  "tier": "premium",
  "duration": "monthly"
}
```

**Request Body** (Amount-based):
```json
{
  "amount": 29900,
  "currency": "INR",
  "notes": {
    "description": "Custom payment"
  }
}
```

**Request Fields**:
- `tier` (string, optional): Subscription tier - `starter`, `premium`, or `ultimate`
- `duration` (string, optional): `monthly` or `yearly` (default: `monthly`)
- `amount` (integer, optional): Amount in paise (required if `tier` not provided)
- `currency` (string, optional): Currency code (default: `INR`)
- `notes` (object, optional): Additional metadata

**Response** (201 Created):
```json
{
  "status": "success",
  "order": {
    "order_id": "order_abc123",
    "amount": 29900,
    "currency": "INR",
    "key_id": "rzp_test_xyz789",
    "tier": "premium",
    "duration": "monthly"
  }
}
```

**Response Fields**:
- `order_id` (string): Razorpay order identifier
- `amount` (integer): Order amount in paise
- `currency` (string): Currency code
- `key_id` (string): Razorpay public key for checkout
- `tier` (string): Subscription tier (if tier-based order)
- `duration` (string): Subscription duration (if tier-based order)

**Error Responses**:
- `400 MISSING_DATA`: Request body is empty
- `400 MISSING_AMOUNT`: Neither `amount` nor `tier` provided
- `400 INVALID_TIER`: Invalid tier name
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 ORDER_ERROR`: Internal server error

---

### Verify Payment

Verify payment signature after successful Razorpay payment.

**Endpoint**: `POST /api/payment/verify`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "razorpay_order_id": "order_abc123",
  "razorpay_payment_id": "pay_xyz789",
  "razorpay_signature": "signature_hash",
  "amount": 29900,
  "currency": "INR",
  "method": "card"
}
```

**Request Fields**:
- `razorpay_order_id` (string, required): Razorpay order ID
- `razorpay_payment_id` (string, required): Razorpay payment ID
- `razorpay_signature` (string, required): Payment signature from Razorpay
- `amount` (integer, optional): Payment amount in paise
- `currency` (string, optional): Currency code
- `method` (string, optional): Payment method used

**Response** (200 OK):
```json
{
  "status": "success",
  "verified": true,
  "message": "Payment verified successfully"
}
```

**Error Responses**:
- `400 MISSING_FIELDS`: Required fields not provided
- `400 INVALID_SIGNATURE`: Signature verification failed
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 VERIFICATION_ERROR`: Internal server error

---

### Payment Webhook

Handle Razorpay webhooks for payment and subscription events.

**Endpoint**: `POST /api/payment/webhook`

**Authentication**: Not required (uses webhook signature)

**Request Headers**:
```
X-Razorpay-Signature: <webhook_signature>
Content-Type: application/json
```

**Request Body**: Razorpay webhook payload (varies by event type)

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Webhook processed successfully"
}
```

**Error Responses**:
- `400 Missing signature`: Signature header not provided
- `400 signature_verification_failed`: Invalid webhook signature
- `500 Webhook processing failed`: Internal server error

**Supported Events**:
- `payment.captured`: Payment successfully captured
- `payment.failed`: Payment failed
- `subscription.activated`: Subscription activated
- `subscription.charged`: Subscription payment charged
- `subscription.cancelled`: Subscription cancelled
- `subscription.completed`: Subscription completed

---

### Get Payment History

Retrieve the user's payment transaction history.

**Endpoint**: `GET /api/payment/history`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Query Parameters**:
- `limit` (integer, optional): Number of payments to return (default: 10, max: 100)
- `payment_type` (string, optional): Filter by type - `subscription`, `upgrade`, `one_time`
- `status` (string, optional): Filter by status - `pending`, `captured`, `failed`, `refunded`

**Example Request**:
```
GET /api/payment/history?limit=20&payment_type=subscription&status=captured
```

**Response** (200 OK):
```json
{
  "payments": [
    {
      "payment_id": "pay_abc123",
      "razorpay_payment_id": "pay_xyz789",
      "amount": 29900,
      "currency": "INR",
      "status": "captured",
      "payment_type": "subscription",
      "payment_method": "card",
      "created_at": "2024-01-01T10:30:00Z",
      "metadata": {
        "tier": "premium",
        "duration": "monthly"
      }
    }
  ],
  "count": 1
}
```

**Response Fields**:
- `payments` (array): List of payment records
  - `payment_id` (string): Internal payment identifier
  - `razorpay_payment_id` (string): Razorpay payment identifier
  - `amount` (integer): Payment amount in paise
  - `currency` (string): Currency code
  - `status` (string): Payment status
  - `payment_type` (string): Type of payment
  - `payment_method` (string): Payment method used
  - `created_at` (string): ISO 8601 timestamp
  - `metadata` (object): Additional payment information
- `count` (integer): Number of payments returned

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 HISTORY_ERROR`: Internal server error

---

## Usage Endpoints

### Get Current Usage

Retrieve the user's usage information for the current day.

**Endpoint**: `GET /api/usage/current`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "usage": {
    "usage_id": "usage_abc123",
    "user_id": "user_xyz789",
    "date": "2024-01-15",
    "query_count": 25,
    "queries_limit": 200,
    "queries_remaining": 175,
    "prediction_count": 1,
    "predictions_limit": 5,
    "predictions_remaining": 4,
    "feature_usage": {
      "diagrams": 5,
      "image_solving": 3,
      "mock_tests": 2
    },
    "created_at": "2024-01-15T00:00:00Z",
    "updated_at": "2024-01-15T14:30:00Z"
  },
  "warning": {
    "warning": false,
    "message": null,
    "usage_percentage": 12.5,
    "queries_remaining": 175
  }
}
```

**Response Fields**:
- `usage` (object): Usage information
  - `usage_id` (string): Usage record identifier
  - `user_id` (string): User identifier
  - `date` (string): Date in YYYY-MM-DD format
  - `query_count` (integer): Queries used today
  - `queries_limit` (integer): Daily query limit (-1 for unlimited)
  - `queries_remaining` (integer): Queries remaining today
  - `prediction_count` (integer): Predictions used this month
  - `predictions_limit` (integer): Monthly prediction limit (-1 for unlimited)
  - `predictions_remaining` (integer): Predictions remaining this month
  - `feature_usage` (object): Feature-specific usage counts
  - `created_at` (string): ISO 8601 timestamp
  - `updated_at` (string): ISO 8601 timestamp
- `warning` (object): Usage warning information
  - `warning` (boolean): Whether user is near limit (>80%)
  - `message` (string|null): Warning message
  - `usage_percentage` (float): Percentage of limit used
  - `queries_remaining` (integer): Queries remaining

**Note**: This endpoint uses caching with a 5-minute TTL for performance.

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 USAGE_ERROR`: Internal server error

---

### Get Usage History

Retrieve the user's historical usage data.

**Endpoint**: `GET /api/usage/history`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Query Parameters**:
- `days` (integer, optional): Number of days to fetch (default: 30, max: 90)

**Example Request**:
```
GET /api/usage/history?days=7
```

**Response** (200 OK):
```json
{
  "history": [
    {
      "usage_id": "usage_abc123",
      "date": "2024-01-15",
      "query_count": 25,
      "queries_limit": 200,
      "prediction_count": 1,
      "predictions_limit": 5,
      "feature_usage": {
        "diagrams": 5,
        "image_solving": 3
      }
    },
    {
      "usage_id": "usage_def456",
      "date": "2024-01-14",
      "query_count": 180,
      "queries_limit": 200,
      "prediction_count": 0,
      "predictions_limit": 5,
      "feature_usage": {
        "diagrams": 12,
        "mock_tests": 1
      }
    }
  ],
  "days": 7,
  "count": 2
}
```

**Response Fields**:
- `history` (array): List of usage records
- `days` (integer): Number of days requested
- `count` (integer): Number of records returned

**Note**: This endpoint uses caching with a 5-minute TTL for performance.

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 HISTORY_ERROR`: Internal server error

---

### Get Usage Statistics

Retrieve aggregated usage statistics for dashboard display.

**Endpoint**: `GET /api/usage/stats`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "stats": {
    "total_queries": 1250,
    "total_predictions": 15,
    "avg_daily_queries": 41.67,
    "days_tracked": 30,
    "current_tier": "premium",
    "queries_limit": 200,
    "predictions_limit": 5
  },
  "current_usage": {
    "query_count": 25,
    "queries_remaining": 175,
    "prediction_count": 1,
    "predictions_remaining": 4
  }
}
```

**Response Fields**:
- `stats` (object): Aggregated statistics
  - `total_queries` (integer): Total queries across all tracked days
  - `total_predictions` (integer): Total predictions across all tracked days
  - `avg_daily_queries` (float): Average queries per day
  - `days_tracked` (integer): Number of days with usage data
  - `current_tier` (string): User's current subscription tier
  - `queries_limit` (integer): Current daily query limit
  - `predictions_limit` (integer): Current monthly prediction limit
- `current_usage` (object): Today's usage information

**Note**: This endpoint uses caching with a 5-minute TTL for performance.

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 STATS_ERROR`: Internal server error

---

### Get Remaining Queries

Lightweight endpoint to quickly check remaining queries and predictions.

**Endpoint**: `GET /api/usage/remaining`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "queries_remaining": 175,
  "predictions_remaining": 4,
  "unlimited_queries": false,
  "unlimited_predictions": false
}
```

**Response Fields**:
- `queries_remaining` (integer): Queries remaining today (-1 for unlimited)
- `predictions_remaining` (integer): Predictions remaining this month (-1 for unlimited)
- `unlimited_queries` (boolean): Whether user has unlimited queries
- `unlimited_predictions` (boolean): Whether user has unlimited predictions

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 REMAINING_ERROR`: Internal server error

---

### Get Prediction Usage

Retrieve detailed prediction usage information.

**Endpoint**: `GET /api/usage/predictions`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "current_month": {
    "prediction_count": 3,
    "predictions_limit": 5,
    "predictions_remaining": 2,
    "unlimited": false
  },
  "history": [
    {
      "date": "2024-01-15",
      "prediction_count": 1,
      "predictions_limit": 5
    },
    {
      "date": "2024-01-10",
      "prediction_count": 2,
      "predictions_limit": 5
    }
  ]
}
```

**Response Fields**:
- `current_month` (object): Current month's prediction usage
  - `prediction_count` (integer): Predictions used this month
  - `predictions_limit` (integer): Monthly prediction limit
  - `predictions_remaining` (integer): Predictions remaining
  - `unlimited` (boolean): Whether user has unlimited predictions
- `history` (array): Historical prediction usage (last 30 days)

**Note**: This endpoint uses caching with a 5-minute TTL for performance.

**Error Responses**:
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `500 PREDICTION_USAGE_ERROR`: Internal server error

---

### Increment Usage

Internal endpoint to increment usage counters. Used by query processing system.

**Endpoint**: `POST /api/usage/increment`

**Authentication**: Required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "query_type": "query"
}
```

**Request Fields**:
- `query_type` (string, optional): Type of usage - `query` or `prediction` (default: `query`)

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Query usage incremented",
  "queries_remaining": 174
}
```

**Response Fields**:
- `message` (string): Confirmation message
- `queries_remaining` (integer): Queries remaining after increment

**Error Responses**:
- `400 INVALID_TYPE`: Invalid `query_type` value
- `401 UNAUTHORIZED`: Missing or invalid authentication token
- `429 LIMIT_REACHED`: Daily limit reached
- `500 INCREMENT_ERROR`: Internal server error

---

## Error Response Format

All error responses follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details (optional)"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `MISSING_FIELDS` | 400 | Required fields not provided |
| `INVALID_TIER` | 400 | Invalid subscription tier |
| `LIMIT_REACHED` | 429 | Usage limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Authenticated endpoints**: 100 requests per minute per user
- **Unauthenticated endpoints**: 20 requests per minute per IP
- **Webhook endpoint**: 1000 requests per minute (verified by signature)

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

When rate limit is exceeded, the API returns:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Please try again later.",
    "retry_after": 60
  }
}
```

**HTTP Status**: 429 Too Many Requests

---

## Pagination

Endpoints that return lists support pagination:

**Query Parameters**:
- `limit` (integer): Number of items per page (default varies by endpoint)
- `offset` (integer): Number of items to skip (default: 0)

**Example**:
```
GET /api/payment/history?limit=20&offset=40
```

**Response includes pagination metadata**:
```json
{
  "data": [...],
  "pagination": {
    "limit": 20,
    "offset": 40,
    "total": 150,
    "has_more": true
  }
}
```

---

## Webhooks

### Webhook Security

All webhooks from Razorpay include a signature in the `X-Razorpay-Signature` header. The system verifies this signature before processing the webhook to ensure authenticity.

### Webhook Retry Policy

If webhook processing fails, Razorpay will retry with exponential backoff:
- 1st retry: After 5 minutes
- 2nd retry: After 30 minutes
- 3rd retry: After 2 hours
- 4th retry: After 6 hours
- 5th retry: After 24 hours

### Webhook Events

The system handles the following webhook events:

| Event | Description | Action |
|-------|-------------|--------|
| `payment.captured` | Payment successfully captured | Activate subscription, send confirmation email |
| `payment.failed` | Payment failed | Record failure, send notification email |
| `subscription.activated` | Subscription activated | Update subscription status |
| `subscription.charged` | Subscription payment charged | Extend subscription period |
| `subscription.cancelled` | Subscription cancelled | Update status, send cancellation email |

---

## Testing

### Test Credentials

For testing in sandbox mode, use these Razorpay test credentials:

**Key ID**: `rzp_test_xxxxxxxxxx` (provided in order creation response)

**Test Cards**:
- Success: `4111 1111 1111 1111`
- Failure: `4000 0000 0000 0002`

**CVV**: Any 3 digits
**Expiry**: Any future date

### Test Webhooks

To test webhooks locally, use ngrok or similar tunneling service:

```bash
ngrok http 5000
```

Configure the ngrok URL in Razorpay dashboard:
```
https://your-ngrok-url.ngrok.io/api/payment/webhook
```

---

## SDK Examples

### JavaScript/Node.js

```javascript
// Create payment order
const createOrder = async (tier, duration) => {
  const response = await fetch('/api/payment/order/create', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ tier, duration })
  });
  
  return await response.json();
};

// Get current subscription
const getSubscription = async () => {
  const response = await fetch('/api/subscription/current', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return await response.json();
};

// Check remaining queries
const checkRemaining = async () => {
  const response = await fetch('/api/usage/remaining', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return await response.json();
};
```

### Python

```python
import requests

# Create payment order
def create_order(token, tier, duration):
    response = requests.post(
        'https://your-domain.com/api/payment/order/create',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        },
        json={'tier': tier, 'duration': duration}
    )
    return response.json()

# Get current subscription
def get_subscription(token):
    response = requests.get(
        'https://your-domain.com/api/subscription/current',
        headers={'Authorization': f'Bearer {token}'}
    )
    return response.json()

# Check remaining queries
def check_remaining(token):
    response = requests.get(
        'https://your-domain.com/api/usage/remaining',
        headers={'Authorization': f'Bearer {token}'}
    )
    return response.json()
```

---

## Changelog

### Version 1.0 (2024-01-15)
- Initial API release
- Subscription management endpoints
- Payment processing with Razorpay
- Usage tracking and statistics
- Webhook support for automated subscription management

---

## Support

For API support, contact:
- Email: support@vidyatid.com
- Documentation: https://docs.vidyatid.com
- Status Page: https://status.vidyatid.com

---

*Last Updated: 2024-01-15*
