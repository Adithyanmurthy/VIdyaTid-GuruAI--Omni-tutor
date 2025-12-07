# Requirements Document

## Introduction

This document outlines the requirements for implementing a subscription-based pricing system for VidyaTid (GuruAI), an AI-powered educational platform for JEE/NEET preparation. The system will support four pricing tiers: Free (10 queries/day), Starter (₹99/month), Premium (₹299/month), and Ultimate (₹499/month). Each tier provides progressively more features and capabilities to support students at different levels of commitment.

## Glossary

- **System**: The VidyaTid subscription management system
- **User**: A student using the VidyaTid platform
- **Query**: A single question asked by a user to the AI system
- **Tier**: A subscription level (Free, Starter, Premium, Ultimate)
- **Daily Limit**: Maximum number of queries allowed per 24-hour period
- **Feature Gate**: A mechanism that enables or disables features based on subscription tier
- **Razorpay**: The payment gateway integration for processing subscriptions
- **Active Subscription**: A subscription with status 'active' and end_date in the future
- **Upgrade**: Moving from a lower tier to a higher tier
- **Downgrade**: Moving from a higher tier to a lower tier
- **Usage Counter**: A daily reset counter tracking queries used

## Requirements

### Requirement 1

**User Story:** As a new user, I want to try the platform with a free tier that has 10 queries per day, so that I can evaluate the platform before committing to a paid plan.

#### Acceptance Criteria

1. WHEN a user signs up THEN the System SHALL create a free tier subscription with 10 queries per day limit
2. WHEN a free tier user submits a query THEN the System SHALL decrement their daily query counter
3. WHEN a free tier user reaches 10 queries in a day THEN the System SHALL prevent additional queries until the next day
4. WHEN the daily reset time occurs THEN the System SHALL reset all free tier users' query counters to 10
5. WHEN a free tier user attempts to access premium features THEN the System SHALL display an upgrade prompt

### Requirement 2

**User Story:** As a student, I want to purchase a Starter plan for ₹99/month with 50 queries per day, so that I can access more queries and basic premium features.

#### Acceptance Criteria

1. WHEN a user selects the Starter plan THEN the System SHALL display the Razorpay payment interface with ₹99 amount
2. WHEN payment is successful THEN the System SHALL create an active Starter subscription for 30 days
3. WHEN a Starter tier user submits a query THEN the System SHALL decrement their daily query counter from 50
4. WHEN a Starter tier user accesses diagrams THEN the System SHALL display the diagram content
5. WHEN a Starter tier user accesses previous year papers from 2015-2024 THEN the System SHALL provide access to those papers

### Requirement 3

**User Story:** As a serious aspirant, I want to purchase a Premium plan for ₹299/month with 200 queries per day, so that I can access advanced features like image-based doubt solving and mock tests.

#### Acceptance Criteria

1. WHEN a user selects the Premium plan THEN the System SHALL display the Razorpay payment interface with ₹299 amount
2. WHEN payment is successful THEN the System SHALL create an active Premium subscription for 30 days
3. WHEN a Premium tier user submits a query THEN the System SHALL decrement their daily query counter from 200
4. WHEN a Premium tier user uploads an image for doubt solving THEN the System SHALL process the image and provide solutions
5. WHEN a Premium tier user accesses mock tests THEN the System SHALL generate JEE/NEET pattern mock tests
6. WHEN a Premium tier user accesses previous year papers from 2010-2024 THEN the System SHALL provide access to those papers

### Requirement 4

**User Story:** As a top performer, I want to purchase an Ultimate plan for ₹499/month with unlimited queries, so that I can access all features without any restrictions.

#### Acceptance Criteria

1. WHEN a user selects the Ultimate plan THEN the System SHALL display the Razorpay payment interface with ₹499 amount
2. WHEN payment is successful THEN the System SHALL create an active Ultimate subscription for 30 days
3. WHEN an Ultimate tier user submits a query THEN the System SHALL not decrement any counter
4. WHEN an Ultimate tier user accesses any feature THEN the System SHALL grant access without restrictions
5. WHEN an Ultimate tier user accesses progress analytics THEN the System SHALL display detailed chapter-wise reports and performance tracking

### Requirement 5

**User Story:** As a user with an active subscription, I want my subscription to be automatically managed, so that I don't lose access unexpectedly and can renew or cancel as needed.

#### Acceptance Criteria

1. WHEN a subscription end_date is reached THEN the System SHALL update the subscription status to 'expired'
2. WHEN a user with an expired subscription attempts to use premium features THEN the System SHALL prompt them to renew
3. WHEN a user cancels their subscription THEN the System SHALL set auto_renew to false and record the cancellation timestamp
4. WHEN a cancelled subscription reaches its end_date THEN the System SHALL downgrade the user to free tier
5. WHEN a user upgrades their subscription THEN the System SHALL immediately activate the new tier and adjust the end_date proportionally

### Requirement 6

**User Story:** As a user, I want to see my current subscription status and usage, so that I can track how many queries I have left and when my subscription renews.

#### Acceptance Criteria

1. WHEN a user views their account page THEN the System SHALL display their current tier, queries remaining, and subscription end date
2. WHEN a user views their usage history THEN the System SHALL display daily query usage for the past 30 days
3. WHEN a user's subscription is about to expire THEN the System SHALL display a renewal reminder 7 days before expiration
4. WHEN a user reaches 80% of their daily query limit THEN the System SHALL display a warning notification
5. WHEN a user views available plans THEN the System SHALL highlight their current plan and show upgrade options

### Requirement 7

**User Story:** As a system administrator, I want payment webhooks to automatically update subscriptions, so that users receive immediate access after successful payment.

#### Acceptance Criteria

1. WHEN Razorpay sends a payment success webhook THEN the System SHALL verify the webhook signature
2. WHEN the webhook signature is valid THEN the System SHALL create or update the user's subscription record
3. WHEN the webhook signature is invalid THEN the System SHALL reject the webhook and log the security event
4. WHEN a payment fails THEN the System SHALL record the failed payment and notify the user
5. WHEN a subscription payment is captured THEN the System SHALL send a confirmation email to the user

### Requirement 8

**User Story:** As a user, I want feature gates to enforce tier restrictions, so that I only access features included in my subscription plan.

#### Acceptance Criteria

1. WHEN a free tier user attempts to access diagrams THEN the System SHALL block access and display an upgrade prompt
2. WHEN a Starter tier user attempts to access image doubt solving THEN the System SHALL block access and display an upgrade prompt
3. WHEN a Starter tier user attempts to access mock tests THEN the System SHALL block access and display an upgrade prompt
4. WHEN a Premium tier user accesses any Premium or lower feature THEN the System SHALL grant access
5. WHEN an Ultimate tier user accesses any feature THEN the System SHALL grant access without restrictions

### Requirement 9

**User Story:** As a user, I want to upgrade or downgrade my subscription, so that I can adjust my plan based on my changing needs.

#### Acceptance Criteria

1. WHEN a user upgrades to a higher tier THEN the System SHALL calculate prorated credit from the current subscription
2. WHEN a user upgrades to a higher tier THEN the System SHALL charge the difference and activate the new tier immediately
3. WHEN a user downgrades to a lower tier THEN the System SHALL schedule the downgrade for the next billing cycle
4. WHEN a downgrade is scheduled THEN the System SHALL maintain current tier access until the end_date
5. WHEN the billing cycle ends with a scheduled downgrade THEN the System SHALL activate the lower tier

### Requirement 10

**User Story:** As a system, I want to track all payment transactions, so that I can maintain accurate financial records and handle disputes.

#### Acceptance Criteria

1. WHEN a payment is initiated THEN the System SHALL create a payment record with status 'pending'
2. WHEN a payment is captured THEN the System SHALL update the payment record status to 'captured'
3. WHEN a payment fails THEN the System SHALL update the payment record status to 'failed'
4. WHEN a refund is processed THEN the System SHALL create a new payment record with status 'refunded'
5. WHEN an administrator views payment records THEN the System SHALL display all transactions with user details and timestamps

### Requirement 11

**User Story:** As a user, I want my query usage to reset daily at midnight, so that I can use my full daily quota each day.

#### Acceptance Criteria

1. WHEN the system clock reaches midnight UTC THEN the System SHALL reset all users' daily query counters
2. WHEN a user's counter is reset THEN the System SHALL set the counter to their tier's daily limit
3. WHEN a user submits a query after reset THEN the System SHALL use the new daily quota
4. WHEN the reset process encounters an error THEN the System SHALL log the error and retry the reset
5. WHEN a user checks their remaining queries after reset THEN the System SHALL display the full daily limit

### Requirement 12

**User Story:** As a user, I want to receive email notifications about my subscription, so that I stay informed about payments, renewals, and expirations.

#### Acceptance Criteria

1. WHEN a user successfully subscribes THEN the System SHALL send a welcome email with subscription details
2. WHEN a subscription is 7 days from expiration THEN the System SHALL send a renewal reminder email
3. WHEN a subscription expires THEN the System SHALL send an expiration notification email
4. WHEN a payment fails THEN the System SHALL send a payment failure notification email
5. WHEN a user cancels their subscription THEN the System SHALL send a cancellation confirmation email

### Requirement 13

**User Story:** As a student, I want to access AI-powered question paper predictions based on my subscription tier, so that I can prepare more effectively for upcoming exams.

#### Acceptance Criteria

1. WHEN a free tier user attempts to access prediction features THEN the System SHALL block access and display an upgrade prompt
2. WHEN a Starter tier user accesses chapter analysis THEN the System SHALL display chapter-wise frequency and probability data
3. WHEN a Starter tier user generates a prediction THEN the System SHALL decrement their monthly prediction counter from 2
4. WHEN a Premium tier user generates a smart paper THEN the System SHALL create a personalized practice paper based on weak areas
5. WHEN an Ultimate tier user generates predictions THEN the System SHALL not enforce any monthly limit

### Requirement 14

**User Story:** As a Premium or Ultimate user, I want to generate complete predicted NEET/JEE papers, so that I can practice with realistic future question patterns.

#### Acceptance Criteria

1. WHEN a Premium tier user requests a complete paper prediction THEN the System SHALL generate a full 200-question NEET paper with confidence scores
2. WHEN a prediction is generated THEN the System SHALL provide chapter-wise insights and recommended focus areas
3. WHEN a user reaches their monthly prediction limit THEN the System SHALL prevent additional predictions until the next month
4. WHEN the monthly reset occurs THEN the System SHALL reset all users' prediction counters based on their tier
5. WHEN a user views prediction history THEN the System SHALL display all previously generated predictions with timestamps

### Requirement 15

**User Story:** As a system deployed on Cloudflare, I want to leverage Cloudflare services for optimal performance, so that users experience fast and reliable service globally.

#### Acceptance Criteria

1. WHEN the application is deployed THEN the System SHALL use Cloudflare D1 for database operations
2. WHEN AI inference is needed THEN the System SHALL use Cloudflare Workers AI for LLM and embedding generation
3. WHEN files are accessed THEN the System SHALL serve content from Cloudflare R2 with CDN caching
4. WHEN subscription data is queried frequently THEN the System SHALL cache results in Cloudflare KV
5. WHEN the frontend is accessed THEN the System SHALL serve static assets from Cloudflare Pages with edge optimization

### Requirement 16

**User Story:** As a user, I want an intuitive and responsive UI/UX, so that I can easily navigate subscription options and manage my account.

#### Acceptance Criteria

1. WHEN a user views the pricing page THEN the System SHALL display pricing cards with clear feature comparisons and highlight the user's current plan
2. WHEN a user views their dashboard THEN the System SHALL display real-time usage statistics with progress bars and visual indicators
3. WHEN a user encounters a feature gate THEN the System SHALL display a modal with clear upgrade options and feature explanations
4. WHEN a user completes a payment THEN the System SHALL show a success animation and confirmation details
5. WHEN a user accesses the site on mobile THEN the System SHALL display a responsive layout optimized for touch interactions
