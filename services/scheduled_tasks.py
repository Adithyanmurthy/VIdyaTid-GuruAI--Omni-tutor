"""
Scheduled Tasks Service for VidyaTid.

Manages periodic background tasks including:
- Daily usage resets at midnight UTC
- Hourly subscription expiration checks
- Daily renewal reminders
- Monthly prediction counter resets

Uses APScheduler for reliable task scheduling.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

from models.database import SessionLocal
from models.user import User
from models.subscription import Subscription
from services.usage_tracker import UsageTracker
from services.subscription_service import SubscriptionService
from services.email_service import EmailService

logger = logging.getLogger(__name__)


class ScheduledTasksService:
    """Service for managing scheduled background tasks"""
    
    def __init__(self):
        """Initialize the scheduled tasks service"""
        self.scheduler = BackgroundScheduler()
        self.email_service = EmailService()
        self.is_running = False
        
        logger.info("ScheduledTasksService initialized")
    
    def daily_reset_task(self):
        """
        Daily reset task - runs at midnight UTC.
        Resets all users' daily query counters.
        
        Requirements: 1.4, 11.1, 11.2, 11.4
        """
        logger.info("Starting daily reset task...")
        
        try:
            # Create database session
            db = SessionLocal()
            
            try:
                # Create usage tracker
                usage_tracker = UsageTracker(db)
                
                # Reset daily usage for all users
                reset_count = usage_tracker.reset_daily_usage()
                
                logger.info(f"Daily reset completed successfully. Reset {reset_count} users.")
                
                # Check if failure rate exceeds threshold (1%)
                # Get total user count
                total_users = db.query(User).count()
                
                if total_users > 0:
                    failure_rate = ((total_users - reset_count) / total_users) * 100
                    
                    if failure_rate > 1.0:
                        # Send alert if failure rate exceeds threshold
                        logger.error(f"Daily reset failure rate: {failure_rate:.2f}% (threshold: 1%)")
                        
                        # Send alert email to admin
                        admin_email = "admin@vidyatid.com"  # TODO: Get from config
                        self.email_service.send_email(
                            to_email=admin_email,
                            subject="⚠️ Daily Reset Failure Alert",
                            html_content=f"""
                            <h2>Daily Reset Failure Alert</h2>
                            <p>The daily reset task has exceeded the failure threshold.</p>
                            <ul>
                                <li>Total Users: {total_users}</li>
                                <li>Successfully Reset: {reset_count}</li>
                                <li>Failed: {total_users - reset_count}</li>
                                <li>Failure Rate: {failure_rate:.2f}%</li>
                            </ul>
                            <p>Please investigate the issue immediately.</p>
                            """
                        )
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in daily reset task: {e}", exc_info=True)
            
            # Send critical alert
            try:
                admin_email = "admin@vidyatid.com"
                self.email_service.send_email(
                    to_email=admin_email,
                    subject="🚨 Critical: Daily Reset Task Failed",
                    html_content=f"""
                    <h2>Critical Error: Daily Reset Task Failed</h2>
                    <p>The daily reset task encountered a critical error and could not complete.</p>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>Please investigate and resolve immediately.</p>
                    """
                )
            except Exception as email_error:
                logger.error(f"Failed to send alert email: {email_error}")
    
    def subscription_expiration_check_task(self):
        """
        Subscription expiration check task - runs hourly.
        Checks for expired subscriptions and executes scheduled downgrades.
        
        Requirements: 5.1, 5.2, 5.4, 9.5
        """
        logger.info("Starting subscription expiration check task...")
        
        try:
            # Create database session
            db = SessionLocal()
            
            try:
                # Create subscription service
                subscription_service = SubscriptionService(db)
                
                # Check and expire subscriptions
                expired_count = subscription_service.check_and_expire_subscriptions()
                
                logger.info(f"Subscription expiration check completed. Processed {expired_count} subscriptions.")
                
                # Send expiration notification emails
                # Get all subscriptions that just expired
                expired_subs = db.query(Subscription).filter(
                    Subscription.status == 'expired',
                    Subscription.updated_at >= datetime.utcnow() - timedelta(hours=1)
                ).all()
                
                for subscription in expired_subs:
                    try:
                        # Get user email
                        user = db.query(User).filter_by(user_id=subscription.user_id).first()
                        
                        if user and user.email:
                            # Send expiration notification
                            self.email_service.send_subscription_expired(
                                user_email=user.email,
                                tier=subscription.tier
                            )
                            logger.info(f"Sent expiration notification to user {user.user_id}")
                    
                    except Exception as email_error:
                        logger.error(f"Error sending expiration email for user {subscription.user_id}: {email_error}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in subscription expiration check task: {e}", exc_info=True)
    
    def renewal_reminder_task(self):
        """
        Renewal reminder task - runs daily.
        Sends renewal reminders to users whose subscriptions expire in 7 days.
        
        Requirements: 6.3, 12.2
        """
        logger.info("Starting renewal reminder task...")
        
        try:
            # Create database session
            db = SessionLocal()
            
            try:
                # Calculate target date (7 days from now)
                target_date = datetime.utcnow() + timedelta(days=7)
                
                # Find subscriptions expiring in 7 days
                # We look for subscriptions expiring between 7 days and 7 days + 1 hour from now
                expiring_subs = db.query(Subscription).filter(
                    Subscription.status == 'active',
                    Subscription.end_date >= target_date,
                    Subscription.end_date < target_date + timedelta(hours=1)
                ).all()
                
                logger.info(f"Found {len(expiring_subs)} subscriptions expiring in 7 days")
                
                # Send renewal reminders
                reminder_count = 0
                
                for subscription in expiring_subs:
                    try:
                        # Get user email
                        user = db.query(User).filter_by(user_id=subscription.user_id).first()
                        
                        if user and user.email:
                            # Calculate days remaining
                            days_remaining = (subscription.end_date - datetime.utcnow()).days
                            
                            # Send renewal reminder
                            success = self.email_service.send_subscription_renewal_reminder(
                                user_email=user.email,
                                tier=subscription.tier,
                                days_remaining=days_remaining
                            )
                            
                            if success:
                                reminder_count += 1
                                logger.info(f"Sent renewal reminder to user {user.user_id}")
                            else:
                                logger.warning(f"Failed to send renewal reminder to user {user.user_id}")
                    
                    except Exception as email_error:
                        logger.error(f"Error sending renewal reminder for user {subscription.user_id}: {email_error}")
                
                logger.info(f"Renewal reminder task completed. Sent {reminder_count} reminders.")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in renewal reminder task: {e}", exc_info=True)
    
    def monthly_prediction_reset_task(self):
        """
        Monthly prediction reset task - runs on the 1st of each month.
        Resets prediction counters for all users.
        
        Requirements: 14.4
        """
        logger.info("Starting monthly prediction reset task...")
        
        try:
            # Create database session
            db = SessionLocal()
            
            try:
                # Create usage tracker
                usage_tracker = UsageTracker(db)
                
                # Reset monthly predictions for all users
                reset_count = usage_tracker.reset_monthly_predictions()
                
                logger.info(f"Monthly prediction reset completed successfully. Reset {reset_count} users.")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in monthly prediction reset task: {e}", exc_info=True)
    
    def start(self):
        """Start all scheduled tasks"""
        if self.is_running:
            logger.warning("Scheduled tasks are already running")
            return
        
        try:
            # Task 1: Daily reset at midnight UTC
            self.scheduler.add_job(
                func=self.daily_reset_task,
                trigger=CronTrigger(hour=0, minute=0, timezone='UTC'),
                id='daily_reset',
                name='Daily Usage Reset',
                replace_existing=True,
                misfire_grace_time=300  # 5 minutes grace time
            )
            logger.info("Scheduled daily reset task (midnight UTC)")
            
            # Task 2: Subscription expiration check every hour
            self.scheduler.add_job(
                func=self.subscription_expiration_check_task,
                trigger=IntervalTrigger(hours=1),
                id='subscription_expiration_check',
                name='Subscription Expiration Check',
                replace_existing=True,
                misfire_grace_time=300
            )
            logger.info("Scheduled subscription expiration check task (hourly)")
            
            # Task 3: Renewal reminder daily at 9 AM UTC
            self.scheduler.add_job(
                func=self.renewal_reminder_task,
                trigger=CronTrigger(hour=9, minute=0, timezone='UTC'),
                id='renewal_reminder',
                name='Renewal Reminder',
                replace_existing=True,
                misfire_grace_time=300
            )
            logger.info("Scheduled renewal reminder task (daily at 9 AM UTC)")
            
            # Task 4: Monthly prediction reset on 1st of month at midnight UTC
            self.scheduler.add_job(
                func=self.monthly_prediction_reset_task,
                trigger=CronTrigger(day=1, hour=0, minute=0, timezone='UTC'),
                id='monthly_prediction_reset',
                name='Monthly Prediction Reset',
                replace_existing=True,
                misfire_grace_time=300
            )
            logger.info("Scheduled monthly prediction reset task (1st of month at midnight UTC)")
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info("All scheduled tasks started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scheduled tasks: {e}", exc_info=True)
            raise
    
    def stop(self):
        """Stop all scheduled tasks"""
        if not self.is_running:
            logger.warning("Scheduled tasks are not running")
            return
        
        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("All scheduled tasks stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping scheduled tasks: {e}", exc_info=True)
    
    def get_job_status(self):
        """Get status of all scheduled jobs"""
        if not self.is_running:
            return {"status": "not_running", "jobs": []}
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "status": "running",
            "jobs": jobs
        }


# Global instance
_scheduled_tasks_service: Optional[ScheduledTasksService] = None


def get_scheduled_tasks_service() -> ScheduledTasksService:
    """
    Get or create the global scheduled tasks service instance.
    
    Returns:
        ScheduledTasksService instance
    """
    global _scheduled_tasks_service
    
    if _scheduled_tasks_service is None:
        _scheduled_tasks_service = ScheduledTasksService()
    
    return _scheduled_tasks_service


def start_scheduled_tasks():
    """Start all scheduled tasks"""
    service = get_scheduled_tasks_service()
    service.start()
    logger.info("Scheduled tasks service started")


def stop_scheduled_tasks():
    """Stop all scheduled tasks"""
    global _scheduled_tasks_service
    
    if _scheduled_tasks_service is not None:
        _scheduled_tasks_service.stop()
        _scheduled_tasks_service = None
        logger.info("Scheduled tasks service stopped")
