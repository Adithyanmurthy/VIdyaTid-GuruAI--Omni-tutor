"""
Email Service for VidyaTid using SendGrid.
Handles transactional emails for payments, subscriptions, and notifications.
"""
import os
import logging
from typing import Optional, Dict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    logging.warning("SendGrid not installed. Run: pip install sendgrid")

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via SendGrid"""
    
    def __init__(self):
        """Initialize SendGrid client"""
        self.api_key = os.getenv('SENDGRID_API_KEY')
        self.from_email = os.getenv('SENDGRID_FROM_EMAIL', 'noreply@vidyatid.com')
        self.from_name = os.getenv('SENDGRID_FROM_NAME', 'VidyaTid')
        
        if not self.api_key or not SENDGRID_AVAILABLE:
            logger.warning("SendGrid not configured or not installed")
            self.client = None
        else:
            self.client = SendGridAPIClient(self.api_key)
            logger.info("SendGrid email service initialized")
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send an email via SendGrid.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text content (optional)
            
        Returns:
            True if sent successfully
        """
        if not self.client:
            logger.error("SendGrid not configured")
            return False
        
        try:
            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=subject,
                html_content=Content("text/html", html_content)
            )
            
            if text_content:
                message.add_content(Content("text/plain", text_content))
            
            response = self.client.send(message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent to {to_email}: {subject}")
                return True
            else:
                logger.error(f"Failed to send email: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_welcome_email(self, user_email: str, username: str) -> bool:
        """
        Send welcome email to new user.
        
        Args:
            user_email: User's email address
            username: User's name
            
        Returns:
            True if sent successfully
        """
        subject = "Welcome to VidyaTid! 🎓"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .features {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .feature {{ margin: 10px 0; padding-left: 25px; position: relative; }}
                .feature:before {{ content: "✓"; position: absolute; left: 0; color: #10b981; font-weight: bold; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to VidyaTid!</h1>
                    <p>Your AI-Powered JEE & NEET Preparation Partner</p>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    
                    <p>Welcome to VidyaTid! We're excited to help you ace your JEE/NEET preparation with AI-powered learning.</p>
                    
                    <div class="features">
                        <h3>What You Can Do:</h3>
                        <div class="feature">Ask any question from NCERT textbooks</div>
                        <div class="feature">Upload diagrams for instant explanations</div>
                        <div class="feature">Practice with previous year papers</div>
                        <div class="feature">Track your progress and performance</div>
                        <div class="feature">Get personalized study recommendations</div>
                    </div>
                    
                    <center>
                        <a href="https://vidyatid.com/chat" class="button">Start Learning Now</a>
                    </center>
                    
                    <p><strong>Quick Tips:</strong></p>
                    <ul>
                        <li>Start with topics you find challenging</li>
                        <li>Use the AI tutor for detailed explanations</li>
                        <li>Practice regularly with previous papers</li>
                        <li>Track your progress to stay motivated</li>
                    </ul>
                    
                    <p>Need help? Reply to this email or visit our <a href="https://vidyatid.com/help">Help Center</a>.</p>
                    
                    <p>Happy Learning!<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_payment_confirmation(
        self,
        user_email: str,
        username: str,
        payment_details: Dict
    ) -> bool:
        """
        Send payment confirmation email.
        
        Args:
            user_email: User's email
            username: User's name
            payment_details: Dict with payment info
            
        Returns:
            True if sent successfully
        """
        subject = "Payment Successful - VidyaTid"
        
        plan_name = payment_details.get('plan_name', 'Premium')
        amount = payment_details.get('amount', 0)
        payment_id = payment_details.get('payment_id', 'N/A')
        date = payment_details.get('date', datetime.now().strftime('%B %d, %Y'))
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .receipt {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .receipt-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
                .receipt-row:last-child {{ border-bottom: none; font-weight: bold; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✓ Payment Successful!</h1>
                    <p>Thank you for subscribing to VidyaTid</p>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    
                    <p>Your payment has been processed successfully. Welcome to VidyaTid {plan_name}!</p>
                    
                    <div class="receipt">
                        <h3>Payment Receipt</h3>
                        <div class="receipt-row">
                            <span>Plan:</span>
                            <span>{plan_name}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Amount:</span>
                            <span>₹{amount}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Payment ID:</span>
                            <span>{payment_id}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Date:</span>
                            <span>{date}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Status:</span>
                            <span style="color: #10b981;">Paid</span>
                        </div>
                    </div>
                    
                    <p><strong>Your subscription is now active!</strong></p>
                    
                    <center>
                        <a href="https://vidyatid.com/chat" class="button">Start Learning</a>
                    </center>
                    
                    <p>If you have any questions about your payment or subscription, please contact us at support@vidyatid.com</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_subscription_activated(
        self,
        user_email: str,
        username: str,
        subscription_details: Dict
    ) -> bool:
        """
        Send subscription activation email.
        
        Args:
            user_email: User's email
            username: User's name
            subscription_details: Dict with subscription info
            
        Returns:
            True if sent successfully
        """
        subject = "Your VidyaTid Subscription is Active! 🎉"
        
        tier = subscription_details.get('tier', 'Premium')
        end_date = subscription_details.get('end_date', 'N/A')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .features {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .feature {{ margin: 10px 0; padding-left: 25px; position: relative; }}
                .feature:before {{ content: "✓"; position: absolute; left: 0; color: #10b981; font-weight: bold; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Subscription Activated!</h1>
                    <p>Welcome to VidyaTid {tier}</p>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    
                    <p>Great news! Your VidyaTid {tier} subscription is now active.</p>
                    
                    <div class="features">
                        <h3>You Now Have Access To:</h3>
                        <div class="feature">Unlimited AI tutor questions</div>
                        <div class="feature">All previous year papers (JEE & NEET)</div>
                        <div class="feature">Full-length mock tests</div>
                        <div class="feature">Advanced progress analytics</div>
                        <div class="feature">Priority support</div>
                        <div class="feature">Offline access to content</div>
                    </div>
                    
                    <p><strong>Subscription Details:</strong></p>
                    <ul>
                        <li>Plan: {tier}</li>
                        <li>Valid Until: {end_date}</li>
                        <li>Auto-Renewal: Enabled</li>
                    </ul>
                    
                    <center>
                        <a href="https://vidyatid.com/chat" class="button">Start Learning Now</a>
                    </center>
                    
                    <p>Make the most of your subscription by:</p>
                    <ul>
                        <li>Setting daily study goals</li>
                        <li>Taking regular mock tests</li>
                        <li>Tracking your progress</li>
                        <li>Focusing on weak areas</li>
                    </ul>
                    
                    <p>Questions? We're here to help at support@vidyatid.com</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                    <p><a href="https://vidyatid.com/unsubscribe">Unsubscribe</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_subscription_cancelled(
        self,
        user_email: str,
        username: str,
        end_date: str
    ) -> bool:
        """
        Send subscription cancellation confirmation.
        
        Args:
            user_email: User's email
            username: User's name
            end_date: When subscription ends
            
        Returns:
            True if sent successfully
        """
        subject = "Subscription Cancelled - VidyaTid"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #f59e0b; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Subscription Cancelled</h1>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    
                    <p>We've received your request to cancel your VidyaTid subscription.</p>
                    
                    <p><strong>Important:</strong> You'll continue to have access to all premium features until {end_date}.</p>
                    
                    <p>We're sorry to see you go! We'd love to know why you're leaving:</p>
                    
                    <center>
                        <a href="https://vidyatid.com/feedback" class="button">Share Feedback</a>
                    </center>
                    
                    <p>Changed your mind? You can reactivate your subscription anytime:</p>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Reactivate Subscription</a>
                    </center>
                    
                    <p>Thank you for being part of VidyaTid. We hope to see you again!</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_payment_failed(
        self,
        user_email: str,
        username: str,
        reason: str
    ) -> bool:
        """
        Send payment failure notification.
        
        Args:
            user_email: User's email
            username: User's name
            reason: Failure reason
            
        Returns:
            True if sent successfully
        """
        subject = "Payment Failed - Action Required"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #ef4444; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Payment Failed</h1>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    
                    <p>We were unable to process your payment for VidyaTid subscription.</p>
                    
                    <p><strong>Reason:</strong> {reason}</p>
                    
                    <p><strong>What to do next:</strong></p>
                    <ul>
                        <li>Check your payment method details</li>
                        <li>Ensure sufficient balance</li>
                        <li>Try a different payment method</li>
                        <li>Contact your bank if issue persists</li>
                    </ul>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Retry Payment</a>
                    </center>
                    
                    <p>Need help? Contact us at support@vidyatid.com or call 1800-XXX-XXXX</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_password_reset(
        self,
        user_email: str,
        username: str,
        reset_link: str
    ) -> bool:
        """
        Send password reset email.
        
        Args:
            user_email: User's email
            username: User's name
            reset_link: Password reset link
            
        Returns:
            True if sent successfully
        """
        subject = "Reset Your VidyaTid Password"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Password Reset Request</h1>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    
                    <p>We received a request to reset your VidyaTid password.</p>
                    
                    <center>
                        <a href="{reset_link}" class="button">Reset Password</a>
                    </center>
                    
                    <p>This link will expire in 1 hour for security reasons.</p>
                    
                    <div class="warning">
                        <strong>⚠️ Security Notice:</strong><br>
                        If you didn't request this password reset, please ignore this email. Your password will remain unchanged.
                    </div>
                    
                    <p>For security, never share your password with anyone, including VidyaTid staff.</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    # ========== Subscription Email Templates ==========
    
    def send_subscription_welcome(
        self,
        user_email: str,
        tier: str,
        end_date: str
    ) -> bool:
        """
        Send welcome email when subscription is activated.
        
        Args:
            user_email: User's email address
            tier: Subscription tier (free, starter, premium, ultimate)
            end_date: Subscription end date (formatted string)
            
        Returns:
            True if sent successfully
        """
        from services.tier_config import get_tier_display_name, get_queries_per_day
        
        tier_name = get_tier_display_name(tier)
        queries_per_day = get_queries_per_day(tier)
        queries_text = "Unlimited" if queries_per_day == -1 else f"{queries_per_day} per day"
        
        subject = f"Welcome to VidyaTid {tier_name}! 🎉"
        
        # Tier-specific features
        features_html = ""
        if tier == 'free':
            features_html = """
                <div class="feature">10 AI tutor queries per day</div>
                <div class="feature">Access to NCERT content</div>
                <div class="feature">Basic study materials</div>
            """
        elif tier == 'starter':
            features_html = """
                <div class="feature">50 AI tutor queries per day</div>
                <div class="feature">Access to all diagrams</div>
                <div class="feature">Previous year papers (2015-2024)</div>
                <div class="feature">Progress tracking</div>
            """
        elif tier == 'premium':
            features_html = """
                <div class="feature">200 AI tutor queries per day</div>
                <div class="feature">Image-based doubt solving</div>
                <div class="feature">Full-length mock tests</div>
                <div class="feature">Previous year papers (2010-2024)</div>
                <div class="feature">Advanced analytics</div>
            """
        elif tier == 'ultimate':
            features_html = """
                <div class="feature">Unlimited AI tutor queries</div>
                <div class="feature">All premium features</div>
                <div class="feature">Personalized study plans</div>
                <div class="feature">Priority support</div>
                <div class="feature">Previous year papers (2005-2024)</div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .features {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .feature {{ margin: 10px 0; padding-left: 25px; position: relative; }}
                .feature:before {{ content: "✓"; position: absolute; left: 0; color: #10b981; font-weight: bold; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .info-box {{ background: #e0e7ff; border-left: 4px solid #667eea; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Welcome to VidyaTid {tier_name}!</h1>
                    <p>Your subscription is now active</p>
                </div>
                <div class="content">
                    <p>Congratulations! Your VidyaTid {tier_name} subscription is now active and ready to use.</p>
                    
                    <div class="info-box">
                        <strong>Subscription Details:</strong><br>
                        Plan: {tier_name}<br>
                        Queries: {queries_text}<br>
                        Valid Until: {end_date}
                    </div>
                    
                    <div class="features">
                        <h3>Your Features:</h3>
                        {features_html}
                    </div>
                    
                    <center>
                        <a href="https://vidyatid.com/chat" class="button">Start Learning Now</a>
                    </center>
                    
                    <p><strong>Make the most of your subscription:</strong></p>
                    <ul>
                        <li>Set daily study goals and track your progress</li>
                        <li>Practice with previous year papers regularly</li>
                        <li>Use the AI tutor for detailed explanations</li>
                        <li>Take mock tests to assess your preparation</li>
                    </ul>
                    
                    <p>Need help getting started? Check out our <a href="https://vidyatid.com/help">Help Center</a> or reply to this email.</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_subscription_renewal_reminder(
        self,
        user_email: str,
        tier: str,
        days_remaining: int
    ) -> bool:
        """
        Send renewal reminder email before subscription expires.
        
        Args:
            user_email: User's email address
            tier: Subscription tier (free, starter, premium, ultimate)
            days_remaining: Number of days until expiration
            
        Returns:
            True if sent successfully
        """
        from services.tier_config import get_tier_display_name
        
        tier_name = get_tier_display_name(tier)
        subject = f"Your VidyaTid {tier_name} subscription expires in {days_remaining} days"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning-box {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⏰ Subscription Expiring Soon</h1>
                    <p>{days_remaining} days remaining</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>Your VidyaTid {tier_name} subscription will expire in <strong>{days_remaining} days</strong>.</p>
                    
                    <div class="warning-box">
                        <strong>Don't lose access to:</strong><br>
                        • Your current learning progress<br>
                        • Premium features and content<br>
                        • Personalized study insights<br>
                        • Mock tests and analytics
                    </div>
                    
                    <p><strong>Renew now to continue your preparation without interruption!</strong></p>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Renew Subscription</a>
                    </center>
                    
                    <p>If you have auto-renewal enabled, your subscription will automatically renew. You can manage your subscription settings in your account.</p>
                    
                    <center>
                        <a href="https://vidyatid.com/settings" style="color: #667eea; text-decoration: none;">Manage Subscription Settings</a>
                    </center>
                    
                    <p>Questions? We're here to help at support@vidyatid.com</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_subscription_expired(
        self,
        user_email: str,
        tier: str
    ) -> bool:
        """
        Send notification when subscription has expired.
        
        Args:
            user_email: User's email address
            tier: Expired subscription tier (starter, premium, ultimate)
            
        Returns:
            True if sent successfully
        """
        from services.tier_config import get_tier_display_name
        
        tier_name = get_tier_display_name(tier)
        subject = f"Your VidyaTid {tier_name} subscription has expired"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .info-box {{ background: #fee2e2; border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Subscription Expired</h1>
                    <p>Your {tier_name} plan has ended</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>Your VidyaTid {tier_name} subscription has expired. You've been moved to the Free plan.</p>
                    
                    <div class="info-box">
                        <strong>What this means:</strong><br>
                        • You now have 10 queries per day (instead of unlimited)<br>
                        • Premium features are no longer accessible<br>
                        • Your progress data is safely stored<br>
                        • You can still access basic NCERT content
                    </div>
                    
                    <p><strong>Want to continue where you left off?</strong></p>
                    
                    <p>Renew your subscription to regain access to all premium features and continue your preparation without limits.</p>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Renew Now</a>
                    </center>
                    
                    <p>We'd love to have you back! If you have any questions or need help choosing a plan, reply to this email.</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_subscription_cancelled(
        self,
        user_email: str,
        tier: str,
        end_date: str
    ) -> bool:
        """
        Send confirmation when user cancels their subscription.
        
        Args:
            user_email: User's email address
            tier: Subscription tier being cancelled
            end_date: Date when access will end (formatted string)
            
        Returns:
            True if sent successfully
        """
        from services.tier_config import get_tier_display_name
        
        tier_name = get_tier_display_name(tier)
        subject = "Subscription Cancelled - VidyaTid"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .info-box {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Subscription Cancelled</h1>
                    <p>We're sorry to see you go</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>We've received your request to cancel your VidyaTid {tier_name} subscription.</p>
                    
                    <div class="info-box">
                        <strong>Important Information:</strong><br>
                        • Your subscription remains active until <strong>{end_date}</strong><br>
                        • You'll continue to have full access until then<br>
                        • No further charges will be made<br>
                        • Your data will be safely stored
                    </div>
                    
                    <p><strong>We'd love to know why you're leaving.</strong></p>
                    
                    <p>Your feedback helps us improve VidyaTid for all students. Please take a moment to share your thoughts:</p>
                    
                    <center>
                        <a href="https://vidyatid.com/feedback" class="button">Share Feedback</a>
                    </center>
                    
                    <p><strong>Changed your mind?</strong></p>
                    
                    <p>You can reactivate your subscription anytime before {end_date} to continue without interruption.</p>
                    
                    <center>
                        <a href="https://vidyatid.com/settings" style="color: #667eea; text-decoration: none;">Reactivate Subscription</a>
                    </center>
                    
                    <p>Thank you for being part of VidyaTid. We hope to see you again soon!</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    # ========== Payment Notification Emails ==========
    
    def send_payment_failed(
        self,
        user_email: str,
        amount: int,
        reason: str
    ) -> bool:
        """
        Send notification when payment fails.
        
        Args:
            user_email: User's email address
            amount: Payment amount in paise
            reason: Reason for payment failure
            
        Returns:
            True if sent successfully
        """
        amount_rupees = amount / 100
        subject = "Payment Failed - Action Required"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .error-box {{ background: #fee2e2; border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>❌ Payment Failed</h1>
                    <p>We couldn't process your payment</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>We were unable to process your payment of <strong>₹{amount_rupees:.2f}</strong> for your VidyaTid subscription.</p>
                    
                    <div class="error-box">
                        <strong>Reason:</strong> {reason}
                    </div>
                    
                    <p><strong>What to do next:</strong></p>
                    <ul>
                        <li>Verify your payment method details are correct</li>
                        <li>Ensure you have sufficient balance in your account</li>
                        <li>Try using a different payment method</li>
                        <li>Contact your bank if the issue persists</li>
                    </ul>
                    
                    <p>You can retry the payment anytime:</p>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Retry Payment</a>
                    </center>
                    
                    <p><strong>Need help?</strong></p>
                    <p>If you continue to experience issues, please contact our support team:</p>
                    <ul>
                        <li>Email: support@vidyatid.com</li>
                        <li>Phone: 1800-XXX-XXXX (Mon-Sat, 9 AM - 6 PM)</li>
                    </ul>
                    
                    <p>We're here to help you get started!</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_payment_success(
        self,
        user_email: str,
        amount: int,
        tier: str
    ) -> bool:
        """
        Send notification when payment is successful.
        
        Args:
            user_email: User's email address
            amount: Payment amount in paise
            tier: Subscription tier purchased
            
        Returns:
            True if sent successfully
        """
        from services.tier_config import get_tier_display_name
        
        amount_rupees = amount / 100
        tier_name = get_tier_display_name(tier)
        subject = f"Payment Successful - VidyaTid {tier_name}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .success-box {{ background: #d1fae5; border-left: 4px solid #10b981; padding: 15px; margin: 20px 0; }}
                .receipt {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .receipt-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
                .receipt-row:last-child {{ border-bottom: none; font-weight: bold; font-size: 18px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✓ Payment Successful!</h1>
                    <p>Thank you for your payment</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>Your payment has been processed successfully! Your VidyaTid {tier_name} subscription is now active.</p>
                    
                    <div class="success-box">
                        <strong>✓ Payment Confirmed</strong><br>
                        Your subscription has been activated and you can start using all features immediately.
                    </div>
                    
                    <div class="receipt">
                        <h3>Payment Receipt</h3>
                        <div class="receipt-row">
                            <span>Plan:</span>
                            <span>{tier_name}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Amount Paid:</span>
                            <span>₹{amount_rupees:.2f}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Payment Date:</span>
                            <span>{datetime.now().strftime('%B %d, %Y')}</span>
                        </div>
                        <div class="receipt-row">
                            <span>Status:</span>
                            <span style="color: #10b981;">Paid</span>
                        </div>
                    </div>
                    
                    <center>
                        <a href="https://vidyatid.com/chat" class="button">Start Learning Now</a>
                    </center>
                    
                    <p><strong>What's next?</strong></p>
                    <ul>
                        <li>Explore all the features available in your plan</li>
                        <li>Set up your study goals and preferences</li>
                        <li>Start practicing with previous year papers</li>
                        <li>Track your progress in the dashboard</li>
                    </ul>
                    
                    <p>Need your invoice? You can download it from your <a href="https://vidyatid.com/settings">account settings</a>.</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_upgrade_confirmation(
        self,
        user_email: str,
        old_tier: str,
        new_tier: str
    ) -> bool:
        """
        Send confirmation when user upgrades their subscription.
        
        Args:
            user_email: User's email address
            old_tier: Previous subscription tier
            new_tier: New subscription tier
            
        Returns:
            True if sent successfully
        """
        from services.tier_config import get_tier_display_name
        
        old_tier_name = get_tier_display_name(old_tier)
        new_tier_name = get_tier_display_name(new_tier)
        subject = f"Subscription Upgraded to {new_tier_name}! 🎉"
        
        # New features based on upgrade
        new_features_html = ""
        if new_tier == 'starter':
            new_features_html = """
                <div class="feature">50 queries per day (up from 10)</div>
                <div class="feature">Access to all diagrams</div>
                <div class="feature">Previous year papers (2015-2024)</div>
            """
        elif new_tier == 'premium':
            new_features_html = """
                <div class="feature">200 queries per day</div>
                <div class="feature">Image-based doubt solving</div>
                <div class="feature">Full-length mock tests</div>
                <div class="feature">Advanced analytics</div>
            """
        elif new_tier == 'ultimate':
            new_features_html = """
                <div class="feature">Unlimited queries</div>
                <div class="feature">Personalized study plans</div>
                <div class="feature">Priority support</div>
                <div class="feature">All premium features</div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .upgrade-box {{ background: #ede9fe; border-left: 4px solid #8b5cf6; padding: 15px; margin: 20px 0; }}
                .features {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .feature {{ margin: 10px 0; padding-left: 25px; position: relative; }}
                .feature:before {{ content: "✓"; position: absolute; left: 0; color: #10b981; font-weight: bold; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Subscription Upgraded!</h1>
                    <p>Welcome to {new_tier_name}</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>Great news! Your subscription has been successfully upgraded from <strong>{old_tier_name}</strong> to <strong>{new_tier_name}</strong>.</p>
                    
                    <div class="upgrade-box">
                        <strong>✓ Upgrade Activated</strong><br>
                        All new features are now available in your account. Start using them right away!
                    </div>
                    
                    <div class="features">
                        <h3>New Features Unlocked:</h3>
                        {new_features_html}
                    </div>
                    
                    <center>
                        <a href="https://vidyatid.com/chat" class="button">Explore New Features</a>
                    </center>
                    
                    <p><strong>Make the most of your upgrade:</strong></p>
                    <ul>
                        <li>Try out the newly unlocked features</li>
                        <li>Increase your daily study time with more queries</li>
                        <li>Take advantage of advanced analytics</li>
                        <li>Practice with more previous year papers</li>
                    </ul>
                    
                    <p>Your billing has been adjusted to reflect the upgrade. You can view the details in your <a href="https://vidyatid.com/settings">account settings</a>.</p>
                    
                    <p>Thank you for choosing VidyaTid {new_tier_name}!</p>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    # ========== Usage Notification Emails ==========
    
    def send_usage_limit_warning(
        self,
        user_email: str,
        queries_remaining: int
    ) -> bool:
        """
        Send warning when user reaches 80% of daily query limit.
        
        Args:
            user_email: User's email address
            queries_remaining: Number of queries remaining for the day
            
        Returns:
            True if sent successfully
        """
        subject = "⚠️ Query Limit Warning - VidyaTid"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning-box {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }}
                .progress-bar {{ background: #e5e7eb; border-radius: 10px; height: 20px; margin: 20px 0; overflow: hidden; }}
                .progress-fill {{ background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); height: 100%; width: 80%; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ Query Limit Warning</h1>
                    <p>You're running low on queries</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>You've used 80% of your daily query limit. You have <strong>{queries_remaining} queries remaining</strong> for today.</p>
                    
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    
                    <div class="warning-box">
                        <strong>What happens when you run out?</strong><br>
                        Your queries will reset at midnight UTC. You'll get your full daily quota back then.
                    </div>
                    
                    <p><strong>Want unlimited queries?</strong></p>
                    
                    <p>Upgrade to a higher tier to get more queries per day or unlimited access:</p>
                    <ul>
                        <li><strong>Starter:</strong> 50 queries/day (₹99/month)</li>
                        <li><strong>Premium:</strong> 200 queries/day (₹299/month)</li>
                        <li><strong>Ultimate:</strong> Unlimited queries (₹499/month)</li>
                    </ul>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Upgrade Now</a>
                    </center>
                    
                    <p><strong>Tips to make the most of your remaining queries:</strong></p>
                    <ul>
                        <li>Focus on your most challenging topics</li>
                        <li>Ask comprehensive questions to get detailed answers</li>
                        <li>Review previous answers before asking new questions</li>
                        <li>Use the search feature to find existing explanations</li>
                    </ul>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)
    
    def send_prediction_limit_warning(
        self,
        user_email: str,
        predictions_remaining: int
    ) -> bool:
        """
        Send warning when user is running low on monthly predictions.
        
        Args:
            user_email: User's email address
            predictions_remaining: Number of predictions remaining for the month
            
        Returns:
            True if sent successfully
        """
        subject = "⚠️ Prediction Limit Warning - VidyaTid"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning-box {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ Prediction Limit Warning</h1>
                    <p>Running low on predictions</p>
                </div>
                <div class="content">
                    <p>Hi there,</p>
                    
                    <p>You have <strong>{predictions_remaining} prediction{"s" if predictions_remaining != 1 else ""} remaining</strong> for this month.</p>
                    
                    <div class="warning-box">
                        <strong>What happens when you run out?</strong><br>
                        Your prediction quota will reset on the 1st of next month. Plan your remaining predictions wisely!
                    </div>
                    
                    <p><strong>Need more predictions?</strong></p>
                    
                    <p>Upgrade to get more predictions per month or unlimited access:</p>
                    <ul>
                        <li><strong>Starter:</strong> 2 predictions/month (₹99/month)</li>
                        <li><strong>Premium:</strong> 10 predictions/month (₹299/month)</li>
                        <li><strong>Ultimate:</strong> Unlimited predictions (₹499/month)</li>
                    </ul>
                    
                    <center>
                        <a href="https://vidyatid.com/pricing" class="button">Upgrade Now</a>
                    </center>
                    
                    <p><strong>Make the most of your remaining predictions:</strong></p>
                    <ul>
                        <li>Focus on subjects where you need the most guidance</li>
                        <li>Use chapter analysis to identify weak areas</li>
                        <li>Generate smart papers for targeted practice</li>
                        <li>Review previous predictions before generating new ones</li>
                    </ul>
                    
                    <p>Best regards,<br>
                    <strong>The VidyaTid Team</strong></p>
                </div>
                <div class="footer">
                    <p>VidyaTid - AI-Powered Education Platform</p>
                    <p>© 2025 VidyaTid. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)


# Global instance
_email_service = None


def get_email_service() -> EmailService:
    """Get or create EmailService instance"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
