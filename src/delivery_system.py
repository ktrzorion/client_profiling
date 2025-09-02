import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import base64
from typing import Dict, List
import json

class DeliverySystem:
    def __init__(self, sendgrid_key: str, slack_token: str):
        self.sg = sendgrid.SendGridAPIClient(api_key=sendgrid_key)
        self.slack = WebClient(token=slack_token)
    
    def deliver_briefing(self, briefing: Dict, recipient: str, 
                        delivery_method: str = 'email') -> bool:
        """Deliver briefing via specified channel"""
        
        if delivery_method == 'email':
            return self.send_email(briefing, recipient)
        elif delivery_method == 'slack':
            return self.send_slack(briefing, recipient)
        elif delivery_method == 'both':
            email_sent = self.send_email(briefing, recipient)
            slack_sent = self.send_slack(briefing, recipient)
            return email_sent and slack_sent
        
        return False
    
    def send_email(self, briefing: Dict, recipient_email: str) -> bool:
        """Send briefing via email"""
        
        message = Mail(
            from_email='intelligence@yourcompany.com',
            to_emails=recipient_email,
            subject=f"Meeting Intelligence Brief - {briefing['meeting_title']}",
            html_content=briefing['html']
        )
        
        # Attach markdown version as file
        encoded_file = base64.b64encode(briefing['markdown'].encode()).decode()
        attached_file = Attachment(
            FileContent(encoded_file),
            FileName(f"briefing_{briefing['meeting_date']}.md"),
            FileType('text/markdown')
        )
        message.attachment = attached_file
        
        try:
            response = self.sg.send(message)
            return response.status_code == 202
        except Exception as e:
            print(f"Email send error: {e}")
            return False
    
    def send_slack(self, briefing: Dict, channel: str) -> bool:
        """Send briefing via Slack"""
        
        # Format for Slack
        blocks = self.format_for_slack(briefing)
        
        try:
            response = self.slack.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=f"Meeting Intelligence Brief - {briefing['meeting_title']}"
            )
            return response['ok']
        except SlackApiError as e:
            print(f"Slack send error: {e}")
            return False
    
    def format_for_slack(self, briefing: Dict) -> List[Dict]:
        """Format briefing for Slack blocks"""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📋 Meeting Brief: {briefing['name']}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{briefing['title']}* at {briefing['organization']}\n📅 {briefing['meeting_date']}"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*🎯 Key Pitch Points:*\n" + "\n".join([f"• {p}" for p in briefing['pitch_points'][:5]])
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🤝 Top Connection:*\n{briefing['connections']['top_connection']}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Full Brief"},
                        "url": briefing.get('full_brief_url', '#')
                    }
                ]
            }
        ]
        return blocks