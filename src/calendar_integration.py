import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Dict, Optional
import re

class CalendarIntegration:
    def __init__(self, credentials_path: str):
        self.creds = self._authenticate(credentials_path)
        self.service = build('calendar', 'v3', credentials=self.creds)
    
    def _authenticate(self, credentials_path: str) -> Credentials:
        """Authenticate with Google Calendar API"""
        # Implementation for OAuth2 flow
        pass
    
    def get_upcoming_meetings(self, days_ahead: int = 3) -> List[Dict]:
        """Fetch meetings for the next N days"""
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        end_time = (datetime.datetime.utcnow() + 
                   datetime.timedelta(days=days_ahead)).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=end_time,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            meetings = []
            for event in events:
                if self._is_external_meeting(event):
                    meeting_data = self._extract_meeting_data(event)
                    meetings.append(meeting_data)
            
            return meetings
            
        except HttpError as error:
            print(f'An error occurred: {error}')
            return []
    
    def _is_external_meeting(self, event: Dict) -> bool:
        """Determine if meeting has external attendees"""
        attendees = event.get('attendees', [])
        if not attendees:
            return False
        
        # Check for external domains
        internal_domain = "@yourcompany.com"
        for attendee in attendees:
            email = attendee.get('email', '')
            if not email.endswith(internal_domain):
                return True
        return False
    
    def _extract_meeting_data(self, event: Dict) -> Dict:
        """Extract relevant meeting information"""
        attendees = event.get('attendees', [])
        external_attendees = []
        
        for attendee in attendees:
            email = attendee.get('email', '')
            if not email.endswith("@yourcompany.com"):
                # Extract name from email or display name
                name = attendee.get('displayName', '')
                if not name:
                    name = email.split('@')[0].replace('.', ' ').title()
                
                external_attendees.append({
                    'name': name,
                    'email': email,
                    'organization': self._extract_organization(email)
                })
        
        return {
            'meeting_id': event['id'],
            'title': event.get('summary', 'Meeting'),
            'start_time': event['start'].get('dateTime', event['start'].get('date')),
            'attendees': external_attendees,
            'description': event.get('description', ''),
            'location': event.get('location', '')
        }
    
    def _extract_organization(self, email: str) -> str:
        """Extract organization from email domain"""
        domain = email.split('@')[1] if '@' in email else ''
        # Remove common suffixes
        org = domain.replace('.com', '').replace('.org', '').replace('.gov.in', '')
        return org.title()