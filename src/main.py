# src/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import yaml
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.orm import declarative_base  # Updated import
from sqlalchemy.orm import sessionmaker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI first (before loading modules that might fail)
app = FastAPI(title="Meeting Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if config exists
config_path = 'config/config.yaml'
if not os.path.exists(config_path):
    logger.warning(f"Config file not found at {config_path}. Creating default config...")
    os.makedirs('config', exist_ok=True)
    
    default_config = {
        'api_keys': {
            'openai': 'your-openai-api-key',
            'google_calendar': 'config/credentials.json',
            'sendgrid': 'your-sendgrid-key',
            'slack': 'your-slack-webhook',
            'linkedin': 'your-linkedin-credentials'
        },
        'data_sources': {
            'priority_order': [
                'official_websites',
                'linkedin',
                'news_articles',
                'press_releases',
                'company_pages'
            ]
        },
        'meeting_settings': {
            'advance_notice_hours': 48,
            'minimum_notice_hours': 24,
            'batch_processing_interval': 3600
        },
        'database': {
            'url': 'sqlite:///meeting_intelligence.db'  # Using SQLite for simplicity
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logger.info("Default config created. Please update with your API keys.")

# Load configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Database setup with updated SQLAlchemy syntax
Base = declarative_base()

class MeetingIntelligence(Base):
    __tablename__ = "meeting_intelligence"
    
    meeting_id = Column(String, primary_key=True)
    attendee_name = Column(String, primary_key=True)
    briefing_data = Column(JSON)
    generated_at = Column(DateTime)
    delivered = Column(String)

# Create database
engine = create_engine(config['database']['url'])
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize components with error handling
components_loaded = {}

try:
    from calendar_integration import CalendarIntegration
    if os.path.exists(config['api_keys']['google_calendar']):
        calendar = CalendarIntegration(config['api_keys']['google_calendar'])
        components_loaded['calendar'] = True
        logger.info("✓ Calendar integration loaded")
    else:
        components_loaded['calendar'] = False
        logger.warning("✗ Calendar credentials not found")
        calendar = None
except Exception as e:
    components_loaded['calendar'] = False
    logger.error(f"✗ Calendar integration failed: {e}")
    calendar = None

try:
    from data_collector import DataCollector
    collector = DataCollector()
    components_loaded['collector'] = True
    logger.info("✓ Data collector loaded")
except Exception as e:
    components_loaded['collector'] = False
    logger.error(f"✗ Data collector failed: {e}")
    collector = None

try:
    from ai_analyzer import AIAnalyzer
    if config['api_keys']['openai'] != 'your-openai-api-key':
        analyzer = AIAnalyzer(config['api_keys']['openai'])
        components_loaded['analyzer'] = True
        logger.info("✓ AI analyzer loaded")
    else:
        components_loaded['analyzer'] = False
        logger.warning("✗ OpenAI API key not configured")
        analyzer = None
except Exception as e:
    components_loaded['analyzer'] = False
    logger.error(f"✗ AI analyzer failed: {e}")
    analyzer = None

try:
    from briefing_generator import BriefingGenerator
    generator = BriefingGenerator()
    components_loaded['generator'] = True
    logger.info("✓ Briefing generator loaded")
except Exception as e:
    components_loaded['generator'] = False
    logger.error(f"✗ Briefing generator failed: {e}")
    generator = None

try:
    from delivery_system import DeliverySystem
    if (config['api_keys']['sendgrid'] != 'your-sendgrid-key' or 
        config['api_keys']['slack'] != 'your-slack-webhook'):
        delivery = DeliverySystem(
            config['api_keys']['sendgrid'],
            config['api_keys']['slack']
        )
        components_loaded['delivery'] = True
        logger.info("✓ Delivery system loaded")
    else:
        components_loaded['delivery'] = False
        logger.warning("✗ Delivery credentials not configured")
        delivery = None
except Exception as e:
    components_loaded['delivery'] = False
    logger.error(f"✗ Delivery system failed: {e}")
    delivery = None

# Pydantic models
class ProcessRequest(BaseModel):
    meeting_id: Optional[str] = None
    attendee_name: Optional[str] = None
    force_refresh: bool = False

class TestDataRequest(BaseModel):
    name: str
    title: Optional[str] = ""
    organization: Optional[str] = ""

# API Endpoints
@app.get("/")
async def health_check():
    """Health check endpoint with component status"""
    return {
        "status": "healthy",
        "service": "Meeting Intelligence System",
        "components": components_loaded,
        "message": "System is running. Some components may need configuration."
    }

@app.get("/setup-instructions")
async def setup_instructions():
    """Provide setup instructions for missing components"""
    instructions = {}
    
    if not components_loaded.get('calendar'):
        instructions['calendar'] = {
            "status": "Not configured",
            "steps": [
                "1. Go to https://console.cloud.google.com/",
                "2. Create a new project or select existing",
                "3. Enable the Google Calendar API",
                "4. Create credentials (OAuth 2.0 Client ID)",
                "5. Download as 'credentials.json'",
                "6. Place in 'config/' directory"
            ]
        }
    
    if not components_loaded.get('analyzer'):
        instructions['openai'] = {
            "status": "Not configured",
            "steps": [
                "1. Go to https://platform.openai.com/api-keys",
                "2. Create an API key",
                "3. Update config/config.yaml with your key"
            ]
        }
    
    if not components_loaded.get('delivery'):
        instructions['delivery'] = {
            "status": "Not configured", 
            "steps": [
                "For SendGrid: Get API key from https://app.sendgrid.com/",
                "For Slack: Create webhook at https://api.slack.com/apps",
                "Update config/config.yaml with credentials"
            ]
        }
    
    return instructions

@app.post("/process_meeting")
async def process_meeting(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process a specific meeting or all upcoming meetings"""
    
    if not calendar:
        raise HTTPException(
            status_code=503, 
            detail="Calendar integration not configured. See /setup-instructions"
        )
    
    if request.meeting_id:
        background_tasks.add_task(process_single_meeting, request.meeting_id)
        return {"status": "processing", "meeting_id": request.meeting_id}
    else:
        background_tasks.add_task(process_all_meetings)
        return {"status": "processing_all_upcoming_meetings"}

@app.get("/upcoming_meetings")
async def get_upcoming_meetings():
    """Get list of upcoming meetings"""
    if not calendar:
        raise HTTPException(
            status_code=503,
            detail="Calendar integration not configured. See /setup-instructions"
        )
    
    try:
        meetings = calendar.get_upcoming_meetings(days_ahead=3)
        return {"count": len(meetings), "meetings": meetings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_data_collection")
async def test_data_collection(request: TestDataRequest):
    """Test data collection for a specific person"""
    if not collector:
        raise HTTPException(
            status_code=503,
            detail="Data collector not available"
        )
    
    try:
        data = collector.collect_person_data(
            name=request.name,
            title=request.title,
            organization=request.organization
        )
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_analysis")
async def test_analysis(request: TestDataRequest):
    """Test the complete analysis pipeline for a person"""
    if not all([collector, analyzer, generator]):
        missing = []
        if not collector: missing.append("collector")
        if not analyzer: missing.append("analyzer")
        if not generator: missing.append("generator")
        
        raise HTTPException(
            status_code=503,
            detail=f"Required components not configured: {', '.join(missing)}"
        )
    
    try:
        # Collect data
        raw_data = collector.collect_person_data(
            name=request.name,
            title=request.title,
            organization=request.organization
        )
        
        # Analyze
        meeting_context = {
            'meeting_title': 'Test Meeting',
            'meeting_date': datetime.now().isoformat(),
            'meeting_description': 'Test meeting for analysis',
            'other_attendees': []
        }
        
        analysis = analyzer.analyze_person_profile(raw_data, meeting_context)
        
        # Generate briefing
        analysis['name'] = request.name
        analysis['title'] = request.title or 'Executive'
        analysis['organization'] = request.organization or 'Unknown'
        
        meeting_data = {
            'title': 'Test Meeting',
            'start_time': datetime.now().isoformat(),
        }
        
        briefing = generator.generate_briefing(analysis, meeting_data)
        
        return {
            "status": "success",
            "briefing": briefing,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"Test analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/briefing/{meeting_id}/{attendee_name}")
async def get_briefing(meeting_id: str, attendee_name: str):
    """Retrieve a generated briefing"""
    db = SessionLocal()
    try:
        briefing = db.query(MeetingIntelligence).filter_by(
            meeting_id=meeting_id,
            attendee_name=attendee_name
        ).first()
        
        if briefing:
            return briefing.briefing_data
        else:
            raise HTTPException(status_code=404, detail="Briefing not found")
    finally:
        db.close()

# Background processing functions
async def process_single_meeting(meeting_id: str):
    """Process intelligence for a single meeting"""
    if not all([calendar, collector, analyzer, generator]):
        logger.error("Required components not available for processing")
        return
    
    try:
        meetings = calendar.get_upcoming_meetings()
        meeting = next((m for m in meetings if m['meeting_id'] == meeting_id), None)
        
        if not meeting:
            logger.warning(f"Meeting {meeting_id} not found")
            return
        
        for attendee in meeting['attendees']:
            await process_attendee(attendee, meeting)
            
    except Exception as e:
        logger.error(f"Error processing meeting {meeting_id}: {e}")

async def process_attendee(attendee: Dict, meeting: Dict):
    """Generate intelligence for a single attendee"""
    
    if not all([collector, analyzer, generator]):
        logger.error("Required components not available")
        return
    
    db = SessionLocal()
    try:
        # Check for existing recent briefing
        existing = db.query(MeetingIntelligence).filter_by(
            meeting_id=meeting['meeting_id'],
            attendee_name=attendee['name']
        ).first()
        
        if existing and existing.generated_at > datetime.now() - timedelta(hours=24):
            logger.info(f"Recent briefing exists for {attendee['name']}")
            return
        
        logger.info(f"Processing {attendee['name']} from {attendee.get('organization', 'Unknown')}")
        
        # Collect data
        raw_data = collector.collect_person_data(
            name=attendee['name'],
            title=attendee.get('title', ''),
            organization=attendee.get('organization', '')
        )
        
        # Analyze
        meeting_context = {
            'meeting_title': meeting['title'],
            'meeting_date': meeting['start_time'],
            'meeting_description': meeting.get('description', ''),
            'other_attendees': [a['name'] for a in meeting['attendees'] if a != attendee]
        }
        
        analysis = analyzer.analyze_person_profile(raw_data, meeting_context)
        
        # Add attendee info
        analysis['name'] = attendee['name']
        analysis['title'] = attendee.get('title', 'Executive')
        analysis['organization'] = attendee.get('organization', 'Unknown')
        
        # Generate briefing
        briefing = generator.generate_briefing(analysis, meeting)
        
        # Store in database
        meeting_intel = MeetingIntelligence(
            meeting_id=meeting['meeting_id'],
            attendee_name=attendee['name'],
            briefing_data=briefing,
            generated_at=datetime.now(),
            delivered='pending'
        )
        
        # Use merge to update or insert
        db.merge(meeting_intel)
        db.commit()
        
        # Deliver if configured
        if delivery:
            delivery_success = delivery.deliver_briefing(
                briefing=briefing,
                recipient='sales-team@yourcompany.com',
                delivery_method='email'
            )
            
            if delivery_success:
                meeting_intel.delivered = 'sent'
                db.commit()
                logger.info(f"Briefing delivered for {attendee['name']}")
        
    except Exception as e:
        logger.error(f"Error processing attendee {attendee['name']}: {e}")
    finally:
        db.close()

async def process_all_meetings():
    """Process all upcoming meetings"""
    if not calendar:
        logger.warning("Calendar not configured, skipping meeting processing")
        return
    
    try:
        meetings = calendar.get_upcoming_meetings(days_ahead=3)
        logger.info(f"Processing {len(meetings)} upcoming meetings")
        
        for meeting in meetings:
            meeting_time = datetime.fromisoformat(
                meeting['start_time'].replace('Z', '+00:00') if 'Z' in meeting['start_time'] 
                else meeting['start_time']
            )
            hours_until_meeting = (meeting_time - datetime.now()).total_seconds() / 3600
            
            min_hours = config['meeting_settings']['minimum_notice_hours']
            max_hours = config['meeting_settings']['advance_notice_hours']
            
            if min_hours < hours_until_meeting < max_hours:
                await process_single_meeting(meeting['meeting_id'])
                await asyncio.sleep(1)  # Rate limiting
        
    except Exception as e:
        logger.error(f"Error processing meetings: {e}")

# Startup message
@app.on_event("startup")
async def startup_event():
    """Display startup information"""
    print("\n" + "="*60)
    print("🚀 Meeting Intelligence System Started")
    print("="*60)
    print(f"📍 API URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"⚙️  Setup: http://localhost:8000/setup-instructions")
    print("\nComponent Status:")
    for component, loaded in components_loaded.items():
        status = "✓" if loaded else "✗"
        print(f"  {status} {component.capitalize()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")