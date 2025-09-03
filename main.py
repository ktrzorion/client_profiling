# AI-Powered Meeting Intelligence System
# Core Implementation with Scraping and Report Generation

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
import re

# FastAPI and async
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Web scraping
import requests
from bs4 import BeautifulSoup
import aiohttp
from urllib.parse import quote, urlparse

# Data processing
import pandas as pd
from functools import lru_cache

# OpenAI
from openai import OpenAI

# Database (using SQLite for simplicity, can switch to PostgreSQL)
import sqlite3
from contextlib import contextmanager

# Redis cache (optional - using in-memory cache for now)
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()

# Environment variables (set these in .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
SERP_API_KEY = os.getenv("SERP_API_KEY", "your-serp-api-key")  # For Google search
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your-news-api-key")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Meeting Intelligence System")

# Cache for API responses (TTL = 24 hours)
cache = TTLCache(maxsize=1000, ttl=86400)

# ====================== DATA MODELS ======================

class MeetingRequest(BaseModel):
    attendee_name: str = Field(..., example="Abhishek Singh")
    title: str = Field(..., example="CEO, IndiaAI Mission")
    organization: str = Field(..., example="Government of India")
    meeting_date: Optional[str] = Field(None, example="2024-12-20")
    our_company: str = Field(default="TechCorp", example="Your Company Name")
    our_solutions: List[str] = Field(default=["AI Solutions", "Digital Transformation"])

class PersonProfile(BaseModel):
    name: str
    title: str
    organization: str
    background: Dict[str, Any]
    recent_activities: List[str]
    portfolio: List[str]
    initiatives: List[str]
    raw_data: Dict[str, Any]

class MeetingBrief(BaseModel):
    prospect_info: Dict[str, str]
    key_pitch_points: List[str]
    background_education: List[str]
    recent_highlights: List[str]
    portfolio_departments: List[str]
    major_initiatives: List[str]
    connection_opportunities: List[str]
    generated_at: str
    confidence_score: float

# ====================== DATABASE SETUP ======================

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('meeting_intelligence.db')
    cursor = conn.cursor()
    
    # Profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_hash TEXT UNIQUE,
            name TEXT,
            title TEXT,
            organization TEXT,
            profile_data TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')
    
    # Briefings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS briefings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_hash TEXT,
            briefing_data TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (person_hash) REFERENCES profiles (person_hash)
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    """Database connection context manager"""
    conn = sqlite3.connect('meeting_intelligence.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ====================== WEB SCRAPING MODULE ======================

class IntelligenceScraper:
    """Multi-source web scraping for person intelligence"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_google(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search Google for information about the person"""
        cache_key = f"google_{hashlib.md5(query.encode()).hexdigest()}"
        
        if cache_key in cache:
            return cache[cache_key]
        
        results = []
        try:
            # Using Google Custom Search API (alternative: use SerpAPI or ScraperAPI)
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': SERP_API_KEY,
                'q': query,
                'num': num_results
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', []):
                    results.append({
                        'title': item.get('title'),
                        'link': item.get('link'),
                        'snippet': item.get('snippet')
                    })
        except Exception as e:
            print(f"Google search error: {e}")
        
        # Fallback to web scraping if API fails
        if not results:
            results = self._scrape_google_fallback(query)
        
        cache[cache_key] = results
        return results
    
    def _scrape_google_fallback(self, query: str) -> List[Dict]:
        """Fallback Google scraping method"""
        results = []
        try:
            search_url = f"https://www.google.com/search?q={quote(query)}"
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for result in soup.select('.g')[:5]:
                title_elem = result.select_one('h3')
                link_elem = result.select_one('a')
                snippet_elem = result.select_one('.VwiC3b')
                
                if title_elem and link_elem:
                    results.append({
                        'title': title_elem.text,
                        'link': link_elem.get('href', ''),
                        'snippet': snippet_elem.text if snippet_elem else ''
                    })
        except Exception as e:
            print(f"Google scraping fallback error: {e}")
        
        return results
    
    def scrape_linkedin(self, name: str, organization: str) -> Dict:
        """Scrape LinkedIn for professional information"""
        # Note: LinkedIn has strict anti-scraping measures
        # In production, use LinkedIn API or services like Proxycurl
        
        search_query = f"{name} {organization} site:linkedin.com"
        search_results = self.search_google(search_query, num_results=3)
        
        linkedin_data = {
            'profile_url': None,
            'summary': None,
            'experience': [],
            'education': []
        }
        
        for result in search_results:
            if 'linkedin.com/in/' in result.get('link', ''):
                linkedin_data['profile_url'] = result['link']
                linkedin_data['summary'] = result.get('snippet', '')
                break
        
        return linkedin_data
    
    def scrape_news(self, name: str, organization: str, days_back: int = 90) -> List[Dict]:
        """Fetch recent news articles about the person"""
        cache_key = f"news_{hashlib.md5(f'{name}{organization}'.encode()).hexdigest()}"
        
        if cache_key in cache:
            return cache[cache_key]
        
        articles = []
        try:
            # Using NewsAPI
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{name}" AND "{organization}"',
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': NEWS_API_KEY,
                'pageSize': 10
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'url': article.get('url'),
                        'published_at': article.get('publishedAt'),
                        'source': article.get('source', {}).get('name')
                    })
        except Exception as e:
            print(f"News API error: {e}")
        
        # Fallback to Google News search
        if not articles:
            query = f"{name} {organization} news"
            search_results = self.search_google(query, num_results=5)
            for result in search_results:
                articles.append({
                    'title': result.get('title'),
                    'description': result.get('snippet'),
                    'url': result.get('link'),
                    'published_at': None,
                    'source': urlparse(result.get('link', '')).netloc
                })
        
        cache[cache_key] = articles
        return articles
    
    def scrape_organization_website(self, organization: str, person_name: str) -> Dict:
        """Scrape organization's official website for person information"""
        org_data = {
            'official_bio': None,
            'department': None,
            'responsibilities': []
        }
        
        # Search for person on organization website
        query = f"{person_name} site:{organization.lower().replace(' ', '')}.com OR site:{organization.lower().replace(' ', '')}.org"
        search_results = self.search_google(query, num_results=3)
        
        for result in search_results:
            if any(keyword in result.get('snippet', '').lower() 
                   for keyword in ['director', 'chief', 'head', 'lead', 'officer']):
                org_data['official_bio'] = result.get('snippet')
                break
        
        return org_data
    
    def aggregate_data(self, name: str, title: str, organization: str) -> Dict:
        """Aggregate data from all sources"""
        print(f"Scraping data for {name}...")
        
        # Parallel scraping can be implemented with asyncio
        data = {
            'person': {
                'name': name,
                'title': title,
                'organization': organization
            },
            'linkedin': self.scrape_linkedin(name, organization),
            'news': self.scrape_news(name, organization),
            'organization': self.scrape_organization_website(organization, name),
            'google_results': self.search_google(f"{name} {title} {organization}", num_results=10)
        }
        
        return data

# ====================== AI ANALYSIS MODULE ======================

class AIAnalyzer:
    """GPT-4 powered analysis engine"""
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model  # Use gpt-4 for better results
    
    def extract_background(self, raw_data: Dict) -> Dict:
        """Extract and structure background information"""
        
        prompt = f"""
        Based on the following information about {raw_data['person']['name']}:
        
        LinkedIn: {json.dumps(raw_data.get('linkedin', {}), indent=2)}
        Organization: {json.dumps(raw_data.get('organization', {}), indent=2)}
        Search Results: {json.dumps(raw_data.get('google_results', [])[:3], indent=2)}
        
        Extract and provide:
        1. Educational background (institutions, degrees, years if available)
        2. Career progression (key positions, companies, timeline)
        3. Areas of expertise
        4. Notable achievements
        5. Current responsibilities
        
        Format as JSON with keys: education, career_progression, expertise, achievements, current_role
        """
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing professional backgrounds and extracting key information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            # Parse JSON response
            try:
                return json.loads(result)
            except:
                return {"raw_response": result}
                
        except Exception as e:
            print(f"AI extraction error: {e}")
            return {}
    
    def analyze_recent_activities(self, raw_data: Dict) -> List[str]:
        """Analyze recent activities and engagements"""
        
        news_items = raw_data.get('news', [])
        news_text = "\n".join([f"- {item['title']}: {item['description']}" 
                               for item in news_items[:5]])
        
        prompt = f"""
        Based on these recent news and activities about {raw_data['person']['name']}:
        
        {news_text}
        
        Provide 5 key recent activities, speeches, or initiatives in bullet points.
        Focus on the most impactful and recent items.
        Each point should be specific and actionable for a business meeting context.
        """
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying key professional activities and initiatives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            # Parse bullet points
            activities = [line.strip('- •').strip() 
                         for line in result.split('\n') 
                         if line.strip().startswith(('-', '•'))]
            
            return activities[:5]
            
        except Exception as e:
            print(f"AI recent activities error: {e}")
            return []
    
    def generate_pitch_points(self, person_data: Dict, company_context: Dict) -> List[str]:
        """Generate 5 personalized pitch points"""
        
        prompt = f"""
        You are preparing a sales meeting brief. Generate 5 highly personalized pitch points for meeting with:
        
        Person: {person_data['person']['name']}
        Title: {person_data['person']['title']}
        Organization: {person_data['person']['organization']}
        
        Background: {json.dumps(person_data.get('background', {}), indent=2)}
        Recent Activities: {json.dumps(person_data.get('recent_activities', []), indent=2)}
        
        Our Company: {company_context['company']}
        Our Solutions: {', '.join(company_context['solutions'])}
        
        Create 5 strategic talking points that:
        1. Connect our solutions to their specific initiatives
        2. Reference their recent work or statements
        3. Align with their organization's goals
        4. Show understanding of their challenges
        5. Propose specific value propositions
        
        Make each point specific, actionable, and directly relevant to this person.
        Format as 5 clear bullet points.
        """
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales strategist who creates highly personalized pitch points."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=600
            )
            
            result = response.choices[0].message.content
            # Parse bullet points
            points = [line.strip('- •').strip() 
                     for line in result.split('\n') 
                     if line.strip().startswith(('-', '•', '1.', '2.', '3.', '4.', '5.'))]
            
            # Clean up numbering if present
            points = [re.sub(r'^\d+\.\s*', '', point) for point in points]
            
            return points[:5]
            
        except Exception as e:
            print(f"AI pitch points error: {e}")
            return ["Error generating pitch points"]
    
    def identify_connections(self, person_data: Dict, company_context: Dict) -> List[str]:
        """Identify connection opportunities and mutual interests"""
        
        prompt = f"""
        Identify connection opportunities between our company and {person_data['person']['name']}:
        
        Their Background: {json.dumps(person_data.get('background', {}), indent=2)}
        Their Organization: {person_data['person']['organization']}
        Their Initiatives: {json.dumps(person_data.get('initiatives', []), indent=2)}
        
        Our Company: {company_context['company']}
        Our Solutions: {', '.join(company_context['solutions'])}
        
        Provide 5 connection opportunities such as:
        - Shared alumni networks or educational backgrounds
        - Regional or geographic alignments
        - Common technology interests
        - Partnership opportunities
        - Mutual contacts or organizations
        - Industry initiatives alignment
        
        Be specific and actionable.
        """
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying business connection opportunities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=400
            )
            
            result = response.choices[0].message.content
            connections = [line.strip('- •').strip() 
                          for line in result.split('\n') 
                          if line.strip().startswith(('-', '•'))]
            
            return connections[:5]
            
        except Exception as e:
            print(f"AI connections error: {e}")
            return []

# ====================== BRIEFING GENERATOR ======================

class BriefingGenerator:
    """Generate formatted meeting briefs"""
    
    def __init__(self):
        self.scraper = IntelligenceScraper()
        self.analyzer = AIAnalyzer()
    
    async def generate_brief(self, request: MeetingRequest) -> MeetingBrief:
        """Generate complete meeting brief"""
        
        # Step 1: Aggregate data from all sources
        raw_data = self.scraper.aggregate_data(
            request.attendee_name,
            request.title,
            request.organization
        )
        
        # Step 2: AI Analysis
        background = self.analyzer.extract_background(raw_data)
        recent_activities = self.analyzer.analyze_recent_activities(raw_data)
        
        # Add to raw_data for pitch point generation
        raw_data['background'] = background
        raw_data['recent_activities'] = recent_activities
        
        # Step 3: Generate personalized content
        company_context = {
            'company': request.our_company,
            'solutions': request.our_solutions
        }
        
        pitch_points = self.analyzer.generate_pitch_points(raw_data, company_context)
        connections = self.analyzer.identify_connections(raw_data, company_context)
        
        # Step 4: Extract structured information
        portfolio = self._extract_portfolio(raw_data)
        initiatives = self._extract_initiatives(raw_data)
        
        # Step 5: Format the brief
        brief = MeetingBrief(
            prospect_info={
                'name': request.attendee_name,
                'title': request.title,
                'organization': request.organization,
                'meeting_date': request.meeting_date or 'TBD'
            },
            key_pitch_points=pitch_points,
            background_education=self._format_background(background),
            recent_highlights=recent_activities,
            portfolio_departments=portfolio,
            major_initiatives=initiatives,
            connection_opportunities=connections,
            generated_at=datetime.now().isoformat(),
            confidence_score=self._calculate_confidence(raw_data)
        )
        
        # Step 6: Store in database
        self._store_brief(request, brief, raw_data)
        
        return brief
    
    def _extract_portfolio(self, raw_data: Dict) -> List[str]:
        """Extract portfolio and department information"""
        portfolio = []
        
        # From organization data
        if raw_data.get('organization', {}).get('responsibilities'):
            portfolio.extend(raw_data['organization']['responsibilities'])
        
        # From title parsing
        title = raw_data['person']['title']
        if title:
            portfolio.append(f"Primary Role: {title}")
        
        # From background analysis
        if raw_data.get('background', {}).get('current_role'):
            portfolio.append(raw_data['background']['current_role'])
        
        return list(set(portfolio))[:5]  # Deduplicate and limit
    
    def _extract_initiatives(self, raw_data: Dict) -> List[str]:
        """Extract major initiatives and projects"""
        initiatives = []
        
        # From recent activities
        if raw_data.get('recent_activities'):
            initiatives.extend(raw_data['recent_activities'][:3])
        
        # From news
        for article in raw_data.get('news', [])[:3]:
            if 'launch' in article.get('title', '').lower() or \
               'initiative' in article.get('title', '').lower():
                initiatives.append(article['title'])
        
        return initiatives[:5]
    
    def _format_background(self, background: Dict) -> List[str]:
        """Format background information into bullet points"""
        formatted = []
        
        if background.get('education'):
            if isinstance(background['education'], list):
                formatted.extend([f"Education: {edu}" for edu in background['education'][:2]])
            else:
                formatted.append(f"Education: {background['education']}")
        
        if background.get('career_progression'):
            if isinstance(background['career_progression'], list):
                formatted.extend(background['career_progression'][:3])
            else:
                formatted.append(background['career_progression'])
        
        if background.get('achievements'):
            if isinstance(background['achievements'], list):
                formatted.extend([f"Achievement: {ach}" for ach in background['achievements'][:2]])
        
        return formatted[:5]
    
    def _calculate_confidence(self, raw_data: Dict) -> float:
        """Calculate confidence score based on data availability"""
        score = 0.0
        
        # Check data completeness
        if raw_data.get('linkedin', {}).get('profile_url'):
            score += 0.2
        if raw_data.get('news', []):
            score += 0.2
        if raw_data.get('organization', {}).get('official_bio'):
            score += 0.2
        if raw_data.get('google_results', []):
            score += 0.2
        if raw_data.get('background', {}):
            score += 0.2
        
        return min(score, 1.0)
    
    def _store_brief(self, request: MeetingRequest, brief: MeetingBrief, raw_data: Dict):
        """Store brief in database"""
        person_hash = hashlib.md5(
            f"{request.attendee_name}{request.organization}".encode()
        ).hexdigest()
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Store or update profile
            cursor.execute('''
                INSERT OR REPLACE INTO profiles 
                (person_hash, name, title, organization, profile_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_hash,
                request.attendee_name,
                request.title,
                request.organization,
                json.dumps(raw_data),
                datetime.now(),
                datetime.now()
            ))
            
            # Store briefing
            cursor.execute('''
                INSERT INTO briefings (person_hash, briefing_data, created_at)
                VALUES (?, ?, ?)
            ''', (
                person_hash,
                json.dumps(brief.dict()),
                datetime.now()
            ))
            
            conn.commit()

# ====================== API ENDPOINTS ======================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    print("Meeting Intelligence System started successfully!")

@app.post("/api/generate-brief", response_model=MeetingBrief)
async def generate_meeting_brief(
    request: MeetingRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a comprehensive meeting brief for an attendee
    """
    try:
        generator = BriefingGenerator()
        brief = await generator.generate_brief(request)
        return brief
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quick-profile")
async def get_quick_profile(request: MeetingRequest):
    """
    Get a quick profile without full analysis (cached data only)
    """
    person_hash = hashlib.md5(
        f"{request.attendee_name}{request.organization}".encode()
    ).hexdigest()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT profile_data FROM profiles WHERE person_hash = ?",
            (person_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            return json.loads(row[0])
        else:
            return {"message": "No cached profile found. Please generate a brief first."}

@app.get("/api/recent-briefs")
async def get_recent_briefs(limit: int = 10):
    """
    Get recently generated briefs
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT b.briefing_data, b.created_at, p.name, p.organization
            FROM briefings b
            JOIN profiles p ON b.person_hash = p.person_hash
            ORDER BY b.created_at DESC
            LIMIT ?
        ''', (limit,))
        
        briefs = []
        for row in cursor.fetchall():
            brief_data = json.loads(row[0])
            brief_data['created_at'] = row[1]
            brief_data['person_name'] = row[2]
            brief_data['organization'] = row[3]
            briefs.append(brief_data)
        
        return briefs

@app.post("/api/batch-briefs")
async def generate_batch_briefs(requests: List[MeetingRequest]):
    """
    Generate briefs for multiple attendees
    """
    generator = BriefingGenerator()
    results = []
    
    for request in requests:
        try:
            brief = await generator.generate_brief(request)
            results.append({
                "status": "success",
                "person": request.attendee_name,
                "brief": brief
            })
        except Exception as e:
            results.append({
                "status": "error",
                "person": request.attendee_name,
                "error": str(e)
            })
    
    return results

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(cache),
        "database": "connected"
    }

# ====================== EXPORT FUNCTIONALITY ======================

@app.get("/api/export-brief/{person_name}")
async def export_brief(person_name: str, format: str = "json"):
    """
    Export a brief in different formats (json, html, markdown)
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT b.briefing_data 
            FROM briefings b
            JOIN profiles p ON b.person_hash = p.person_hash
            WHERE p.name = ?
            ORDER BY b.created_at DESC
            LIMIT 1
        ''', (person_name,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Brief not found")
        
        brief_data = json.loads(row[0])
        
        if format == "json":
            return brief_data
        elif format == "markdown":
            return {"content": _format_as_markdown(brief_data)}
        elif format == "html":
            return {"content": _format_as_html(brief_data)}
        else:
            raise HTTPException(status_code=400, detail="Invalid format")

def _format_as_markdown(brief: Dict) -> str:
    """Convert brief to markdown format"""
    md = f"""# Meeting Intelligence Brief

## PROSPECT: {brief['prospect_info']['name']}, {brief['prospect_info']['title']}, {brief['prospect_info']['organization']}

### 1. KEY PITCH POINTS
"""
    for point in brief['key_pitch_points']:
        md += f"- {point}\n"
    
    md += "\n### 2. BACKGROUND & EDUCATION\n"
    for item in brief['background_education']:
        md += f"- {item}\n"
    
    md += "\n### 3. RECENT HIGHLIGHTS\n"
    for item in brief['recent_highlights']:
        md += f"- {item}\n"
    
    md += "\n### 4. PORTFOLIO & DEPARTMENTS\n"
    for item in brief['portfolio_departments']:
        md += f"- {item}\n"
    
    md += "\n### 5. MAJOR INITIATIVES\n"
    for item in brief['major_initiatives']:
        md += f"- {item}\n"
    
    md += "\n### CONNECTION OPPORTUNITIES\n"
    for item in brief['connection_opportunities']:
        md += f"- {item}\n"
    
    md += f"\n---\n*Generated: {brief['generated_at']}*\n"
    md += f"*Confidence Score: {brief['confidence_score']:.1%}*\n"
    
    return md

def _format_as_html(brief: Dict) -> str:
    """Convert brief to HTML format"""
    # Convert markdown to HTML (simplified version)
    md_content = _format_as_markdown(brief)
    html = md_content.replace('\n', '<br>\n')
    html = html.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
    html = html.replace('## ', '<h2>').replace('\n', '</h2>\n', 1)
    html = html.replace('### ', '<h3>').replace('\n', '</h3>\n')
    html = html.replace('- ', '<li>').replace('\n', '</li>\n')
    
    return f"""
    <html>
    <head>
        <title>Meeting Brief - {brief['prospect_info']['name']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; border-bottom: 2px solid #007bff; }}
            h2 {{ color: #007bff; }}
            h3 {{ color: #555; }}
            li {{ margin: 10px 0; }}
            .confidence {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        {html}
        <div class="confidence">
            <p>Confidence Score: {brief['confidence_score']:.1%}</p>
        </div>
    </body>
    </html>
    """

# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )