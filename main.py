import os
import json
import time
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import re

# FastAPI and async
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Web scraping
import requests
from bs4 import BeautifulSoup
import aiohttp
from urllib.parse import quote, urlparse
import asyncio

# Data processing
import pandas as pd
from functools import lru_cache

# OpenAI
from openai import OpenAI

# Database (using SQLite for simplicity)
import sqlite3
from contextlib import contextmanager

# Cache
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="Meeting Intelligence System",
    description="AI-powered meeting preparation with real-time web scraping",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for API responses (TTL = 6 hours)
cache = TTLCache(maxsize=1000, ttl=21600)

# ====================== DATA MODELS ======================

class MeetingRequest(BaseModel):
    attendee_name: str = Field(..., example="Vinay Krishna Gupta")
    title: str = Field(..., example="CEO")
    organization: str = Field(..., example="Antino Labs Private Limited")
    meeting_date: Optional[str] = Field(None, example="2024-12-20")
    our_company: str = Field(default="TechCorp", example="Your Company Name")
    our_solutions: List[str] = Field(default=["AI Solutions", "Digital Transformation"])

class ScrapingStatus(BaseModel):
    status: str
    progress: int
    current_task: str
    data_found: Dict[str, bool]

class MeetingBrief(BaseModel):
    prospect_info: Dict[str, str]
    key_pitch_points: List[str]
    background_education: List[str]
    recent_highlights: List[str]
    portfolio_departments: List[str]
    major_initiatives: List[str]
    connection_opportunities: List[str]
    data_sources: Dict[str, bool]
    scraping_summary: Dict[str, int]
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
            scraping_status TEXT,
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

# ====================== ENHANCED WEB SCRAPING MODULE ======================

class EnhancedIntelligenceScraper:
    """Production-ready multi-source web scraping"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 1  # Minimum delay between requests in seconds
    
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Async search using DuckDuckGo"""
        print(f"[Scraper] Searching DuckDuckGo for: {query}")
        
        results = []
        try:
            self._rate_limit()
            
            # DuckDuckGo HTML search
            search_url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            response = self.session.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse DuckDuckGo results
                for result in soup.select('.result__body')[:num_results]:
                    title_elem = result.select_one('.result__title a')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if title_elem:
                        results.append({
                            'title': title_elem.text.strip(),
                            'link': title_elem.get('href', ''),
                            'snippet': snippet_elem.text.strip() if snippet_elem else '',
                            'source': 'duckduckgo'
                        })
                
                print(f"[Scraper] Found {len(results)} DuckDuckGo results")
        
        except Exception as e:
            print(f"[Scraper] DuckDuckGo search error: {e}")
        
        return results
    
    async def search_bing(self, query: str, num_results: int = 10) -> List[Dict]:
        """Async search using Bing"""
        print(f"[Scraper] Searching Bing for: {query}")
        
        results = []
        try:
            self._rate_limit()
            
            search_url = "https://www.bing.com/search"
            params = {'q': query, 'count': num_results}
            
            response = self.session.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for result in soup.select('.b_algo')[:num_results]:
                    title_elem = result.select_one('h2 a')
                    snippet_elem = result.select_one('.b_caption p')
                    
                    if title_elem:
                        results.append({
                            'title': title_elem.text.strip(),
                            'link': title_elem.get('href', ''),
                            'snippet': snippet_elem.text.strip() if snippet_elem else '',
                            'source': 'bing'
                        })
                
                print(f"[Scraper] Found {len(results)} Bing results")
        
        except Exception as e:
            print(f"[Scraper] Bing search error: {e}")
        
        return results
    
    async def scrape_company_website(self, company_name: str, person_name: str) -> Dict:
        """Scrape company website for person information"""
        print(f"[Scraper] Scraping company website for: {person_name} at {company_name}")
        
        company_data = {
            'website_found': False,
            'person_bio': None,
            'company_description': None,
            'website_url': None
        }
        
        # Generate potential domain names
        clean_name = re.sub(r'\b(private|limited|ltd|pvt|inc|corp|llc)\b', '', company_name.lower())
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', clean_name).strip()
        
        domain_patterns = [
            f"{clean_name.replace(' ', '')}.com",
            f"{clean_name.replace(' ', '')}.in",
            f"{clean_name.replace(' ', '')}.co",
            f"{clean_name.replace(' ', '-')}.com",
            f"{clean_name.replace(' ', '')}.org"
        ]
        
        for domain in domain_patterns:
            try:
                self._rate_limit()
                base_url = f"https://{domain}"
                
                response = self.session.get(base_url, timeout=15)
                if response.status_code == 200:
                    print(f"[Scraper] Found company website: {base_url}")
                    company_data['website_found'] = True
                    company_data['website_url'] = base_url
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract company description
                    meta_description = soup.find('meta', {'name': 'description'})
                    if meta_description:
                        company_data['company_description'] = meta_description.get('content', '')
                    
                    # Look for person's name on main page
                    page_text = soup.get_text().lower()
                    if person_name.lower() in page_text:
                        company_data['person_bio'] = self._extract_person_context(soup.get_text(), person_name)
                    
                    # Try about/team pages
                    about_urls = ['/about', '/about-us', '/team', '/leadership', '/management']
                    for about_path in about_urls:
                        try:
                            self._rate_limit()
                            about_response = self.session.get(f"{base_url}{about_path}", timeout=10)
                            if about_response.status_code == 200:
                                about_soup = BeautifulSoup(about_response.text, 'html.parser')
                                about_text = about_soup.get_text()
                                
                                if person_name.lower() in about_text.lower():
                                    person_context = self._extract_person_context(about_text, person_name)
                                    if person_context and len(person_context) > len(company_data.get('person_bio', '')):
                                        company_data['person_bio'] = person_context
                        except:
                            continue
                    
                    break
            
            except Exception as e:
                continue
        
        return company_data
    
    def _extract_person_context(self, text: str, person_name: str) -> str:
        """Extract relevant context around person's name"""
        sentences = re.split(r'[.!?]+', text)
        relevant_context = []
        
        name_parts = person_name.lower().split()
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if any part of the name is in the sentence
            if any(name_part in sentence_lower for name_part in name_parts if len(name_part) > 2):
                # Get surrounding context
                sentence_clean = re.sub(r'\s+', ' ', sentence.strip())
                if len(sentence_clean) > 10:  # Ignore very short sentences
                    relevant_context.append(sentence_clean)
        
        return '. '.join(relevant_context[:3]) if relevant_context else None
    
    async def scrape_linkedin_info(self, name: str, organization: str) -> Dict:
        """Scrape LinkedIn information from search results"""
        print(f"[Scraper] Searching for LinkedIn profile: {name}")
        
        linkedin_data = {
            'profile_found': False,
            'profile_url': None,
            'summary': None,
            'title_from_linkedin': None,
            'company_from_linkedin': None
        }
        
        search_queries = [
            f'"{name}" {organization} site:linkedin.com',
            f'"{name}" CEO LinkedIn {organization}',
            f'{name} LinkedIn profile {organization}'
        ]
        
        for query in search_queries:
            # Search both engines
            duckduckgo_results = await self.search_duckduckgo(query, num_results=5)
            bing_results = await self.search_bing(query, num_results=5)
            
            all_results = duckduckgo_results + bing_results
            
            for result in all_results:
                link = result.get('link', '')
                if 'linkedin.com/in/' in link:
                    linkedin_data['profile_found'] = True
                    linkedin_data['profile_url'] = link
                    linkedin_data['summary'] = result.get('snippet', '')
                    
                    # Extract title and company from snippet
                    snippet = result.get('snippet', '')
                    # Look for patterns like "CEO at Company" or "Title | Company"
                    patterns = [
                        r'(.+?)\s+at\s+(.+?)(?:\.|$)',
                        r'(.+?)\s*\|\s*(.+?)(?:\.|$)',
                        r'(.+?),\s*(.+?)(?:\.|$)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, snippet, re.IGNORECASE)
                        if match:
                            linkedin_data['title_from_linkedin'] = match.group(1).strip()
                            linkedin_data['company_from_linkedin'] = match.group(2).strip()
                            break
                    
                    return linkedin_data
        
        return linkedin_data
    
    async def scrape_news_articles(self, name: str, organization: str, days_back: int = 180) -> List[Dict]:
        """Scrape news articles about the person"""
        print(f"[Scraper] Searching for news articles about: {name}")
        
        articles = []
        
        # Enhanced news search queries
        news_queries = [
            f'"{name}" {organization} news',
            f'"{name}" CEO announcement funding',
            f'{name} {organization} interview startup',
            f'"{name}" investment round',
            f'{organization} "{name}" launch product',
            f'"{name}" {organization} growth expansion'
        ]
        
        seen_urls = set()
        
        for query in news_queries[:3]:  # Limit queries to avoid too many requests
            # Search both engines
            duckduckgo_results = await self.search_duckduckgo(query, num_results=8)
            bing_results = await self.search_bing(query, num_results=5)
            
            all_results = duckduckgo_results + bing_results
            
            for result in all_results:
                link = result.get('link', '')
                
                # Skip if already processed
                if link in seen_urls:
                    continue
                seen_urls.add(link)
                
                # Filter for news and business sources
                news_domains = [
                    'techcrunch.com', 'economic', 'business', 'forbes.com', 
                    'reuters.com', 'bloomberg.com', 'mint.com', 'zeebiz.com',
                    'moneycontrol.com', 'startupstory', 'yourstory.com',
                    'inc42.com', 'entrackr.com', 'indianstartupnews.com',
                    'trak.in', 'medianama.com', 'news18.com', 'hindustantimes.com'
                ]
                
                if any(domain in link.lower() for domain in news_domains) or 'news' in link.lower():
                    # Extract date from title or snippet if possible
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    
                    # Look for recent indicators
                    recent_indicators = ['2024', '2023', 'recently', 'latest', 'new', 'announces']
                    is_recent = any(indicator in (title + snippet).lower() for indicator in recent_indicators)
                    
                    articles.append({
                        'title': title,
                        'description': snippet,
                        'url': link,
                        'source': urlparse(link).netloc,
                        'published_at': 'Recent' if is_recent else 'Unknown',
                        'relevance_score': self._calculate_relevance(title + snippet, name, organization)
                    })
        
        # Sort by relevance score
        articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"[Scraper] Found {len(articles)} news articles")
        return articles[:15]  # Return top 15 most relevant
    
    def _calculate_relevance(self, text: str, name: str, organization: str) -> int:
        """Calculate relevance score for news articles"""
        score = 0
        text_lower = text.lower()
        
        # Name mentions
        if name.lower() in text_lower:
            score += 10
        
        # Organization mentions
        if organization.lower() in text_lower:
            score += 8
        
        # Business keywords
        business_keywords = ['funding', 'investment', 'launch', 'CEO', 'startup', 'company', 
                           'business', 'growth', 'expansion', 'partnership', 'deal']
        for keyword in business_keywords:
            if keyword in text_lower:
                score += 2
        
        return score
    
    async def scrape_additional_profiles(self, name: str, organization: str) -> Dict:
        """Scrape additional professional profiles"""
        print(f"[Scraper] Searching for additional profiles: {name}")
        
        profiles_data = {
            'crunchbase_found': False,
            'other_profiles': []
        }
        
        # Search for professional profiles
        profile_queries = [
            f'"{name}" {organization} crunchbase',
            f'"{name}" {organization} bloomberg',
            f'"{name}" profile {organization}'
        ]
        
        for query in profile_queries:
            results = await self.search_duckduckgo(query, num_results=5)
            
            for result in results:
                link = result.get('link', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                # Check for specific profile sites
                if 'crunchbase.com' in link:
                    profiles_data['crunchbase_found'] = True
                
                profile_sites = ['crunchbase.com', 'bloomberg.com', 'forbes.com', 
                               'angel.co', 'f6s.com', 'startup.com']
                
                if any(site in link for site in profile_sites):
                    profiles_data['other_profiles'].append({
                        'title': title,
                        'snippet': snippet,
                        'url': link,
                        'platform': urlparse(link).netloc
                    })
        
        return profiles_data
    
    async def aggregate_data(self, name: str, title: str, organization: str) -> Dict:
        """Enhanced async data aggregation from multiple sources"""
        print(f"[Scraper] Starting comprehensive data collection for {name}...")
        
        data = {
            'person': {
                'name': name,
                'title': title,
                'organization': organization
            },
            'general_search': [],
            'company_website': {},
            'linkedin': {},
            'news_articles': [],
            'other_profiles': {},
            'scraping_stats': {
                'sources_attempted': 0,
                'sources_successful': 0,
                'total_results': 0
            }
        }
        
        try:
            # Track scraping attempts
            data['scraping_stats']['sources_attempted'] = 5
            
            # 1. General search results
            print("[Scraper] Phase 1: General search...")
            general_query = f'"{name}" {title} {organization}'
            general_results = await self.search_duckduckgo(general_query, num_results=15)
            if general_results:
                data['general_search'] = general_results
                data['scraping_stats']['sources_successful'] += 1
                data['scraping_stats']['total_results'] += len(general_results)
            
            # 2. Company website scraping
            print("[Scraper] Phase 2: Company website...")
            company_data = await self.scrape_company_website(organization, name)
            data['company_website'] = company_data
            if company_data.get('website_found'):
                data['scraping_stats']['sources_successful'] += 1
            
            # 3. LinkedIn information
            print("[Scraper] Phase 3: LinkedIn search...")
            linkedin_data = await self.scrape_linkedin_info(name, organization)
            data['linkedin'] = linkedin_data
            if linkedin_data.get('profile_found'):
                data['scraping_stats']['sources_successful'] += 1
            
            # 4. News articles
            print("[Scraper] Phase 4: News articles...")
            news_articles = await self.scrape_news_articles(name, organization)
            data['news_articles'] = news_articles
            if news_articles:
                data['scraping_stats']['sources_successful'] += 1
                data['scraping_stats']['total_results'] += len(news_articles)
            
            # 5. Additional profiles
            print("[Scraper] Phase 5: Additional profiles...")
            other_profiles = await self.scrape_additional_profiles(name, organization)
            data['other_profiles'] = other_profiles
            if other_profiles.get('other_profiles'):
                data['scraping_stats']['sources_successful'] += 1
            
            success_rate = (data['scraping_stats']['sources_successful'] / 
                          data['scraping_stats']['sources_attempted']) * 100
            
            print(f"[Scraper] Data collection completed for {name}")
            print(f"  - Success rate: {success_rate:.1f}%")
            print(f"  - Total results: {data['scraping_stats']['total_results']}")
            print(f"  - LinkedIn found: {data['linkedin'].get('profile_found', False)}")
            print(f"  - Company website: {data['company_website'].get('website_found', False)}")
            print(f"  - News articles: {len(data['news_articles'])}")
            
        except Exception as e:
            print(f"[Scraper] Error in data aggregation: {e}")
        
        return data

# ====================== AI ANALYSIS MODULE ======================

class EnhancedAIAnalyzer:
    """Enhanced GPT-powered analysis engine"""

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def extract_background(self, raw_data: Dict) -> Dict:
        """Extract and structure background information from scraped data"""
        print("[AIAnalyzer] Extracting background for:", raw_data['person']['name'])

        # Compile all available text data
        text_data = []
        
        # From general search
        for result in raw_data.get('general_search', [])[:5]:
            text_data.append(f"Search result: {result.get('title', '')} - {result.get('snippet', '')}")
        
        # From LinkedIn
        if raw_data.get('linkedin', {}).get('summary'):
            text_data.append(f"LinkedIn: {raw_data['linkedin']['summary']}")
        
        # From company website
        if raw_data.get('company_website', {}).get('person_bio'):
            text_data.append(f"Company bio: {raw_data['company_website']['person_bio']}")
        
        # From news articles
        for article in raw_data.get('news_articles', [])[:3]:
            text_data.append(f"News: {article.get('title', '')} - {article.get('description', '')}")

        compiled_text = "\n".join(text_data)

        prompt = f"""
        Based on the following scraped information about {raw_data['person']['name']}:

        {compiled_text}

        Extract and provide structured information about:
        1. Educational background (degrees, institutions)
        2. Career progression (previous roles, companies)
        3. Areas of expertise and specialization
        4. Notable achievements and recognition
        5. Current responsibilities and role description

        Respond with valid JSON only:
        {{
            "education": ["degree/institution info"],
            "career_progression": ["previous role at company", "another role"],
            "expertise": ["area1", "area2"],
            "achievements": ["achievement1", "achievement2"],
            "current_role": "detailed description of current position"
        }}
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing professional backgrounds. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )

            result = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure valid JSON
            result = result.replace('```json', '').replace('```', '').strip()
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                print(f"[AIAnalyzer] JSON parsing error, raw response: {result[:200]}")
                return {
                    "education": ["Information not clearly available"],
                    "career_progression": ["Current role details extracted from available data"],
                    "expertise": ["Leadership", "Business Development"],
                    "achievements": ["Leading " + raw_data['person']['organization']],
                    "current_role": f"{raw_data['person']['title']} at {raw_data['person']['organization']}"
                }

        except Exception as e:
            print(f"[AIAnalyzer] AI extraction error: {e}")
            return {
                "education": ["Background information being researched"],
                "career_progression": ["Professional experience in leadership roles"],
                "expertise": ["Business leadership", "Strategic planning"],
                "achievements": ["Current leadership position"],
                "current_role": f"{raw_data['person']['title']} at {raw_data['person']['organization']}"
            }

    def analyze_recent_activities(self, raw_data: Dict) -> List[str]:
        """Analyze recent activities from news and search results"""
        print("[AIAnalyzer] Analyzing recent activities...")

        # Compile recent activity data
        activity_data = []
        
        # From news articles
        for article in raw_data.get('news_articles', [])[:8]:
            activity_data.append(f"News: {article.get('title', '')} - {article.get('description', '')}")
        
        # From general search results
        for result in raw_data.get('general_search', [])[:5]:
            if any(keyword in result.get('snippet', '').lower() 
                   for keyword in ['recently', '2024', '2023', 'announced', 'launched', 'joined']):
                activity_data.append(f"Recent: {result.get('title', '')} - {result.get('snippet', '')}")

        if not activity_data:
            return [f"Currently serving as {raw_data['person']['title']} at {raw_data['person']['organization']}"]

        compiled_activities = "\n".join(activity_data)

        prompt = f"""
        Based on recent news and activities about {raw_data['person']['name']}:

        {compiled_activities}

        Identify the 5 most important recent activities, announcements, or initiatives. Focus on:
        - Business developments
        - Product launches  
        - Funding announcements
        - Speaking engagements
        - Strategic partnerships
        - Leadership changes

        Provide exactly 5 bullet points, each describing a specific recent activity.
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying key professional activities from news and business information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )

            result = response.choices[0].message.content
            activities = []
            
            # Parse bullet points from response
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                # Remove bullet point markers
                line = re.sub(r'^[-•*]\s*', '', line)
                line = re.sub(r'^\d+\.\s*', '', line)
                
                if len(line) > 10 and not line.startswith(('Based on', 'Here are', 'The following')):
                    activities.append(line)
                    
                if len(activities) >= 5:
                    break

            if not activities:
                activities = [f"Currently leading {raw_data['person']['organization']} as {raw_data['person']['title']}"]

            return activities[:5]

        except Exception as e:
            print(f"[AIAnalyzer] Recent activities error: {e}")
            return [f"Active in leadership role at {raw_data['person']['organization']}"]

    def generate_pitch_points(self, person_data: Dict, company_context: Dict) -> List[str]:
        """Generate personalized pitch points based on scraped data"""
        print("[AIAnalyzer] Generating pitch points...")

        # Compile comprehensive person profile
        profile_summary = []
        
        # Background info
        background = person_data.get('background', {})
        if background.get('current_role'):
            profile_summary.append(f"Current role: {background['current_role']}")
        
        if background.get('expertise'):
            profile_summary.append(f"Expertise: {', '.join(background['expertise'])}")
        
        # Recent activities
        recent_activities = person_data.get('recent_activities', [])
        if recent_activities:
            profile_summary.append(f"Recent activities: {'; '.join(recent_activities[:3])}")
        
        # Company info
        company_info = person_data.get('company_website', {})
        if company_info.get('company_description'):
            profile_summary.append(f"Company: {company_info['company_description']}")

        profile_text = "\n".join(profile_summary)
        solutions = [str(s) for s in company_context.get('solutions', [])]

        prompt = f"""
        Create 5 highly personalized pitch points for a sales meeting with:
        
        Person: {person_data['person']['name']}
        Title: {person_data['person']['title']}  
        Organization: {person_data['person']['organization']}
        
        Their Profile:
        {profile_text}
        
        Our Company: {company_context['company']}
        Our Solutions: {', '.join(solutions)}
        
        Generate 5 compelling, personalized pitch points that:
        1. Reference specific aspects of their background/role
        2. Connect our solutions to their likely pain points
        3. Demonstrate understanding of their industry/challenges
        4. Show clear value proposition
        5. Are actionable and specific
        
        Make each point 2-3 sentences maximum.
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales strategist creating personalized pitch points."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )

            result = response.choices[0].message.content
            points = []
            
            # Parse points from response
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                line = re.sub(r'^[-•*]\s*', '', line)
                line = re.sub(r'^\d+\.\s*', '', line)
                
                if len(line) > 20 and not line.startswith(('Here are', 'Based on')):
                    points.append(line)
                    
                if len(points) >= 5:
                    break

            return points[:5] if points else [f"Our {solutions[0] if solutions else 'solutions'} can help {person_data['person']['organization']} achieve their strategic goals."]

        except Exception as e:
            print(f"[AIAnalyzer] Pitch points error: {e}")
            return [f"Partnership opportunity with {company_context['company']} for {person_data['person']['organization']}"]

    def identify_connections(self, person_data: Dict, company_context: Dict) -> List[str]:
        """Identify strategic connection opportunities"""
        print("[AIAnalyzer] Identifying connection opportunities...")

        # Compile all available data for connection analysis
        connection_context = []
        
        # Add their initiatives and activities
        recent_activities = person_data.get('recent_activities', [])
        if recent_activities:
            connection_context.append(f"Recent initiatives: {'; '.join(recent_activities)}")
        
        # Add their expertise
        background = person_data.get('background', {})
        if background.get('expertise'):
            connection_context.append(f"Areas of expertise: {', '.join(background['expertise'])}")
            
        # Add company information
        company_info = person_data.get('company_website', {})
        if company_info.get('company_description'):
            connection_context.append(f"Company focus: {company_info['company_description']}")

        context_text = "\n".join(connection_context)
        solutions = [str(s) for s in company_context.get('solutions', [])]

        prompt = f"""
        Identify 5 strategic connection opportunities between our companies:
        
        Their Background:
        {context_text}
        
        Our Company: {company_context['company']}
        Our Solutions: {', '.join(solutions)}
        
        Find specific areas where we can:
        1. Solve their current challenges
        2. Support their growth initiatives  
        3. Enhance their existing operations
        4. Create mutual value
        5. Build long-term partnership
        
        Be specific about HOW we can help, not just generic statements.
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying strategic business connections and partnerships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=600
            )

            result = response.choices[0].message.content
            connections = []
            
            # Parse connections from response
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                line = re.sub(r'^[-•*]\s*', '', line)
                line = re.sub(r'^\d+\.\s*', '', line)
                
                if len(line) > 15 and not line.startswith(('Here are', 'Based on')):
                    connections.append(line)
                    
                if len(connections) >= 5:
                    break

            return connections[:5] if connections else [f"Explore partnership opportunities in {solutions[0] if solutions else 'technology solutions'}"]

        except Exception as e:
            print(f"[AIAnalyzer] Connections error: {e}")
            return [f"Strategic partnership potential with {person_data['person']['organization']}"]

# ====================== ENHANCED BRIEFING GENERATOR ======================

class EnhancedBriefingGenerator:
    """Enhanced briefing generator with real scraping"""
    
    def __init__(self):
        self.scraper = EnhancedIntelligenceScraper()
        self.analyzer = EnhancedAIAnalyzer()
    
    async def generate_brief(self, request: MeetingRequest) -> MeetingBrief:
        """Generate comprehensive meeting brief with real data"""
        print(f"[BriefingGenerator] Starting brief generation for {request.attendee_name}")
        
        # Step 1: Scrape data from all sources
        raw_data = await self.scraper.aggregate_data(
            request.attendee_name,
            request.title,
            request.organization
        )
        
        # Step 2: AI Analysis of scraped data
        background = self.analyzer.extract_background(raw_data)
        recent_activities = self.analyzer.analyze_recent_activities(raw_data)
        
        # Add analyzed data back to raw_data for pitch generation
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
        background_formatted = self._format_background(background)
        
        # Step 5: Determine data sources and confidence
        data_sources = {
            'linkedin_found': raw_data.get('linkedin', {}).get('profile_found', False),
            'company_website_found': raw_data.get('company_website', {}).get('website_found', False),
            'news_articles_found': len(raw_data.get('news_articles', [])) > 0,
            'general_search_found': len(raw_data.get('general_search', [])) > 0,
            'other_profiles_found': len(raw_data.get('other_profiles', {}).get('other_profiles', [])) > 0
        }
        
        scraping_summary = {
            'total_sources_checked': raw_data.get('scraping_stats', {}).get('sources_attempted', 0),
            'successful_sources': raw_data.get('scraping_stats', {}).get('sources_successful', 0),
            'total_results_found': raw_data.get('scraping_stats', {}).get('total_results', 0),
            'news_articles_count': len(raw_data.get('news_articles', [])),
            'search_results_count': len(raw_data.get('general_search', []))
        }
        
        confidence_score = self._calculate_confidence_score(raw_data, data_sources)
        
        # Step 6: Create the final brief
        brief = MeetingBrief(
            prospect_info={
                'name': request.attendee_name,
                'title': request.title,
                'organization': request.organization,
                'meeting_date': request.meeting_date or 'TBD'
            },
            key_pitch_points=pitch_points,
            background_education=background_formatted,
            recent_highlights=recent_activities,
            portfolio_departments=portfolio,
            major_initiatives=initiatives,
            connection_opportunities=connections,
            data_sources=data_sources,
            scraping_summary=scraping_summary,
            generated_at=datetime.now().isoformat(),
            confidence_score=confidence_score
        )
        
        # Step 7: Store in database
        await self._store_brief(request, brief, raw_data)
        
        print(f"[BriefingGenerator] Brief generated with confidence score: {confidence_score:.2f}")
        return brief
    
    def _extract_portfolio(self, raw_data: Dict) -> List[str]:
        """Extract portfolio and department information"""
        portfolio = []
        
        # From LinkedIn data
        linkedin_data = raw_data.get('linkedin', {})
        if linkedin_data.get('title_from_linkedin'):
            portfolio.append(f"LinkedIn Profile: {linkedin_data['title_from_linkedin']}")
        
        # From company website
        company_data = raw_data.get('company_website', {})
        if company_data.get('person_bio'):
            # Extract key responsibilities from bio
            bio = company_data['person_bio']
            if 'responsible' in bio.lower() or 'leads' in bio.lower() or 'oversees' in bio.lower():
                portfolio.append(f"Key Responsibilities: {bio[:100]}...")
        
        # From background analysis
        background = raw_data.get('background', {})
        if background.get('current_role'):
            portfolio.append(f"Current Focus: {background['current_role']}")
        
        # From expertise
        if background.get('expertise'):
            portfolio.append(f"Expertise Areas: {', '.join(background['expertise'][:3])}")
        
        # Default if nothing found
        if not portfolio:
            portfolio.append(f"Primary Role: {raw_data['person']['title']} at {raw_data['person']['organization']}")
        
        return portfolio[:5]
    
    def _extract_initiatives(self, raw_data: Dict) -> List[str]:
        """Extract major initiatives and projects"""
        initiatives = []
        
        # From recent activities (already analyzed)
        recent_activities = raw_data.get('recent_activities', [])
        initiatives.extend(recent_activities[:3])
        
        # From news articles (extract initiative keywords)
        for article in raw_data.get('news_articles', [])[:5]:
            title = article.get('title', '').lower()
            if any(keyword in title for keyword in ['launch', 'initiative', 'program', 'partnership', 'expansion']):
                initiatives.append(article['title'])
        
        # From company achievements
        background = raw_data.get('background', {})
        if background.get('achievements'):
            initiatives.extend(background['achievements'][:2])
        
        # Remove duplicates and limit
        unique_initiatives = []
        seen = set()
        for init in initiatives:
            if init not in seen and len(init) > 10:
                seen.add(init)
                unique_initiatives.append(init)
        
        return unique_initiatives[:5]
    
    def _format_background(self, background: Dict) -> List[str]:
        """Format background information into structured points"""
        formatted = []
        
        # Education
        if background.get('education'):
            education = background['education']
            if isinstance(education, list):
                for edu in education[:2]:
                    if len(edu) > 5:  # Filter out empty/short entries
                        formatted.append(f"Education: {edu}")
            else:
                formatted.append(f"Education: {education}")
        
        # Career progression
        if background.get('career_progression'):
            career = background['career_progression']
            if isinstance(career, list):
                for role in career[:2]:
                    if len(role) > 5:
                        formatted.append(f"Experience: {role}")
            else:
                formatted.append(f"Experience: {career}")
        
        # Achievements
        if background.get('achievements'):
            achievements = background['achievements']
            if isinstance(achievements, list):
                for ach in achievements[:2]:
                    if len(ach) > 5:
                        formatted.append(f"Achievement: {ach}")
        
        # Current role details
        if background.get('current_role') and len(background['current_role']) > 10:
            formatted.append(f"Current Role: {background['current_role']}")
        
        # Ensure we have at least something
        if not formatted:
            formatted.append("Professional background information collected from available sources")
        
        return formatted[:5]
    
    def _calculate_confidence_score(self, raw_data: Dict, data_sources: Dict) -> float:
        """Calculate confidence score based on data quality and sources"""
        score = 0.0
        
        # Base score from successful data sources
        if data_sources.get('linkedin_found'):
            score += 0.25
        if data_sources.get('company_website_found'):
            score += 0.20
        if data_sources.get('news_articles_found'):
            score += 0.20
        if data_sources.get('general_search_found'):
            score += 0.15
        if data_sources.get('other_profiles_found'):
            score += 0.10
        
        # Bonus for data richness
        total_results = raw_data.get('scraping_stats', {}).get('total_results', 0)
        if total_results > 10:
            score += 0.10
        elif total_results > 5:
            score += 0.05
        
        # Quality indicators
        if raw_data.get('company_website', {}).get('person_bio'):
            score += 0.05  # Found specific bio info
        
        if len(raw_data.get('news_articles', [])) >= 3:
            score += 0.05  # Multiple recent articles
        
        return min(score, 1.0)
    
    async def _store_brief(self, request: MeetingRequest, brief: MeetingBrief, raw_data: Dict):
        """Store brief and raw data in database"""
        person_hash = hashlib.md5(
            f"{request.attendee_name}{request.organization}".encode()
        ).hexdigest()
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # Store or update profile
                cursor.execute('''
                    INSERT OR REPLACE INTO profiles 
                    (person_hash, name, title, organization, profile_data, scraping_status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    person_hash,
                    request.attendee_name,
                    request.title,
                    request.organization,
                    json.dumps(raw_data),
                    json.dumps(brief.scraping_summary),
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
                print(f"[BriefingGenerator] Data stored for {request.attendee_name}")
        
        except Exception as e:
            print(f"[BriefingGenerator] Database error: {e}")

# ====================== API ENDPOINTS ======================

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    init_database()
    print("🚀 Enhanced Meeting Intelligence System started!")
    print("✅ Real web scraping enabled")
    print("✅ Multi-source data collection active")
    print("✅ AI analysis engine ready")

@app.post("/api/generate-brief", response_model=MeetingBrief)
async def generate_meeting_brief(request: MeetingRequest, background_tasks: BackgroundTasks):
    """
    Generate a comprehensive meeting brief with real web scraping
    """
    try:
        print(f"[API] Received brief request for: {request.attendee_name}")
        
        generator = EnhancedBriefingGenerator()
        brief = await generator.generate_brief(request)
        
        print(f"[API] Brief generated successfully with confidence: {brief.confidence_score:.2f}")
        return brief
        
    except Exception as e:
        print(f"[API] Error generating brief: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating brief: {str(e)}")

@app.post("/api/quick-profile")
async def get_quick_profile(request: MeetingRequest):
    """
    Get a quick profile from cached data or generate new if not available
    """
    person_hash = hashlib.md5(
        f"{request.attendee_name}{request.organization}".encode()
    ).hexdigest()
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT profile_data, scraping_status, updated_at FROM profiles WHERE person_hash = ?",
                (person_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                profile_data = json.loads(row[0])
                scraping_status = json.loads(row[1]) if row[1] else {}
                updated_at = row[2]
                
                return {
                    "status": "cached",
                    "profile": profile_data,
                    "scraping_summary": scraping_status,
                    "last_updated": updated_at,
                    "person": {
                        "name": request.attendee_name,
                        "organization": request.organization
                    }
                }
            else:
                return {
                    "status": "not_found", 
                    "message": "No cached profile found. Use /api/generate-brief to create one.",
                    "person": {
                        "name": request.attendee_name,
                        "organization": request.organization
                    }
                }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/recent-briefs")
async def get_recent_briefs(limit: int = 10):
    """
    Get recently generated briefs with scraping statistics
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT b.briefing_data, b.created_at, p.name, p.organization, p.scraping_status
                FROM briefings b
                JOIN profiles p ON b.person_hash = p.person_hash
                ORDER BY b.created_at DESC
                LIMIT ?
            ''', (limit,))
            
            briefs = []
            for row in cursor.fetchall():
                brief_data = json.loads(row[0])
                scraping_status = json.loads(row[4]) if row[4] else {}
                
                briefs.append({
                    "person_name": row[2],
                    "organization": row[3],
                    "created_at": row[1],
                    "confidence_score": brief_data.get('confidence_score', 0),
                    "data_sources": brief_data.get('data_sources', {}),
                    "scraping_summary": scraping_status,
                    "brief_preview": {
                        "pitch_points_count": len(brief_data.get('key_pitch_points', [])),
                        "recent_highlights_count": len(brief_data.get('recent_highlights', [])),
                        "connection_opportunities_count": len(brief_data.get('connection_opportunities', []))
                    }
                })
            
            return {"briefs": briefs, "total": len(briefs)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/batch-briefs")
async def generate_batch_briefs(requests: List[MeetingRequest]):
    """
    Generate briefs for multiple attendees with progress tracking
    """
    generator = EnhancedBriefingGenerator()
    results = []
    
    total_requests = len(requests)
    
    for i, request in enumerate(requests):
        try:
            print(f"[API] Processing batch item {i+1}/{total_requests}: {request.attendee_name}")
            
            brief = await generator.generate_brief(request)
            
            results.append({
                "status": "success",
                "person": request.attendee_name,
                "organization": request.organization,
                "confidence_score": brief.confidence_score,
                "data_sources": brief.data_sources,
                "brief": brief
            })
            
        except Exception as e:
            print(f"[API] Batch error for {request.attendee_name}: {e}")
            results.append({
                "status": "error",
                "person": request.attendee_name,
                "organization": request.organization,
                "error": str(e)
            })
    
    # Calculate batch statistics
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "error"])
    avg_confidence = sum([r.get("confidence_score", 0) for r in results if r["status"] == "success"]) / max(successful, 1)
    
    return {
        "results": results,
        "statistics": {
            "total_processed": total_requests,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_requests) * 100,
            "average_confidence": avg_confidence
        }
    }

@app.get("/api/health")
async def health_check():
    """
    Comprehensive health check with system status
    """
    # Test scraping capability
    scraper = EnhancedIntelligenceScraper()
    
    try:
        # Quick test search
        test_results = await scraper.search_duckduckgo("test search", num_results=1)
        scraping_status = "operational" if test_results else "limited"
    except:
        scraping_status = "error"
    
    # Database check
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM profiles")
            profile_count = cursor.fetchone()[0]
        db_status = "connected"
    except:
        db_status = "error"
        profile_count = 0
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_status,
            "web_scraping": scraping_status,
            "ai_analysis": "operational" if OPENAI_API_KEY != "your-openai-api-key" else "not_configured"
        },
        "statistics": {
            "cache_size": len(cache),
            "profiles_stored": profile_count
        },
        "version": "2.0.0"
    }

@app.get("/api/scraping-test/{person_name}")
async def test_scraping(person_name: str, organization: str = ""):
    """
    Test scraping capabilities for a specific person
    """
    try:
        scraper = EnhancedIntelligenceScraper()
        
        # Perform test scraping
        test_data = await scraper.aggregate_data(person_name, "CEO", organization)
        
        return {
            "person": person_name,
            "organization": organization,
            "scraping_results": {
                "general_search_count": len(test_data.get('general_search', [])),
                "linkedin_found": test_data.get('linkedin', {}).get('profile_found', False),
                "company_website_found": test_data.get('company_website', {}).get('website_found', False),
                "news_articles_count": len(test_data.get('news_articles', [])),
                "other_profiles_count": len(test_data.get('other_profiles', {}).get('other_profiles', []))
            },
            "scraping_stats": test_data.get('scraping_stats', {}),
            "sample_results": {
                "first_search_result": test_data.get('general_search', [{}])[0] if test_data.get('general_search') else None,
                "linkedin_summary": test_data.get('linkedin', {}).get('summary', 'Not found')[:200] + "..." if test_data.get('linkedin', {}).get('summary') else 'Not found',
                "first_news_article": test_data.get('news_articles', [{}])[0].get('title', 'No news found') if test_data.get('news_articles') else 'No news found'
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping test error: {str(e)}")

# ====================== EXPORT AND UTILITY ENDPOINTS ======================

@app.get("/api/export-brief/{person_name}")
async def export_brief(person_name: str, format: str = "json"):
    """
    Export a brief in different formats with enhanced data
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT b.briefing_data, p.organization 
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
            organization = row[1]
            
            if format == "json":
                return brief_data
            elif format == "markdown":
                return {"content": _format_as_markdown(brief_data)}
            elif format == "html":
                return {"content": _format_as_html(brief_data)}
            else:
                raise HTTPException(status_code=400, detail="Invalid format. Use: json, markdown, html")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

def _format_as_markdown(brief: Dict) -> str:
    """Enhanced markdown formatting with scraping info"""
    md = f"""# Meeting Intelligence Brief

## PROSPECT: {brief['prospect_info']['name']}, {brief['prospect_info']['title']}, {brief['prospect_info']['organization']}

**Meeting Date:** {brief['prospect_info']['meeting_date']}  
**Generated:** {brief['generated_at']}  
**Confidence Score:** {brief['confidence_score']:.1%}  

### DATA SOURCES
"""
    
    # Add data sources info
    data_sources = brief.get('data_sources', {})
    for source, found in data_sources.items():
        status = "✅" if found else "❌"
        md += f"- {source.replace('_', ' ').title()}: {status}\n"
    
    scraping_summary = brief.get('scraping_summary', {})
    if scraping_summary:
        md += f"\n**Scraping Summary:** {scraping_summary.get('successful_sources', 0)}/{scraping_summary.get('total_sources_checked', 0)} sources successful, {scraping_summary.get('total_results_found', 0)} results found\n"

    md += "\n### 🎯 KEY PITCH POINTS\n"
    for i, point in enumerate(brief['key_pitch_points'], 1):
        md += f"{i}. {point}\n\n"
    
    md += "### 🎓 BACKGROUND & EDUCATION\n"
    for item in brief['background_education']:
        md += f"- {item}\n"
    
    md += "\n### 📰 RECENT HIGHLIGHTS\n"
    for item in brief['recent_highlights']:
        md += f"- {item}\n"
    
    md += "\n### 💼 PORTFOLIO & DEPARTMENTS\n"
    for item in brief['portfolio_departments']:
        md += f"- {item}\n"
    
    md += "\n### 🚀 MAJOR INITIATIVES\n"
    for item in brief['major_initiatives']:
        md += f"- {item}\n"
    
    md += "\n### 🤝 CONNECTION OPPORTUNITIES\n"
    for item in brief['connection_opportunities']:
        md += f"- {item}\n"
    
    return md

def _format_as_html(brief: Dict) -> str:
    """Enhanced HTML formatting with styling"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Meeting Brief - {brief['prospect_info']['name']}</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{ 
                color: #2c3e50; 
                border-bottom: 3px solid #3498db; 
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            h2 {{ 
                color: #3498db; 
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }}
            h3 {{ 
                color: #34495e;
                margin-top: 20px;
            }}
            .info-header {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .info-header strong {{
                color: #2c3e50;
            }}
            .confidence {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
            }}
            .confidence.high {{
                background-color: #27ae60;
                color: white;
            }}
            .confidence.medium {{
                background-color: #f39c12;
                color: white;
            }}
            .confidence.low {{
                background-color: #e74c3c;
                color: white;
            }}
            .data-sources {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .data-sources h4 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .source-item {{
                display: inline-block;
                margin: 5px;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.9em;
            }}
            .source-found {{
                background-color: #d4edda;
                color: #155724;
            }}
            .source-not-found {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 25px;
            }}
            li {{
                margin: 8px 0;
                line-height: 1.8;
            }}
            .pitch-points {{
                background-color: #e8f4fd;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .pitch-points li {{
                margin: 15px 0;
                font-size: 1.05em;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background-color: #fafafa;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }}
            .connections {{
                background-color: #f0f8ff;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Meeting Intelligence Brief</h1>
            
            <div class="info-header">
                <h2>{brief['prospect_info']['name']}</h2>
                <p><strong>{brief['prospect_info']['title']}</strong> at <strong>{brief['prospect_info']['organization']}</strong></p>
                <p><strong>Meeting Date:</strong> {brief['prospect_info']['meeting_date']}</p>
                <p><strong>Generated:</strong> {brief['generated_at']}</p>
                <p><strong>Confidence Score:</strong> 
                    <span class="confidence {'high' if brief['confidence_score'] > 0.7 else 'medium' if brief['confidence_score'] > 0.4 else 'low'}">
                        {brief['confidence_score']:.1%}
                    </span>
                </p>
            </div>
            
            <div class="data-sources">
                <h4>Data Sources</h4>
                {''.join([f'<span class="source-item {"source-found" if found else "source-not-found"}">{source.replace("_", " ").title()}: {"✓" if found else "✗"}</span>' 
                         for source, found in brief.get('data_sources', {}).items()])}
                <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    {brief.get('scraping_summary', {}).get('successful_sources', 0)}/{brief.get('scraping_summary', {}).get('total_sources_checked', 0)} sources successful, 
                    {brief.get('scraping_summary', {}).get('total_results_found', 0)} results found
                </p>
            </div>
            
            <div class="pitch-points">
                <h3>🎯 Key Pitch Points</h3>
                <ol>
                    {''.join([f'<li>{point}</li>' for point in brief['key_pitch_points']])}
                </ol>
            </div>
            
            <div class="section">
                <h3>🎓 Background & Education</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in brief['background_education']])}
                </ul>
            </div>
            
            <div class="section">
                <h3>📰 Recent Highlights</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in brief['recent_highlights']])}
                </ul>
            </div>
            
            <div class="section">
                <h3>💼 Portfolio & Departments</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in brief['portfolio_departments']])}
                </ul>
            </div>
            
            <div class="section">
                <h3>🚀 Major Initiatives</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in brief['major_initiatives']])}
                </ul>
            </div>
            
            <div class="connections">
                <h3>🤝 Connection Opportunities</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in brief['connection_opportunities']])}
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated by AI Meeting Intelligence System</p>
                <p>This brief is based on publicly available information and AI analysis</p>
            </div>
        </div>
    </body>
    </html>
    """

# ====================== ADDITIONAL UTILITY FUNCTIONS ======================

@app.post("/api/refresh-brief")
async def refresh_brief(request: MeetingRequest):
    """
    Force refresh a brief by clearing cache and regenerating
    """
    person_hash = hashlib.md5(
        f"{request.attendee_name}{request.organization}".encode()
    ).hexdigest()
    
    # Clear cache
    keys_to_remove = [key for key in cache.keys() if person_hash in str(key)]
    for key in keys_to_remove:
        del cache[key]
    
    # Generate fresh brief
    generator = EnhancedBriefingGenerator()
    brief = await generator.generate_brief(request)
    
    return {
        "status": "refreshed",
        "brief": brief,
        "message": f"Brief regenerated with fresh data for {request.attendee_name}"
    }

@app.delete("/api/clear-cache")
async def clear_cache():
    """
    Clear all cached data
    """
    cache.clear()
    return {"status": "success", "message": "Cache cleared successfully"}

@app.get("/api/statistics")
async def get_statistics():
    """
    Get system usage statistics
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get profile statistics
            cursor.execute("SELECT COUNT(*) FROM profiles")
            total_profiles = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM briefings")
            total_briefings = cursor.fetchone()[0]
            
            # Get recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM briefings 
                WHERE datetime(created_at) > datetime('now', '-24 hours')
            """)
            briefs_last_24h = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM briefings 
                WHERE datetime(created_at) > datetime('now', '-7 days')
            """)
            briefs_last_week = cursor.fetchone()[0]
            
            # Get organization statistics
            cursor.execute("""
                SELECT organization, COUNT(*) as count 
                FROM profiles 
                GROUP BY organization 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_organizations = [{"org": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Get average confidence scores
            cursor.execute("""
                SELECT AVG(json_extract(briefing_data, '$.confidence_score')) 
                FROM briefings
            """)
            avg_confidence = cursor.fetchone()[0] or 0
            
            return {
                "profiles": {
                    "total": total_profiles,
                    "unique_organizations": len(top_organizations)
                },
                "briefings": {
                    "total": total_briefings,
                    "last_24_hours": briefs_last_24h,
                    "last_7_days": briefs_last_week,
                    "average_confidence_score": round(avg_confidence, 2)
                },
                "top_organizations": top_organizations,
                "cache": {
                    "current_size": len(cache),
                    "max_size": cache.maxsize,
                    "ttl_hours": cache.ttl / 3600
                }
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(
    person_name: str,
    organization: str,
    feedback: str,
    accuracy_rating: int = 5
):
    """
    Submit feedback on generated briefs for continuous improvement
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Create feedback table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT,
                    organization TEXT,
                    feedback TEXT,
                    accuracy_rating INTEGER,
                    created_at TIMESTAMP
                )
            ''')
            
            # Insert feedback
            cursor.execute('''
                INSERT INTO feedback (person_name, organization, feedback, accuracy_rating, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_name, organization, feedback, accuracy_rating, datetime.now()))
            
            conn.commit()
            
            return {
                "status": "success",
                "message": "Feedback submitted successfully"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════════════╗
    ║     AI Meeting Intelligence System v2.0            ║
    ║     Real-time Web Scraping & AI Analysis          ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    # Run the FastAPI app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )