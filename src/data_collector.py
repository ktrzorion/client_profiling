import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from typing import Dict, List, Optional
import json
import time
from newspaper import Article
import re

class DataCollector:
    def __init__(self):
        self.setup_selenium()
        self.sources_priority = [
            self.search_government_sites,
            self.search_linkedin,
            self.search_company_pages,
            self.search_news,
            self.search_press_releases
        ]
    
    def setup_selenium(self):
        """Setup headless Chrome for dynamic content"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def collect_person_data(self, name: str, title: str = "", 
                           organization: str = "") -> Dict:
        """Main collection orchestrator"""
        person_data = {
            'name': name,
            'title': title,
            'organization': organization,
            'sources': [],
            'background': {},
            'recent_activities': [],
            'portfolio': [],
            'initiatives': [],
            'education': [],
            'career_history': [],
            'speeches': [],
            'news_mentions': [],
            'social_profiles': {}
        }
        
        # Search across all sources
        for search_method in self.sources_priority:
            try:
                source_data = search_method(name, title, organization)
                if source_data:
                    person_data = self.merge_data(person_data, source_data)
                    person_data['sources'].append(source_data.get('source', 'unknown'))
            except Exception as e:
                print(f"Error in {search_method.__name__}: {e}")
                continue
        
        return person_data
    
    def search_government_sites(self, name: str, title: str, org: str) -> Dict:
        """Search Indian government websites"""
        gov_domains = [
            "india.gov.in",
            "meity.gov.in",
            "digitalindia.gov.in",
            "mygov.in",
            "startupindia.gov.in"
        ]
        
        results = {
            'source': 'government_sites',
            'data': {}
        }
        
        for domain in gov_domains:
            search_query = f"site:{domain} {name} {title}"
            search_url = f"https://www.google.com/search?q={search_query}"
            
            try:
                response = requests.get(search_url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract relevant information
                for result in soup.find_all('div', class_='g')[:3]:
                    link = result.find('a')
                    if link:
                        page_data = self.scrape_page(link.get('href'))
                        if page_data:
                            results['data'] = self.merge_data(results['data'], page_data)
            except:
                continue
        
        return results
    
    def search_linkedin(self, name: str, title: str, org: str) -> Dict:
        """Search LinkedIn profiles"""
        # Note: LinkedIn API requires authentication
        # This is a simplified version
        search_url = f"https://www.linkedin.com/search/results/people/?keywords={name}%20{org}"
        
        try:
            self.driver.get(search_url)
            time.sleep(3)  # Wait for page load
            
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            profile_data = {
                'source': 'linkedin',
                'education': [],
                'experience': [],
                'skills': [],
                'connections': []
            }
            
            # Extract profile information (simplified)
            # In production, use LinkedIn API or more sophisticated scraping
            
            return profile_data
        except Exception as e:
            print(f"LinkedIn search error: {e}")
            return {}
    
    def search_news(self, name: str, title: str, org: str) -> Dict:
        """Search recent news articles"""
        news_sources = [
            "https://news.google.com/search?q=",
            "https://economictimes.indiatimes.com/search?q=",
            "https://www.livemint.com/search?q="
        ]
        
        articles_data = {
            'source': 'news',
            'articles': [],
            'quotes': [],
            'mentions': []
        }
        
        for source in news_sources:
            search_url = f"{source}{name}+{org}"
            try:
                article = Article(search_url)
                article.download()
                article.parse()
                
                if name.lower() in article.text.lower():
                    articles_data['articles'].append({
                        'title': article.title,
                        'date': article.publish_date,
                        'summary': article.summary[:500] if hasattr(article, 'summary') else '',
                        'url': search_url
                    })
            except:
                continue
        
        return articles_data
    
    def scrape_page(self, url: str) -> Dict:
        """Generic page scraper"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract structured data
            data = {
                'title': soup.find('title').text if soup.find('title') else '',
                'paragraphs': [p.text for p in soup.find_all('p')[:10]],
                'headers': [h.text for h in soup.find_all(['h1', 'h2', 'h3'])[:10]]
            }
            
            return data
        except:
            return {}
    
    def merge_data(self, existing: Dict, new: Dict) -> Dict:
        """Intelligently merge data from multiple sources"""
        merged = existing.copy()
        
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list):
                merged[key].extend(value)
            elif isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            elif value and not merged[key]:
                merged[key] = value
        
        return merged