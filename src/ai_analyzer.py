import openai
from typing import Dict, List, Optional
import json
import re
from datetime import datetime

class AIAnalyzer:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-4-turbo-preview"
    
    def analyze_person_profile(self, raw_data: Dict, meeting_context: Dict) -> Dict:
        """Main analysis pipeline"""
        
        # Step 1: Clean and structure the raw data
        structured_data = self.structure_raw_data(raw_data)
        
        # Step 2: Generate background analysis
        background = self.analyze_background(structured_data)
        
        # Step 3: Identify recent activities
        recent_activities = self.extract_recent_activities(structured_data)
        
        # Step 4: Map portfolio and responsibilities
        portfolio = self.map_portfolio(structured_data)
        
        # Step 5: Extract major initiatives
        initiatives = self.identify_initiatives(structured_data)
        
        # Step 6: Find connection opportunities
        connections = self.find_connections(structured_data, meeting_context)
        
        # Step 7: Generate pitch points
        pitch_points = self.generate_pitch_points(
            structured_data, 
            background, 
            portfolio, 
            initiatives, 
            connections,
            meeting_context
        )
        
        return {
            'pitch_points': pitch_points,
            'background': background,
            'recent_activities': recent_activities,
            'portfolio': portfolio,
            'initiatives': initiatives,
            'connections': connections,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def structure_raw_data(self, raw_data: Dict) -> Dict:
        """Use GPT to structure and clean raw scraped data"""
        
        prompt = f"""
        Structure the following raw data about a person into a clean JSON format:
        
        Raw Data: {json.dumps(raw_data, indent=2)}
        
        Output a structured JSON with these fields:
        - full_name
        - current_title
        - organization
        - education (list of degrees/institutions)
        - career_history (list of positions)
        - recent_speeches (list with dates)
        - key_responsibilities
        - known_initiatives
        - public_statements
        - professional_focus_areas
        
        Extract only factual information. If information is not available, use null.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data structuring expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_background(self, data: Dict) -> Dict:
        """Deep analysis of professional background"""
        
        prompt = f"""
        Analyze this person's professional background and provide insights:
        
        Data: {json.dumps(data, indent=2)}
        
        Provide:
        1. Career progression narrative (2-3 sentences)
        2. Key expertise areas
        3. Notable achievements
        4. Leadership style indicators
        5. Domain expertise
        6. Geographical focus/experience
        
        Format as JSON with clear, professional language.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional analyst specializing in executive profiles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        return json.loads(response.choices[0].message.content)
    
    def generate_pitch_points(self, data: Dict, background: Dict, 
                            portfolio: Dict, initiatives: Dict, 
                            connections: Dict, meeting_context: Dict) -> List[str]:
        """Generate 5 strategic pitch points"""
        
        # Gather company context (you'd load this from your database)
        company_context = {
            "solutions": "AI-powered enterprise solutions",
            "focus_areas": ["Digital transformation", "AI/ML", "Government tech"],
            "case_studies": ["Smart City implementations", "Digital India projects"],
            "differentiators": ["Local presence", "Government experience", "Scalable solutions"]
        }
        
        prompt = f"""
        Generate 5 strategic pitch points for a meeting with this person:
        
        PERSON PROFILE:
        {json.dumps(data, indent=2)}
        
        BACKGROUND ANALYSIS:
        {json.dumps(background, indent=2)}
        
        CURRENT PORTFOLIO:
        {json.dumps(portfolio, indent=2)}
        
        INITIATIVES:
        {json.dumps(initiatives, indent=2)}
        
        OUR COMPANY:
        {json.dumps(company_context, indent=2)}
        
        MEETING CONTEXT:
        {json.dumps(meeting_context, indent=2)}
        
        Generate 5 pitch points that:
        1. Are highly personalized to their background and current role
        2. Align our solutions with their initiatives
        3. Reference specific opportunities for collaboration
        4. Build on shared connections or experiences
        5. Are strategic, not generic sales points
        
        Each point should be:
        - One clear, actionable sentence
        - Specific and relevant
        - Building rapport while positioning our value
        
        Format as a JSON array of 5 strings.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strategic sales intelligence expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('pitch_points', [])
    
    def extract_recent_activities(self, data: Dict) -> List[Dict]:
        """Extract and analyze recent activities"""
        
        prompt = f"""
        From this data, extract recent activities (last 6 months):
        
        {json.dumps(data, indent=2)}
        
        For each activity, provide:
        - date (approximate if needed)
        - type (speech, announcement, launch, etc.)
        - description
        - key_message
        - relevance_to_our_business
        
        Return as JSON array, most recent first.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an intelligence analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def find_connections(self, data: Dict, meeting_context: Dict) -> Dict:
        """Identify connection opportunities"""
        
        # This would connect to your CRM/database for richer matching
        prompt = f"""
        Identify connection opportunities between us and this person:
        
        Person: {json.dumps(data, indent=2)}
        Our context: {json.dumps(meeting_context, indent=2)}
        
        Find:
        1. Shared educational backgrounds (specific schools/programs)
        2. Common professional networks
        3. Mutual connections (if any mentioned)
        4. Regional/geographical alignments
        5. Industry/domain overlaps
        6. Shared interests or initiatives
        
        Be specific with names, places, and programs.
        Format as JSON.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a relationship mapping expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        return json.loads(response.choices[0].message.content)