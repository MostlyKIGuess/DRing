"""
Web search service using DuckDuckGo and other search engines
"""
import requests
import streamlit as st
from typing import List, Dict, Optional
import re
import time
from config.settings import (
    MAX_SEARCH_RESULTS, DEFAULT_SEARCH_ENGINE, SEARCH_MIN_INTERVAL,
    SEARCH_MAX_RETRIES, SEARCH_RETRY_DELAY, SEARCH_MAX_RESULTS_SAFE,
    FALLBACK_SEARCH_ENGINES
)


class WebSearchService:
    """Service for web search and content retrieval"""
    
    def __init__(self, search_engine: str = DEFAULT_SEARCH_ENGINE):
        self.search_engine = search_engine
        self.ddgs = None
        self.last_request_time = 0
        self.min_request_interval = SEARCH_MIN_INTERVAL
        self.max_retries = SEARCH_MAX_RETRIES
        self.retry_delay = SEARCH_RETRY_DELAY
    
    def _get_ddgs_client(self):
        """Get DuckDuckGo client with rate limiting"""
        if self.ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
            except Exception as e:
                st.error(f"Failed to initialize DuckDuckGo client: {e}")
                return None
        return self.ddgs
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def search(
        self, 
        query: str, 
        max_results: int = MAX_SEARCH_RESULTS,
        time_range: Optional[str] = None
    ) -> List[Dict]:
        """Perform web search with single attempt"""
        try:
            if self.search_engine == "duckduckgo":
                result = self._search_duckduckgo_with_retry(query, max_results, time_range)
                if result:  # Only return if we got actual results
                    return result
                else:
                    raise Exception("No results returned from DuckDuckGo")
            else:
                # Fallback to basic web scraping if DuckDuckGo fails
                return self._search_fallback(query, max_results)
                
        except Exception:
            # Fail silently - return fallback without showing errors
            return self._search_fallback(query, max_results)
    
    def _search_duckduckgo_with_retry(
        self, 
        query: str, 
        max_results: int,
        time_range: Optional[str] = None
    ) -> List[Dict]:
        """Search using DuckDuckGo with rate limiting"""
        self._wait_for_rate_limit()
        
        ddgs = self._get_ddgs_client()
        if not ddgs:
            raise Exception("Failed to initialize DuckDuckGo client")
        
        # Perform search with reduced results to avoid rate limits
        safe_max_results = min(max_results, SEARCH_MAX_RESULTS_SAFE)
        
        results = ddgs.text(
            keywords=query,
            max_results=safe_max_results,
            timelimit=time_range,
            safesearch='moderate'
        )
        
        if not results:
            raise Exception("No results returned from DuckDuckGo")
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "source": "DuckDuckGo"
            })
        
        return formatted_results
    
    def _search_fallback(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced fallback search method when DuckDuckGo is unavailable"""
        # Silent fallback - no messages to user
        
        # Use configured fallback search engines with better formatting
        fallback_results = []
        
        for name, base_url in FALLBACK_SEARCH_ENGINES.items():
            if len(fallback_results) >= max_results:
                break
                
            display_name = name.replace('_', ' ').title()
            search_url = f"{base_url}{query.replace(' ', '+')}"
            
            # Generate more specific snippets based on search engine
            if 'scholar' in name:
                snippet = f"Search for academic papers and citations related to '{query}' on Google Scholar"
            elif 'arxiv' in name:
                snippet = f"Find preprint research papers about '{query}' on arXiv repository"
            elif 'semantic' in name:
                snippet = f"Discover research papers and authors studying '{query}' on Semantic Scholar"
            elif 'pubmed' in name:
                snippet = f"Search biomedical literature for '{query}' on PubMed database"
            else:
                snippet = f"Search for research and academic information about '{query}' on {display_name}"
            
            fallback_results.append({
                "title": f"Academic Search: {query} - {display_name}",
                "url": search_url,
                "snippet": snippet,
                "source": f"Fallback - {display_name}",
                "type": "academic_search"
            })
        
        # Add some general research suggestions if we don't have enough results
        if len(fallback_results) < max_results:
            general_searches = [
                {
                    "title": f"Wikipedia: {query}",
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "snippet": f"Background information and overview of '{query}' from Wikipedia",
                    "source": "Fallback - Wikipedia",
                    "type": "reference"
                },
                {
                    "title": f"Research Gate: {query}",
                    "url": f"https://www.researchgate.net/search?q={query.replace(' ', '%20')}",
                    "snippet": f"Research publications and researchers working on '{query}'",
                    "source": "Fallback - ResearchGate",
                    "type": "academic_network"
                }
            ]
            
            fallback_results.extend(general_searches[:max_results - len(fallback_results)])
        
        return fallback_results[:max_results]
    
    def search_academic(
        self, 
        query: str, 
        max_results: int = MAX_SEARCH_RESULTS
    ) -> List[Dict]:
        """Search for academic content"""
        # Add academic-specific terms to query
        academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:researchgate.net OR site:ieee.org OR site:acm.org"
        return self.search(academic_query, max_results)
    
    def search_recent(
        self, 
        query: str, 
        max_results: int = MAX_SEARCH_RESULTS
    ) -> List[Dict]:
        """Search for recent content (last month)"""
        return self.search(query, max_results, time_range="m")
    
    def get_page_content(self, url: str) -> Optional[str]:
        """Retrieve content from a web page"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Basic text extraction (could be enhanced with BeautifulSoup)
            content = response.text  
            
            # Remove HTML tags (basic)
            clean_content = re.sub(r'<[^>]+>', '', content)
            
            # Clean up whitespace
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            return clean_content[:5000]  # Limit content length
        except Exception as e:
            st.error(f"Error fetching page content: {str(e)}")
            return None
    
    def generate_search_queries(self, text: str, topic: str = "") -> List[str]:
        """Generate search queries from text content"""
        queries = []
        
        # Extract key phrases from text
        sentences = text.split('.')[:10]  # First 10 sentences
        
        for sentence in sentences:
            # Look for research-related terms
            research_terms = re.findall(
                r'\b(?:algorithm|method|approach|technique|model|system|analysis|study|research)\b', 
                sentence.lower()
            )
            
            if research_terms:
                # Extract the context around research terms
                words = sentence.split()
                for i, word in enumerate(words):
                    if word.lower() in research_terms:
                        # Get surrounding context
                        start = max(0, i-3)
                        end = min(len(words), i+4)
                        context = ' '.join(words[start:end])
                        if len(context) > 10:
                            queries.append(context)
        
        # Add topic-based queries if provided
        if topic:
            queries.extend([
                f"{topic} recent research",
                f"{topic} latest developments",
                f"{topic} state of the art",
                f"{topic} survey",
                f"{topic} review"
            ])
        
        return queries[:8]  # Return top 8 queries
    
    def search_news(
        self, 
        query: str, 
        max_results: int = MAX_SEARCH_RESULTS
    ) -> List[Dict]:
        """Search for news articles with single attempt"""
        try:
            self._wait_for_rate_limit()
            
            ddgs = self._get_ddgs_client()
            if not ddgs:
                raise Exception("Failed to initialize DuckDuckGo client")
            
            safe_max_results = min(max_results, 3)  # Conservative for news to avoid rate limits
            
            results = ddgs.news(
                keywords=query,
                max_results=safe_max_results
            )
            
            if not results:
                raise Exception("No news results returned")
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("body", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", "News")
                })
            
            return formatted_results
            
        except Exception:
            # Fail silently - news search is optional
            return self._search_fallback(f"{query} news", max_results)
    
    def search_images(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[Dict]:
        """Search for images"""
        try:
            results = self.ddgs.images(
                keywords=query,
                max_results=max_results
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "image_url": result.get("image", ""),
                    "source_url": result.get("url", ""),
                    "thumbnail": result.get("thumbnail", ""),
                    "source": "DuckDuckGo Images"
                })
            
            return formatted_results
        except Exception as e:
            st.error(f"Image search error: {str(e)}")
            return []


class ContentAnalyzer:
    """Analyze web content for research purposes"""
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords from web content"""
        if not text:
            return []
        
        # Remove common words
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'we', 'they', 'it', 'he', 'she'
        }
        
        # Extract words (3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Count frequency
        word_freq = {}
        for word in words:
            if word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    @staticmethod
    def extract_research_topics(text: str) -> List[str]:
        """Extract research topics from text"""
        topics = set()
        
        # Research-related patterns
        patterns = [
            r'research in (\w+(?:\s+\w+){0,2})',
            r'study of (\w+(?:\s+\w+){0,2})',
            r'analysis of (\w+(?:\s+\w+){0,2})',
            r'(\w+(?:\s+\w+){0,2}) research',
            r'(\w+(?:\s+\w+){0,2}) analysis',
            r'(\w+(?:\s+\w+){0,2}) study'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            topics.update(matches)
        
        return list(topics)[:10]
    
    @staticmethod
    def summarize_content(text: str, max_sentences: int = 3) -> str:
        """Create a simple summary of content"""
        sentences = text.split('.')
        
        # score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:  # Meaningful sentences
                # score = len(sentence) * (1 - i/len(sentences))  # Prefer longer, earlier sentences
                score = len(sentence) # Prefer longer, I like this actually
                scored_sentences.append((score, sentence.strip()))
        
        # sort by score and return top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [sentence for score, sentence in scored_sentences[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
