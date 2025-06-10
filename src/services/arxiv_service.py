"""
arXiv research paper discovery service
"""
import arxiv
import streamlit as st
from typing import List, Dict, Optional
import re
from datetime import datetime, timedelta
from config.settings import MAX_ARXIV_RESULTS


class ArxivService:
    """Service for discovering and analyzing arXiv papers"""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(
        self, 
        query: str, 
        max_results: int = MAX_ARXIV_RESULTS,
        category: Optional[str] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict]:
        """Search for papers on arXiv"""
        try:
            # Construct search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND {query}"
            
            # Create search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                papers.append(self._format_paper(result))
            
            return papers
        except Exception as e:
            st.error(f"Error searching arXiv: {str(e)}")
            return []
    
    def search_recent_papers(
        self, 
        query: str, 
        days: int = 30,
        max_results: int = MAX_ARXIV_RESULTS
    ) -> List[Dict]:
        """Search for recent papers (last N days)"""
        try:
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=days)
            
            # Search papers
            search = arxiv.Search(
                query=query,
                max_results=max_results * 2,  # Get more to filter by date
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                if result.published >= date_threshold:
                    papers.append(self._format_paper(result))
                    if len(papers) >= max_results:
                        break
            
            return papers
        except Exception as e:
            st.error(f"Error searching recent papers: {str(e)}")
            return []
    
    def search_by_keywords(
        self, 
        keywords: List[str], 
        max_results: int = MAX_ARXIV_RESULTS
    ) -> List[Dict]:
        """Search papers using multiple keywords"""
        if not keywords:
            return []
        
        # Construct query from keywords
        query = " OR ".join([f'"{keyword}"' for keyword in keywords[:5]])  # Limit to top 5 keywords
        
        return self.search_papers(query, max_results)
    
    def get_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """Get detailed information about a specific paper"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            for result in self.client.results(search):
                return self._format_paper(result, detailed=True)
            return None
        except Exception as e:
            st.error(f"Error getting paper details: {str(e)}")
            return None
    
    def _format_paper(self, result: arxiv.Result, detailed: bool = False) -> Dict:
        """Format arXiv result into standardized dictionary"""
        paper = {
            "id": result.entry_id.split('/')[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
            "categories": result.categories,
            "primary_category": result.primary_category,
            "pdf_url": result.pdf_url,
            "arxiv_url": result.entry_id,
            "doi": result.doi,
        }
        
        if detailed:
            paper.update({
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "links": [{"href": link.href, "title": link.title} for link in result.links]
            })
        
        return paper
    
    def generate_search_queries(self, text: str, max_queries: int = 5) -> List[str]:
        """Generate search queries from document text"""
        # Extract key phrases and terms
        # This is a simple implementation - could be enhanced with NLP
        
        # Remove common words
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract meaningful phrases
        sentences = re.split(r'[.!?]+', text)
        phrases = []
        
        for sentence in sentences[:20]:  # Analyze first 20 sentences
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            meaningful_words = [w for w in words if w not in common_words]
            
            if len(meaningful_words) >= 2:
                # Create phrases from consecutive meaningful words
                for i in range(len(meaningful_words) - 1):
                    phrase = f"{meaningful_words[i]} {meaningful_words[i+1]}"
                    phrases.append(phrase)
        
        # Count phrase frequency and return top queries
        phrase_count = {}
        for phrase in phrases:
            phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
        
        # Sort by frequency and return top queries
        sorted_phrases = sorted(phrase_count.items(), key=lambda x: x[1], reverse=True)
        queries = [phrase for phrase, count in sorted_phrases[:max_queries]]
        
        return queries
    
    def categorize_papers(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize papers by subject area"""
        categories = {}
        
        category_map = {
            'cs': 'Computer Science',
            'math': 'Mathematics',
            'physics': 'Physics',
            'q-bio': 'Quantitative Biology',
            'q-fin': 'Quantitative Finance',
            'stat': 'Statistics',
            'eess': 'Electrical Engineering',
            'econ': 'Economics',
        }
        
        for paper in papers:
            primary_cat = paper.get('primary_category', '').split('.')[0]
            category_name = category_map.get(primary_cat, 'Other')
            
            if category_name not in categories:
                categories[category_name] = []
            categories[category_name].append(paper)
        
        return categories
    
    def extract_citations_from_papers(self, papers: List[Dict]) -> List[str]:
        """Extract formatted citations from papers"""
        citations = []
        
        for paper in papers:
            # Create a simple citation format
            authors = paper.get('authors', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            elif len(authors) > 1:
                author_str = ", ".join(authors[:-1]) + f" and {authors[-1]}"
            else:
                author_str = authors[0] if authors else "Unknown"
            
            year = paper.get('published', '').split('-')[0]
            title = paper.get('title', '')
            arxiv_id = paper.get('id', '')
            
            citation = f"{author_str} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."
            citations.append(citation)
        
        return citations


class ResearchQueryGenerator:
    """Generate research queries for literature discovery"""
    
    @staticmethod
    def generate_from_keywords(keywords: List[str]) -> List[str]:
        """Generate research queries from keywords"""
        if not keywords:
            return []
        
        queries = []
        
        # Single keyword queries
        for keyword in keywords[:3]:
            queries.append(keyword)
        
        # Combined keyword queries
        if len(keywords) >= 2:
            for i in range(min(3, len(keywords))):
                for j in range(i+1, min(5, len(keywords))):
                    queries.append(f"{keywords[i]} {keywords[j]}")
        
        tech_terms = ['machine learning', 'deep learning', 'artificial intelligence', 
                     'neural network', 'algorithm', 'optimization', 'analysis']
        
        for keyword in keywords[:2]:
            for tech in tech_terms:
                if tech.lower() not in keyword.lower():
                    queries.append(f"{keyword} {tech}")
        
        return queries[:10]  
    
    @staticmethod
    def generate_from_abstract(abstract: str) -> List[str]:
        """Generate research queries from paper abstract"""
        # Extract key noun phrases and technical terms
        # this is a very naive implementation 
        # TODO: Enhance with NLP techniques for better extraction, ppl if you know this shit pls help out
        
        # Common research phrases to look for
        research_patterns = [
            r'(\w+\s+learning)',
            r'(\w+\s+algorithm)',
            r'(\w+\s+method)',
            r'(\w+\s+approach)',
            r'(\w+\s+technique)',
            r'(\w+\s+model)',
            r'(\w+\s+system)',
            r'(\w+\s+analysis)',
        ]
        
        queries = set()
        for pattern in research_patterns:
            matches = re.findall(pattern, abstract.lower())
            queries.update(matches)
        
        return list(queries)[:8]
