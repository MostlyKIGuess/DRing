"""
Research analysis service for comprehensive document analysis
"""
from typing import List, Dict
import re
from datetime import datetime
from src.core.gemini_client import GeminiClient
from src.services.arxiv_service import ArxivService
from src.services.search_service import WebSearchService


class ResearchAnalyzer:
    """Comprehensive research analysis service"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.arxiv_service = ArxivService()
        self.search_service = WebSearchService()
    
    def analyze_document(
        self, 
        pdf_bytes: bytes, 
        analysis_type: str = "comprehensive"
    ) -> Dict:
        """Perform comprehensive document analysis"""
        
        analysis_prompts = {
            "comprehensive": """Analyze this research document comprehensively. Provide:
            1. Main research topic and objectives
            2. Key findings and contributions
            3. Methodology used
            4. Important technical terms and concepts
            5. Research gaps or limitations mentioned
            6. Future work suggestions
            7. Key statistics or data points
            8. Most important citations or references
            
            Format your response as a structured analysis with clear sections.""",
            
            "literature_review": """Analyze this document for literature review purposes. Identify:
            1. Research domain and field
            2. Key concepts and terminology
            3. Main research questions addressed
            4. Methodological approaches used
            5. Significant findings and conclusions
            6. Gaps in current research
            7. Related work and citations
            8. Keywords for finding similar papers
            
            Provide analysis suitable for conducting a literature review.""",
            
            "methodology_focus": """Focus on the methodological aspects of this research. Analyze:
            1. Research design and approach
            2. Data collection methods
            3. Analysis techniques used
            4. Tools and technologies employed
            5. Experimental setup or procedure
            6. Evaluation metrics and criteria
            7. Limitations of the methodology
            8. Potential improvements or alternatives
            
            Provide detailed methodological insights.""",
            
            "research_gaps": """Identify research gaps and opportunities from this document. Focus on:
            1. Explicitly stated limitations
            2. Areas marked for future work
            3. Unresolved questions or challenges
            4. Methodological limitations
            5. Scope limitations
            6. Potential extensions or improvements
            7. Related problems not addressed
            8. Emerging research directions mentioned
            
            Highlight opportunities for future research."""
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])
        
        analysis_result = self.gemini_client.analyze_pdf(pdf_bytes, prompt)
        
        if not analysis_result:
            return {"error": "Failed to analyze document"}
        
        keywords = self._extract_keywords_from_analysis(analysis_result)
        research_topics = self._extract_research_topics(analysis_result)
        
        return {
            "analysis": analysis_result,
            "keywords": keywords,
            "research_topics": research_topics,
            "timestamp": datetime.now().isoformat()
        }
    
    def discover_related_papers(
        self, 
        keywords: List[str], 
        research_topics: List[str],
        max_papers: int = 10
    ) -> Dict:
        """Discover related papers from arXiv"""
        
        all_papers = []
        search_queries = []
        
        # Search using keywords
        if keywords:
            keyword_papers = self.arxiv_service.search_by_keywords(keywords[:5], max_papers//2)
            all_papers.extend(keyword_papers)
            search_queries.extend(keywords[:5])
        
        # Search using research topics
        if research_topics:
            for topic in research_topics[:3]:
                topic_papers = self.arxiv_service.search_papers(topic, max_papers//3)
                all_papers.extend(topic_papers)
                search_queries.append(topic)
        
        unique_papers = {}
        for paper in all_papers:
            paper_id = paper.get('id')
            if paper_id not in unique_papers:
                unique_papers[paper_id] = paper
        
        papers_list = list(unique_papers.values())[:max_papers]
        
        categorized_papers = self.arxiv_service.categorize_papers(papers_list)
        
        return {
            "papers": papers_list,
            "categorized_papers": categorized_papers,
            "search_queries": search_queries,
            "total_found": len(papers_list)
        }
    
    def generate_literature_queries(self, analysis_text: str) -> List[str]:
        """Generate literature search queries from analysis"""
        
        query_prompt = f"""Based on this research analysis, generate 8-10 specific search queries that would help find related literature and papers. 
        Focus on key concepts, methodologies, and research areas mentioned.
        
        Analysis: {analysis_text[:2000]}
        
        Return a simple list of search queries, one per line."""
        
        response = self.gemini_client.generate_content([query_prompt])
        
        if response:
            # Extract queries from response
            lines = response.strip().split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 5:
                    # Clean up formatting
                    query = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                    query = re.sub(r'^[-*]\s*', '', query)   # Remove bullet points
                    if query:
                        queries.append(query)
            
            return queries[:10]
        
        return []
    
    def search_web_resources(
        self, 
        queries: List[str], 
        max_results_per_query: int = 3
    ) -> Dict:
        """Search web for additional research resources"""
        
        # Try only the first query to avoid spam and rate limits
        if not queries:
            return {"web_results": [], "academic_results": [], "queries_used": []}
        
        try:
            # Only try the most relevant query (first one)
            query = queries[0]
            
            # Single web search attempt - fail silently if it doesn't work
            web_results = self.search_service.search(query, max_results_per_query)
            
            return {
                "web_results": web_results,
                "academic_results": [],  # Skip academic search for now to avoid multiple failures
                "queries_used": [query]
            }
        except Exception:
            # Fail silently - web search is optional
            return {"web_results": [], "academic_results": [], "queries_used": []}
    
    def generate_research_summary(
        self, 
        original_analysis: str, 
        related_papers: List[Dict],
        web_results: List[Dict]
    ) -> str:
        """Generate comprehensive research summary"""
        
        # Prepare paper summaries
        paper_summaries = []
        for paper in related_papers[:5]:
            summary = f"- {paper.get('title', 'Unknown')} ({paper.get('published', 'Unknown date')}): {paper.get('abstract', '')[:200]}..."
            paper_summaries.append(summary)
        
        # Prepare web resource summaries
        web_summaries = []
        for result in web_results[:5]:
            summary = f"- {result.get('title', 'Unknown')}: {result.get('snippet', '')[:150]}..."
            web_summaries.append(summary)
        
        summary_prompt = f"""Create a comprehensive research summary that synthesizes the following information:

        Original Document Analysis:
        {original_analysis[:1500]}

        Related Papers Found:
        {chr(10).join(paper_summaries)}

        Additional Web Resources:
        {chr(10).join(web_summaries)}

        Please provide:
        1. Key research themes identified
        2. Current state of research in this area
        3. Main methodological approaches
        4. Research gaps and opportunities
        5. Recommended next steps for research
        6. Important resources for further study

        Format as a structured research summary."""
        
        response = self.gemini_client.generate_content([summary_prompt])
        return response if response else "Failed to generate research summary."
    
    def compare_documents(self, doc_analyses: List[str]) -> str:
        """Compare multiple document analyses"""
        
        if len(doc_analyses) < 2:
            return "Need at least 2 documents for comparison."
        
        comparison_prompt = f"""Compare these research document analyses and provide:

        1. Common themes and topics
        2. Different approaches or methodologies
        3. Complementary findings
        4. Conflicting results or viewpoints
        5. Research gaps that appear across documents
        6. Synthesis opportunities
        7. Recommendations for comprehensive understanding

        Document Analyses:
        {chr(10).join([f"Document {i+1}: {analysis[:800]}" for i, analysis in enumerate(doc_analyses)])}

        Provide a structured comparative analysis."""
        
        response = self.gemini_client.generate_content([comparison_prompt])
        return response if response else "Failed to generate comparison."
    
    def _extract_keywords_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract meaningful keywords from analysis text using AI"""
        
        # Use AI to extract better keywords instead of basic regex
        keyword_prompt = f"""Extract 15-20 key technical terms, concepts, and important phrases from this research analysis. 
        Focus on:
        - Technical terminology
        - Research methods and approaches
        - Key concepts and findings
        - Important domain-specific terms
        - Significant proper nouns (technologies, frameworks, etc.)
        
        Return only the keywords/phrases, one per line, without explanations.
        
        Analysis text:
        {analysis_text[:2000]}"""
        
        try:
            response = self.gemini_client.generate_content([keyword_prompt])
            if response:
                # Parse keywords from response
                keywords = []
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Clean up formatting
                        keyword = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                        keyword = re.sub(r'^[-*•]\s*', '', keyword)   # Remove bullet points
                        keyword = keyword.strip('"\'')  # Remove quotes
                        if keyword and len(keyword) > 2:
                            keywords.append(keyword)
                return keywords[:20]
        except Exception:
            pass
        
        # Fallback to basic extraction if AI fails
        return self._basic_keyword_extraction(analysis_text)
    
    def _basic_keyword_extraction(self, analysis_text: str) -> List[str]:
        """Fallback basic keyword extraction"""
        # Look for technical terms and key concepts
        technical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b\w+(?:ing|tion|sion|ness|ment)\b',   # Terms with common suffixes
            r'\b(?:algorithm|method|approach|technique|model|system|analysis|framework)\b'
        ]
        
        keywords = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, analysis_text)
            keywords.update([match.lower() for match in matches if len(match) > 3])
        
        return list(keywords)[:15]
    
    def _extract_research_topics(self, analysis_text: str) -> List[str]:
        """Extract research topics from analysis text using AI"""
        
        topic_prompt = f"""Extract 8-12 specific research topics, domains, and areas of study from this analysis.
        Focus on:
        - Research domains and fields
        - Specific application areas
        - Methodological approaches
        - Technology domains
        - Problem areas being addressed
        
        Return only the topic names, one per line, without explanations.
        
        Analysis text:
        {analysis_text[:2000]}"""
        
        try:
            response = self.gemini_client.generate_content([topic_prompt])
            if response:
                # Parse topics from response
                topics = []
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Clean up formatting
                        topic = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                        topic = re.sub(r'^[-*•]\s*', '', topic)   # Remove bullet points
                        topic = topic.strip('"\'')  # Remove quotes
                        if topic and len(topic) > 3:
                            topics.append(topic)
                return topics[:12]
        except Exception:
            pass
        
        # Fallback to basic extraction if AI fails
        return self._basic_topic_extraction(analysis_text)
    
    def _basic_topic_extraction(self, analysis_text: str) -> List[str]:
        """Fallback basic topic extraction"""
        # Look for research area indicators
        topic_patterns = [
            r'research in (\w+(?:\s+\w+){0,2})',
            r'field of (\w+(?:\s+\w+){0,2})',
            r'area of (\w+(?:\s+\w+){0,2})',
            r'domain of (\w+(?:\s+\w+){0,2})',
            r'(\w+(?:\s+\w+){0,2}) research'
        ]
        
        topics = set()
        for pattern in topic_patterns:
            matches = re.findall(pattern, analysis_text.lower())
            topics.update(matches)
        
        return list(topics)[:8]


class CitationManager:
    """Manage and format citations"""
    
    @staticmethod
    def format_arxiv_citation(paper: Dict, style: str = "apa") -> str:
        """Format arXiv paper citation"""
        authors = paper.get('authors', [])
        title = paper.get('title', '')
        year = paper.get('published', '').split('-')[0] if paper.get('published') else ''
        arxiv_id = paper.get('id', '')
        
        if style == "apa":
            if len(authors) > 6:
                author_str = f"{authors[0]} et al."
            elif len(authors) > 1:
                author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
            else:
                author_str = authors[0] if authors else "Unknown Author"
            
            return f"{author_str} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."
        
        return f"{authors[0] if authors else 'Unknown'} et al. ({year}). {title}. arXiv:{arxiv_id}"
    
    @staticmethod
    def generate_bibliography(papers: List[Dict], style: str = "apa") -> str:
        """Generate bibliography from papers"""
        citations = []
        for paper in papers:
            citation = CitationManager.format_arxiv_citation(paper, style)
            citations.append(citation)
        
        return "\n\n".join(citations)
    
    @staticmethod
    def extract_paper_metadata(paper: Dict) -> Dict:
        """Extract key metadata from paper"""
        return {
            "title": paper.get('title', ''),
            "authors": paper.get('authors', []),
            "year": paper.get('published', '').split('-')[0],
            "abstract_summary": paper.get('abstract', '')[:200] + "...",
            "categories": paper.get('categories', []),
            "arxiv_id": paper.get('id', ''),
            "pdf_url": paper.get('pdf_url', '')
        }
