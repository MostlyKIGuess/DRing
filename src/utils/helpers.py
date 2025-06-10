"""
Utility functions for Dread Rising
"""
import re
import streamlit as st
from typing import List, Dict, Optional, Any
import hashlib
import json
from datetime import datetime, timedelta
import time


class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\-.,;:!?()]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        if not text:
            return []
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 10:  # Meaningful sentences only
                clean_sentences.append(clean_sentence)
        
        return clean_sentences
    
    @staticmethod
    def extract_technical_terms(text: str) -> List[str]:
        """Extract technical terms from text"""
        if not text:
            return []
        
        # Patterns for technical terms
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b\w+(?:ing|tion|sion|ness|ment|ity|ism)\b',  # Technical suffixes
            r'\b(?:algorithm|method|approach|technique|model|system|framework|architecture)\b'
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update([match.lower() for match in matches if len(match) > 3])
        
        return list(terms)
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calculate simple readability score"""
        if not text:
            return 0.0
        
        sentences = TextProcessor.extract_sentences(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability score (lower is easier)
        score = (avg_sentence_length * 0.39) + (avg_word_length * 11.8) - 15.59
        return max(0.0, min(100.0, score))


class CacheManager:
    """Simple caching mechanism for API responses"""
    
    def __init__(self, cache_duration_hours: int = 24):
        self.cache_duration = timedelta(hours=cache_duration_hours)
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_entry = st.session_state.cache.get(key)
        if cache_entry:
            timestamp, value = cache_entry
            if datetime.now() - timestamp < self.cache_duration:
                return value
            else:
                # Remove expired entry
                del st.session_state.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        st.session_state.cache[key] = (datetime.now(), value)
    
    def clear(self):
        """Clear all cached values"""
        st.session_state.cache = {}


class FileValidator:
    """File validation utilities"""
    
    @staticmethod
    def validate_pdf(file_bytes: bytes) -> bool:
        """Validate PDF file"""
        if not file_bytes:
            return False
        
        # Check PDF header
        pdf_header = file_bytes[:4]
        return pdf_header == b'%PDF'
    
    @staticmethod
    def get_file_size_mb(file_bytes: bytes) -> float:
        """Get file size in MB"""
        return len(file_bytes) / (1024 * 1024)
    
    @staticmethod
    def validate_file_size(file_bytes: bytes, max_size_mb: float = 50) -> bool:
        """Validate file size"""
        size_mb = FileValidator.get_file_size_mb(file_bytes)
        return size_mb <= max_size_mb


class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    def handle_api_error(error: Exception, service_name: str = "API"):
        """Handle API errors gracefully"""
        error_msg = str(error)
        
        if "rate limit" in error_msg.lower():
            st.error(f"🚦 {service_name} rate limit exceeded. Please wait a moment and try again.")
        elif "unauthorized" in error_msg.lower() or "invalid api key" in error_msg.lower():
            st.error(f"Invalid API key for {service_name}. Please check your API key.")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            st.error(f"Network error with {service_name}. Please check your connection.")
        else:
            st.error(f"❌ {service_name} error: {error_msg}")
    
    @staticmethod
    def log_error(error: Exception, context: str = ""):
        """Log error for debugging"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        }
        
        # Store in session state for debugging
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append(error_info)
        
        # Keep only last 50 errors
        if len(st.session_state.error_log) > 50:
            st.session_state.error_log = st.session_state.error_log[-50:]


class RateLimiter:
    """Rate limiting utilities"""
    
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        return len(self.request_times) < self.requests_per_minute
    
    def record_request(self):
        """Record a request"""
        self.request_times.append(time.time())
    
    def get_wait_time(self) -> int:
        """Get time to wait before next request"""
        if not self.request_times:
            return 0
        
        oldest_request = min(self.request_times)
        wait_time = 60 - (time.time() - oldest_request)
        return max(0, int(wait_time))


class ProgressTracker:
    """Progress tracking utilities"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = None
        self.status_text = None
    
    def start(self):
        """Start progress tracking"""
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, step: int, message: str):
        """Update progress"""
        self.current_step = step
        progress = step / self.total_steps
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text:
            self.status_text.text(f"Step {step}/{self.total_steps}: {message}")
    
    def complete(self, message: str = "Complete!"):
        """Mark as complete"""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        
        if self.status_text:
            self.status_text.success(message)


class DataExporter:
    """Data export utilities"""
    
    @staticmethod
    def export_analysis_to_json(analysis_data: Dict) -> str:
        """Export analysis data to JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "dread_rising_version": "1.0.0",
            "data": analysis_data
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    @staticmethod
    def export_papers_to_csv(papers: List[Dict]) -> str:
        """Export papers to CSV format"""
        if not papers:
            return ""
        
        csv_lines = ["Title,Authors,Published,Category,Abstract,arXiv_ID,PDF_URL"]
        
        for paper in papers:
            title = paper.get('title', '').replace(',', ';')
            authors = '; '.join(paper.get('authors', []))
            published = paper.get('published', '')
            category = paper.get('primary_category', '')
            abstract = paper.get('abstract', '').replace(',', ';').replace('\n', ' ')[:200]
            arxiv_id = paper.get('id', '')
            pdf_url = paper.get('pdf_url', '')
            
            csv_line = f'"{title}","{authors}","{published}","{category}","{abstract}","{arxiv_id}","{pdf_url}"'
            csv_lines.append(csv_line)
        
        return '\n'.join(csv_lines)
    
    @staticmethod
    def create_research_report(
        analysis: str,
        papers: List[Dict],
        web_results: List[Dict],
        summary: str = ""
    ) -> str:
        """Create comprehensive research report"""
        report = f"""
# Dread Rising Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Document Analysis
{analysis}

## Related Literature ({len(papers)} papers found)
"""
        
        for i, paper in enumerate(papers[:10], 1):
            authors = ', '.join(paper.get('authors', [])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += ' et al.'
            
            report += f"""
### {i}. {paper.get('title', 'Unknown Title')}
**Authors:** {authors}
**Published:** {paper.get('published', 'Unknown')}
**Category:** {paper.get('primary_category', 'Unknown')}
**Abstract:** {paper.get('abstract', 'No abstract available')[:300]}...
**arXiv ID:** {paper.get('id', 'Unknown')}
**PDF:** {paper.get('pdf_url', 'Not available')}
"""
        
        if web_results:
            report += f"\n## Web Resources ({len(web_results)} found)\n"
            for i, result in enumerate(web_results[:5], 1):
                report += f"""
### {i}. {result.get('title', 'Unknown Title')}
**URL:** {result.get('url', 'Unknown')}
**Summary:** {result.get('snippet', 'No summary available')}
"""
        
        if summary:
            report += f"\n## Research Summary\n{summary}\n"
        
        report += "\n---\nGenerated by Dread Rising - Your AI-Powered Research Companion"
        
        return report


class ColorUtils:
    """Color utility functions"""
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to RGB tuple (0-1 range)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
    
    @staticmethod
    def get_highlight_colors() -> Dict[str, str]:
        """Get predefined highlight colors"""
        return {
            "Yellow": "#FFFF00",
            "Green": "#00FF00",
            "Blue": "#00BFFF",
            "Pink": "#FF69B4",
            "Orange": "#FFA500",
            "Purple": "#9370DB",
            "Red": "#FF6B6B"
        }


class SessionManager:
    """Manage Streamlit session state"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        defaults = {
            'analysis_result': {},
            'papers_result': {},
            'web_results': {},
            'highlights': [],
            'highlighted_pdf': None,
            'research_summary': '',
            'error_log': [],
            'cache': {},
            'chat_messages': [],  # Global chat history
            'current_context': '',  # Current research context
            'user_preferences': {}  # User preferences and settings
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_session():
        """Clear session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    @staticmethod
    def get_session_info() -> Dict:
        """Get session information"""
        return {
            "session_keys": list(st.session_state.keys()),
            "cache_size": len(st.session_state.get('cache', {})),
            "error_count": len(st.session_state.get('error_log', [])),
            "has_analysis": bool(st.session_state.get('analysis_result')),
            "has_papers": bool(st.session_state.get('papers_result')),
            "has_highlights": bool(st.session_state.get('highlights'))
        }
    
    @staticmethod
    def export_session() -> str:
        """Export complete session data as JSON string"""
        session_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "analyzed_documents": st.session_state.get('analyzed_documents', []),
            "analysis_result": st.session_state.get('analysis_result', {}),
            "papers_result": st.session_state.get('papers_result', {}),
            "web_results": st.session_state.get('web_results', {}),
            "highlights": st.session_state.get('highlights', []),
            "chat_messages": st.session_state.get('chat_messages', []),
            "current_context": st.session_state.get('current_context', ''),
            "uploaded_documents": [],  # Don't include file bytes in export
            "current_document_name": st.session_state.get('current_document_name', ''),
            "use_google_search": st.session_state.get('use_google_search', False)
        }
        
        # Add document metadata (without actual file bytes)
        uploaded_docs = st.session_state.get('uploaded_documents', [])
        for doc in uploaded_docs:
            doc_meta = {
                'name': doc.get('name', ''),
                'hash': doc.get('hash', ''),
                'uploaded_at': doc.get('uploaded_at', ''),
                'size_mb': len(doc.get('bytes', b'')) / (1024 * 1024) if doc.get('bytes') else 0
            }
            session_data['uploaded_documents'].append(doc_meta)
        
        return json.dumps(session_data, indent=2, default=str)
    
    @staticmethod
    def import_session(session_json: str) -> bool:
        """Import session data from JSON string"""
        try:
            session_data = json.loads(session_json)
            
            # Validate version
            if session_data.get("version") != "1.0":
                st.warning(f"Session version {session_data.get('version')} may not be fully compatible")
            
            # Restore session state
            st.session_state.analyzed_documents = session_data.get('analyzed_documents', [])
            st.session_state.analysis_result = session_data.get('analysis_result', {})
            st.session_state.papers_result = session_data.get('papers_result', {})
            st.session_state.web_results = session_data.get('web_results', {})
            st.session_state.highlights = session_data.get('highlights', [])
            st.session_state.chat_messages = session_data.get('chat_messages', [])
            st.session_state.current_context = session_data.get('current_context', '')
            st.session_state.current_document_name = session_data.get('current_document_name', '')
            st.session_state.use_google_search = session_data.get('use_google_search', False)
            
            # Note: uploaded_documents with file bytes are not restored
            # Users will need to re-upload files if they want to analyze new documents
            
            return True
            
        except json.JSONDecodeError as e:
            st.error(f"Invalid session file format: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error importing session: {str(e)}")
            return False
    
    @staticmethod
    def get_session_summary() -> str:
        """Get a summary of current session data"""
        analyzed_docs = len(st.session_state.get('analyzed_documents', []))
        total_papers = st.session_state.get("papers_result", {}).get("total_found", 0)
        chat_messages = len(st.session_state.get('chat_messages', []))
        highlights = len(st.session_state.get('highlights', []))
        
        summary = f"""
**Current Session Summary:**
- Documents analyzed: {analyzed_docs}
- Related papers found: {total_papers}
- Chat messages: {chat_messages}
- Highlights: {highlights}
        """
        return summary.strip()


class ChatContextManager:
    """Manage chat context and continuity across sessions"""
    
    @staticmethod
    def update_context(new_analysis: str = "", papers: list = None, web_results: list = None, document_name: str = ""):
        """Update the global research context - accumulate instead of replace"""
        
        # Initialize documents list if not exists
        if 'analyzed_documents' not in st.session_state:
            st.session_state.analyzed_documents = []
        
        context_parts = []
        
        if new_analysis and document_name:
            # Add to documents list
            doc_entry = {
                'name': document_name,
                'analysis': new_analysis[:500] + "..." if len(new_analysis) > 500 else new_analysis,
                'keywords_count': len(st.session_state.get("analysis_result", {}).get("keywords", [])),
                'analyzed_at': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            # Check if document already exists, update or add
            existing_doc_index = None
            for i, doc in enumerate(st.session_state.analyzed_documents):
                if doc['name'] == document_name:
                    existing_doc_index = i
                    break
            
            if existing_doc_index is not None:
                st.session_state.analyzed_documents[existing_doc_index] = doc_entry
            else:
                st.session_state.analyzed_documents.append(doc_entry)
            
            # Limit to last 3 documents to avoid too much context 
            # TODO : Make this configurable
            # or allow user to clear context
            if len(st.session_state.analyzed_documents) > 3:
                st.session_state.analyzed_documents = st.session_state.analyzed_documents[-3:]
        
        # Build context from all analyzed documents
        if st.session_state.get('analyzed_documents'):
            context_parts.append("=== ANALYZED DOCUMENTS ===")
            for doc in st.session_state.analyzed_documents:
                context_parts.append(f"Document: {doc['name']} (analyzed: {doc['analyzed_at']})")
                context_parts.append(f"Key insights: {doc['analysis']}")
                context_parts.append(f"Keywords found: {doc['keywords_count']}")
                context_parts.append("---")
        
        total_papers = st.session_state.get("papers_result", {}).get("total_found", 0)
        if total_papers > 0:
            context_parts.append("=== RESEARCH PAPERS ===")
            context_parts.append(f"Total papers found: {total_papers}")
            papers = st.session_state.get("papers_result", {}).get("papers", [])
            if papers:
                recent_papers = papers[:3]
                for paper in recent_papers:
                    context_parts.append(f"• {paper.get('title', 'Unknown')[:60]}...")
        
        web_count = len(st.session_state.get("web_results", {}).get("web_results", []))
        if web_count > 0:
            context_parts.append("=== WEB RESOURCES ===")
            context_parts.append(f"Additional web sources: {web_count}")
        
        st.session_state.current_context = "\n".join(context_parts)
    
    @staticmethod
    def get_context() -> str:
        """Get current research context"""
        return st.session_state.get('current_context', '')
    
    @staticmethod
    def add_chat_message(role: str, content: str, metadata: Dict = None):
        """Add a message to global chat history with optional metadata"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
        
        st.session_state.chat_messages.append(message)
    
    @staticmethod
    def get_chat_history(limit: int = 10) -> list:
        """Get recent chat history"""
        messages = st.session_state.get('chat_messages', [])
        return messages[-limit:] if len(messages) > limit else messages
    
    @staticmethod
    def clear_chat():
        """Clear chat history"""
        st.session_state.chat_messages = []
    
    @staticmethod
    def export_chat() -> str:
        """Export chat history as text"""
        messages = st.session_state.get('chat_messages', [])
        
        export_text = "Dread Rising - Chat Export\n"
        export_text += "=" * 40 + "\n\n"
        
        for msg in messages:
            role = "You" if msg["role"] == "user" else "DRing Assistant"
            timestamp = msg.get("timestamp", "")
            content = msg["content"]
            
            export_text += f"[{timestamp}] {role}:\n{content}\n\n"
        
        return export_text
