"""
PDF processing utilities for Dread Rising
"""
import fitz  # PyMuPDF
import streamlit as st
from typing import List, Dict, Tuple
import re
import json


class PDFProcessor:
    """Handle PDF processing operations"""
    
    def __init__(self):
        self.document = None
    
    def load_pdf(self, pdf_bytes: bytes) -> bool:
        """Load PDF from bytes"""
        try:
            self.document = fitz.open("pdf", pdf_bytes)
            return True
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False
    
    def extract_text(self) -> str:
        """Extract all text from PDF"""
        if not self.document:
            return ""
        
        try:
            text = ""
            for page in self.document:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def get_page_count(self) -> int:
        """Get number of pages in PDF"""
        if not self.document:
            return 0
        return len(self.document)
    
    def get_metadata(self) -> Dict:
        """Get PDF metadata"""
        if not self.document:
            return {}
        
        try:
            metadata = self.document.metadata
            return { # ignore the type warnings here lmao plsssss 
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "pages": self.get_page_count()
            }
        except Exception as e:
            st.error(f"Error getting metadata: {str(e)}")
            return {}
    
    def highlight_text(
        self, 
        highlights: List[Dict[str, str]], 
        highlight_color: Tuple[float, float, float] = (1, 1, 0)
    ) -> bytes:
        """Highlight specified text in PDF"""
        if not self.document:
            return b""
        
        try:
            for highlight_info in highlights:
                text_to_highlight = highlight_info.get("text", "").strip()
                if not text_to_highlight:
                    continue
                
                self._highlight_text_instances(text_to_highlight, highlight_color)
            
            # modified PDF bytes
            modified_pdf_bytes = self.document.write()
            return modified_pdf_bytes
        except Exception as e:
            st.error(f"Error highlighting PDF: {str(e)}")
            return b""
    
    def _highlight_text_instances(
        self, 
        text: str, 
        color: Tuple[float, float, float]
    ):
        """Find and highlight all instances of text"""
        for page_num in range(len(self.document)):
            page = self.document[page_num]
            
            # Try exact match first
            text_instances = page.search_for(text)
            
            # If no exact match, try cleaned text
            if not text_instances:
                cleaned_text = ' '.join(text.split())
                text_instances = page.search_for(cleaned_text)
            
            # Try sentence-by-sentence
            if not text_instances:
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
                for sentence in sentences:
                    if len(sentence) > 10:
                        sentence_instances = page.search_for(sentence)
                        if sentence_instances:
                            text_instances.extend(sentence_instances)
            
            # Try phrase matching
            if not text_instances:
                words = text.split()
                if len(words) >= 5:
                    for i in range(len(words) - 4):
                        phrase = ' '.join(words[i:i+5])
                        phrase_instances = page.search_for(phrase)
                        if phrase_instances:
                            text_instances.extend(phrase_instances)
                            break
            
            # Highlight instances
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors({"stroke": color})
                highlight.update()
    
    def extract_citations(self) -> List[str]:
        """Extract potential citations from PDF"""
        text = self.extract_text()
        if not text:
            return []
        
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+, \d{4})\)',  # (Author, 2023)
            r'([A-Z][a-z]+ et al\., \d{4})',  # Author et al., 2023
            r'([A-Z][a-z]+ & [A-Z][a-z]+, \d{4})',  # Author & Author, 2023
        ]
        
        citations = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.update(matches)
        
        return list(citations)
    
    def extract_keywords(self, max_keywords: int = 20) -> List[str]:
        """Extract potential keywords from PDF"""
        text = self.extract_text().lower()
        if not text:
            return []
        
        # Remove common words and extract meaningful terms
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'we', 'they', 'it', 'he', 'she'
        }
        
        # Extract words (3+ characters, alphanumeric)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        word_freq = {}
        for word in words:
            if word not in common_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def close(self):
        """Close the PDF document"""
        if self.document:
            self.document.close()
            self.document = None


class HighlightParser:
    """Parse highlighting instructions from AI responses"""
    
    @staticmethod
    def parse_response(response_text: str) -> List[Dict[str, str]]:
        """Parse AI response to extract highlighting instructions"""
        try:
            #  DEBUG
            # st.write(f"Debug: Parsing response of length {len(response_text)}")
            # st.write(f"Response preview: {response_text[:300]}...")
            
            json_patterns = [
                r'\[[\s\S]*?\]',  # Standard array pattern
                r'```json\s*(\[[\s\S]*?\])\s*```',  # Markdown code block
                r'```\s*(\[[\s\S]*?\])\s*```',  # Generic code block
                r'(\[[\s\S]*?\])',  # Simple array capture
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if json_match.groups() else json_match.group(0)
                    json_str = json_str.strip()
                    
                    try:
                        json_str = json_str.replace('```json', '').replace('```', '').strip()
                        
                        highlights = json.loads(json_str)
                        
                        if isinstance(highlights, list) and all(
                            isinstance(h, dict) and 'text' in h and 'reason' in h 
                            for h in highlights
                        ):
                            return highlights
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            # Fallback 1: Try to extract and fix JSON manually
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    # Common fixes for malformed JSON
                    json_str = json_str.strip()
                    # Remove trailing commas before closing brackets
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    highlights = json.loads(json_str)
                    if isinstance(highlights, list):
                        return highlights
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Fallback 2: Parse line by line for structured text
            lines = response_text.strip().split('\n')
            highlights = []
            current_text = ""
            current_reason = ""
            
            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    if line.lower().startswith('text:') or line.lower().startswith('"text"'):
                        current_text = line.split(':', 1)[1].strip().strip('"').strip("'")
                    elif line.lower().startswith('reason:') or line.lower().startswith('"reason"'):
                        current_reason = line.split(':', 1)[1].strip().strip('"').strip("'")
                        if current_text and current_reason:
                            highlights.append({
                                "text": current_text,
                                "reason": current_reason
                            })
                            current_text = ""
                            current_reason = ""
            
            # Fallback 3: Extract any quoted strings as potential highlights
            if not highlights:
                quoted_texts = re.findall(r'"([^"]+)"', response_text)
                for i, text in enumerate(quoted_texts):
                    if len(text) > 10:  # Only meaningful text
                        highlights.append({
                            "text": text,
                            "reason": f"Identified content {i+1}"
                        })
            
            return highlights
            
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
            # Return empty list but log what we tried to parse
            st.error(f"Response content preview: {response_text[:500]}...")
            return []
    
    @staticmethod
    def get_highlight_prompts() -> Dict[str, str]:
        """Get predefined prompts for different highlight types"""
        return {
            "key_points": """Analyze this PDF document and identify the most important key points, main ideas, and critical information that should be highlighted. 
            Return a JSON list of objects with 'text' (exact complete sentences or phrases to highlight) and 'reason' (why it's important). 
            Focus on: main arguments, conclusions, key statistics, important definitions, and critical findings.
            Highlight complete sentences or meaningful phrases, not individual words.
            For longer documents, identify 20-30 key highlights to ensure comprehensive coverage.
            Format: [{"text": "complete sentence or phrase from document", "reason": "why important"}, ...]""",
            
            "technical_terms": """Analyze this PDF document and identify technical terms, jargon, acronyms, and specialized vocabulary that should be highlighted for better understanding.
            Return a JSON list of objects with 'text' (complete sentences or phrases containing technical terms) and 'reason' (explanation of the term).
            Highlight the full sentence or context around technical terms, not just the term itself.
            For longer documents, identify 20-30 technical highlights to ensure comprehensive coverage.
            Format: [{"text": "complete sentence or phrase from document", "reason": "definition or explanation"}, ...]""",
            
            "action_items": """Analyze this PDF document and identify action items, tasks, deadlines, requirements, and actionable information that should be highlighted.
            Return a JSON list of objects with 'text' (complete sentences or phrases describing actions) and 'reason' (what action is needed).
            Highlight complete sentences that contain actionable information, not individual words.
            For longer documents, identify 20-30 action-related highlights to ensure comprehensive coverage.
            Format: [{"text": "complete sentence or phrase from document", "reason": "action required"}, ...]""",
            
            "important_data": """Analyze this PDF document and identify important data points, statistics, numbers, dates, and quantitative information that should be highlighted.
            Return a JSON list of objects with 'text' (complete sentences or phrases containing important data) and 'reason' (significance of the data).
            Highlight the full sentence or context around important data, not just the numbers.
            For longer documents, identify 20-30 data-related highlights to ensure comprehensive coverage.
            Format: [{"text": "complete sentence or phrase from document", "reason": "why this data matters"}, ...]""",
            
            "research_gaps": """Analyze this research document and identify potential research gaps, limitations, future work suggestions, and areas for further investigation.
            Return a JSON list of objects with 'text' (sentences mentioning gaps or limitations) and 'reason' (what research opportunity this represents).
            Focus on: stated limitations, future work sections, unresolved questions, and areas needing further study.
            Format: [{"text": "complete sentence or phrase from document", "reason": "research opportunity identified"}, ...]""",
            
            "methodology": """Analyze this research document and identify methodological information, experimental procedures, data collection methods, and analytical approaches.
            Return a JSON list of objects with 'text' (sentences describing methods) and 'reason' (type of methodological information).
            Focus on: experimental design, data collection, analysis methods, tools used, and procedural details.
            Format: [{"text": "complete sentence or phrase from document", "reason": "methodological aspect"}, ...]"""
        }
