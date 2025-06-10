"""
Main application pages for Dread Rising
"""
import streamlit as st
from typing import Dict, List
from datetime import datetime
import plotly.express as px
import pandas as pd
from src.ui.components import UIComponents
from src.core.gemini_client import GeminiClient
from src.core.pdf_processor import PDFProcessor, HighlightParser
from src.services.research_service import ResearchAnalyzer
from src.services.arxiv_service import ArxivService
from src.services.search_service import WebSearchService
from src.utils.helpers import ChatContextManager


class UnifiedResearchPage:
    """Unified research page - single interface for all research needs"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.highlight_parser = HighlightParser()
        self.arxiv_service = ArxivService()
        self.search_service = WebSearchService()
    
    def render(self, config: Dict):
        """Render the unified research interface"""
        UIComponents.render_header()
        
        if not config.get("api_key"):
            st.error("Please enter your Gemini API key in the sidebar to get started")
            st.markdown("### Quick Start:")
            st.markdown("1. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
            st.markdown("2. Enter it in the sidebar")
            st.markdown("3. Upload a PDF and start chatting!")
            return
        
        gemini_client = GeminiClient(config["api_key"])
        if not gemini_client.is_connected():
            st.error("Failed to connect to Gemini API. Please check your API key.")
            return
        
        st.session_state.gemini_client_instance = gemini_client
        st.session_state.current_config = config
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_document_upload(gemini_client, config)
        
        with col2:
            self._render_chat_interface(gemini_client)
        
        self._render_research_insights()
    
    def _render_document_upload(self, gemini_client: GeminiClient, config: Dict):
        """Render document upload and quick actions"""
        st.markdown("### Upload & Analyze")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload any research paper or document"
        )
        
        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.success(f"File: {uploaded_file.name}")
            st.caption(f"Size: {file_size:.1f} MB")
            
            # Track uploaded documents
            if 'uploaded_documents' not in st.session_state:
                st.session_state.uploaded_documents = []
            
            current_file_hash = hash(uploaded_file.getvalue())
            if not any(doc['hash'] == current_file_hash for doc in st.session_state.uploaded_documents):
                doc_info = {
                    'name': uploaded_file.name,
                    'hash': current_file_hash,
                    'bytes': uploaded_file.getvalue(),
                    'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.uploaded_documents.append(doc_info)
            
            if st.button("Analyze Document", type="primary", use_container_width=True):
                self._auto_analyze_document(uploaded_file.getvalue(), gemini_client, config, uploaded_file.name)
        
        if st.session_state.get('uploaded_documents'):
            st.markdown("---")
            st.markdown("### Previous Documents")
            for i, doc in enumerate(st.session_state.uploaded_documents[-3:]):  # Show last 3
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"File: {doc['name']} ({doc['uploaded_at']})")
                with col2:
                    if st.button("Re-analyze", key=f"reanalyze_{i}", use_container_width=True):
                        self._auto_analyze_document(doc['bytes'], gemini_client, config, doc['name'])
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.session_state.get('analysis_result'):
            if st.button("Find More Similar Papers", use_container_width=True):
                self._find_similar_papers(gemini_client, config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Session", use_container_width=True, help="Save current research session"):
                from src.utils.helpers import SessionManager
                session_data = SessionManager.export_session()
                st.download_button(
                    "Download Session File",
                    data=session_data,
                    file_name=f"research_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="main_download_session"
                )
        
        with col2:
            if st.button("Start Fresh", use_container_width=True, help="Clear all data and start over"):
                for key in ["analysis_result", "highlights", "papers_result", "web_results", "chat_messages", "analyzed_documents", "current_context"]:
                    if key in st.session_state:
                        del st.session_state[key]
                ChatContextManager.clear_chat()
                st.rerun()
        
        if st.session_state.get('chat_messages'):
            chat_export = ChatContextManager.export_chat()
            st.download_button(
                "Export Chat Only",
                data=chat_export,
                file_name=f"research_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Export just the chat conversation"
            )
        
        # Context summary - Enhanced to show all analyzed documents
        context = ChatContextManager.get_context()
        if context:
            st.markdown("### Research Summary")
            
            # Show analyzed documents in tabs
            analyzed_docs = st.session_state.get('analyzed_documents', [])
            if analyzed_docs:
                if len(analyzed_docs) == 1:
                    doc = analyzed_docs[0]
                    st.markdown(f"**Document:** {doc['name']}")
                    st.markdown(f"**Analyzed:** {doc['analyzed_at']}")
                    st.markdown(f"**Keywords:** {doc['keywords_count']}")
                    with st.expander("Analysis Preview", expanded=False):
                        st.text(doc['analysis'])
                else:
                    # Multiple documents - use tabs
                    tab_names = [f"{doc['name'][:20]}..." if len(doc['name']) > 20 else doc['name'] for doc in analyzed_docs]
                    tabs = st.tabs(tab_names)
                    
                    for i, (tab, doc) in enumerate(zip(tabs, analyzed_docs)):
                        with tab:
                            st.markdown(f"**Analyzed:** {doc['analyzed_at']}")
                            st.markdown(f"**Keywords:** {doc['keywords_count']}")
                            st.text_area("Analysis", doc['analysis'], height=100, disabled=True, key=f"doc_analysis_{i}")
            
            # Quick stats
            total_papers = st.session_state.get("papers_result", {}).get("total_found", 0)
            web_resources = len(st.session_state.get("web_results", {}).get("web_results", []))
            
            if total_papers > 0 or web_resources > 0:
                col1, col2 = st.columns(2)
                with col1:
                    if total_papers > 0:
                        st.metric("Related Papers", total_papers)
                with col2:
                    if web_resources > 0:
                        st.metric("Web Resources", web_resources)
        else:
            st.info("Upload and analyze documents to see research summary here")
    
    def _render_chat_interface(self, gemini_client: GeminiClient):
        """Render the main chat interface"""
        st.markdown("### DRing Assistant")
        
        # Get current research context
        context = ChatContextManager.get_context()
        
        # Status indicator
        if context:
            st.success("Ready to discuss your research!")
        else:
            st.info("Upload a document or ask me anything!")
        
        # Google Search toggle
        use_search = st.checkbox(
            "Use Google Search for real-time information",
            value=st.session_state.get('use_google_search', False),
            help="Enable to get up-to-date information from the web"
        )
        st.session_state.use_google_search = use_search
        
        # Main chat interface
        UIComponents.render_chat_interface(gemini_client, context, use_search)
    
    def _auto_analyze_document(self, pdf_bytes: bytes, gemini_client: GeminiClient, config: Dict, filename: str = "document"):
        """Automatically analyze document and discover related research"""
        with st.status("Analyzing your document...", expanded=True) as status:
            research_analyzer = ResearchAnalyzer(gemini_client)
            
            # Step 1: Document Analysis
            st.write("Analyzing document content...")
            analysis_result = research_analyzer.analyze_document(pdf_bytes, config["analysis_type"])
            
            if "error" in analysis_result:
                st.error(f"Analysis failed: {analysis_result['error']}")
                return
            
            st.session_state.analysis_result = analysis_result
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.current_document_name = filename
            
            # Step 2: Generate Highlights
            st.write("Finding key highlights...")
            highlights = self._generate_highlights(pdf_bytes, gemini_client, config)
            if highlights:
                st.session_state.highlights = highlights
            
            # Step 3: Smart Research Discovery
            st.write("Discovering related research...")
            papers_result = research_analyzer.discover_related_papers(
                analysis_result.get("keywords", []),
                analysis_result.get("research_topics", []),
                config["max_papers"]
            )
            st.session_state.papers_result = papers_result
            
            # Step 4: Web Research (if enabled)
            web_results = {}
            if config.get("search_web_resources"):
                st.write("Finding additional resources...")
                queries = research_analyzer.generate_literature_queries(analysis_result["analysis"])
                web_results = research_analyzer.search_web_resources(queries)
                st.session_state.web_results = web_results
            
            # Update chat context
            ChatContextManager.update_context(
                new_analysis=analysis_result.get("analysis", ""),
                papers=papers_result.get("papers", []),
                web_results=web_results.get("web_results", []),
                document_name=filename
            )
            
            # Add AI message about the analysis
            analysis_summary = f"I've analyzed '{filename}'! Here's what I found:\n\n"
            analysis_summary += f"**Key Insights:** {len(analysis_result.get('keywords', []))} keywords, {len(analysis_result.get('research_topics', []))} research topics\n"
            analysis_summary += f"**Related Papers:** Found {papers_result.get('total_found', 0)} relevant papers\n"
            if highlights:
                analysis_summary += f"**Highlights:** {len(highlights)} important sections identified\n"
            analysis_summary += "\nWhat would you like to explore first?"
            
            ChatContextManager.add_chat_message("assistant", analysis_summary)
            
            status.update(label="Analysis complete!", state="complete")
        
        st.rerun()
    
    def _generate_highlights(self, pdf_bytes: bytes, gemini_client: GeminiClient, config: Dict):
        """Generate highlights for PDF"""
        highlight_prompts = self.highlight_parser.get_highlight_prompts()
        prompt = highlight_prompts.get(config["highlight_type"])
        
        if not prompt:
            return []
        
        response = gemini_client.analyze_pdf(pdf_bytes, prompt)
        if response:
            return self.highlight_parser.parse_response(response)
        
        return []
    
    def _render_research_insights(self):
        """Render research insights if available"""
        if not any(key in st.session_state for key in ["analysis_result", "papers_result", "highlights", "analyzed_documents"]):
            return
        
        # Show all analyzed documents summary
        analyzed_docs = st.session_state.get('analyzed_documents', [])
        if analyzed_docs:
            st.markdown("---")
            if len(analyzed_docs) == 1:
                st.markdown(f"### Analysis of '{analyzed_docs[0]['name']}'")
                doc = analyzed_docs[0]
                
                # Display key findings from analysis
                analysis_result = st.session_state.get("analysis_result", {})
                if analysis_result.get("analysis"):
                    with st.expander("Detailed Analysis", expanded=True):
                        st.markdown(analysis_result["analysis"])
                
                # Show keywords and topics for current document
                col1, col2 = st.columns(2)
                with col1:
                    keywords = analysis_result.get("keywords", [])
                    if keywords:
                        st.markdown("**Key Terms Found:**")
                        keyword_tags = " ".join([f"`{keyword}`" for keyword in keywords[:10]])
                        st.markdown(keyword_tags)
                
                with col2:
                    topics = analysis_result.get("research_topics", [])
                    if topics:
                        st.markdown("**Research Topics:**")
                        topic_tags = " ".join([f"`{topic}`" for topic in topics[:8]])
                        st.markdown(topic_tags)
            else:
                st.markdown(f"### Analysis of {len(analyzed_docs)} Documents")
                
                # Show summary of all documents
                for i, doc in enumerate(analyzed_docs):
                    with st.expander(f"{doc['name']} ({doc['analyzed_at']})", expanded=i == len(analyzed_docs)-1):
                        st.markdown(f"**Keywords found:** {doc['keywords_count']}")
                        st.markdown("**Analysis preview:**")
                        st.markdown(doc['analysis'])
                
                # Show combined keywords and topics from current session
                analysis_result = st.session_state.get("analysis_result", {})
                if analysis_result:
                    col1, col2 = st.columns(2)
                    with col1:
                        keywords = analysis_result.get("keywords", [])
                        if keywords:
                            st.markdown("**Latest Document - Key Terms:**")
                            keyword_tags = " ".join([f"`{keyword}`" for keyword in keywords[:10]])
                            st.markdown(keyword_tags)
                    
                    with col2:
                        topics = analysis_result.get("research_topics", [])
                        if topics:
                            st.markdown("**Latest Document - Research Topics:**")
                            topic_tags = " ".join([f"`{topic}`" for topic in topics[:8]])
                            st.markdown(topic_tags)
        
        # Fallback for legacy single document view
        elif st.session_state.get("current_document_name"):
            st.markdown("---")
            st.markdown(f"### Analysis of '{st.session_state.current_document_name}'")
            
            # Display key findings from analysis
            analysis_result = st.session_state.get("analysis_result", {})
            if analysis_result.get("analysis"):
                with st.expander("Detailed Analysis", expanded=True):
                    st.markdown(analysis_result["analysis"])
            
            # Show keywords and topics
            col1, col2 = st.columns(2)
            with col1:
                keywords = analysis_result.get("keywords", [])
                if keywords:
                    st.markdown("**Key Terms Found:**")
                    keyword_tags = " ".join([f"`{keyword}`" for keyword in keywords[:10]])
                    st.markdown(keyword_tags)
            
            with col2:
                topics = analysis_result.get("research_topics", [])
                if topics:
                    st.markdown("**Research Topics:**")
                    topic_tags = " ".join([f"`{topic}`" for topic in topics[:8]])
                    st.markdown(topic_tags)
        
        st.markdown("---")
        st.markdown("### Research Dashboard")
        
        # Calculate comprehensive metrics from ALL analyzed documents
        analyzed_docs = st.session_state.get('analyzed_documents', [])
        
        # Total keywords across all documents
        total_keywords = 0
        if analyzed_docs:
            for doc in analyzed_docs:
                total_keywords += doc.get('keywords_count', 0)
        else:
            # Fallback to current session if no analyzed_documents
            total_keywords = len(st.session_state.get("analysis_result", {}).get("keywords", []))
        
        # Papers count (cumulative)
        papers_count = st.session_state.get("papers_result", {}).get("total_found", 0)
        
        # Highlights count (cumulative)
        highlights_count = len(st.session_state.get("highlights", []))
        
        # Web resources count (cumulative)
        web_count = len(st.session_state.get("web_results", {}).get("web_results", []))
        
        # Documents analyzed count
        docs_analyzed = len(analyzed_docs)
        
        # Summary metrics - now showing comprehensive data
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Documents", docs_analyzed, help="Total documents analyzed in this session")
        
        with col2:
            st.metric("Keywords", total_keywords, help="Total keywords found across all documents")
        
        with col3:
            st.metric("Papers Found", papers_count, help="Related research papers discovered")
        
        with col4:
            st.metric("Highlights", highlights_count, help="Important sections highlighted")
        
        with col5:
            st.metric("Web Resources", web_count, help="Additional web resources found")
        
        # Enhanced papers display with charts
        if papers_count > 0:
            # Create papers by category chart
            papers = st.session_state.papers_result.get("papers", [])
            categorized = st.session_state.papers_result.get("categorized_papers", {})
            
            if categorized and len(categorized) > 1:
                try:
                    # Category distribution chart
                    cat_data = [(cat, len(papers_list)) for cat, papers_list in categorized.items()]
                    df = pd.DataFrame(cat_data, columns=['Category', 'Count'])
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        fig = px.pie(df, values='Count', names='Category', 
                                   title=f"Distribution of {papers_count} Related Papers")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig_bar = px.bar(df, x='Category', y='Count', 
                                       title="Papers by Research Area")
                        st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    st.info("Charts not available - visualization libraries need setup")
            
            # Show ALL papers
            with st.expander(f"View All {papers_count} Related Papers"):
                for i, paper in enumerate(papers):
                    st.markdown(f"**{i+1}. {paper.get('title', 'Unknown Title')}**")
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        authors = paper.get('authors', [])
                        if authors:
                            author_str = ", ".join(authors[:2])
                            if len(authors) > 2:
                                author_str += " et al."
                            st.caption(f"Authors: {author_str} | Published: {paper.get('published', 'Unknown')[:4]}")
                    
                    with col2:
                        if paper.get('pdf_url'):
                            st.link_button("PDF", paper['pdf_url'])
                    
                    with col3:
                        st.caption(f"Category: {paper.get('primary_category', 'Unknown')}")
                    
                    # Show abstract preview
                    abstract = paper.get('abstract', '')
                    if abstract:
                        st.text(abstract[:150] + "..." if len(abstract) > 150 else abstract)
                    
                    if i < len(papers) - 1:
                        st.divider()
        
        if st.session_state.get("highlights") or st.session_state.get("analysis_result"):
            st.markdown("### Downloads")
            
            # Debug information for troubleshooting
            # with st.expander("Debug Information", expanded=False):
            #     st.markdown("**Troubleshooting 'Generate Highlighted PDF' button:**")
                
            #     # Check highlights
            #     highlights = st.session_state.get("highlights", [])
            #     has_highlights = bool(highlights)
            #     st.markdown(f"- **Highlights available:** {'Yes' if has_highlights else 'No'} ({len(highlights)} highlights)")
                
            #     # Check PDF bytes
            #     pdf_bytes = st.session_state.get("pdf_bytes")
            #     has_pdf = bool(pdf_bytes)
            #     st.markdown(f"- **PDF bytes stored:** {'Yes' if has_pdf else 'No'}")
                
            #     # Show what config was used for highlight generation
            #     current_config = st.session_state.get("current_config", {})
            #     highlight_type = current_config.get("highlight_type", "Not set")
            #     st.markdown(f"- **Highlight type:** {highlight_type}")
                
            #     # Show analysis result status
            #     analysis_result = st.session_state.get("analysis_result", {})
            #     has_analysis = bool(analysis_result)
            #     st.markdown(f"- **Analysis completed:** {'Yes' if has_analysis else 'No'}")
                
            #     if not has_highlights and has_analysis:
            #         st.warning("**Issue detected:** Document was analyzed but no highlights were generated. This could be due to:")
            #         st.markdown("1. Highlight generation failed during analysis")
            #         st.markdown("2. AI response couldn't be parsed")
            #         st.markdown("3. No suitable content found for highlighting")
                    
            #         if highlight_type not in ["key_points", "research_gaps", "methodology", "technical_terms", "action_items", "important_data"]:
            #             st.error(f"**Configuration issue:** highlight_type '{highlight_type}' is not supported")
                
            #     if not has_pdf:
            #         st.warning("**Issue detected:** No PDF bytes stored. Try re-uploading and re-analyzing the document.")
            
            # Standalone highlight generation section
            if st.session_state.get("pdf_bytes") and st.session_state.get("gemini_client_instance"):
                st.markdown("**Generate Highlights:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    manual_highlight_type = st.selectbox(
                        "Choose highlight type:",
                        ["key_points", "research_gaps", "methodology", "technical_terms", "action_items", "important_data"],
                        key="manual_highlight_type"
                    )
                with col_b:
                    if st.button("Generate Highlights", key="manual_highlights", use_container_width=True):
                        with st.spinner("Generating highlights..."):
                            manual_config = {"highlight_type": manual_highlight_type}
                            highlights = self._generate_highlights(
                                st.session_state.pdf_bytes, 
                                st.session_state.gemini_client_instance, 
                                manual_config
                            )
                            if highlights:
                                st.session_state.highlights = highlights
                                st.success(f"Successfully generated {len(highlights)} highlights!")
                                st.rerun()
                            else:
                                st.error("Failed to generate highlights. Check the debug info above.")
                with col_c:
                    if st.button("Highlight All Types", key="highlight_all", use_container_width=True):
                        with st.spinner("Generating highlights for all types..."):
                            all_highlights = []
                            highlight_types = ["key_points", "research_gaps", "methodology", "technical_terms", "action_items", "important_data"]
                            
                            for highlight_type in highlight_types:
                                manual_config = {"highlight_type": highlight_type}
                                highlights = self._generate_highlights(
                                    st.session_state.pdf_bytes, 
                                    st.session_state.gemini_client_instance, 
                                    manual_config
                                )
                                if highlights:
                                    # Add type prefix to each highlight for identification
                                    for highlight in highlights:
                                        highlight['type'] = highlight_type
                                    all_highlights.extend(highlights)
                            
                            if all_highlights:
                                st.session_state.highlights = all_highlights
                                st.success(f"Successfully generated {len(all_highlights)} highlights across all types!")
                                st.rerun()
                            else:
                                st.error("Failed to generate highlights for any type. Check the debug info above.")
            elif st.session_state.get("analysis_result"):
                st.info("Re-upload your document to enable highlight generation.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                has_highlights = bool(st.session_state.get("highlights"))
                has_pdf = bool(st.session_state.get("pdf_bytes"))
                
                if has_highlights and has_pdf:
                    if st.button("Generate Highlighted PDF", use_container_width=True):
                        current_config = st.session_state.get("current_config", {"highlight_color": "#FFFF00"})
                        self._create_highlighted_pdf(current_config)
                else:
                    missing_items = []
                    if not has_highlights:
                        missing_items.append("highlights")
                    if not has_pdf:
                        missing_items.append("PDF data")
                    
                    st.button(
                        "Generate Highlighted PDF",
                        use_container_width=True, 
                        disabled=True,
                        help=f"Missing: {', '.join(missing_items)}. Use the 'Generate Highlights' section above to create highlights first."
                    )
                    
                    # Show what's missing
                    if missing_items:
                        st.caption(f"Missing: {', '.join(missing_items)}")
                
                # Show download button if highlighted PDF exists
                if st.session_state.get("highlighted_pdf"):
                    st.download_button(
                        "Download Highlighted PDF",
                        data=st.session_state.highlighted_pdf,
                        file_name="highlighted_document.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            with col2:
                if st.session_state.get("analysis_result"):
                    analysis_markdown = self._create_analysis_markdown()
                    st.download_button(
                        "Download Analysis Report",
                        data=analysis_markdown,
                        file_name="analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with col3:
                if st.session_state.get("papers_result"):
                    papers = st.session_state.papers_result.get("papers", [])
                    if papers:
                        bibliography = self._create_bibliography_markdown(papers)
                        st.download_button(
                            "Download Bibliography",
                            data=bibliography,
                            file_name="bibliography.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
    
    def _create_highlighted_pdf(self, config: Dict):
        """Create highlighted PDF"""
        if "highlights" not in st.session_state or "pdf_bytes" not in st.session_state:
            st.error("No highlights or PDF available")
            return
        
        with st.spinner("Generating highlighted PDF..."):
            try:
                # Convert hex color to RGB
                hex_color = config.get("highlight_color", "#FFFF00").lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
                
                # Load and highlight PDF
                self.pdf_processor.load_pdf(st.session_state.pdf_bytes)
                highlighted_pdf = self.pdf_processor.highlight_text(
                    st.session_state.highlights,
                    rgb_color
                )
                
                if highlighted_pdf:
                    st.session_state.highlighted_pdf = highlighted_pdf
                    st.success("Highlighted PDF generated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to generate highlighted PDF")
            except Exception as e:
                st.error(f"Error generating highlighted PDF: {str(e)}")
    
    def _find_similar_papers(self, gemini_client: GeminiClient, config: Dict):
        """Find more similar papers based on current analysis"""
        if not st.session_state.get("analysis_result"):
            st.error("No analysis available to find similar papers")
            return
        
        with st.spinner("Finding more similar papers..."):
            research_analyzer = ResearchAnalyzer(gemini_client)
            analysis_result = st.session_state.analysis_result
            
            # Use existing keywords and topics to find more papers
            papers_result = research_analyzer.discover_related_papers(
                keywords=analysis_result.get("keywords", []),
                research_topics=analysis_result.get("research_topics", []),
                max_papers=10
            )
            
            if papers_result and papers_result.get("papers"):
                # Check for existing papers to avoid duplicates
                existing_papers = st.session_state.get("papers_result", {}).get("papers", [])
                existing_ids = {paper.get("id") for paper in existing_papers}
                
                # Filter out duplicates
                new_papers = [
                    paper for paper in papers_result["papers"] 
                    if paper.get("id") not in existing_ids
                ]
                
                if new_papers:
                    # Update session state with combined papers
                    all_papers = existing_papers + new_papers
                    st.session_state.papers_result = {
                        "papers": all_papers,
                        "total_found": len(all_papers),
                        "categorized_papers": self.arxiv_service.categorize_papers(all_papers)
                    }
                    
                    st.success(f"Found {len(new_papers)} additional similar papers!")
                    st.rerun()
                else:
                    st.info("No new similar papers found. Try adjusting your search terms.")
            else:
                st.warning("No additional papers found.")
    
    def _create_analysis_markdown(self) -> str:
        """Create markdown formatted analysis report"""
        analysis_result = st.session_state.get("analysis_result", {})
        document_name = st.session_state.get("current_document_name", "Unknown Document")
        
        markdown = "# Research Analysis Report\n\n"
        markdown += f"**Document:** {document_name}\n"
        markdown += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Main analysis
        if analysis_result.get("analysis"):
            markdown += "## Detailed Analysis\n\n"
            markdown += analysis_result["analysis"] + "\n\n"
        
        # Keywords
        keywords = analysis_result.get("keywords", [])
        if keywords:
            markdown += "## Key Terms\n\n"
            for keyword in keywords:
                markdown += f"- `{keyword}`\n"
            markdown += "\n"
        
        # Research topics
        topics = analysis_result.get("research_topics", [])
        if topics:
            markdown += "## Research Topics\n\n"
            for topic in topics:
                markdown += f"- {topic}\n"
            markdown += "\n"
        
        # Highlights summary
        highlights = st.session_state.get("highlights", [])
        if highlights:
            markdown += "## Key Highlights\n\n"
            for i, highlight in enumerate(highlights, 1):
                markdown += f"### Highlight {i}\n"
                markdown += f"**Text:** {highlight.get('text', '')}\n\n"
                if highlight.get('reason'):
                    markdown += f"**Reason:** {highlight.get('reason', '')}\n\n"
        
        # Related papers summary
        papers_result = st.session_state.get("papers_result", {})
        if papers_result.get("papers"):
            papers_count = papers_result.get("total_found", 0)
            markdown += f"## Related Literature ({papers_count} papers)\n\n"
            
            papers = papers_result["papers"][:5]  # Top 5 for summary
            for i, paper in enumerate(papers, 1):
                markdown += f"### {i}. {paper.get('title', 'Unknown Title')}\n"
                authors = paper.get('authors', [])
                if authors:
                    author_str = ", ".join(authors[:3])
                    if len(authors) > 3:
                        author_str += " et al."
                    markdown += f"**Authors:** {author_str}\n"
                markdown += f"**Published:** {paper.get('published', 'Unknown')}\n"
                markdown += f"**Category:** {paper.get('primary_category', 'Unknown')}\n"
                if paper.get('pdf_url'):
                    markdown += f"**PDF:** [{paper['pdf_url']}]({paper['pdf_url']})\n"
                markdown += "\n"
        
        return markdown
    
    def _create_bibliography_markdown(self, papers: List[Dict]) -> str:
        """Create markdown formatted bibliography"""
        markdown = "# Bibliography\n\n"
        markdown += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        markdown += f"**Total Papers:** {len(papers)}\n\n"
        
        for i, paper in enumerate(papers, 1):
            authors = paper.get('authors', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            elif len(authors) > 1:
                author_str = ", ".join(authors[:-1]) + f" & {authors[-1]}"
            else:
                author_str = authors[0] if authors else "Unknown Author"
            
            year = paper.get('published', '').split('-')[0] if paper.get('published') else 'Unknown'
            title = paper.get('title', 'Unknown Title')
            arxiv_id = paper.get('id', '')
            
            markdown += f"## {i}. {title}\n\n"
            markdown += f"**Authors:** {author_str}\n\n"
            markdown += f"**Year:** {year}\n\n"
            markdown += f"**arXiv ID:** {arxiv_id}\n\n"
            
            if paper.get('abstract'):
                abstract = paper['abstract'][:300] + "..." if len(paper['abstract']) > 300 else paper['abstract']
                markdown += f"**Abstract:** {abstract}\n\n"
            
            if paper.get('pdf_url'):
                markdown += f"**PDF Link:** [{paper['pdf_url']}]({paper['pdf_url']})\n\n"
            
            markdown += "---\n\n"
        
        return markdown
