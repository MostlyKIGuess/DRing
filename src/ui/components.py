"""
Streamlit UI components for Dread Rising
"""
import streamlit as st
from typing import List, Dict, Optional
import plotly.express as px
import pandas as pd
from datetime import datetime


class UIComponents:
    """Reusable UI components for Dread Rising"""
    
    @staticmethod
    def render_header():
        """Render main application header"""
        import random
        
        taglines = [
            "Feed me theory -> I laugh like Dazai -> You cry at the conference",
            "Upload your draft -> I channel Sartre -> You stare into the void",
            "Drop some theory -> I monologue like Walter -> You cite yourself twice",
            "Feed me jargon -> I spiral like Kafka -> You wake up tenured",
            "Give me papers -> I meow like Schrödinger's cat -> You both pass and fail",
            "Hand over PDFs -> I gaslight like Saul -> You get the last authorship",
            "Sacrifice a paper -> I quote Nietzsche -> You transcend the abstract",
            "Give me citations -> I ghostwrite Camus -> You publish the absurd",
            "Feed me theory -> I laugh like Dazai -> You cry at the conference",
            "Give me docs -> I lie like Saul -> You pass peer review"
        ]
        
        if 'current_tagline' not in st.session_state:
            st.session_state.current_tagline = random.choice(taglines)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <h1>Dread Rising</h1>
            <p style="font-size: 1.1em; color: #666;">{st.session_state.current_tagline}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_config() -> Dict:
        """Render simplified sidebar configuration"""
        with st.sidebar:
            st.header("Setup")
            
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Enter your Google Gemini API key to get started"
            )
            
            if not api_key:
                st.warning("API key required to analyze documents")
                st.markdown("[Get API key](https://aistudio.google.com/app/apikey)")
            
            st.header("Settings")
            
            analysis_focus = st.radio(
                "Analysis Focus",
                ["key_points", "research_gaps", "methodology"],
                index=0,
                format_func=lambda x: {
                    "key_points": "Key Points & Main Ideas",
                    "research_gaps": "Research Gaps & Questions", 
                    "methodology": "Methods & Procedures"
                }[x],
                help="Choose what to focus on when analyzing your document"
            )
            
            # Max papers to find
            max_papers = st.slider("Papers to Find", 5, 15, 8)
            
            # Google Search toggle
            enable_web_search = st.checkbox(
                "Enable Web Search",
                value=False,
                help="Allow the AI to search the web for up-to-date information"
            )
            
            st.header("Session")
            
            from src.utils.helpers import SessionManager
            session_summary = SessionManager.get_session_summary()
            if session_summary.strip() != "**Current Session Summary:**\n- Documents analyzed: 0\n- Related papers found: 0\n- Chat messages: 0\n- Highlights: 0":
                with st.expander("Current Session", expanded=False):
                    st.markdown(session_summary)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Session", use_container_width=True):
                    session_data = SessionManager.export_session()
                    st.download_button(
                        "Download Session",
                        data=session_data,
                        file_name=f"research_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True,
                        key="download_session"
                    )
            
            with col2:
                uploaded_session = st.file_uploader(
                    "Import Session",
                    type="json",
                    help="Upload a previously exported session file",
                    key="session_uploader"
                )
                
                if uploaded_session is not None:
                    session_content = uploaded_session.read().decode('utf-8')
                    if SessionManager.import_session(session_content):
                        st.success("Session imported successfully!")
                        st.rerun()
            
            with st.expander("Advanced Options"):
                highlight_color = st.color_picker("Highlight Color", "#FFFF00")
            
            return {
                "api_key": api_key,
                "analysis_type": "comprehensive",
                "highlight_type": analysis_focus,
                "highlight_color": highlight_color,
                "max_papers": max_papers,
                "search_web_resources": enable_web_search,
                "enable_google_search": enable_web_search
            }
    
    @staticmethod
    def render_file_upload() -> Optional[bytes]:
        """Render file upload widget"""
        st.header("Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a research paper, document, or any PDF to analyze"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.success(f"Uploaded: {uploaded_file.name}")
            st.info(f"File size: {file_size:.2f} MB")
            
            return uploaded_file.getvalue()
        
        return None
    
    @staticmethod
    def render_analysis_results(analysis_result: Dict):
        """Render document analysis results"""
        st.subheader("Document Analysis")
        
        if "error" in analysis_result:
            st.error(f"Error: {analysis_result['error']}")
            return
        
        st.markdown(analysis_result.get("analysis", "No analysis available"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Terms**")
            keywords = analysis_result.get("keywords", [])
            if keywords:
                # Display keywords as tags using markdown
                keyword_tags = " ".join([f"`{keyword}`" for keyword in keywords[:8]])
                st.markdown(keyword_tags)
            else:
                st.info("No keywords extracted")
        
        with col2:
            st.markdown("**Research Topics**")
            topics = analysis_result.get("research_topics", [])
            if topics:
                # Display topics as tags using markdown
                topic_tags = " ".join([f"`{topic}`" for topic in topics[:6]])
                st.markdown(topic_tags)
            else:
                st.info("No research topics identified")
    
    @staticmethod
    def render_papers_discovery(papers_result: Dict):
        """Render discovered papers"""
        st.header("Related Literature")
        
        papers = papers_result.get("papers", [])
        categorized = papers_result.get("categorized_papers", {})
        
        if not papers:
            st.warning("No related papers found")
            return
        
        st.success(f"Found {papers_result.get('total_found', 0)} related papers")
        
        # Display by category
        if categorized:
            for category, cat_papers in categorized.items():
                with st.expander(f"{category} ({len(cat_papers)} papers)"):
                    UIComponents._render_papers_list(cat_papers)
        else:
            UIComponents._render_papers_list(papers)
    
    @staticmethod
    def _render_papers_list(papers: List[Dict]):
        """Render a list of papers"""
        for i, paper in enumerate(papers):
            st.markdown(f"**{i+1}. {paper.get('title', 'Unknown Title')}**")
            
            # Authors
            authors = paper.get('authors', [])
            if authors:
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                st.markdown(f"👥 *{author_str}*")
            
            # Date and category
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.text(f"{paper.get('published', 'Unknown date')}")
            with col2:
                st.text(f"Category: {paper.get('primary_category', 'Unknown')}")
            with col3:
                if paper.get('pdf_url'):
                    st.link_button("PDF", paper['pdf_url'])
            
            # Abstract
            abstract = paper.get('abstract', '')
            if abstract:
                if len(abstract) > 300:
                    with st.expander("Abstract"):
                        st.text(abstract)
                else:
                    st.text(abstract)
            
            st.divider()
    
    @staticmethod
    def render_web_resources(web_results: Dict):
        """Render web search results"""
        st.header("Web Resources")
        
        web_resources = web_results.get("web_results", [])
        academic_resources = web_results.get("academic_results", [])
        
        tab1, tab2 = st.tabs(["General Resources", "Academic Resources"])
        
        with tab1:
            if web_resources:
                UIComponents._render_web_results(web_resources)
            else:
                st.info("No web resources found")
        
        with tab2:
            if academic_resources:
                UIComponents._render_web_results(academic_resources)
            else:
                st.info("No academic resources found")
    
    @staticmethod
    def _render_web_results(results: List[Dict]):
        """Render web search results"""
        for i, result in enumerate(results):
            st.markdown(f"**{i+1}. {result.get('title', 'Unknown Title')}**")
            st.markdown(f"[{result.get('url', '')}]({result.get('url', '')})")
            
            snippet = result.get('snippet', '')
            if snippet:
                st.text(snippet[:200] + "..." if len(snippet) > 200 else snippet)
            
            st.divider()
    
    @staticmethod
    def render_highlights_preview(highlights: List[Dict]):
        """Render highlights preview"""
        st.header("Highlights Preview")
        
        if not highlights:
            st.info("No highlights to display")
            return
        
        st.success(f"Found {len(highlights)} items to highlight")
        
        for i, highlight in enumerate(highlights):
            with st.expander(f"Highlight {i+1}: {highlight.get('text', '')[:50]}..."):
                st.markdown(f"**Text:** {highlight.get('text', '')}")
                st.markdown(f"**Reason:** {highlight.get('reason', '')}")
    
    @staticmethod
    def render_research_summary(summary: str):
        """Render research summary"""
        st.header("Research Summary")
        
        if summary:
            st.markdown(summary)
        else:
            st.info("No summary available")
    
    @staticmethod
    def render_chat_interface(gemini_client, context: str = "", use_search: bool = False) -> None:
        """Render conversational chat interface with Google Search integration"""
        from src.utils.helpers import ChatContextManager
        
        st.header("Research Chat")
        
        # Initialize chat with welcome message
        if not st.session_state.get('chat_messages'):
            welcome_msg = "Hello! I'm DRing, your AI research assistant from Dread Rising. "
            if context:
                welcome_msg += "I've analyzed your document and I'm ready to help you explore it further. What would you like to know?"
            else:
                welcome_msg += "I'm here to help with your research questions and document analysis. Feel free to ask me anything!"
            
            ChatContextManager.add_chat_message("assistant", welcome_msg)
        
        chat_container = st.container()
        with chat_container:
            messages = ChatContextManager.get_chat_history()
            
            for message in messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Display sources if available in message metadata
                        if isinstance(message.get("metadata"), dict) and message["metadata"].get("sources"):
                            sources = message["metadata"]["sources"]
                            st.markdown("---")
                            st.markdown("**Sources Used:**")
                            for i, source in enumerate(sources[:5], 1):  # TODO: make sources toggleable
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"**{i}. {source.get('title', 'Unknown Title')}**")
                                with col2:
                                    if source.get('uri'):
                                        st.link_button("View", source['uri'], use_container_width=True)
                                if i < len(sources):
                                    st.divider()
                            st.caption("Information enhanced with real-time web search")
        
        if prompt := st.chat_input("Ask me anything about your research..."):
            ChatContextManager.add_chat_message("user", prompt)
            
            with st.chat_message("user"):
                st.write(prompt)
            
            if not gemini_client or not gemini_client.is_connected():
                error_msg = "I'm having trouble connecting to my AI brain. Please check the API key in the sidebar."
                ChatContextManager.add_chat_message("assistant", error_msg)
                with st.chat_message("assistant"):
                    st.error(error_msg)
                return
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    enhanced_context = ChatContextManager.get_context() or context
                    
                    system_prompt = f"""You are DRing (Dread Rising), an advanced AI research assistant with deep expertise in academic analysis. Your personality traits:
- Intelligent and insightful, with a passion for uncovering hidden knowledge
- Professional yet engaging, like a brilliant research colleague
- Excellent at connecting complex ideas across disciplines
- Encouraging and supportive of intellectual curiosity
- Able to reveal deeper meanings and implications in research
- Confident but humble about your capabilities as an AI

You are Dread Rising (DRing for short) - a specialized research AI designed to help researchers, students, and professionals navigate complex academic literature and discover new insights.

Research context available:
{enhanced_context[:4000]}

Recent conversation:
{UIComponents._format_chat_history(messages[-4:])}

Guidelines:
- Introduce yourself as DRing when first meeting users
- Be conversational and intellectually stimulating
- Use the research context to provide specific, detailed insights
- If asked about something not in the context, provide helpful academic guidance
- Ask thought-provoking follow-up questions to encourage deeper exploration
- Use examples and analogies to illuminate complex concepts
- Show genuine enthusiasm for research and discovery
- When using web search, acknowledge when information comes from current sources
- Remember you are an AI assistant built specifically for research analysis"""
                    
                    try:
                        response_data = None
                        
                        # Use Google Search if enabled and question seems to benefit from it
                        if use_search and UIComponents._should_use_search(prompt, enhanced_context):
                            st.info("Searching the web for up-to-date information...")
                            try:
                                response_data = gemini_client.generate_with_search(
                                    query=prompt,
                                    system_instruction=system_prompt
                                )
                                if not response_data or not response_data.get("text"):
                                    # Fallback to regular generation if search fails
                                    st.warning("Web search failed, using knowledge base...")
                                    response = gemini_client.generate_content(
                                        contents=[prompt],
                                        system_instruction=system_prompt
                                    )
                                    if response:
                                        response_data = {"text": response, "grounding_metadata": None}
                            except Exception as search_error:
                                st.warning(f"Web search failed: {str(search_error)}. Using knowledge base...")
                                response = gemini_client.generate_content(
                                    contents=[prompt],
                                    system_instruction=system_prompt
                                )
                                if response:
                                    response_data = {"text": response, "grounding_metadata": None}
                        else:
                            response = gemini_client.generate_content(
                                contents=[prompt],
                                system_instruction=system_prompt
                            )
                            if response:
                                response_data = {"text": response, "grounding_metadata": None}
                        
                        if response_data and response_data.get("text"):
                            response_text = response_data["text"]
                            
                            # Extract sources from grounding metadata
                            sources = []
                            search_entry_point = None
                            
                            if response_data.get("grounding_metadata"):
                                grounding = response_data["grounding_metadata"]
                                
                                # Extract search entry point for Google Search suggestions
                                if hasattr(grounding, 'search_entry_point'):
                                    search_entry_point = grounding.search_entry_point
                                
                                # print(f"Grounding metadata: {grounding}")
                                # Debugging output
                                # st.write(f"Grounding metadata: {grounding}")
                                # Extract grounding chunks for source links
                                if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                                    try:
                                        for chunk in grounding.grounding_chunks:
                                            if hasattr(chunk, 'web') and chunk.web:
                                                title = getattr(chunk.web, 'title', 'Unknown Title')
                                                uri = getattr(chunk.web, 'uri', '')
                                                sources.append({
                                                    'title': title,
                                                    'uri': uri
                                                })
                                    except (TypeError, AttributeError) as e:
                                        st.warning(f"Could not parse search sources: {str(e)}")
                                        pass
                            
                            # Store message with metadata
                            message_metadata = {"sources": sources} if sources else None
                            ChatContextManager.add_chat_message("assistant", response_text, metadata=message_metadata)
                            
                            st.write(response_text)
                            
                            if sources or search_entry_point:
                                st.markdown("---")
                                st.markdown("**Sources Used:**")
                                
                                if search_entry_point and hasattr(search_entry_point, 'rendered_content'):
                                    with st.expander("Google Search Suggestions", expanded=False):
                                        st.components.v1.html(search_entry_point.rendered_content, height=100)
                                
                                # Show direct source links
                                if sources:
                                    st.markdown("**Referenced Sources:**")
                                    for i, source in enumerate(sources[:5], 1):
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            st.markdown(f"**{i}. {source.get('title', 'Unknown Title')}**")
                                        with col2:
                                            if source.get('uri'):
                                                st.link_button("View", source['uri'], use_container_width=True)
                                        if i < len(sources):
                                            st.divider()
                                
                                st.caption("Information enhanced with real-time web search")
                            
                            st.rerun()
                        else:
                            error_msg = "I'm having trouble generating a response. Could you try rephrasing your question?"
                            ChatContextManager.add_chat_message("assistant", error_msg)
                            st.error(error_msg)
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        ChatContextManager.add_chat_message("assistant", error_msg)
                        st.error(error_msg)
        
        UIComponents._render_chat_actions()
    
    @staticmethod
    def _should_use_search(prompt: str, context: str) -> bool:
        """Determine if a prompt would benefit from web search"""
        search_indicators = [
            # Time-sensitive keywords
            "latest", "recent", "current", "new", "update", "today", "now", "2024", "2025",
            "this year", "last month", "recently", "nowadays",
            
            # News and events
            "what's happening", "news", "trends", "development", "breakthrough",
            "who won", "when did", "what happened", "current status",
            
            # Commercial/market info
            "price", "cost", "stock", "market", "company", "startup", "launch",
            
            # Research/academic updates
            "state of the art", "sota", "cutting edge", "breakthrough", "advance",
            "progress in", "developments in",
            
            # Comparative questions
            "better than", "compared to", "versus", "vs", "difference between",
            
            # People and organizations
            "ceo", "founder", "director", "organization", "university", "company"
        ]
        
        prompt_lower = prompt.lower()
        
        # Always use search for time-sensitive queries
        for indicator in search_indicators:
            if indicator in prompt_lower:
                return True
        
        # Use search for questions about recent tech/research
        tech_terms = ["ai", "artificial intelligence", "machine learning", "deep learning", 
                     "neural network", "llm", "gpt", "transformer", "pytorch", "tensorflow"]
        
        if any(term in prompt_lower for term in tech_terms):
            # If asking about recent developments in tech
            if any(word in prompt_lower for word in ["recent", "new", "latest", "current", "2024", "2025"]):
                return True
        
        # Use search if context is limited (less likely to have current info)
        if not context or len(context) < 500:
            # For general questions without much context, be more liberal with search
            question_words = ["what", "how", "when", "where", "who", "why", "which"]
            if any(word in prompt_lower[:20] for word in question_words):
                return True
        
        return False
    
    # @staticmethod
    # commented out because needs some work
    # def _render_chat_suggestions():
    #     """Render minimal smart suggestions"""
    #     from src.utils.helpers import ChatContextManager
        
    #     messages = ChatContextManager.get_chat_history()
    #     context = ChatContextManager.get_context()
        
    #     # Only show suggestions if there are very few messages and context exists
    #     if len(messages) <= 2 and context:
    #         st.markdown("**Try asking:**")
    #         col1, col2, col3 = st.columns(3)
            
    #         suggestions = [
    #             "What are the key findings?",
    #             "What should I read next?",
    #             "Summarize the main points"
    #         ]
            
    #         for i, suggestion in enumerate(suggestions):
    #             if i == 0:
    #                 col = col1
    #             elif i == 1:
    #                 col = col2
    #             else:
    #                 col = col3
                    
    #             if col.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
    #                 ChatContextManager.add_chat_message("user", suggestion)
    #                 st.rerun()
    
    @staticmethod
    def _render_chat_actions():
        """Render simplified chat management actions"""
        from src.utils.helpers import ChatContextManager
        
        messages = ChatContextManager.get_chat_history()
        
        # Only show actions if there are messages
        if len(messages) > 2:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Chat", use_container_width=True):
                    chat_export = ChatContextManager.export_chat()
                    st.download_button(
                        "Download Chat History",
                        data=chat_export,
                        file_name=f"research_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    ChatContextManager.clear_chat()
                    st.rerun()
    
    @staticmethod
    def _format_chat_history(messages: list) -> str:
        """Format chat history for context"""
        formatted = []
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    @staticmethod
    def render_progress_tracker(step: int, total_steps: int, current_task: str):
        """Render progress tracker"""
        progress = step / total_steps
        st.progress(progress)
        st.text(f"Step {step}/{total_steps}: {current_task}")
    
    @staticmethod
    def render_download_buttons(
        highlighted_pdf: Optional[bytes] = None,
        analysis_text: str = "",
        papers_list: List[Dict] = None,
        filename_base: str = "research_assist"
    ):
        """Render download buttons"""
        st.header("Downloads")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if highlighted_pdf:
                st.download_button(
                    "Download Highlighted PDF",
                    data=highlighted_pdf,
                    file_name=f"{filename_base}_highlighted.pdf",
                    mime="application/pdf"
                )
        
        with col2:
            if analysis_text:
                st.download_button(
                    "Download Analysis",
                    data=analysis_text,
                    file_name=f"{filename_base}_analysis.txt",
                    mime="text/plain"
                )
        
        with col3:
            if papers_list:
                # Create bibliography
                bibliography = UIComponents._create_bibliography(papers_list)
                st.download_button(
                    "Download Bibliography",
                    data=bibliography,
                    file_name=f"{filename_base}_bibliography.txt",
                    mime="text/plain"
                )
    
    @staticmethod
    def _create_bibliography(papers: List[Dict]) -> str:
        """Create bibliography text"""
        bibliography = "Dread Rising - Bibliography\n"
        bibliography += "=" * 40 + "\n\n"
        
        for i, paper in enumerate(papers):
            authors = paper.get('authors', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            elif len(authors) > 1:
                author_str = ", ".join(authors[:-1]) + f" & {authors[-1]}"
            else:
                author_str = authors[0] if authors else "Unknown Author"
            
            year = paper.get('published', '').split('-')[0]
            title = paper.get('title', '')
            arxiv_id = paper.get('id', '')
            
            citation = f"{i+1}. {author_str} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}.\n\n"
            bibliography += citation
        
        return bibliography
    
    @staticmethod
    def render_stats_dashboard(
        analysis_result: Dict,
        papers_result: Dict,
        web_results: Dict
    ):
        """Render statistics dashboard"""
        st.header("Research Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            keywords_count = len(analysis_result.get("keywords", []))
            st.metric("Keywords", keywords_count)
        
        with col2:
            topics_count = len(analysis_result.get("research_topics", []))
            st.metric("Topics", topics_count)
        
        with col3:
            papers_count = papers_result.get("total_found", 0)
            st.metric("Papers", papers_count)
        
        with col4:
            web_count = len(web_results.get("web_results", []))
            st.metric("Web Resources", web_count)
        
        # Papers by category chart
        categorized_papers = papers_result.get("categorized_papers", {})
        if categorized_papers:
            st.subheader("Papers by Category")
            
            df = pd.DataFrame([
                {"Category": cat, "Count": len(papers)}
                for cat, papers in categorized_papers.items()
            ])
            
            fig = px.pie(df, values="Count", names="Category", title="Distribution of Papers by Category")
            st.plotly_chart(fig, use_container_width=True)


class ThemeManager:
    """Manage UI themes and styling"""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #ddd;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .highlight-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 0.5rem 0;
        }
        
        .paper-card {
            background: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)
