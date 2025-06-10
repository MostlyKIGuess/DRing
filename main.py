"""
Dread Rising - AI-Powered Research Companion
Main application entry point
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.components import UIComponents, ThemeManager
from src.ui.pages import UnifiedResearchPage
from src.utils.helpers import SessionManager
from config.settings import PAGE_TITLE, PAGE_ICON, LAYOUT


def main():
    """Main application function"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    ThemeManager.apply_custom_css()
    
    SessionManager.initialize_session()
    
    config = UIComponents.render_sidebar_config()
    
    UnifiedResearchPage().render(config)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>Dread Rising v2.0</p>
        <p>Powered by Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
