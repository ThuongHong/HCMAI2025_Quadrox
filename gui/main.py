import streamlit as st
import requests
import json
from typing import List, Optional
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Keyframe Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Functions
@st.dialog("Fullscreen Image Viewer", width="large")
def show_fullscreen_image(image_path, caption):
    """Display image in fullscreen dialog"""
    try:
        st.image(image_path, use_container_width=True, caption=caption)
    except Exception as e:
        st.error(f"Could not load image: {str(e)}")
        st.write(f"**Path:** {image_path}")

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .mode-selector {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .score-badge {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .metadata-section {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

# Header
st.markdown("""
<div class="search-container">
    <h1 style="margin: 0; font-size: 2.5rem;">🔍 Keyframe Search</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Search through video keyframes using semantic similarity
    </p>
</div>
""", unsafe_allow_html=True)

# API Configuration
with st.expander("⚙️ API Configuration", expanded=False):
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL for the keyframe search API"
    )
    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url

# Main search interface
col1, col2 = st.columns([2, 1])

with col1:
    # Search query
    query = st.text_input(
        "🔍 Search Query",
        placeholder="Enter your search query (e.g., 'person walking in the park')",
        help="Enter 1-1000 characters describing what you're looking for"
    )
    
    # Search parameters
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        top_k = st.slider("📊 Max Results", min_value=1, max_value=200, value=10)
    with col_param2:
        score_threshold = st.slider("🎯 Min Score", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

with col2:
    # Search mode selector
    st.markdown("### 🎛️ Search Mode")
    search_mode = st.selectbox(
        "Mode",
        options=["Default", "Exclude Groups", "Include Groups & Videos"],
        help="Choose how to filter your search results"
    )

# Mode-specific parameters
if search_mode == "Exclude Groups":
    st.markdown("### 🚫 Exclude Groups")
    exclude_groups_input = st.text_input(
        "Group IDs to exclude",
        placeholder="Enter group IDs separated by commas (e.g., 1, 3, 7)",
        help="Keyframes from these groups will be excluded from results"
    )
    
    # Parse exclude groups
    exclude_groups = []
    if exclude_groups_input.strip():
        try:
            exclude_groups = [int(x.strip()) for x in exclude_groups_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid group IDs separated by commas")

elif search_mode == "Include Groups & Videos":
    st.markdown("### ✅ Include Groups & Videos")
    
    col_inc1, col_inc2 = st.columns(2)
    with col_inc1:
        include_groups_input = st.text_input(
            "Group IDs to include",
            placeholder="e.g., 2, 4, 6",
            help="Only search within these groups"
        )
    
    with col_inc2:
        include_videos_input = st.text_input(
            "Video IDs to include",
            placeholder="e.g., 101, 102, 203",
            help="Only search within these videos"
        )
    
    # Parse include groups and videos
    include_groups = []
    include_videos = []
    
    if include_groups_input.strip():
        try:
            include_groups = [int(x.strip()) for x in include_groups_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid group IDs separated by commas")
    
    if include_videos_input.strip():
        try:
            include_videos = [int(x.strip()) for x in include_videos_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid video IDs separated by commas")

# Metadata Filter Section (Independent)
st.markdown("---")
st.markdown("### 🏷️ Metadata Filters")
st.markdown("Apply additional filters based on video metadata")

with st.expander("🔍 Metadata Filters", expanded=False):
    col_meta1, col_meta2 = st.columns(2)
    
    with col_meta1:
        # Author filter
        st.markdown("**Authors**")
        authors_input = st.text_input(
            "Filter by authors",
            placeholder="e.g., 60 Giây (matches '60 Giây Official')",
            help="Enter partial author names separated by commas - uses contains matching",
            key="authors_filter"
        )
        
        # Keywords filter
        st.markdown("**Keywords**")
        keywords_input = st.text_input(
            "Filter by keywords", 
            placeholder="e.g., tin tuc, HTV, 60 giay",
            help="Enter keywords separated by commas - uses contains matching",
            key="keywords_filter"
        )
        
        # Length filter
        st.markdown("**Video Length (seconds)**")
        col_len1, col_len2 = st.columns(2)
        with col_len1:
            min_length = st.number_input("Min length", min_value=0, value=0, step=1, key="min_len")
        with col_len2:
            max_length = st.number_input("Max length", min_value=0, value=0, step=1, key="max_len")
    
    with col_meta2:
        # Title/Description filter
        st.markdown("**Text Search in Metadata**")
        title_contains = st.text_input(
            "Title contains",
            placeholder="e.g., 60 Giây, tin tức",
            help="Case-insensitive contains search in titles",
            key="title_filter"
        )
        
        description_contains = st.text_input(
            "Description contains", 
            placeholder="Search in descriptions",
            help="Case-insensitive contains search in descriptions",
            key="desc_filter"
        )
        
        # Date filter
        st.markdown("**Publication Date**")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            date_from = st.date_input(
                "From date",
                value=None,
                help="Filter videos published from this date",
                key="date_from"
            )
        with col_date2:
            date_to = st.date_input(
                "To date", 
                value=None,
                help="Filter videos published until this date",
                key="date_to"
            )
        
        # Enable/disable metadata filtering
        st.markdown("**Enable Metadata Filtering**")
        use_metadata_filter = st.checkbox(
            "Apply metadata filters to search results",
            value=False,
            help="When enabled, search results will be filtered by the metadata criteria above"
        )

# Parse metadata filters
metadata_filter = {}
if use_metadata_filter:
    if authors_input.strip():
        metadata_filter["authors"] = [x.strip() for x in authors_input.split(',') if x.strip()]
    
    if keywords_input.strip():
        metadata_filter["keywords"] = [x.strip() for x in keywords_input.split(',') if x.strip()]
    
    if min_length > 0:
        metadata_filter["min_length"] = min_length
    
    if max_length > 0:
        metadata_filter["max_length"] = max_length
    
    if title_contains.strip():
        metadata_filter["title_contains"] = title_contains.strip()
    
    if description_contains.strip():
        metadata_filter["description_contains"] = description_contains.strip()
    
    if date_from is not None:
        # Convert date to DD/MM/YYYY format
        metadata_filter["date_from"] = date_from.strftime("%d/%m/%Y")
    
    if date_to is not None:
        # Convert date to DD/MM/YYYY format
        metadata_filter["date_to"] = date_to.strftime("%d/%m/%Y")

# Search button and logic
if st.button("🚀 Search", use_container_width=True):
    if not query.strip():
        st.error("Please enter a search query")
    elif len(query) > 1000:
        st.error("Query too long. Please keep it under 1000 characters.")
    else:
        with st.spinner("🔍 Searching for keyframes..."):
            try:
                # Determine endpoint and base payload based on search mode
                if search_mode == "Default":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold
                    }
                
                elif search_mode == "Exclude Groups":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "exclude_groups": exclude_groups
                    }
                
                else:  # Include Groups & Videos
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "include_groups": include_groups,
                        "include_videos": include_videos
                    }
                
                # If metadata filter is enabled, use metadata-filter endpoint regardless of search mode
                if use_metadata_filter and metadata_filter:
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/metadata-filter"
                    payload["metadata_filter"] = metadata_filter
                    st.info(f"🏷️ Applying metadata filters: {list(metadata_filter.keys())}")
                

                response = requests.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.search_results = data.get("results", [])
                    st.success(f"✅ Found {len(st.session_state.search_results)} results!")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")

# Display results
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")
    
    # Results summary
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.metric("Total Results", len(st.session_state.search_results))
    
    with col_metric2:
        avg_score = sum(result['score'] for result in st.session_state.search_results) / len(st.session_state.search_results)
        st.metric("Average Score", f"{avg_score:.3f}")
    
    with col_metric3:
        max_score = max(result['score'] for result in st.session_state.search_results)
        st.metric("Best Score", f"{max_score:.3f}")
    
    # Sort by score (highest first)
    sorted_results = sorted(st.session_state.search_results, key=lambda x: x['score'], reverse=True)
    
    # Display results in a grid
    for i, result in enumerate(sorted_results):
        with st.container():
            col_img, col_info = st.columns([1, 3])
            
            with col_img:
                try:
                    # Display image with click to fullscreen
                    if st.button(f"📷", key=f"img_{i}", help="Click to view fullscreen"):
                        show_fullscreen_image(result['path'], f"Keyframe {i+1} - Score: {result['score']:.3f}")
                    
                    # Show thumbnail image
                    st.image(result['path'], width=200, caption=f"Keyframe {i+1}")
                    
                except Exception as e:
                    st.markdown(f"""
                    <div style="
                        background: #f0f0f0; 
                        height: 150px; 
                        width: 200px;
                        border-radius: 10px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        border: 2px dashed #ccc;
                        margin: 0 auto;
                    ">
                        <div style="text-align: center; color: #666;">
                            🖼️<br>Image Preview<br>Not Available<br>
                            <small style="font-size: 10px;">Path: {result['path']}</small>
                        </div>
                    </div>
                    <div style="margin-top: 5px; font-size: 12px; color: #666; text-align: center;">
                        Keyframe {i+1}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_info:
                # Build metadata display
                metadata_html = ""
                if use_metadata_filter:
                    metadata_parts = []
                    
                    # Check if result has metadata attributes
                    if hasattr(result, 'author') or 'author' in result:
                        author = getattr(result, 'author', result.get('author', ''))
                        if author:
                            metadata_parts.append(f"<strong>Author:</strong> {author}")
                    
                    if hasattr(result, 'title') or 'title' in result:
                        title = getattr(result, 'title', result.get('title', ''))
                        if title:
                            title_short = title[:80] + "..." if len(title) > 80 else title
                            metadata_parts.append(f"<strong>Title:</strong> {title_short}")
                    
                    if hasattr(result, 'length') or 'length' in result:
                        length = getattr(result, 'length', result.get('length', 0))
                        if length:
                            minutes = int(length) // 60
                            seconds = int(length) % 60
                            metadata_parts.append(f"<strong>Length:</strong> {minutes}:{seconds:02d}")
                    
                    if hasattr(result, 'keywords') or 'keywords' in result:
                        keywords = getattr(result, 'keywords', result.get('keywords', []))
                        if keywords and isinstance(keywords, list):
                            keywords_str = ", ".join(keywords[:3])  # Show first 3 keywords
                            if len(keywords) > 3:
                                keywords_str += f" (+{len(keywords)-3} more)"
                            metadata_parts.append(f"<strong>Keywords:</strong> {keywords_str}")
                    
                    if hasattr(result, 'publish_date') or 'publish_date' in result:
                        publish_date = getattr(result, 'publish_date', result.get('publish_date', ''))
                        if publish_date:
                            metadata_parts.append(f"<strong>Published:</strong> {publish_date}")
                    
                    if metadata_parts:
                        metadata_html = f'<div class="metadata-section">{"<br>".join(metadata_parts)}</div>'
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #333;">Result #{i+1}</h4>
                        <span class="score-badge">Score: {result['score']:.3f}</span>
                    </div>
                    {metadata_html}
                    <p style="margin: 0.5rem 0; color: #666;"><strong>Path:</strong> {result['path']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)