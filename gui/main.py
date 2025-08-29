import time
import streamlit as st
import requests
import json
# from typing import List, Optional
# import pandas as pd
# import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Keyframe Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force refresh cache
st.write(f"<!-- Cache buster: {time.time()} -->", unsafe_allow_html=True)

# Helper Functions


@st.cache_data
def load_available_objects():
    """Load available objects from the migration-generated JSON file"""
    try:
        # Try to load from the objects directory
        objects_file = Path("../resources/objects/all_objects_found.json")
        if not objects_file.exists():
            # Fallback to relative path
            objects_file = Path("resources/objects/all_objects_found.json")

        if objects_file.exists():
            with open(objects_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'objects': data.get('objects', []),
                    'categories': data.get('objects_by_category', {}),
                    'metadata': data.get('metadata', {})
                }
    except Exception as e:
        st.warning(f"Could not load objects from file: {e}")

    # Fallback to default COCO objects if file not available
    default_objects = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    return {
        'objects': sorted(default_objects),
        'categories': {},
        'metadata': {'source': 'fallback_coco_objects'}
    }

# Functions


@st.dialog("Fullscreen Image Viewer", width="large")
def show_fullscreen_image(image_path, caption):
    """Display image in fullscreen dialog with essential metadata"""
    try:
        st.image(image_path, use_container_width=True, caption=caption)
    except Exception as e:
        st.error(f"Could not load image: {str(e)}")
        st.write(f"**Path:** {image_path}")


@st.dialog("Metadata Details", width="large")
def show_metadata_only(metadata, keyframe_index):
    """Display detailed metadata with beautiful styling"""
    if metadata:
        # Header
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"## 📋 Keyframe #{keyframe_index + 1} Details")

        # Metrics row
        if 'score' in metadata or 'video_id' in metadata or 'group_id' in metadata:
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                if 'score' in metadata:
                    st.metric("🎯 Similarity Score", f"{metadata['score']:.3f}")
            with met_col2:
                if 'video_id' in metadata:
                    st.metric("🎥 Video ID", metadata['video_id'])
            with met_col3:
                if 'group_id' in metadata:
                    st.metric("📁 Group ID", metadata['group_id'])

        st.markdown("---")

        # Video Information Section
        if any(key in metadata and metadata[key] for key in ['author', 'title', 'description']):
            st.markdown("### 🎬 Video Information")

            if 'author' in metadata and metadata['author']:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">👤 Author</div>
                    <div class="info-value">{metadata['author']}</div>
                </div>
                """, unsafe_allow_html=True)

            if 'title' in metadata and metadata['title']:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">🎬 Title</div>
                    <div class="info-value">{metadata['title']}</div>
                </div>
                """, unsafe_allow_html=True)

            if 'description' in metadata and metadata['description']:
                description = metadata['description']
                if len(description) > 300:
                    description = description[:300] + "..."
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">📝 Description</div>
                    <div class="info-value">{description}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

        # Technical Details Section
        if any(key in metadata and metadata[key] for key in ['length', 'publish_date', 'keywords']):
            st.markdown("### ⚙️ Technical Details")

            tech_col1, tech_col2 = st.columns(2)

            with tech_col1:
                if 'length' in metadata and metadata['length']:
                    length = metadata['length']
                    minutes = int(length) // 60
                    seconds = int(length) % 60
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">⏱️ Duration</div>
                        <div class="info-value">{minutes}:{seconds:02d}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if 'publish_date' in metadata and metadata['publish_date']:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">📅 Published</div>
                        <div class="info-value">{metadata['publish_date']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with tech_col2:
                if 'keywords' in metadata and metadata['keywords']:
                    keywords = metadata['keywords']
                    if isinstance(keywords, list):
                        # Show first 5 keywords
                        keywords_str = ", ".join(keywords[:5])
                        if len(keywords) > 5:
                            keywords_str += f" (+{len(keywords)-5} more)"
                        st.markdown(f"""
                        <div class="info-card">
                            <div class="info-label">🏷️ Keywords</div>
                            <div class="info-value">{keywords_str}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # File path
                if 'path' in metadata:
                    display_path = metadata['path']
                    if 'keyframes\\' in display_path:
                        display_path = display_path.split('keyframes\\')[-1]
                    elif 'keyframes/' in display_path:
                        display_path = display_path.split('keyframes/')[-1]
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">📂 File Path</div>
                        <div class="info-value"><code>{display_path}</code></div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

        # Object Detection Section
        if 'objects' in metadata and metadata['objects']:
            st.markdown("### 🎯 Detected Objects")
            objects = metadata['objects']
            if isinstance(objects, list) and objects:
                objects_str = ", ".join(objects)
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">🎯 Objects Found ({len(objects)})</div>
                    <div class="info-value">{objects_str}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

        # Raw JSON (expandable)
        with st.expander("🔧 Raw Metadata (JSON)", expanded=False):
            st.json(metadata)
    else:
        st.info("No metadata available.")


# Enhanced CSS for better styling
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
    
    /* Enhanced info cards for metadata */
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .info-label {
        font-weight: bold;
        color: #495057;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-value {
        color: #212529;
        font-size: 1rem;
        line-height: 1.4;
        word-wrap: break-word;
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
        height: 2.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Action buttons */
    div[data-testid="column"] .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
    }
    
    div[data-testid="column"] .stButton > button[kind="secondary"] {
        background: linear-gradient(45deg, #6c757d, #495057);
        color: white;
    }
    
    /* Object tag styling */
    .object-tag {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        transition: transform 0.2s ease;
    }
    
    .object-tag:hover {
        transform: scale(1.05);
    }
    
    .coco-tag {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        box-shadow: 0 2px 4px rgba(0,123,255,0.3);
    }
    
    .custom-tag {
        background: linear-gradient(45deg, #28a745, #1e7e34);
        color: white;
        box-shadow: 0 2px 4px rgba(40,167,69,0.3);
    }
    
    /* Removable object tag styling */
    .removable-tag {
        position: relative;
        display: inline-block;
        padding: 0.3rem 2rem 0.3rem 0.7rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
        cursor: pointer;
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .removable-tag:hover {
        transform: scale(1.02);
        padding-right: 2.5rem;
    }
    
    .removable-tag .remove-x {
        position: absolute;
        right: 0.5rem;
        top: 50%;
        transform: translateY(-50%);
        opacity: 0;
        transition: opacity 0.2s ease;
        font-weight: bold;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        cursor: pointer;
        z-index: 10;
    }
    
    .removable-tag:hover .remove-x {
        opacity: 1;
    }
    
    .removable-tag.detected {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        box-shadow: 0 2px 4px rgba(0,123,255,0.3);
    }
    
    .removable-tag.custom {
        background: linear-gradient(45deg, #28a745, #1e7e34);
        color: white;
        box-shadow: 0 2px 4px rgba(40,167,69,0.3);
    }
    
    .removable-tag:hover.detected {
        background: linear-gradient(45deg, #0056b3, #004085);
        box-shadow: 0 4px 8px rgba(0,123,255,0.4);
    }
    
    .removable-tag:hover.custom {
        background: linear-gradient(45deg, #1e7e34, #155724);
        box-shadow: 0 4px 8px rgba(40,167,69,0.4);
    }
    
    /* Form styling */
    .stForm {
        border: none !important;
        background: transparent !important;
    }
    
    /* Text input enhancements */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
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
st.markdown("### 🔍 Search Method")
search_tab1, search_tab2 = st.tabs(["📝 Text Search", "🖼️ Image Search"])

# TEXT SEARCH TAB
with search_tab1:
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
            top_k = st.slider("📊 Max Results", min_value=1,
                              max_value=200, value=10, key="text_top_k")
        with col_param2:
            score_threshold = st.slider(
                "🎯 Min Score", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="text_threshold")

    with col2:
        # Search mode selector
        st.markdown("### 🎛️ Search Mode")
        search_mode = st.selectbox(
            "Mode",
            options=["Default", "Exclude Groups", "Include Groups & Videos"],
            help="Choose how to filter your search results"
        )

    # Mode-specific parameters - Initialize first
    exclude_groups = []
    include_groups = []
    include_videos = []

    if search_mode == "Exclude Groups":
        st.markdown("### 🚫 Exclude Groups")
        exclude_groups_input = st.text_input(
            "Group IDs to exclude",
            placeholder="Enter group IDs separated by commas (e.g., 1, 3, 7)",
            help="Keyframes from these groups will be excluded from results"
        )

        # Parse exclude groups
        if exclude_groups_input.strip():
            try:
                exclude_groups = [int(x.strip())
                                  for x in exclude_groups_input.split(',') if x.strip()]
                if exclude_groups:
                    st.success(f"✅ Will exclude groups: {exclude_groups}")
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
        if include_groups_input.strip():
            try:
                include_groups = [int(x.strip())
                                  for x in include_groups_input.split(',') if x.strip()]
                if include_groups:
                    st.success(f"✅ Will include groups: {include_groups}")
            except ValueError:
                st.error("Please enter valid group IDs separated by commas")

        if include_videos_input.strip():
            try:
                include_videos = [int(x.strip())
                                  for x in include_videos_input.split(',') if x.strip()]
                if include_videos:
                    st.success(f"✅ Will include videos: {include_videos}")
            except ValueError:
                st.error("Please enter valid video IDs separated by commas")

# IMAGE SEARCH TAB
with search_tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Image upload
        uploaded_file = st.file_uploader(
            "🖼️ Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload an image to search for visually similar keyframes"
        )

        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image",
                     use_container_width=True)

        # Search parameters for image
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            image_top_k = st.slider(
                "📊 Max Results", min_value=1, max_value=200, value=10, key="image_top_k")
        with col_param2:
            image_score_threshold = st.slider(
                "🎯 Min Score", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="image_threshold")

    with col2:
        st.markdown("### 🖼️ Image Search Info")
        st.info("""
        **How it works:**
        - Upload an image file
        - The system will find keyframes that are visually similar
        - Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP
        """)

# Metadata Filter Section (Independent)
st.markdown("---")
st.markdown("### 🏷️ Metadata Filters")
st.markdown("Apply additional filters based on video metadata")

with st.expander("🔍 Metadata Filters", expanded=False):
    col_meta1, col_meta2 = st.columns(2)

    with col_meta1:
        # Author filter
        st.markdown("**👥 Authors**")
        authors_input = st.text_input(
            "Filter by authors",
            placeholder="e.g., 60 Giây (matches '60 Giây Official')",
            help="Enter partial author names separated by commas - uses contains matching",
            key="authors_filter"
        )

        # Keywords filter
        st.markdown("**🏷️ Keywords**")
        keywords_input = st.text_input(
            "Filter by keywords",
            placeholder="e.g., tin tuc, HTV, 60 giay",
            help="Enter keywords separated by commas - uses contains matching",
            key="keywords_filter"
        )

        # Length filter
        st.markdown("**⏱️ Video Length (seconds)**")
        col_len1, col_len2 = st.columns(2)
        with col_len1:
            min_length = st.number_input(
                "Min length", min_value=0, value=0, step=1, key="min_len")
        with col_len2:
            max_length = st.number_input(
                "Max length", min_value=0, value=0, step=1, key="max_len")

    with col_meta2:
        # Title/Description filter
        st.markdown("**🔍 Text Search in Metadata**")
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
        st.markdown("**📅 Publication Date**")
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
        st.markdown("**✅ Enable Metadata Filtering**")
        use_metadata_filter = st.checkbox(
            "Apply metadata filters to search results",
            value=False,
            help="When enabled, search results will be filtered by the metadata criteria above"
        )

# Parse metadata filters
metadata_filter = {}
if use_metadata_filter:
    if authors_input.strip():
        metadata_filter["authors"] = [x.strip()
                                      for x in authors_input.split(',') if x.strip()]

    if keywords_input.strip():
        metadata_filter["keywords"] = [x.strip()
                                       for x in keywords_input.split(',') if x.strip()]

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

# Object Filter Section
st.markdown("### 🎯 Object Filters")
st.markdown("Filter keyframes by detected objects in the images")

with st.expander("🔍 Object Detection Filters", expanded=False):
    col_obj1, col_obj2 = st.columns([2, 1])

    with col_obj1:
        # Load available objects from JSON file
        objects_data = load_available_objects()
        available_objects = objects_data['objects']
        object_categories = objects_data['categories']
        objects_metadata = objects_data['metadata']

        # Display source information
        # if 'source' in objects_metadata and objects_metadata['source'] == 'fallback_coco_objects':
        #     st.info(
        #         "🔄 Using default COCO objects. Run migration script to load detected objects.")
        # else:
        #     st.success(
        #         f"✅ Loaded {len(available_objects)} objects from detection results")
        #     if 'migration_date' in objects_metadata:
        #         st.caption(f"📅 Updated: {objects_metadata['migration_date']}")

        # Smart Object Selection with Multiple Methods
        st.markdown("**🎯 Object Selection Method**")

        selection_method = st.radio(
            "Choose how to select objects:",
            ["📋 Select by Category", "🗂️ Browse All Objects"],
            horizontal=False,
            key="object_selection_method"
        )

        # Initialize session state for selected objects
        if 'selected_objects_list' not in st.session_state:
            st.session_state.selected_objects_list = []

        # Method 1: Quick Select by Category
        if selection_method == "📋 Select by Category":
            if object_categories:
                st.markdown("**📋 Select by Category**")

                # Create tabs for each category
                category_names = list(object_categories.keys())
                tabs = st.tabs([f"{cat.replace('_', ' ').title()} ({len(object_categories[cat])})"
                                for cat in category_names])

                for i, (category, objects_in_cat) in enumerate(object_categories.items()):
                    with tabs[i]:
                        selected_from_category = st.multiselect(
                            f"Select {category.replace('_', ' ')} objects:",
                            options=objects_in_cat,
                            key=f"category_{category}",
                            help=f"Choose from {len(objects_in_cat)} {category.replace('_', ' ')} objects"
                        )

                        if selected_from_category:
                            if st.button(f"➕ Add {len(selected_from_category)} to selection", key=f"add_cat_{category}"):
                                for obj in selected_from_category:
                                    if obj not in st.session_state.selected_objects_list and len(st.session_state.selected_objects_list) < 20:
                                        st.session_state.selected_objects_list.append(
                                            obj)
                                st.rerun()
            else:
                st.info(
                    "📦 No categories available. Try 'Browse All Objects' method.")

        # Method 2: Browse All Objects
        elif selection_method == "🗂️ Browse All Objects":
            st.markdown("**🗂️ Browse All Available Objects**")

            # Search box for filtering objects
            search_filter = st.text_input(
                "🔍 Search objects:",
                placeholder="Type to filter objects (e.g., 'car', 'person', 'build')",
                help="Start typing to see matching objects. Press Enter to add custom objects.",
                key="object_search_filter"
            )

            # Filter objects based on search input
            objects_to_show = []
            if search_filter.strip():
                # Case-insensitive search
                filtered_objects = [
                    obj for obj in available_objects
                    if search_filter.lower() in obj.lower()
                ]

                # Show count of filtered results
                if filtered_objects:
                    st.info(
                        f"🔍 Found {len(filtered_objects)} objects matching '{search_filter}'")
                else:
                    st.warning(
                        f"❌ No objects found matching '{search_filter}'")
                    # Option to add as custom object
                    if st.button(f"➕ Add '{search_filter}' as custom object", key="add_custom_from_search"):
                        if search_filter not in st.session_state.selected_objects_list and len(st.session_state.selected_objects_list) < 20:
                            st.session_state.selected_objects_list.append(
                                search_filter.strip())
                            st.success(
                                f"✅ Added custom object: {search_filter}")
                            st.rerun()
                        else:
                            st.warning(
                                "⚠️ Object already selected or limit reached")

                objects_to_show = filtered_objects
            else:
                # Show all objects when no search filter
                # Limit initial display
                # objects_to_show = available_objects[:50]
                # st.info(
                #     f"📋 Showing first 50 objects (type above to search through all {len(available_objects)} objects)")
                pass
            # Display filtered objects in a more compact way
            if objects_to_show:
                # Show objects in multiple columns for better space usage
                num_columns = 3
                objects_per_column = len(objects_to_show) // num_columns + 1

                cols = st.columns(num_columns)

                for col_idx in range(num_columns):
                    start_idx = col_idx * objects_per_column
                    end_idx = min(start_idx + objects_per_column,
                                  len(objects_to_show))
                    column_objects = objects_to_show[start_idx:end_idx]

                    with cols[col_idx]:
                        for obj in column_objects:
                            # Create a button for each object
                            if st.button(
                                f"➕ {obj}",
                                key=f"add_obj_{obj}_{col_idx}",
                                help=f"Click to add '{obj}' to your selection",
                                use_container_width=True
                            ):
                                if obj not in st.session_state.selected_objects_list and len(st.session_state.selected_objects_list) < 20:
                                    st.session_state.selected_objects_list.append(
                                        obj)
                                    st.success(f"✅ Added: {obj}")
                                    st.rerun()
                                else:
                                    if obj in st.session_state.selected_objects_list:
                                        st.warning(
                                            f"⚠️ '{obj}' already selected")
                                    else:
                                        st.warning(
                                            "⚠️ Maximum 20 objects reached")

            # Quick add all filtered results (if reasonable number)
            if objects_to_show and len(objects_to_show) <= 10 and search_filter.strip():
                if st.button(f"📥 Add all {len(objects_to_show)} filtered objects", key="add_all_filtered"):
                    added_count = 0
                    for obj in objects_to_show:
                        if obj not in st.session_state.selected_objects_list and len(st.session_state.selected_objects_list) < 20:
                            st.session_state.selected_objects_list.append(obj)
                            added_count += 1
                    if added_count > 0:
                        st.success(f"✅ Added {added_count} objects")
                        st.rerun()
                    if len(st.session_state.selected_objects_list) >= 20:
                        st.warning("⚠️ Maximum 20 objects reached")

        # Display currently selected objects
        if st.session_state.selected_objects_list:
            st.markdown("---")
            st.markdown("**🏷️ Currently Selected Objects:**")

            # Display objects in rows with integrated remove buttons
            objects_to_remove = []

            # Always use 4 columns for consistent layout
            objects_per_row = 4
            for i in range(0, len(st.session_state.selected_objects_list), objects_per_row):
                row_objects = st.session_state.selected_objects_list[i:i+objects_per_row]

                # Always create 4 columns for consistent spacing
                cols = st.columns(4)

                for j in range(4):  # Always iterate through 4 columns
                    with cols[j]:
                        if j < len(row_objects):  # If we have an object for this position
                            obj = row_objects[j]

                            # Determine tag style based on object type
                            if obj in available_objects:
                                # Detected object - blue style
                                tag_emoji = "🎯"
                                bg_color = "linear-gradient(45deg, #007bff, #0056b3)"
                                hover_bg = "linear-gradient(45deg, #0056b3, #004085)"
                            else:
                                # Custom object - green style
                                tag_emoji = "✨"
                                bg_color = "linear-gradient(45deg, #28a745, #1e7e34)"
                                hover_bg = "linear-gradient(45deg, #1e7e34, #155724)"

                            # Create compact object tag name
                            display_name = obj if len(
                                obj) <= 8 else obj[:8] + "..."

                            # Create integrated tag button that looks like a tag
                            button_clicked = st.button(
                                f"{tag_emoji} {display_name} ×",
                                key=f"remove_obj_{i}_{j}_{obj}",
                                help=f"Click to remove '{obj}' from selection",
                                use_container_width=True
                            )

                            if button_clicked:
                                objects_to_remove.append(obj)
                        else:
                            # Empty column for consistent spacing
                            st.write("")

            # Add global CSS for tag-like button styling
            st.markdown(
                '''
                <style>
                /* Style all remove buttons to look like integrated tags */
                div[data-testid="column"] .stButton > button {
                    background: linear-gradient(45deg, #007bff, #0056b3) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 12px !important;
                    font-size: 0.75rem !important;
                    font-weight: 500 !important;
                    padding: 0.3rem 0.6rem !important;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
                    transition: all 0.2s ease !important;
                    min-height: 2.2rem !important;
                    white-space: nowrap !important;
                }
                
                /* Hover effect for tag buttons */
                div[data-testid="column"] .stButton > button:hover {
                    background: linear-gradient(45deg, #0056b3, #004085) !important;
                    transform: scale(1.02) !important;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
                }
                
                /* Green style for custom objects - will be overridden by JavaScript */
                div[data-testid="column"] .stButton > button[title*="custom"] {
                    background: linear-gradient(45deg, #28a745, #1e7e34) !important;
                }
                
                div[data-testid="column"] .stButton > button[title*="custom"]:hover {
                    background: linear-gradient(45deg, #1e7e34, #155724) !important;
                }
                </style>
                
                <script>
                // Apply different colors based on emoji in button text
                setTimeout(function() {
                    const buttons = document.querySelectorAll('div[data-testid="column"] .stButton > button');
                    buttons.forEach(button => {
                        const text = button.textContent || button.innerText;
                        if (text.includes('✨')) {
                            // Custom object - green
                            button.style.background = 'linear-gradient(45deg, #28a745, #1e7e34)';
                            button.addEventListener('mouseenter', function() {
                                this.style.background = 'linear-gradient(45deg, #1e7e34, #155724)';
                            });
                            button.addEventListener('mouseleave', function() {
                                this.style.background = 'linear-gradient(45deg, #28a745, #1e7e34)';
                            });
                        } else if (text.includes('🎯')) {
                            // Detected object - blue (default)
                            button.style.background = 'linear-gradient(45deg, #007bff, #0056b3)';
                            button.addEventListener('mouseenter', function() {
                                this.style.background = 'linear-gradient(45deg, #0056b3, #004085)';
                            });
                            button.addEventListener('mouseleave', function() {
                                this.style.background = 'linear-gradient(45deg, #007bff, #0056b3)';
                            });
                        }
                    });
                }, 100);
                </script>
                ''',
                unsafe_allow_html=True
            )

            # Remove objects that were marked for removal
            for obj_to_remove in objects_to_remove:
                if obj_to_remove in st.session_state.selected_objects_list:
                    st.session_state.selected_objects_list.remove(
                        obj_to_remove)
                    st.success(f"✅ Removed '{obj_to_remove}' from selection")
                    st.rerun()

            # Management buttons
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("🗑️ Clear All", help="Remove all selected objects"):
                    st.session_state.selected_objects_list = []
                    st.rerun()
            with col_clear2:
                if st.button("📤 Export List", help="Show selected objects as text"):
                    st.text_area("Selected Objects (copy this):",
                                 value=", ".join(
                                     st.session_state.selected_objects_list),
                                 height=100, key="export_objects")

        # Final selection for the filter (for backward compatibility)
        # Limit to 20
        selected_objects = st.session_state.selected_objects_list[:20]

    with col_obj2:
        # Filter mode
        st.markdown("**⚙️ Filter Mode**")
        object_filter_mode = st.radio(
            "Mode:",
            ["any", "all"],
            help="'any': keyframe contains at least one object, 'all': keyframe contains all objects",
            key="object_mode"
        )

        # Enable/disable object filtering
        st.markdown("**✅ Enable Object Filtering**")
        use_object_filter = st.checkbox(
            "Apply object filters to search results",
            value=False,
            help="When enabled, search results will be filtered by detected objects",
            key="use_object_filter"
        )

        # Show selected objects count and info
        if selected_objects:
            st.info(f"🎯 Selected: {len(selected_objects)} objects")

            # Show breakdown of object types
            detected_count = sum(
                1 for obj in selected_objects if obj in available_objects)
            custom_count = len(selected_objects) - detected_count

            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("🎯 Detected Objects", detected_count)
            with col_info2:
                st.metric("✨ Custom Objects", custom_count)
            with col_info3:
                st.metric("📊 Total", len(selected_objects))

# Parse object filters
object_filter = {}
if use_object_filter and selected_objects:
    object_filter = {
        "objects": selected_objects,
        "mode": object_filter_mode
    }

# Search button and logic
col_search1, col_search2 = st.columns(2)

with col_search1:
    if st.button("🚀 Text Search", use_container_width=True):
        # Ensure search mode variables exist (defined in Text Search tab)
        if 'search_mode' not in locals():
            search_mode = "Default"
        if 'exclude_groups' not in locals():
            exclude_groups = []
        if 'include_groups' not in locals():
            include_groups = []
        if 'include_videos' not in locals():
            include_videos = []
            
        if not query.strip():
            st.error("Please enter a search query")
        elif len(query) > 1000:
            st.error("Query too long. Please keep it under 1000 characters.")
        else:
            with st.spinner("🔍 Searching for keyframes..."):
                try:
                    # Use text search parameters
                    current_top_k = top_k if 'top_k' in locals() else image_top_k
                    current_threshold = score_threshold if 'score_threshold' in locals(
                    ) else image_score_threshold

                    # Show what search mode is being used
                    if search_mode == "Exclude Groups" and exclude_groups:
                        st.info(f"🚫 Excluding groups: {exclude_groups}")
                    elif search_mode == "Include Groups & Videos":
                        if include_groups or include_videos:
                            filter_parts = []
                            if include_groups:
                                filter_parts.append(f"groups: {include_groups}")
                            if include_videos:
                                filter_parts.append(f"videos: {include_videos}")
                            st.info(f"✅ Including {', '.join(filter_parts)}")

                    # Determine endpoint and base payload based on search mode
                    if search_mode == "Default":
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                        payload = {
                            "query": query,
                            "top_k": current_top_k,
                            "score_threshold": current_threshold
                        }

                    elif search_mode == "Exclude Groups":
                        if not exclude_groups:
                            st.warning("⚠️ No groups to exclude specified. Using default search.")
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold
                            }
                        else:
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold,
                                "exclude_groups": exclude_groups
                            }

                    else:  # Include Groups & Videos
                        if not include_groups and not include_videos:
                            st.warning("⚠️ No groups or videos to include specified. Using default search.")
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold
                            }
                        else:
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold,
                                "include_groups": include_groups,
                                "include_videos": include_videos
                            }

                    # If metadata filter or object filter is enabled, use metadata-filter endpoint regardless of search mode
                    if (use_metadata_filter and metadata_filter) or (use_object_filter and object_filter):
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/metadata-filter"

                        filter_info = []
                        
                        # Add search mode filters to the payload
                        if search_mode == "Exclude Groups" and exclude_groups:
                            payload["exclude_groups"] = exclude_groups
                            filter_info.append(f"exclude groups: {exclude_groups}")
                        elif search_mode == "Include Groups & Videos":
                            if include_groups:
                                payload["include_groups"] = include_groups
                                filter_info.append(f"include groups: {include_groups}")
                            if include_videos:
                                payload["include_videos"] = include_videos
                                filter_info.append(f"include videos: {include_videos}")

                        if metadata_filter:
                            payload["metadata_filter"] = metadata_filter
                            filter_info.append(
                                f"metadata: {list(metadata_filter.keys())}")

                        if object_filter:
                            payload["object_filter"] = object_filter
                            objects_str = ", ".join(
                                object_filter["objects"][:3])
                            if len(object_filter["objects"]) > 3:
                                objects_str += f" (+{len(object_filter['objects'])-3} more)"
                            filter_info.append(
                                f"objects[{object_filter['mode']}]: {objects_str}")

                        st.info(
                            f"🎯 Applying filters: {' | '.join(filter_info)}")

                    # Show the final endpoint being used
                    endpoint_display = endpoint.split('/')[-1]  # Just show the last part
                    st.info(f"🔗 Using endpoint: {endpoint_display}")

                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.search_results = results
                        st.session_state.search_query = query
                        st.rerun()
                    else:
                        st.error(
                            f"Search failed: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with col_search2:
    if st.button("🖼️ Image Search", use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload an image file")
        else:
            with st.spinner("🔍 Searching for visually similar keyframes..."):
                try:
                    # Prepare the image file for upload
                    files = {
                        'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }

                    # Prepare the parameters
                    params = {
                        'top_k': image_top_k,
                        'score_threshold': image_score_threshold
                    }

                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/image"

                    response = requests.post(
                        endpoint,
                        files=files,
                        params=params
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.search_results = results
                        st.session_state.search_query = f"Image: {uploaded_file.name}"
                        st.rerun()
                    else:
                        st.error(
                            f"Image search failed: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Display results
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")

    # Show search query if available
    if hasattr(st.session_state, 'search_query') and st.session_state.search_query:
        st.info(f"🔍 Search Query: **{st.session_state.search_query}**")

    # Handle different response formats
    results_data = st.session_state.search_results
    if isinstance(results_data, dict) and 'results' in results_data:
        results_list = results_data['results']
    else:
        results_list = results_data

    # Results summary
    col_metric1, col_metric2, col_metric3 = st.columns(3)

    with col_metric1:
        st.metric("Total Results", len(results_list))

    with col_metric2:
        if results_list:
            avg_score = sum(result['score']
                            for result in results_list) / len(results_list)
            st.metric("Average Score", f"{avg_score:.3f}")
        else:
            st.metric("Average Score", "N/A")

    with col_metric3:
        if results_list:
            max_score = max(result['score'] for result in results_list)
            st.metric("Best Score", f"{max_score:.3f}")
        else:
            st.metric("Best Score", "N/A")

    # Sort by score (highest first)
    sorted_results = []
    if results_list:
        sorted_results = sorted(
            results_list, key=lambda x: x['score'], reverse=True)

    # Display results in a grid
    for i, result in enumerate(sorted_results):
        with st.container():
            col_img, col_info = st.columns([1, 3])

            with col_img:
                try:
                    # Create button layout with better styling
                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        # Display image with click to fullscreen
                        if st.button(f"🔍 Zoom", key=f"img_{i}", help="View fullscreen with metadata", type="primary", use_container_width=True):
                            show_fullscreen_image(
                                result['path'], f"Keyframe {i+1} - Score: {result['score']:.3f}")

                    with col_btn2:
                        # View metadata details only
                        if st.button(f"📋 Detail", key=f"detail_{i}", help="View detailed metadata", type="secondary", use_container_width=True):
                            show_metadata_only(result, i)

                    # Show thumbnail image
                    st.image(result['path'], width=300,
                             caption=f"Keyframe {i+1}")

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
                # Build enhanced metadata display
                metadata_html = ""
                metadata_parts = []

                # Always show basic info
                if 'video_id' in result:
                    metadata_parts.append(
                        f"<strong>🎥 Video ID:</strong> {result['video_id']}")

                if 'group_id' in result:
                    metadata_parts.append(
                        f"<strong>📁 Group ID:</strong> {result['group_id']}")

                # Check if result has extended metadata attributes
                if 'author' in result and result['author']:
                    metadata_parts.append(
                        f"<strong>👤 Author:</strong> {result['author']}")

                if 'title' in result and result['title']:
                    title_short = result['title'][:60] + \
                        "..." if len(result['title']) > 60 else result['title']
                    metadata_parts.append(
                        f"<strong>🎬 Title:</strong> {title_short}")

                if 'length' in result and result['length']:
                    length = result['length']
                    minutes = int(length) // 60
                    seconds = int(length) % 60
                    metadata_parts.append(
                        f"<strong>⏱️ Length:</strong> {minutes}:{seconds:02d}")

                # if 'description' in result and result['description']:
                #     description = result['description'][:100] + "..." if len(result['description']) > 100 else result['description']
                #     metadata_parts.append(f"<strong>📝 Description:</strong> {description}")

                if 'keywords' in result and result['keywords']:
                    keywords = result['keywords']
                    if isinstance(keywords, list):
                        # Show first 3 keywords
                        keywords_str = ", ".join(keywords[:3])
                        if len(keywords) > 3:
                            keywords_str += f" (+{len(keywords)-3} more)"
                        metadata_parts.append(
                            f"<strong>🏷️ Keywords:</strong> {keywords_str}")

                if 'publish_date' in result and result['publish_date']:
                    metadata_parts.append(
                        f"<strong>📅 Published:</strong> {result['publish_date']}")

                # Show detected objects if available
                if 'objects' in result and result['objects']:
                    objects = result['objects']
                    if isinstance(objects, list) and objects:
                        # Show first 5 objects
                        objects_str = ", ".join(objects[:5])
                        if len(objects) > 5:
                            objects_str += f" (+{len(objects)-5} more)"
                        metadata_parts.append(
                            f"<strong>🎯 Detected Objects:</strong> {objects_str}")

                if metadata_parts:
                    metadata_html = f'<div class="metadata-section">{"<br>".join(metadata_parts)}</div>'

                # Format path for display - show only relative part from keyframes/
                display_path = result['path']
                if 'keyframes\\' in display_path:
                    display_path = display_path.split('keyframes\\')[-1]
                elif 'keyframes/' in display_path:
                    display_path = display_path.split('keyframes/')[-1]

                # Create the result card HTML
                result_html = f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #333;">Result #{i+1}</h4>
                        <span class="score-badge">Score: {result['score']:.3f}</span>
                    </div>
                    {metadata_html}
                    <div style="margin-top: 0.5rem; padding: 0.5rem; background: #f8f9fa; border-radius: 5px;">
                        <strong>📂 Path:</strong> <code style="font-size: 0.85rem;">{display_path}</code>
                    </div>
                </div>
                """

                st.markdown(result_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
