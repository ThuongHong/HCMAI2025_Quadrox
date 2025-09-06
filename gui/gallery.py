import streamlit as st
import pandas as pd
import os
import json
import csv
from pathlib import Path
from datetime import datetime
import re
import time

# Page configuration
st.set_page_config(
    page_title="Keyframe Gallery Browser",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force refresh cache
st.write(f"<!-- Cache buster: {time.time()} -->", unsafe_allow_html=True)

@st.cache_data
def scan_keyframe_structure():
    """Scan keyframe directory structure to find available groups and videos"""
    structure = {}
    
    # Find project root dynamically
    project_root = Path("..")
    keyframes_path = project_root / "resources" / "keyframes"
    
    if not keyframes_path.exists():
        return structure, None
    
    # Scan directory structure: keyframes/{L|K}{group}/{L|K}{group}_V{video}/
    for group_dir in keyframes_path.iterdir():
        if group_dir.is_dir() and re.match(r'[LK]\d+', group_dir.name):
            # Extract group info: L21 -> ('L', 21), K01 -> ('K', 1)
            group_prefix = group_dir.name[0]  # L or K
            group_number = int(group_dir.name[1:])  # 21, 01, etc.
            group_key = f"{group_prefix}{group_number:02d}"  # L21, K01
            structure[group_key] = {}
            
            for video_dir in group_dir.iterdir():
                if video_dir.is_dir() and re.match(r'[LK]\d+_V\d+', video_dir.name):
                    # Extract video number from L21_V001 or K01_V001 -> 1
                    video_match = re.search(r'_V(\d+)', video_dir.name)
                    if video_match:
                        video_id = int(video_match.group(1))
                        
                        # Count keyframes in this directory
                        keyframes = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
                        structure[group_key][video_id] = {
                            'path': video_dir,
                            'full_id': video_dir.name,  # L21_V001 or K01_V001
                            'keyframe_count': len(keyframes),
                            'keyframes': sorted(keyframes, key=lambda x: int(x.stem))
                        }
    
    return structure, keyframes_path

@st.cache_data
def load_keyframe_mapping(video_full_id):
    """Load keyframe mapping CSV for a specific video"""
    project_root = Path("..")
    map_path = project_root / "resources" / "map-keyframes" / f"{video_full_id}.csv"
    
    if map_path.exists():
        try:
            df = pd.read_csv(map_path)
            return df
        except Exception as e:
            st.warning(f"Error loading mapping for {video_full_id}: {e}")
            return None
    
    return None

def apply_video_id_mapping(video_id):
    """Apply video ID mapping rule: L01-L20 -> K01-K20"""
    try:
        # Extract group and video parts: L21_V001 -> ('L', '21', 'V001')
        match = re.match(r'([LK])(\d+)(_V\d+)', video_id)
        if match:
            prefix, group_num, video_part = match.groups()
            group_number = int(group_num)
            
            # Apply mapping rule: L01-L20 -> K01-K20
            if prefix == 'L' and 1 <= group_number <= 20:
                mapped_video_id = f"K{group_number:02d}{video_part}"
                return mapped_video_id
            
        # Return original if no mapping needed
        return video_id
    except Exception:
        # Return original if any error
        return video_id

def export_gallery_selection_to_csv(filename, processed_keyframes, mapping_df=None, append_mode=False, export_mode="KIS", qa_question="", num_scenes=3):
    """Export processed keyframes to CSV in submission format"""
    try:
        project_root = Path("..")
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        filename = str(output_dir / filename)
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Prepare data for CSV based on mode
        csv_data = []
        
        if export_mode == "KIS":
            # Original KIS format: video_id, frame_idx
            for keyframe_info in processed_keyframes:
                # Apply video ID mapping (L01-L20 -> K01-K20)
                mapped_video_id = apply_video_id_mapping(keyframe_info['video_full_id'])
                csv_data.append([
                    mapped_video_id,
                    keyframe_info['real_frame_idx']
                ])
            
            # Create DataFrame with explicit column order
            df = pd.DataFrame(csv_data, columns=['video_id', 'frame_idx'])
            
        elif export_mode == "QA":
            # QA format: video_id, frame_idx, answer (in quotes)
            for keyframe_info in processed_keyframes:
                # Apply video ID mapping (L01-L20 -> K01-K20)
                mapped_video_id = apply_video_id_mapping(keyframe_info['video_full_id'])
                csv_data.append([
                    mapped_video_id,
                    keyframe_info['real_frame_idx'],
                    qa_question  # Don't add quotes here, CSV will handle it
                ])
            
            # Create DataFrame with explicit column order
            df = pd.DataFrame(csv_data, columns=['video_id', 'frame_idx', 'answer'])
            
        elif export_mode == "TRAKE":
            # TRAKE format: dynamic columns based on num_scenes
            # Group keyframes by video
            video_groups = {}
            for keyframe_info in processed_keyframes:
                # Apply video ID mapping (L01-L20 -> K01-K20)
                mapped_video_id = apply_video_id_mapping(keyframe_info['video_full_id'])
                if mapped_video_id not in video_groups:
                    video_groups[mapped_video_id] = []
                video_groups[mapped_video_id].append(keyframe_info['real_frame_idx'])
            
            # Create rows with num_scenes columns
            for video_id, frame_list in video_groups.items():
                # Limit to num_scenes frames
                limited_frames = frame_list[:num_scenes]
                
                # Create row data as list in correct order: [video_id, scene_1, scene_2, ...]
                row_data = [video_id]
                for i in range(num_scenes):
                    if i < len(limited_frames):
                        row_data.append(limited_frames[i])
                    else:
                        row_data.append('')  # Empty for unused scenes
                
                csv_data.append(row_data)
            
            # Create DataFrame with proper column order
            columns = ['video_id'] + [f'scene_{i+1}' for i in range(num_scenes)]
            df = pd.DataFrame(csv_data, columns=columns)
        
        if append_mode and Path(filename).exists():
            # Read existing CSV and append new data
            try:
                if export_mode == "KIS":
                    existing_df = pd.read_csv(filename, header=None, names=['video_id', 'frame_idx'])
                elif export_mode == "QA":
                    existing_df = pd.read_csv(filename, header=None, names=['video_id', 'frame_idx', 'answer'])
                elif export_mode == "TRAKE":
                    # For TRAKE, read with dynamic columns but no header
                    # Generate column names based on current num_scenes
                    columns = ['video_id'] + [f'scene_{i+1}' for i in range(num_scenes)]
                    existing_df = pd.read_csv(filename, header=None, names=columns)
                
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # Remove duplicates only when ALL columns are identical
                combined_df = combined_df.drop_duplicates()
                
                # Save combined data
                if export_mode == "QA":
                    # For QA mode, ensure quotes around answer
                    combined_df['answer'] = combined_df['answer'].apply(lambda x: f'"{x}"')
                combined_df.to_csv(filename, index=False, header=False, encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar='\\')
                
                return True, len(csv_data), filename, len(combined_df)  # Return both new count and total count
            except Exception as e:
                st.warning(f"Could not read existing CSV, creating new file: {e}")
                # Fall back to creating new file
                if export_mode == "QA":
                    # For QA mode, ensure quotes around answer
                    df['answer'] = df['answer'].apply(lambda x: f'"{x}"')
                df.to_csv(filename, index=False, header=False, encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar='\\')
                return True, len(csv_data), filename, len(csv_data)
        else:
            # Create new file or overwrite
            if export_mode == "QA":
                # For QA mode, ensure quotes around answer
                df['answer'] = df['answer'].apply(lambda x: f'"{x}"')
            df.to_csv(filename, index=False, header=False, encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar='\\')
            return True, len(csv_data), filename, len(csv_data)
    
    except Exception as e:
        st.error(f"Error exporting gallery selection: {e}")
        return False, 0, None, 0

# Enhanced CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    .gallery-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #007bff;
    }
    
    .keyframe-card {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .keyframe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .keyframe-card.selected {
        border-color: #007bff;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .keyframe-info {
        text-align: center;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #28a745;
    }
    
    .export-section {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-top: 1.5rem;
    }
    
    .video-summary {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_keyframes' not in st.session_state:
    st.session_state.selected_keyframes = []
if 'gallery_csv_filename' not in st.session_state:
    st.session_state.gallery_csv_filename = f"query-p2-.csv"
# SIMPLE: Only track current video's checked keyframes
if 'current_video_checked' not in st.session_state:
    st.session_state.current_video_checked = set()
# Mode selection
if 'export_mode' not in st.session_state:
    st.session_state.export_mode = "KIS"
# QA mode question
if 'qa_question' not in st.session_state:
    st.session_state.qa_question = ""
# TRAKE mode settings
if 'trake_num_scenes' not in st.session_state:
    st.session_state.trake_num_scenes = 3
if 'trake_video_selections' not in st.session_state:
    st.session_state.trake_video_selections = {}  # {video_id: [frame1, frame2, ...]}

# Header
st.markdown("""
<div class="gallery-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🖼️ Keyframe Gallery Browser</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Browse and select keyframes by Group ID and Video ID for CSV export
    </p>
</div>
""", unsafe_allow_html=True)

# Scan keyframe structure
with st.spinner("🔍 Scanning keyframe directory structure..."):
    structure, keyframes_path = scan_keyframe_structure()

if not structure:
    st.error("❌ No keyframe structure found. Please check if keyframes directory exists.")
    st.info("Expected structure: `resources/keyframes/{L|K}{group}/{L|K}{group}_V{video}/001.jpg, 002.jpg, ...`")
    st.stop()

# Display structure summary
total_videos = sum(len(videos) for videos in structure.values())
st.success(f"✅ Found {len(structure)} groups with {total_videos} total videos")

# Filter Section
st.markdown("""
<div class="filter-section">
    <h3 style="margin-top: 0;">🎯 Select Export Mode, Group and Video</h3>
</div>
""", unsafe_allow_html=True)

# Mode selection row
col_mode1, col_mode2, col_mode3 = st.columns([2, 2, 2])

with col_mode1:
    export_mode = st.selectbox(
        "📋 Export Mode",
        options=["KIS", "QA", "TRAKE"],
        index=["KIS", "QA", "TRAKE"].index(st.session_state.export_mode),
        help="Choose export format:\n• KIS: video_id, frame_idx\n• QA: video_id, frame_idx, answer\n• TRAKE: video_id, scene_1, scene_2, ...",
        key="export_mode_selector"
    )
    st.session_state.export_mode = export_mode

with col_mode2:
    if export_mode == "QA":
        qa_question = st.text_input(
            "❓ Answer",
            value=st.session_state.qa_question,
            placeholder="Enter answer for QA mode...",
            help="Answer will be wrapped in quotes in CSV",
            key="qa_question_input"
        )
        st.session_state.qa_question = qa_question
    elif export_mode == "TRAKE":
        trake_num_scenes = st.number_input(
            "🎬 Number of Scenes",
            min_value=1,
            max_value=10,
            value=st.session_state.trake_num_scenes,
            help="Maximum scenes per video in TRAKE mode",
            key="trake_scenes_input"
        )
        st.session_state.trake_num_scenes = trake_num_scenes

with col_mode3:
    # Mode info display
    if export_mode == "KIS":
        st.info("📊 Format: video_id, frame_idx")
    elif export_mode == "QA":
        st.info("📊 Format: video_id, frame_idx, \"answer\"")
    elif export_mode == "TRAKE":
        st.info(f"📊 Format: video_id, scene_1, ..., scene_{trake_num_scenes}")

col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])

with col_filter1:
    # Group selection
    available_groups = sorted(structure.keys())
    selected_group = st.selectbox(
        "📁 Select Group ID",
        options=available_groups,
        format_func=lambda x: f"{x} ({len(structure[x])} videos)",
        help="Choose a group to see available videos (L21-L30, K01-K20)",
        key="group_selector"
    )

with col_filter2:
    # Video selection
    if selected_group in structure:
        available_videos = sorted(structure[selected_group].keys())
        selected_video = st.selectbox(
            "🎥 Select Video ID",
            options=available_videos,
            format_func=lambda x: f"V{x:03d} ({structure[selected_group][x]['keyframe_count']} keyframes)",
            help="Choose a video to browse its keyframes",
            key="video_selector"
        )
    else:
        selected_video = None
        st.selectbox("🎥 Select Video ID", options=[], disabled=True, key="video_selector_disabled")

with col_filter3:
    # Display selection info
    if selected_group and selected_video:
        video_info = structure[selected_group][selected_video]
        st.markdown(f"""
        <div class="video-summary">
            <h4 style="margin: 0; color: #007bff;">📹 {video_info['full_id']}</h4>
            <p style="margin: 0.5rem 0 0 0;">
                📊 <strong>{video_info['keyframe_count']}</strong> keyframes available
            </p>
        </div>
        """, unsafe_allow_html=True)

# Selection controls and Export
if selected_group and selected_video:
    video_info = structure[selected_group][selected_video]
    video_full_id = video_info['full_id']
    keyframes = video_info['keyframes']
    
    st.markdown("---")
    
    # Load mapping data
    mapping_df = load_keyframe_mapping(video_full_id)
    
    # Export section always visible at the top
    # SIMPLE: Only track current video's keyframes
    checked_count = sum(1 for keyframe_path in keyframes 
                       if st.session_state.get(f"mark_{video_full_id}_{int(keyframe_path.stem)}", False))
    
    # Simple stats for current video only
    checked_stats = {
        'count': checked_count, 
        'current_video': video_full_id if checked_count > 0 else None
    }
    
    checked_count = checked_stats['count']
    
    # Show checked keyframes for current video only
    if checked_count > 0:
        with st.expander(f"📋 Checked Keyframes in {video_full_id} ({checked_count})", expanded=False):
            checked_keyframes = []
            for keyframe_path in keyframes:
                n = int(keyframe_path.stem)
                checkbox_key = f"mark_{video_full_id}_{n}"
                if st.session_state.get(checkbox_key, False):
                    checked_keyframes.append(f"{n:03d}.jpg")
            
            st.text(", ".join(sorted(checked_keyframes)))
    
    # Simple statistics for current video only
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    
    with col_stat1:
        st.metric("📊 Checked", checked_count)
    
    with col_stat2:
        st.metric("🎥 Video", video_full_id if checked_count > 0 else "None")
    
    with col_stat3:
        # Show TRAKE limitation info
        if export_mode == "TRAKE":
            max_scenes = st.session_state.trake_num_scenes
            if checked_count > max_scenes:
                st.warning(f"⚠️ Only first {max_scenes} will be used")
            else:
                st.info(f"✅ Max {max_scenes} scenes")
        else:
            # Append mode toggle
            append_mode = st.checkbox("📝 Append Mode", value=True, help="Append to existing CSV instead of overwriting")
    
    with col_stat4:
        # Export configuration (compact)
        if export_mode != "TRAKE":  # Append mode not shown for TRAKE in col_stat3
            csv_filename = st.text_input(
                "📁 Filename",
                value=st.session_state.gallery_csv_filename,
                help="CSV filename",
                key="csv_filename_input",
                label_visibility="collapsed",
                placeholder="Enter CSV filename..."
            )
        else:
            # For TRAKE mode, also show append mode here
            append_mode = st.checkbox("📝 Append Mode", value=True, help="Append to existing CSV instead of overwriting")
            
        if 'csv_filename' not in locals():
            csv_filename = st.text_input(
                "📁 Filename",
                value=st.session_state.gallery_csv_filename,
                help="CSV filename",
                key="csv_filename_input2" if export_mode == "TRAKE" else "csv_filename_input",
                label_visibility="collapsed",
                placeholder="Enter CSV filename..."
            )
        
        if csv_filename != st.session_state.gallery_csv_filename:
            st.session_state.gallery_csv_filename = csv_filename
    
    with col_stat5:
        # Export button - Compact
        if checked_count > 0:
            # Check QA mode validation
            if export_mode == "QA" and not st.session_state.qa_question.strip():
                button_text = "📤 Need Answer"
                button_disabled = True
            else:
                button_text = f"📤 {'Append' if append_mode else 'Export'} ({checked_count})"
                button_disabled = False
        else:
            button_text = "📤 No items"
            button_disabled = True
    
    if st.button(button_text, type="primary", use_container_width=True, disabled=button_disabled):
        if csv_filename.strip():
            # Validate QA mode
            if export_mode == "QA" and not st.session_state.qa_question.strip():
                st.error("Please enter an answer for QA mode")
            else:
                with st.spinner(f"🔄 {'Appending' if append_mode else 'Exporting'} current video's keyframes..."):
                    # Collect checked keyframes from CURRENT VIDEO ONLY
                    processed_keyframes = []
                    
                    for keyframe_path in keyframes:
                        n = int(keyframe_path.stem)
                        checkbox_key = f"mark_{video_full_id}_{n}"
                        
                        if st.session_state.get(checkbox_key, False):  # Checkbox is checked
                            try:
                                # Get real frame_idx from mapping
                                real_frame_idx = n  # Default fallback
                                if mapping_df is not None:
                                    matching_row = mapping_df[mapping_df['n'] == n]
                                    if not matching_row.empty:
                                        real_frame_idx = int(matching_row.iloc[0]['frame_idx'])
                                
                                processed_keyframes.append({
                                    'video_full_id': video_full_id,
                                    'n': n,
                                    'real_frame_idx': real_frame_idx
                                })
                            except Exception as e:
                                st.warning(f"Error processing {keyframe_path.name}: {e}")
                    
                    # For TRAKE mode, limit to num_scenes
                    if export_mode == "TRAKE":
                        max_scenes = st.session_state.trake_num_scenes
                        if len(processed_keyframes) > max_scenes:
                            processed_keyframes = processed_keyframes[:max_scenes]
                            st.info(f"🎬 Limited to first {max_scenes} scenes for TRAKE mode")
                    
                    # Export processed keyframes with append mode
                    if processed_keyframes:
                        result = export_gallery_selection_to_csv(
                            csv_filename, 
                            processed_keyframes,
                            None,  # We already processed the mapping above
                            append_mode=append_mode,  # Pass append mode
                            export_mode=export_mode,
                            qa_question=st.session_state.qa_question,
                            num_scenes=st.session_state.trake_num_scenes
                        )
                    
                        
                        if len(result) == 4:  # New format with total count
                            success, count, output_path, total_count = result
                        else:  # Fallback to old format
                            success, count, output_path = result
                            total_count = count
                        
                        if success:
                            if append_mode:
                                st.success(f"✅ Appended {count} keyframes from {video_full_id} to {output_path}")
                                st.info(f"📊 Total entries in file: {total_count}")
                            else:
                                st.success(f"✅ Exported {count} keyframes from {video_full_id} to {output_path}")
                            
                            # Clear current video's selections after successful export
                            for keyframe_path in keyframes:
                                n = int(keyframe_path.stem)
                                checkbox_key = f"mark_{video_full_id}_{n}"
                                st.session_state[checkbox_key] = False
                            
                            # Show export preview
                            with st.expander("👀 Export Preview (first 10 rows)", expanded=True):
                                preview_data = []
                                for kf in processed_keyframes[:10]:
                                    row_data = {
                                        'video_id': kf['video_full_id'],
                                        'frame_idx': kf['real_frame_idx'],
                                        'source_n': kf['n']
                                    }
                                    if export_mode == "QA":
                                        row_data['answer'] = f'"{st.session_state.qa_question}"'
                                    preview_data.append(row_data)
                                
                                preview_df = pd.DataFrame(preview_data)
                                st.dataframe(preview_df, use_container_width=True)
                        else:
                            st.error("❌ Failed to export CSV")
                    else:
                        st.error("No keyframes marked for export. Please check some keyframes first.")
        else:
            st.error("Please enter a valid filename")
    
    st.markdown("---")
    
    # Show mapping info and controls
    col_map1, col_map2 = st.columns([3, 2])
    
    with col_map1:
        if mapping_df is not None:
            st.success(f"✅ Loaded mapping data ({len(mapping_df)} entries)")
            
            # Show full mapping data instead of sample
            with st.expander("📋 Full Mapping Data", expanded=False):
                st.dataframe(mapping_df, use_container_width=True)
        else:
            st.warning(f"⚠️ No mapping data found for {video_full_id}")
    
    with col_map2:
        # Selection controls
        st.markdown("**🎯 Selection Controls**")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            if export_mode == "TRAKE":
                button_text = f"✅ Check First {st.session_state.trake_num_scenes}"
                help_text = f"Check first {st.session_state.trake_num_scenes} keyframes (TRAKE limit)"
            else:
                button_text = "✅ Check All"
                help_text = "Check all keyframes in this video"
                
            if st.button(button_text, help=help_text, use_container_width=True):
                if export_mode == "TRAKE":
                    # Only check up to trake_num_scenes
                    max_scenes = st.session_state.trake_num_scenes
                    for i, keyframe_path in enumerate(keyframes[:max_scenes]):
                        n = int(keyframe_path.stem)
                        checkbox_key = f"mark_{video_full_id}_{n}"
                        st.session_state[checkbox_key] = True
                else:
                    # Set all checkbox values to True for current video
                    for keyframe_path in keyframes:
                        n = int(keyframe_path.stem)
                        checkbox_key = f"mark_{video_full_id}_{n}"
                        st.session_state[checkbox_key] = True
                st.rerun()
        
        with col_sel2:
            if st.button("🗑️ Uncheck All", help="Uncheck all keyframes in this video", use_container_width=True):
                # Set all checkbox values to False for current video
                for keyframe_path in keyframes:
                    n = int(keyframe_path.stem)
                    checkbox_key = f"mark_{video_full_id}_{n}"
                    st.session_state[checkbox_key] = False
                st.rerun()
    
    # Display keyframes in grid
    st.markdown(f"### 🖼️ Keyframes for {video_full_id}")
    
    if keyframes:
        # Display keyframes in grid with checkboxes (no rerun needed)
        st.markdown("**📌 Check keyframes to mark for export:**")
        
        # Performance controls
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            show_images = st.checkbox("🖼️ Show Images", value=True, key="show_images_toggle")
        with col_perf2:
            items_per_page = st.selectbox("📄 Items per page", [10, 20, 30, 50], index=1, key="items_per_page")  # Default 20 instead of 50
        with col_perf3:
            if len(keyframes) > items_per_page:
                total_pages = (len(keyframes) - 1) // items_per_page + 1
                current_page = st.selectbox("📖 Page", range(1, total_pages + 1), key="current_page")
            else:
                current_page = 1
                total_pages = 1
        
        # Calculate pagination
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(keyframes))
        current_keyframes = keyframes[start_idx:end_idx]
        
        st.markdown(f"**Showing {len(current_keyframes)} of {len(keyframes)} keyframes (Page {current_page}/{total_pages})**")
        
        # Optimize: Create mapping lookup dictionary once
        mapping_lookup = {}
        if mapping_df is not None:
            for _, row in mapping_df.iterrows():
                mapping_lookup[int(row['n'])] = {
                    'frame_idx': int(row['frame_idx']),
                    'pts_time': f"{row['pts_time']:.1f}s"
                }
        
        # Create grid layout using columns - REDUCE COLUMN COUNT for performance
        keyframes_per_row = 4 if show_images else 6  # Fewer columns to reduce lag
        
        for i in range(0, len(current_keyframes), keyframes_per_row):
            cols = st.columns(keyframes_per_row)
            row_keyframes = current_keyframes[i:i+keyframes_per_row]
            
            for j, keyframe_path in enumerate(row_keyframes):
                with cols[j]:
                    n = int(keyframe_path.stem)  # Get number from filename
                    
                    # Display keyframe conditionally
                    if show_images:
                        try:
                            st.image(str(keyframe_path), width=200)  # Even smaller images
                        except:
                            st.error(f"Cannot load {keyframe_path.name}")
                    
                    # Get real frame_idx and pts_time from mapping (optimized lookup)
                    if n in mapping_lookup:
                        real_frame_idx = mapping_lookup[n]['frame_idx']
                        pts_time = mapping_lookup[n]['pts_time']
                    else:
                        real_frame_idx = n  # Default fallback
                        pts_time = "N/A"
                    
                    # Compact keyframe info
                    if show_images:
                        info_text = f"**{n:03d}.jpg** | F:{real_frame_idx} | {pts_time}"
                    else:
                        info_text = f"**📄 {n:03d}.jpg** | Frame:{real_frame_idx} | Time:{pts_time}"
                    
                    st.markdown(info_text)
                    
                    # Checkbox for marking - SIMPLE VERSION (NO CALLBACK!)
                    checkbox_key = f"mark_{video_full_id}_{n}"
                    current_checked = st.session_state.get(checkbox_key, False)
                    
                    # For TRAKE mode, check if we're at the limit
                    if export_mode == "TRAKE":
                        # Count currently checked for this video
                        video_checked_count = sum(1 for kf_path in keyframes 
                                                if st.session_state.get(f"mark_{video_full_id}_{int(kf_path.stem)}", False))
                        
                        max_scenes = st.session_state.trake_num_scenes
                        
                        # Disable checkbox if at limit and this one isn't already checked
                        if video_checked_count >= max_scenes and not current_checked:
                            st.checkbox(
                                f"Mark (Max {max_scenes})",
                                value=False,
                                disabled=True,
                                key=f"{checkbox_key}_disabled",
                                help=f"Maximum {max_scenes} scenes allowed per video in TRAKE mode"
                            )
                        else:
                            is_checked = st.checkbox(
                                f"Mark ({video_checked_count}/{max_scenes})",
                                value=current_checked,
                                key=checkbox_key,
                                help=f"Selected scenes: {video_checked_count}/{max_scenes}"
                            )
                    else:
                        # Normal checkbox for KIS and QA modes
                        is_checked = st.checkbox(
                            "Mark",  # Shorter label
                            value=current_checked,
                            key=checkbox_key
                        )
    else:
        st.warning(f"No keyframes found in {video_info['path']}")

else:
    # Show empty state when no group/video selected
    st.markdown("---")
    st.info("📋 No keyframes checked yet. Choose a Group ID and Video ID above to start browsing and checking frames.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🖼️ Keyframe Gallery Browser | Built with Streamlit</p>
    <p><small>💡 Tip: Use this tool to quickly browse and select keyframes for submission CSV files</small></p>
</div>
""", unsafe_allow_html=True)
