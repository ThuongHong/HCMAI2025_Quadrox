import time
import streamlit as st
import requests
import json
import pandas as pd
import os
import subprocess
import csv
from pathlib import Path
from datetime import datetime
import re
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Keyframe Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force refresh cache
st.write(f"<!-- Cache buster: {time.time()} -->", unsafe_allow_html=True)

# Check working directory and set up path prefix
current_cwd = os.getcwd()
path_prefix = ""

if current_cwd.endswith('gui') or current_cwd.endswith('gui\\'):
    # If running from gui folder, need to go up one level for resources
    path_prefix = "../"
    print(f"🔧 Running from gui folder, using path prefix: {path_prefix}")
else:
    print(f"🔧 Running from project root: {current_cwd}")

# Verify resources directory exists
resources_path = f"{path_prefix}resources"
if not os.path.exists(resources_path):
    st.error("❌ Resources directory not found. Please run from project root or gui/ folder.")
    st.info(f"Looking for: {os.path.abspath(resources_path)}")
    st.stop()

# Helper Functions


def debug_paths():
    """Debug function to check current working directory and available paths"""
    import os
    current_dir = os.getcwd()
    st.write(f"**Current working directory:** {current_dir}")

    # Check for resources directory
    resources_paths = [
        Path("../resources"),
        Path("resources"),
        Path("d:/Study/HCMAI2025_Quadrox/resources")
    ]

    st.write("**Resource directory checks:**")
    for path in resources_paths:
        exists = path.exists()
        st.write(f"- {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        if exists:
            # Check subdirectories
            for subdir in ['map-keyframes', 'metadata', 'objects']:
                subpath = path / subdir
                st.write(f"  - {subpath}: {'✅' if subpath.exists() else '❌'}")

    # Test the parsing functions with sample data
    st.write("**Function testing:**")
    test_path = "resources/keyframes/L21/L21_V001/002.jpg"
    video_id, frame_idx = get_frame_info_from_keyframe_path(test_path)

    if video_id and frame_idx is not None:
        pts_time = get_pts_time_from_n(video_id, frame_idx)
        if pts_time is not None:
            st.success("✅ Complete parsing and lookup workflow is working!")
            st.write(
                f"Sample: `{video_id}` frame `{frame_idx}` → `{pts_time}s`")
        else:
            st.warning("⚠️ pts_time lookup failed")
    else:
        st.warning("⚠️ Path parsing failed")


def get_frame_info_from_keyframe_path(keyframe_path):
    """Extract video ID and frame index from keyframe path"""
    try:
        # Path structure: .../keyframes/L21/L21_V001/001.jpg
        # Extract the path components
        path_parts = keyframe_path.replace('\\', '/').split('/')

        # Find the pattern in the path
        video_id_full = None
        frame_filename = None

        for i, part in enumerate(path_parts):
            # Look for L{batch}_V{video} pattern in directory names
            if re.match(r'L\d+_V\d+', part):
                video_id_full = part
                # The next part should be the filename
                if i + 1 < len(path_parts):
                    frame_filename = path_parts[i + 1]
                break

        if video_id_full and frame_filename:
            # Extract frame index from filename (e.g., "001.jpg" -> 1)
            filename_without_ext = os.path.splitext(frame_filename)[0]
            try:
                # Convert "001" to 1, "002" to 2, etc.
                frame_idx = int(filename_without_ext)
                return video_id_full, frame_idx
            except ValueError:
                # Fallback: try to extract numbers from filename
                numbers = re.findall(r'\d+', filename_without_ext)
                if numbers:
                    frame_idx = int(numbers[-1])  # Take the last number found
                    return video_id_full, frame_idx

        return None, None
    except Exception as e:
        return None, None


def get_real_frame_idx_from_n(video_id, n):
    """Get real frame_idx from n using map-keyframes CSV"""
    try:
        # Try multiple path combinations to find the map file
        possible_paths = [
            Path("../resources/map-keyframes") / f"{video_id}.csv",
            Path("resources/map-keyframes") / f"{video_id}.csv",
            Path("d:/Study/HCMAI2025_Quadrox/resources/map-keyframes") /
            f"{video_id}.csv",
            Path("../resources/map-keyframes") / f"{video_id.upper()}.csv",
            Path("resources/map-keyframes") / f"{video_id.upper()}.csv",
            Path("d:/Study/HCMAI2025_Quadrox/resources/map-keyframes") /
            f"{video_id.upper()}.csv",
        ]

        map_file_path = None
        for path in possible_paths:
            if path.exists():
                map_file_path = path
                break

        if map_file_path and map_file_path.exists():
            df = pd.read_csv(map_file_path)
            # Find the row where 'n' matches our n value
            matching_row = df[df['n'] == n]
            if not matching_row.empty:
                return int(matching_row.iloc[0]['frame_idx'])

        return None
    except Exception as e:
        return None


def get_pts_time_from_n(video_id, n):
    """Get pts_time from filename number (n) using map-keyframes CSV"""
    try:
        # Try multiple path combinations to find the map file
        possible_paths = [
            Path("../resources/map-keyframes") / f"{video_id}.csv",
            Path("resources/map-keyframes") / f"{video_id}.csv",
            Path("d:/Study/HCMAI2025_Quadrox/resources/map-keyframes") /
            f"{video_id}.csv",
            Path("../resources/map-keyframes") / f"{video_id.upper()}.csv",
            Path("resources/map-keyframes") / f"{video_id.upper()}.csv",
            Path("d:/Study/HCMAI2025_Quadrox/resources/map-keyframes") /
            f"{video_id.upper()}.csv",
        ]

        map_file_path = None
        for path in possible_paths:
            if path.exists():
                map_file_path = path
                break

        if map_file_path and map_file_path.exists():
            df = pd.read_csv(map_file_path)
            # The n from our parsing (e.g., 1 from "001.jpg") corresponds to the 'n' column
            # Find the row where 'n' matches our n value
            matching_row = df[df['n'] == n]
            if not matching_row.empty:
                return float(matching_row.iloc[0]['pts_time'])

        return None
    except Exception as e:
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


def append_to_csv(filename, result_data):
    """Append a single result to CSV file in format: video_id, frame_idx, [vqa_answer]"""
    try:
        output_dir = Path("../output")
        output_dir.mkdir(exist_ok=True)

        # Ensure the filename has .csv extension
        filename = str(output_dir / filename)
        if not filename.endswith('.csv'):
            filename += '.csv'

        # Apply video ID mapping (L01-L20 -> K01-K20)
        video_id = result_data.get('video_id', '')
        mapped_video_id = apply_video_id_mapping(video_id)

        # Data row: video_id, real frame_idx, and optionally vqa_answer
        row_data = {
            'video_id': mapped_video_id,
            # Use real_frame_idx if available
            'frame_idx': result_data.get('real_frame_idx', result_data.get('frame_idx', '')),
        }

        # Add VQA answer if provided
        vqa_text = ""
        if 'vqa_answer' in result_data and result_data['vqa_answer']:
            vqa_text = result_data['vqa_answer'].strip()

        # Write CSV manually to control exact format
        with open(filename, 'a', encoding='utf-8', newline='') as f:
            if vqa_text:
                # Format: video_id,frame_idx,"vqa_answer"
                f.write(
                    f'{row_data["video_id"]},{row_data["frame_idx"]},"{vqa_text}"\n')
            else:
                # Format: video_id,frame_idx
                f.write(f'{row_data["video_id"]},{row_data["frame_idx"]}\n')

        return True
    except Exception as e:
        st.error(f"Error appending to CSV: {e}")
        return False


def export_all_results_to_csv(filename, results_list):
    """Export all search results to CSV file in simple format: video_id, frame_idx"""
    try:
        output_dir = Path("../output")
        output_dir.mkdir(exist_ok=True)
        filename = str(output_dir / filename)
        if not filename.endswith('.csv'):
            filename += '.csv'

        # Prepare simple data
        all_data = []
        for result in results_list:
            # Extract frame info from path (e.g., L21_V001 from "L21/L21_V001/001.jpg")
            video_id, n = get_frame_info_from_keyframe_path(
                result.get('path', ''))

            if video_id and n:
                # Get real frame_idx from CSV mapping
                real_frame_idx = get_real_frame_idx_from_n(video_id, n)
                if real_frame_idx is not None:
                    # Apply video ID mapping (L01-L20 -> K01-K20)
                    mapped_video_id = apply_video_id_mapping(video_id)

                    row_data = {
                        'video_id': mapped_video_id,  # Use mapped video_id
                        'frame_idx': real_frame_idx,  # Use real frame_idx from mapping
                    }
                    all_data.append(row_data)

        # Create DataFrame and save without header with UTF-8 encoding
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False, header=False, encoding='utf-8')

        return True, len(all_data)
    except Exception as e:
        st.error(f"Error exporting to CSV: {e}")
        return False, 0


@st.cache_data
def load_available_objects():
    """Load available objects from the migration-generated JSON file"""
    try:
        # Try multiple possible paths
        possible_paths = [
            Path("../resources/objects/all_objects_found.json"),
            Path("resources/objects/all_objects_found.json"),
            Path("d:/Study/HCMAI2025_Quadrox/resources/objects/all_objects_found.json")
        ]

        for objects_file in possible_paths:
            if objects_file.exists():
                with open(objects_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        'objects': data.get('objects', []),
                        'categories': data.get('objects_by_category', {}),
                        'metadata': data.get('metadata', {})
                    }
                break
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


def extract_video_names_from_results(results):
    """Extract unique video names from search results for progressive filtering"""
    video_names = set()
    for result in results:
        if 'path' in result:
            # Extract video name from path like "D:/...keyframes/L21/L21_V026/230.jpg"
            path_str = str(result['path']).replace('\\', '/')

            # Look for pattern like L##_V###
            import re
            pattern = r'L\d+_V\d+'
            matches = re.findall(pattern, path_str)

            for match in matches:
                video_names.add(match)

        # Also try to extract from video_id if available
        elif 'video_id' in result:
            video_id = result['video_id']
            if isinstance(video_id, str) and '_V' in video_id:
                video_names.add(video_id)

    return sorted(list(video_names))


def extract_video_name_from_path(path):
    """Extract video name from keyframe path"""
    if not path:
        return ""

    path_str = str(path).replace('\\', '/')
    # Look for pattern like L##_V###
    import re
    pattern = r'L\d+_V\d+'
    matches = re.findall(pattern, path_str)

    if matches:
        return matches[0]
    return ""


def extract_frame_number_from_path(path):
    """Extract frame number from keyframe path"""
    if not path:
        return 0

    path_str = str(path).replace('\\', '/')
    # Extract filename without extension and convert to int
    filename = os.path.basename(path_str)
    name_without_ext = os.path.splitext(filename)[0]

    try:
        return int(name_without_ext)
    except (ValueError, TypeError):
        return 0


@st.cache_data
def load_available_keywords():
    """Load available keywords from metadata JSON files"""
    keywords_set = set()
    try:
        # Try multiple possible paths
        metadata_dirs = [
            Path("../resources/metadata"),
            Path("resources/metadata"),
            Path("d:/Study/HCMAI2025_Quadrox/resources/metadata")
        ]

        for metadata_dir in metadata_dirs:
            if metadata_dir.exists():
                for json_file in metadata_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'keywords' in data and isinstance(data['keywords'], list):
                                keywords_set.update(data['keywords'])
                    except Exception as e:
                        continue  # Skip files that can't be parsed
                break  # Use the first directory that exists

    except Exception as e:
        st.warning(f"Could not load keywords from metadata files: {e}")

    return sorted(list(keywords_set))  # Return sorted list


# Functions
def open_in_mpc(video_url, start_time=0):
    """Open video in MPC-HC using yt-dlp"""
    try:
        # Check if yt-dlp is available
        try:
            subprocess.run(['yt-dlp', '--version'],
                           capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            st.error("❌ yt-dlp not found. Please install: pip install yt-dlp")
            st.info("💡 Or download from: https://github.com/yt-dlp/yt-dlp/releases")
            return False

        # Get direct stream URL
        with st.spinner("🔍 Getting video stream URL..."):
            cmd = [
                'yt-dlp',
                '-g',
                # Prefer 720p or best available
                '--format', 'best[height<=720]/best',
                '--no-warnings',
                video_url
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                stream_url = result.stdout.strip()

                # Try different MPC-HC paths
                mpc_paths = [
                    r'C:\Program Files\MPC-HC\mpc-hc64.exe',
                    r'C:\Program Files (x86)\MPC-HC\mpc-hc.exe',
                    r'C:\Program Files\K-Lite Codec Pack\MPC-HC64\mpc-hc64.exe',
                    r'C:\Program Files (x86)\K-Lite Codec Pack\MPC-HC\mpc-hc.exe',
                    'mpc-hc64.exe',  # If in PATH
                    'mpc-hc.exe'     # If in PATH
                ]

                mpc_found = False
                for mpc_path in mpc_paths:
                    try:
                        # Test if MPC exists
                        if mpc_path.startswith('C:'):
                            if not Path(mpc_path).exists():
                                continue

                        # Build command
                        mpc_cmd = [mpc_path, stream_url]

                        # Add start time if specified (MPC-HC uses /start with space, not =)
                        if start_time > 0:
                            # Convert to milliseconds
                            mpc_cmd.extend(
                                ['/start', str(int(start_time * 1000))])

                        # Launch MPC
                        subprocess.Popen(
                            mpc_cmd, creationflags=subprocess.CREATE_NO_WINDOW)
                        mpc_found = True
                        st.success(f"✅ Opened in MPC-HC at {start_time:.1f}s")
                        return True

                    except Exception as e:
                        continue

                if not mpc_found:
                    st.error(
                        "❌ MPC-HC not found. Please install MPC-HC or K-Lite Codec Pack")
                    st.info(
                        "💡 Download from: https://mpc-hc.org/ or https://codecguide.com/download_k-lite_codec_pack_standard.htm")
                    return False

            else:
                st.error(f"❌ Failed to get stream URL: {result.stderr}")
                return False

    except subprocess.TimeoutExpired:
        st.error("❌ Timeout getting video stream URL")
        return False
    except Exception as e:
        st.error(f"❌ Error opening in MPC: {e}")
        return False


@st.dialog("MPC-HC Video Player", width="large")
def show_mpc_video(pts_time, result_data, result_index):
    """Display MPC-HC video controls with start time and frame export"""
    try:
        # Initialize session state for frame navigation
        if f'current_frame_{result_index}' not in st.session_state:
            # Get initial frame index from result data
            video_id, frame_idx = get_frame_info_from_keyframe_path(
                result_data.get('path', ''))
            st.session_state[f'current_frame_{result_index}'] = frame_idx if frame_idx else 1

        # Get video_id from path parsing (e.g., L21_V001 from "L21/L21_V001/001.jpg")
        parsed_video_id, parsed_frame_idx = get_frame_info_from_keyframe_path(
            result_data.get('path', ''))
        video_id = parsed_video_id if parsed_video_id else result_data.get(
            'video_id', 'Unknown')

        current_frame = st.session_state[f'current_frame_{result_index}']
        watch_url = result_data.get('watch_url', 'Not available')

        # Header with video info
        st.markdown(f"### 🎥 MPC-HC Player - Result #{result_index + 1}")

        # Video info columns
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"**Video ID:** {video_id}")
        with col_info2:
            st.markdown(f"**Start Time:** {pts_time:.1f}s")
        with col_info3:
            st.markdown(f"**Current File:** {current_frame}")

        if watch_url != 'Not available':
            st.markdown(f"**Original URL:** {watch_url}")

        # Calculate current time based on frame
        current_pts_time = get_pts_time_from_n(video_id, current_frame)
        if current_pts_time is None:
            current_pts_time = pts_time  # Fallback to original time

        # CSV Export section
        st.markdown("---")
        st.markdown("### 💾 Export Current Frame")

        col_csv1, col_csv2 = st.columns([2, 1])
        with col_csv1:
            # Use session state to persist the filename value
            csv_key = f"csv_filename_{result_index}"
            if csv_key not in st.session_state:
                st.session_state[csv_key] = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            csv_filename_frame = st.text_input(
                "CSV filename for frame export:",
                placeholder="Enter CSV filename",
                key=csv_key,
                help="File to append current frame data"
            )

            # Use the session state value if input is empty
            if not csv_filename_frame:
                csv_filename_frame = st.session_state[csv_key]

        with col_csv2:
            # Frame input for export
            export_frame = st.number_input(
                "Frame to export:",
                min_value=1,
                value=current_frame,
                step=1,
                key=f"export_frame_input_{result_index}",
                help="Enter frame number to export (default: current frame)"
            )

        # VQA Answer input
        vqa_answer = st.text_input(
            "VQA Answer (optional):",
            value="",
            key=f"vqa_answer_{result_index}",
            help="Enter VQA answer if available, will be added as 3rd column in CSV"
        )

        # Export button
        if st.button("📤 Add Frame to CSV", help=f"Add frame {export_frame} to CSV file", key=f"export_frame_{result_index}"):
            # Get real frame_idx from CSV mapping (n -> frame_idx)
            real_frame_idx = get_real_frame_idx_from_n(video_id, export_frame)
            if real_frame_idx is None:
                real_frame_idx = export_frame  # Fallback to input value

            # Prepare frame data for CSV: video_id, real_frame_idx, vqa_answer (if provided)
            frame_data = {
                'video_id': video_id,  # L21_V001 from path parsing
                'frame_idx': real_frame_idx,  # Real frame_idx from CSV mapping
            }

            # Add VQA answer if provided
            if vqa_answer and vqa_answer.strip():
                frame_data['vqa_answer'] = vqa_answer.strip()

            success = append_to_csv(csv_filename_frame, frame_data)
            if success:
                if vqa_answer and vqa_answer.strip():
                    st.success(
                        f"✅ Frame {export_frame} (real frame_idx: {real_frame_idx}) added to {csv_filename_frame} as {video_id}, {real_frame_idx}, {vqa_answer.strip()}")
                else:
                    st.success(
                        f"✅ Frame {export_frame} (real frame_idx: {real_frame_idx}) added to {csv_filename_frame} as {video_id}, {real_frame_idx}")
            else:
                st.error("❌ Failed to export frame data")

        # MPC-HC Quick Launch
        st.markdown("---")
        st.markdown("### 🎥 MPC-HC Player")

        # Large MPC-HC button
        col_mpc1, col_mpc2, col_mpc3 = st.columns([1, 2, 1])
        with col_mpc2:
            if st.button("🎬 Open Video in MPC-HC",
                         help=f"Open video in MPC-HC at frame {current_frame} ({current_pts_time:.1f}s)",
                         key=f"mpc_large_{result_index}",
                         use_container_width=True):
                video_url = result_data.get('watch_url', '')
                if video_url:
                    open_in_mpc(video_url, current_pts_time)
                else:
                    st.error("❌ No video URL available")

        # Quick info about current position
        st.info(
            f"Will open at **Frame {current_frame}** (Time: **{current_pts_time:.1f}s**)")

        # Setup instructions (expandable)
        with st.expander("🔧 Setup MPC-HC", expanded=False):
            st.markdown("""
            **Required:**
            1. **Install yt-dlp:** `pip install yt-dlp`
            2. **Install MPC-HC:** Download from [mpc-hc.org](https://mpc-hc.org/) or install K-Lite Codec Pack
            
            **Benefits of MPC-HC:**
            - ✅ Better video quality and performance
            - ✅ Frame-perfect seeking and navigation
            - ✅ Full keyboard shortcuts support (J/K/L, comma/dot for frame stepping)
            - ✅ No ads or restrictions
            - ✅ Better playback controls and speed adjustment
            """)

        # Show mapping info
        if 'watch_url' in result_data and result_data['watch_url']:
            st.success(f"✅ Using video URL from metadata")
        else:
            st.warning("⚠️ No video URL found in metadata.")

    except Exception as e:
        st.error(f"Could not load video: {str(e)}")


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

    /* Special styling for Video buttons */
    div[data-testid="column"] .stButton > button[title*="Video"] {
        background: linear-gradient(45deg, #ff0000, #cc0000) !important;
        color: white !important;
    }
    
    div[data-testid="column"] .stButton > button[title*="Video"]:hover {
        background: linear-gradient(45deg, #cc0000, #990000) !important;
    }
    
    /* Special styling for CSV export buttons */
    div[data-testid="column"] .stButton > button[title*="CSV"] {
        background: linear-gradient(45deg, #28a745, #1e7e34) !important;
        color: white !important;
    }
    
    div[data-testid="column"] .stButton > button[title*="CSV"]:hover {
        background: linear-gradient(45deg, #1e7e34, #155724) !important;
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
    st.session_state.api_base_url = "http://localhost:8000"  # Main app with OCR integrated
if 'ocr_mode' not in st.session_state:
    st.session_state.ocr_mode = "local"  # local or api
if 'csv_filename' not in st.session_state:
    st.session_state.csv_filename = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

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
    col_config1, col_config2 = st.columns(2)

    with col_config1:
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_base_url,
            help="Base URL for the keyframe search API"
        )
        if api_url != st.session_state.api_base_url:
            st.session_state.api_base_url = api_url

    with col_config2:
        csv_filename = st.text_input(
            "CSV Export Filename",
            value=st.session_state.csv_filename,
            help="Filename for exporting search results to CSV"
        )
        if csv_filename != st.session_state.csv_filename:
            st.session_state.csv_filename = csv_filename

    # Debug section
    if st.button("🔧 Debug Paths", help="Check file system paths for troubleshooting"):
        debug_paths()

# Main search interface
st.markdown("### 🔍 Search Method")
search_tab1, search_tab2, search_tab3 = st.tabs(["📝 Text Search", "🖼️ Image Search", "📄 OCR Search"])

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
                              max_value=200, value=200, key="text_top_k")
        with col_param2:
            score_threshold = st.slider(
                "🎯 Min Score", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="text_threshold")

    with col2:
        # Search mode selector
        st.markdown("### 🎛️ Search Mode")
        search_mode = st.selectbox(
            "Mode",
            options=["Default", "Exclude Groups",
                     "Include Groups & Videos", "Include Video"],
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

    elif search_mode == "Include Video":
        st.markdown("### 🎯 Include Video Selection")
        st.markdown(
            "**Perfect for progressive multi-stage queries!** Use results from previous search to narrow down the scope.")

        # Video scope input
        video_scope_input = st.text_area(
            "🎬 Video Names to Search Within",
            placeholder="Enter video names, one per line or comma-separated:\nL21_V026\nL22_V110\nL23_V045\n\nOr: L21_V026, L22_V110, L23_V045",
            help="Enter video names like L21_V026. You can copy these from 'Video Scope Extraction' section below after getting search results.",
            height=100
        )

        # Parse video scope
        video_scope_list = []
        if video_scope_input.strip():
            # Split by both newlines and commas, then clean up
            raw_videos = []
            for line in video_scope_input.strip().split('\n'):
                if ',' in line:
                    raw_videos.extend([v.strip() for v in line.split(',')])
                else:
                    raw_videos.append(line.strip())

            # Filter and validate video names
            for video in raw_videos:
                video = video.strip()
                if video and (video.startswith('L') and '_V' in video):
                    video_scope_list.append(video)

            if video_scope_list:
                st.success(
                    f"✅ Will search within {len(video_scope_list)} videos: {', '.join(video_scope_list[:5])}")
                if len(video_scope_list) > 5:
                    with st.expander(f"Show all {len(video_scope_list)} videos"):
                        st.write(video_scope_list)
            else:
                st.warning(
                    "⚠️ No valid video names found. Use format like: L21_V026")

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

# OCR SEARCH TAB
with search_tab3:
    col1, col2 = st.columns([2, 1])

    with col1:
        # OCR Search query
        ocr_query = st.text_input(
            "📄 OCR Search Query",
            placeholder="Enter text to search in OCR results (e.g., 'giây', 'HD', 'time')",
            help="Search for text that appears in video keyframes using OCR data",
            key="ocr_query"
        )

        # OCR Search parameters
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            ocr_top_k = st.slider("📊 Max Results", min_value=1,
                                  max_value=500, value=50, key="ocr_top_k")
        with col_param2:
            # Search mode selection (AND/OR)
            ocr_search_mode = st.selectbox(
                "🔍 Search Mode",
                options=["Simple", "AND", "OR", "Phrase"],
                help="Choose how to combine multiple search terms",
                key="ocr_search_mode"
            )
        
        # Video filters for OCR
        st.markdown("**🎬 Video Filters (Optional)**")
        ocr_video_filters = st.text_input(
            "Video IDs to filter",
            placeholder="Enter video IDs separated by commas (e.g., L01_V001, L02_V005)",
            help="Limit search to specific videos only",
            key="ocr_video_filters"
        )

    with col2:
        st.markdown("### 📄 OCR Search Info")
        
        # OCR Mode Selector
        st.markdown("**⚙️ OCR Search Mode**")
        ocr_mode = st.radio(
            "Choose OCR search method:",
            options=["local", "api"],
            format_func=lambda x: {
                "local": "🏠 Local Search (Direct DB)",
                "api": "🌐 API Search (HTTP)"
            }[x],
            help="Local: Direct database access, faster but requires database file. API: Uses OCR server.",
            key="ocr_mode_selector"
        )
        
        # Update session state
        if ocr_mode != st.session_state.ocr_mode:
            st.session_state.ocr_mode = ocr_mode
        
        # Show current mode status
        if st.session_state.ocr_mode == "local":
            st.success("🏠 Using local OCR database")
        else:
            st.info(f"🌐 Using API: `{st.session_state.api_base_url}`")
        
        st.info("""
        **How it works:**
        - Search through text detected in video keyframes
        - Uses OCR (Optical Character Recognition) data
        
        **Search Modes:**
        - **Simple**: Basic text search
        - **AND**: All words must appear: `person AND walking`
        - **OR**: Any word can appear: `person OR people`
        - **Phrase**: Exact phrase search: `"exact phrase"`
        """)

# Metadata Filter Section (Independent)
st.markdown("---")
st.markdown("### 🏷️ Metadata Filters")

# Load available keywords once at the top
available_keywords = load_available_keywords()

with st.expander("🔍 Metadata Filters", expanded=False):
    st.markdown("**✅ Enable Metadata Filtering**")

    # Enable/disable metadata filtering
    use_metadata_filter = st.checkbox(
        "Apply metadata filters to search results",
        value=False,
        help="When enabled, search results will be filtered by the metadata criteria below"
    )

    if use_metadata_filter:
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

            # Keywords selection method
            keywords_method = st.radio(
                "Select keywords method:",
                ["🔍 Search & Select", "📝 Manual Input"],
                horizontal=True,
                key="keywords_method"
            )

            selected_keywords = []
            if keywords_method == "🔍 Search & Select" and available_keywords:
                # Search keywords
                keyword_search = st.text_input(
                    "🔍 Search keywords:",
                    placeholder="Type to filter available keywords",
                    key="keyword_search"
                )

                # Filter keywords based on search
                if keyword_search:
                    filtered_keywords = [
                        k for k in available_keywords if keyword_search.lower() in k.lower()]
                    # Limit to 20 for performance
                    display_keywords = filtered_keywords[:20]
                else:
                    display_keywords = available_keywords  # Show all by default

                selected_keywords = st.multiselect(
                    "Select keywords:",
                    options=display_keywords,
                    help="Choose from detected keywords in metadata",
                    key="keywords_multiselect"
                )

            else:
                # Manual input fallback
                keywords_input = st.text_input(
                    "Manual keyword input",
                    placeholder="e.g., tin tuc, HTV, 60 giay",
                    help="Enter keywords separated by commas",
                    key="keywords_manual"
                )
                if keywords_input.strip():
                    selected_keywords = [
                        k.strip() for k in keywords_input.split(',') if k.strip()]

            # Keywords mode
            if selected_keywords:
                keywords_mode = st.radio(
                    "Keywords matching mode:",
                    ["any", "all"],
                    help="'any': match videos with at least one keyword, 'all': match videos with all keywords",
                    key="keywords_mode",
                    horizontal=True
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

            # Title filter with mode
            title_contains = st.text_input(
                "Title contains",
                placeholder="e.g., 60 Giây, tin tức",
                help="Enter search terms separated by commas for multiple term search",
                key="title_filter"
            )

            # Always show title mode
            title_mode = st.radio(
                "Title matching mode:",
                ["any", "all"],
                help="'any': match titles with at least one term, 'all': match titles with all terms",
                key="title_mode",
                horizontal=True
            )

            # Description filter with mode
            description_contains = st.text_input(
                "Description contains",
                placeholder="Search in descriptions",
                help="Enter search terms separated by commas for multiple term search",
                key="desc_filter"
            )

            # Always show description mode
            description_mode = st.radio(
                "Description matching mode:",
                ["any", "all"],
                help="'any': match descriptions with at least one term, 'all': match descriptions with all terms",
                key="description_mode",
                horizontal=True
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

# Parse metadata filters
metadata_filter = {}
if use_metadata_filter:
    if authors_input.strip():
        metadata_filter["authors"] = [x.strip()
                                      for x in authors_input.split(',') if x.strip()]

    # Keywords with mode support - get from session state
    final_keywords = []
    if 'keywords_multiselect' in st.session_state and st.session_state.keywords_multiselect:
        final_keywords = st.session_state.keywords_multiselect
    elif 'keywords_manual' in st.session_state and st.session_state.keywords_manual.strip():
        final_keywords = [
            k.strip() for k in st.session_state.keywords_manual.split(',') if k.strip()]

    if final_keywords:
        metadata_filter["keywords"] = final_keywords
        # Only add mode if keywords are selected and mode exists
        if len(final_keywords) > 1 and 'keywords_mode' in st.session_state:
            metadata_filter["keywords_mode"] = st.session_state.keywords_mode

    if min_length > 0:
        metadata_filter["min_length"] = min_length

    if max_length > 0:
        metadata_filter["max_length"] = max_length

    # Title with mode support
    if title_contains.strip():
        # Always send both formats for maximum compatibility
        # Original format
        metadata_filter["title_contains"] = title_contains.strip()

        # Also send as terms array
        if ',' in title_contains:
            title_terms = [x.strip()
                           for x in title_contains.split(',') if x.strip()]
        else:
            title_terms = [title_contains.strip()]

        metadata_filter["title_terms"] = title_terms
        if 'title_mode' in st.session_state:
            metadata_filter["title_mode"] = st.session_state.title_mode
        else:
            metadata_filter["title_mode"] = "any"

    # Description with mode support
    if description_contains.strip():
        # Always send both formats for maximum compatibility
        # Original format
        metadata_filter["description_contains"] = description_contains.strip()

        # Also send as terms array
        if ',' in description_contains:
            description_terms = [
                x.strip() for x in description_contains.split(',') if x.strip()]
        else:
            description_terms = [description_contains.strip()]

        metadata_filter["description_terms"] = description_terms
        if 'description_mode' in st.session_state:
            metadata_filter["description_mode"] = st.session_state.description_mode
        else:
            metadata_filter["description_mode"] = "any"

    if date_from is not None:
        # Convert date to DD/MM/YYYY format
        metadata_filter["date_from"] = date_from.strftime("%d/%m/%Y")

    if date_to is not None:
        # Convert date to DD/MM/YYYY format
        metadata_filter["date_to"] = date_to.strftime("%d/%m/%Y")
else:
    # Initialize default values when metadata filter is not enabled
    use_metadata_filter = False
    authors_input = ""
    selected_keywords = []
    keywords_method = "🔍 Search & Select"
    keywords_mode = "any"
    min_length = 0
    max_length = 0
    title_contains = ""
    title_mode = "any"
    description_contains = ""
    description_mode = "any"
    date_from = None
    date_to = None

# Rerank Options Section
st.markdown("### ⚡ Rerank Options")

# Initialize rerank variables with defaults
enable_rerank = False
rerank_superglobal_enabled = False
rerank_final_top_k = 10
sg_top_t = 50


with st.expander("🎯 Multi-Stage Reranking", expanded=False):
    st.markdown(
        "**✅ Enable Multi-Stage Reranking**")

    # Enable/disable reranking
    enable_rerank = st.checkbox(
        "Apply Multi-Stage Reranking",
        value=False,
        help="Apply advanced reranking pipeline (SuperGlobal) to improve result quality"
    )

    if enable_rerank:
        col_rerank1, col_rerank2 = st.columns(2)

        with col_rerank1:
            st.markdown("**🎯 SuperGlobal Reranking**")
            rerank_superglobal_enabled = st.checkbox(
                "Enable SuperGlobal rerank",
                value=True,
                help="Fast reranking using global features"
            )

            if rerank_superglobal_enabled:
                rerank_superglobal_weight = st.slider(
                    "SuperGlobal weight",
                    min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                    help="Weight for SuperGlobal similarity scores"
                )
                sg_top_t = st.slider(
                    "SuperGlobal top_t",
                    min_value=50, max_value=500, value=200, step=25,
                    help="Number of candidates for SuperGlobal reranking"
                )

            st.markdown("**⚙️ Advanced Settings**")
            rerank_cache_enabled = st.checkbox(
                "Enable result caching",
                value=True,
                help="Cache reranking results for faster repeated queries"
            )

            rerank_fallback_enabled = st.checkbox(
                "Enable graceful fallback",
                value=True,
                help="Fallback to simpler methods if advanced reranking fails"
            )

            st.markdown("**📊 Final Results**")
            rerank_final_top_k = st.slider(
                "Final top_k results",
                min_value=5, max_value=100, value=100, step=5,
                help="Final number of results after reranking"
            )

        # Show rerank configuration summary
        if any([rerank_superglobal_enabled]):
            st.markdown("---")
            st.markdown("**🎯 Rerank Pipeline Summary:**")

            stages = []
            if rerank_superglobal_enabled:
                stages.append(
                    f"SuperGlobal (top_t: {sg_top_t})")

            for i, stage in enumerate(stages, 1):
                st.info(f"**Stage {i}:** {stage}")

# Object Filter Section
st.markdown("### 🎯 Object Filters")

with st.expander("🔍 Object Detection Filters", expanded=False):
    # Enable/disable object filtering
    st.markdown("**✅ Enable Object Filtering**")
    use_object_filter = st.checkbox(
        "Apply object filters to search results",
        value=False,
        help="When enabled, search results will be filtered by detected objects below"
    )

    if use_object_filter:
        col_obj1, col_obj2 = st.columns([2, 1])

        with col_obj1:
            # Load available objects from JSON file
            objects_data = load_available_objects()
            available_objects = objects_data['objects']
            object_categories = objects_data['categories']
            objects_metadata = objects_data['metadata']

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
        selected_objects = st.session_state.selected_objects_list[:20] if hasattr(
            st.session_state, 'selected_objects_list') else []

        with col_obj2:
            # Filter mode
            st.markdown("**⚙️ Filter Mode**")
            object_filter_mode = st.radio(
                "Mode:",
                ["any", "all"],
                help="'any': keyframe contains at least one object, 'all': keyframe contains all objects",
                key="object_mode"
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
                    st.metric("🎯 Detected", detected_count)
                with col_info2:
                    st.metric("✨ Custom", custom_count)
                with col_info3:
                    st.metric("📊 Total", len(selected_objects))
            else:
                selected_objects = []

# Parse object filters
object_filter = {}
if use_object_filter and 'selected_objects' in locals() and selected_objects:
    object_filter = {
        "objects": selected_objects,
        "mode": object_filter_mode if 'object_filter_mode' in locals() else "any"
    }

# Toggle for new engine (QExp)
enable_qexp = st.checkbox(
    "**🚀 Enable Advanced Query Enhancement**", value=False,
    help="Use multi-variant fusion and object-aware boost"
)

# Search button and logic
col_search1, col_search2, col_search3 = st.columns(3)

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
                                filter_parts.append(
                                    f"groups: {include_groups}")
                            if include_videos:
                                filter_parts.append(
                                    f"videos: {include_videos}")
                            st.info(f"✅ Including {', '.join(filter_parts)}")
                    elif search_mode == "Include Video" and 'video_scope_list' in locals() and video_scope_list:
                        st.info(
                            f"🎯 Searching in {len(video_scope_list)} videos: {', '.join(video_scope_list[:5])}{'...' if len(video_scope_list) > 5 else ''}")

                    # Determine endpoint and base payload based on search mode
                    if search_mode == "Default":
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                        payload = {
                            "query": query,
                            "top_k": current_top_k,
                            "score_threshold": current_threshold
                        }
                        if enable_qexp:
                            payload["qexp_enable"] = True

                    elif search_mode == "Exclude Groups":
                        if not exclude_groups:
                            st.warning(
                                "⚠️ No groups to exclude specified. Using default search.")
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold
                            }
                            if enable_qexp:
                                payload["qexp_enable"] = True
                        else:
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold,
                                "exclude_groups": exclude_groups
                            }
                            if enable_qexp:
                                payload["qexp_enable"] = True

                    elif search_mode == "Include Groups & Videos":
                        if not include_groups and not include_videos:
                            st.warning(
                                "⚠️ No groups or videos to include specified. Using default search.")
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold
                            }
                            if enable_qexp:
                                payload["qexp_enable"] = True
                        else:
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold,
                                "include_groups": include_groups,
                                "include_videos": include_videos
                            }
                            if enable_qexp:
                                payload["qexp_enable"] = True

                    elif search_mode == "Include Video":
                        if 'video_scope_list' not in locals() or not video_scope_list:
                            st.warning(
                                "⚠️ No video names specified. Using default search.")
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold
                            }
                            if enable_qexp:
                                payload["qexp_enable"] = True
                        else:
                            endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/video-names"
                            payload = {
                                "query": query,
                                "top_k": current_top_k,
                                "score_threshold": current_threshold,
                                "video_names": video_scope_list
                            }
                            if enable_qexp:
                                payload["qexp_enable"] = True

                    # Add rerank parameters if enabled
                    params = None
                    if enable_rerank:
                        # Use advanced search endpoint for reranking
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/advanced"

                        # Build query parameters for GET request
                        params = {
                            "q": query,
                            "top_k": rerank_final_top_k,
                            "score_threshold": current_threshold,
                            "rerank": 1,
                            "rerank_mode": "custom"
                        }

                        # Add individual rerank methods
                        if rerank_superglobal_enabled:
                            params["rr_superglobal"] = 1
                            params["sg_top_m"] = sg_top_t
                        else:
                            params["rr_superglobal"] = 0

                        # Attach filters so backend can filter then rerank
                        filter_info = []

                        # Search mode filters
                        if search_mode == "Exclude Groups" and exclude_groups:
                            params["exclude_groups"] = ",".join(
                                str(x) for x in exclude_groups)
                            filter_info.append(
                                f"exclude groups: {exclude_groups}")
                        elif search_mode == "Include Groups & Videos":
                            if include_groups:
                                params["include_groups"] = ",".join(
                                    str(x) for x in include_groups)
                                filter_info.append(
                                    f"include groups: {include_groups}")
                            if include_videos:
                                params["include_videos"] = ",".join(
                                    str(x) for x in include_videos)
                                filter_info.append(
                                    f"include videos: {include_videos}")
                        elif search_mode == "Include Video" and 'video_scope_list' in locals() and video_scope_list:
                            # Convert video names like 'L21_V026' to integers for include_videos
                            try:
                                video_ids = []
                                for vn in video_scope_list:
                                    if isinstance(vn, str) and "_V" in vn:
                                        part = vn.split("_V", 1)[1]
                                        video_ids.append(int(part))
                                if video_ids:
                                    params["include_videos"] = ",".join(
                                        str(x) for x in video_ids)
                                    filter_info.append(
                                        f"include videos: {video_ids}")
                            except Exception:
                                pass

                        # Metadata/Object filters
                        if use_metadata_filter and metadata_filter:
                            try:
                                params["metadata_filter"] = json.dumps(
                                    metadata_filter)
                                filter_info.append(
                                    f"metadata: {list(metadata_filter.keys())}")
                            except Exception:
                                pass
                        if use_object_filter and object_filter:
                            try:
                                params["object_filter"] = json.dumps(
                                    object_filter)
                                objects_str = ", ".join(
                                    object_filter.get("objects", [])[:3])
                                if len(object_filter.get("objects", [])) > 3:
                                    objects_str += f" (+{len(object_filter['objects'])-3} more)"
                                filter_info.append(
                                    f"objects[{object_filter.get('mode','any')}]: {objects_str}")
                            except Exception:
                                pass

                        # Show rerank info
                        rerank_methods = []
                        if params.get("rr_superglobal"):
                            rerank_methods.append("SuperGlobal")
                        # SG only

                        st.info(
                            f"⚡ Reranking enabled: {' + '.join(rerank_methods)} → {params['top_k']} results")

                        # For rerank, we don't use metadata/object filters (advanced endpoint doesn't support them)
                        # if False and ((use_metadata_filter and metadata_filter) or (use_object_filter and object_filter)):
                        #     st.warning(
                        #         "⚠️ Metadata and object filters are not supported with reranking. Using rerank only.")

                    # Show Query Expansion status
                    if enable_qexp:
                        st.info(
                            "🚀 Query Expansion enabled: Multi-variant fusion and object-aware boost")

                    # If not using reranking but have metadata/object filters, use metadata-filter endpoint
                    elif (use_metadata_filter and metadata_filter) or (use_object_filter and object_filter):
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/metadata-filter"

                        filter_info = []

                        # Add search mode filters to the payload
                        if search_mode == "Exclude Groups" and exclude_groups:
                            payload["exclude_groups"] = exclude_groups
                            filter_info.append(
                                f"exclude groups: {exclude_groups}")
                        elif search_mode == "Include Groups & Videos":
                            if include_groups:
                                payload["include_groups"] = include_groups
                                filter_info.append(
                                    f"include groups: {include_groups}")
                            if include_videos:
                                payload["include_videos"] = include_videos
                                filter_info.append(
                                    f"include videos: {include_videos}")

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

                        # Enable QExp per request in metadata-filter flow
                        if enable_qexp:
                            payload["qexp_enable"] = True

                        st.info(
                            f"🎯 Applying filters: {' | '.join(filter_info)}")

                    # Show the final endpoint being used
                    endpoint_display = endpoint.split(
                        '/')[-1]  # Just show the last part
                    st.info(f"🔗 Using endpoint: {endpoint_display}")

                    # Use GET for rerank with params, POST for regular search with JSON payload
                    if params is not None:
                        # GET request for rerank endpoint
                        response = requests.get(
                            endpoint,
                            params=params
                        )
                    else:
                        # POST request for regular search endpoints
                        response = requests.post(
                            endpoint,
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        )

                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.search_results = results
                        st.session_state.search_query = query
                        # Store rerank settings for later use in metadata display
                        st.session_state.last_search_rerank_enabled = enable_rerank
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
                        # Store rerank settings for image search (no rerank for image search)
                        st.session_state.last_search_rerank_enabled = False
                        st.rerun()
                    else:
                        st.error(
                            f"Image search failed: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with col_search3:
    if st.button("📄 OCR Search", use_container_width=True):
        if not ocr_query.strip():
            st.error("Please enter an OCR search query")
        elif len(ocr_query) > 1000:
            st.error("Query too long. Please keep it under 1000 characters.")
        else:
            with st.spinner("🔍 Searching through OCR text..."):
                try:
                    # Parse video filters
                    video_filter_list = []
                    if ocr_video_filters.strip():
                        video_filter_list = [v.strip() for v in ocr_video_filters.split(',') if v.strip()]
                    
                    # Build query based on search mode
                    processed_query = ocr_query
                    if ocr_search_mode == "AND" and " " in ocr_query:
                        words = ocr_query.split()
                        processed_query = " AND ".join(words)
                    elif ocr_search_mode == "OR" and " " in ocr_query:
                        words = ocr_query.split()
                        processed_query = " OR ".join(words)
                    elif ocr_search_mode == "Phrase":
                        processed_query = f'"{ocr_query}"'
                    
                    # Choose search method based on mode
                    if st.session_state.ocr_mode == "local":
                        # LOCAL OCR SEARCH
                        from gui_ocr_search import GuiOCRSearch
                        
                        ocr_service = GuiOCRSearch()
                        ocr_results = ocr_service.search_text(
                            query=processed_query,
                            limit=ocr_top_k,
                            min_confidence=0.0,
                            video_filters=video_filter_list if video_filter_list else None
                        )
                        
                    else:
                        # API OCR SEARCH
                        import requests
                        
                        payload = {
                            "query": processed_query,
                            "limit": ocr_top_k,
                            "min_confidence": 0.0,
                            "video_filters": video_filter_list if video_filter_list else None,
                            "include_ocr_details": True,
                            "include_video_metadata": True  # Add this to get video URLs
                        }
                        
                        endpoint = f"{st.session_state.api_base_url}/api/v1/ocr/search"
                        
                        response = requests.post(
                            endpoint,
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if response.status_code == 200:
                            ocr_results = response.json()
                        else:
                            st.error(f"API search failed: {response.status_code} - {response.text}")
                            ocr_results = None
                    
                    # Process results if successful
                    if ocr_results and ocr_results.get('results'):
                        # Convert OCR results to keyframe search format for compatibility
                        converted_results = []
                        for ocr_result in ocr_results['results']:
                            video_id = ocr_result['video_id']
                            frame_number = str(ocr_result['frame_number'])
                            
                            # Parse group from video_id (e.g., L01_V001 -> L01)
                            group = video_id.split('_')[0] if '_' in video_id else video_id[:3]
                            
                            # Path with prefix for GUI folder
                            keyframe_path = f"{path_prefix}resources/keyframes/{group}/{video_id}/{frame_number.zfill(3)}.jpg"
                            
                            converted_result = {
                                "path": keyframe_path,
                                "score": ocr_result.get('confidence', 1.0),
                                "video_id": video_id,
                                "frame_id": ocr_result['frame_id'],
                                "ocr_text": ocr_result['ocr_text'],
                                "ocr_details": ocr_result.get('ocr_details', []),
                                # Add video metadata if available
                                "watch_url": ocr_result.get('watch_url'),
                                "title": ocr_result.get('title'),
                                "author": ocr_result.get('author'),
                                "description": ocr_result.get('description'),
                                "thumbnail_url": ocr_result.get('thumbnail_url'),
                                "publish_date": ocr_result.get('publish_date'),
                                "length": ocr_result.get('length')
                            }
                            converted_results.append(converted_result)
                        
                        # Store in session state
                        st.session_state.search_results = converted_results
                        st.session_state.search_query = f"OCR ({st.session_state.ocr_mode}): {ocr_query}"
                        st.session_state.last_search_rerank_enabled = False
                        
                        st.success(f"Found {len(converted_results)} OCR results using {st.session_state.ocr_mode} search")
                        st.rerun()
                    else:
                        st.warning("No OCR results found")
                        
                except Exception as e:
                    st.error(f"OCR search error: {str(e)}")
                    print(f"OCR search failed: {e}")  # Use print instead of logger

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

    # Video Scope Extraction for Multi-Stage Queries
    if results_list:
        with st.expander("🎯 Video Scope Extraction", expanded=False):
            st.markdown("### Extract Video Names for Next Stage Query")
            st.markdown(
                "Copy the video names below to use in Include Video mode for the next search stage:")

            # Extract unique video names from results
            extracted_videos = extract_video_names_from_results(results_list)

            if extracted_videos:
                # Display as comma-separated list for easy copying
                video_names_text = ", ".join(extracted_videos)
                st.text_area(
                    f"📁 {len(extracted_videos)} unique videos found:",
                    value=video_names_text,
                    height=100,
                    help="Copy this text and paste it into the Include Video mode for your next search stage"
                )

                # Also show as a formatted list
                st.markdown("**Video List:**")
                cols = st.columns(3)
                for i, video_name in enumerate(extracted_videos):
                    with cols[i % 3]:
                        st.code(video_name, language=None)
            else:
                st.warning(
                    "No video names could be extracted from the results.")

    # CSV Export Configuration (before results)
    with st.expander("💾 CSV Export Settings", expanded=False):
        col_csv1, col_csv2 = st.columns([2, 1])

        with col_csv1:
            new_filename = st.text_input(
                "📁 CSV Filename",
                value=st.session_state.csv_filename,
                help="Filename for exporting search results. Will automatically add .csv extension if not present."
            )
            if new_filename != st.session_state.csv_filename:
                st.session_state.csv_filename = new_filename

        with col_csv2:
            st.markdown("**📊 Export Options**")
            st.markdown("• Individual: Use 'Add to CSV' buttons")
            st.markdown("• Bulk: Use 'Export All to CSV' button")
            st.markdown("• Includes: Frame index, PTS time, metadata")

    # Results summary
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

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

    with col_metric4:
        # Export all results button
        if st.button("📤 Export All to CSV", help=f"Export all {len(results_list)} results to {st.session_state.csv_filename}", use_container_width=True):
            if results_list:
                success, count = export_all_results_to_csv(
                    st.session_state.csv_filename, results_list)
                if success:
                    st.success(
                        f"✅ Exported {count} results to {st.session_state.csv_filename}")
                else:
                    st.error("❌ Failed to export results")
            else:
                st.warning("⚠️ No results to export")

    # Sort by score (highest first)
    sorted_results = []
    if results_list:
        sorted_results = sorted(
            results_list, key=lambda x: x['score'], reverse=True)

    # Display results in a grid
    for i, result in enumerate(sorted_results):
        # Extract frame info for each result
        video_id, frame_idx = get_frame_info_from_keyframe_path(
            result.get('path', ''))
        pts_time = None
        if video_id and frame_idx is not None:
            pts_time = get_pts_time_from_n(video_id, frame_idx)

        # Add extracted info to result for display and CSV export
        result['frame_idx'] = frame_idx  # n (filename number like 1, 2, 3...)
        result['pts_time'] = pts_time

        # Get real frame_idx for CSV export
        if video_id and frame_idx is not None:
            real_frame_idx = get_real_frame_idx_from_n(video_id, frame_idx)
            result['real_frame_idx'] = real_frame_idx
        else:
            result['real_frame_idx'] = None

        with st.container():
            col_img, col_info = st.columns([1, 3])

            with col_img:
                try:
                    # Enhanced button layout
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

                    # New buttons row for Video and CSV export
                    col_btn3, col_btn4 = st.columns(2)

                    with col_btn3:
                        # MPC-HC button (only show if we have pts_time)
                        if pts_time is not None:
                            if st.button(f"🎥 MPC-HC", key=f"mpc_{i}", help=f"Open Video in MPC-HC at {pts_time:.1f}s", use_container_width=True):
                                show_mpc_video(pts_time, result, i)
                        else:
                            st.button(
                                f"🎥 MPC-HC", key=f"mpc_{i}", help="pts_time not available", disabled=True, use_container_width=True)

                    with col_btn4:
                        # Append to CSV button
                        if st.button(f"💾 Add to CSV", key=f"csv_{i}", help=f"Add this result to {st.session_state.csv_filename}", use_container_width=True):
                            if append_to_csv(st.session_state.csv_filename, result):
                                st.success(
                                    f"✅ Added to {st.session_state.csv_filename}")
                            else:
                                st.error("❌ Failed to add to CSV")

                    # Show thumbnail image
                    try:
                        # Adjust path based on working directory
                        image_path = result['path']
                        if path_prefix and not os.path.isabs(image_path):
                            adjusted_path = f"{path_prefix}{image_path}"
                        else:
                            adjusted_path = image_path
                        
                        # Use adjusted path
                        st.image(adjusted_path, width=300, caption=f"Keyframe {i+1}")
                    except Exception as img_err:
                        # If adjusted path fails, try original path
                        try:
                            st.image(result['path'], width=300, caption=f"Keyframe {i+1}")
                        except Exception as e2:
                            st.error(f"❌ Image load failed")
                            st.text(f"Tried paths:")
                            st.text(f"  1. {path_prefix}{result['path']} (adjusted)")
                            st.text(f"  2. {result['path']} (original)")
                            st.text(f"Working dir: {os.getcwd()}")

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

                # NEW: Show frame information (both filename number and real frame_idx)
                if frame_idx is not None:
                    real_frame_idx = result.get('real_frame_idx')
                    if real_frame_idx is not None:
                        metadata_parts.append(
                            f"<strong>🎬 Frame:</strong> File #{frame_idx} (Video Frame #{real_frame_idx})")
                    else:
                        metadata_parts.append(
                            f"<strong>🎬 Frame:</strong> File #{frame_idx} (Video Frame: Unknown)")
                else:
                    metadata_parts.append(
                        f"<strong>🎬 Frame:</strong> <em>Unable to parse</em>")

                # NEW: Show pts_time information
                pts_minutes = int(
                    pts_time // 60) if pts_time is not None else None
                pts_seconds = int(pts_time %
                                  60) if pts_time is not None else None
                if pts_time is not None:
                    metadata_parts.append(
                        f"<strong>⏱️ PTS Time:</strong> {pts_minutes}:{pts_seconds:02d} ({pts_time:.1f}s)")
                else:
                    metadata_parts.append(
                        f"<strong>⏱️ PTS Time:</strong> <em>Not available</em>")

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

                # Show Video URL if available
                if 'watch_url' in result and result['watch_url']:
                    watch_url = result['watch_url']
                    # Shorten URL for display
                    if len(watch_url) > 50:
                        display_url = watch_url[:47] + "..."
                    else:
                        display_url = watch_url
                    metadata_parts.append(
                        f"<strong>🔗 Video:</strong> <a href='{watch_url}' target='_blank'>{display_url}</a>")

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

                # NEW: Show OCR text if available (for OCR search results)
                if 'ocr_text' in result and result['ocr_text']:
                    ocr_text = result['ocr_text']
                    # Truncate long OCR text for display
                    if len(ocr_text) > 150:
                        display_ocr = ocr_text[:147] + "..."
                    else:
                        display_ocr = ocr_text
                    
                    metadata_parts.append(
                        f"<strong>📄 OCR Text:</strong> <em>'{display_ocr}'</em>")

                if metadata_parts:
                    metadata_html = f'<div class="metadata-section">{"<br>".join(metadata_parts)}</div>'

                # Format path for display - show only relative part from keyframes/
                display_path = result['path'].replace("/", "\\")
                # if 'keyframes\\' in display_path:
                #     display_path = display_path.split('keyframes\\')[-1]
                # elif 'keyframes/' in display_path:
                #     display_path = display_path.split('keyframes/')[-1]

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

    # Temporal Search (beta)
    if sorted_results:
        with st.expander("⏱️ Temporal Search (beta)", expanded=False):
            enable_temporal = st.checkbox("Enable Temporal Search", value=False)
            if enable_temporal:
                mode = st.radio("Mode", options=["Auto", "Interactive"], horizontal=True)

                # Choose pivot result
                pivot_idx = 0
                if mode == "Interactive":
                    pivot_idx = st.number_input(
                        "Pivot result index (0-based)", min_value=0, max_value=len(sorted_results)-1, value=0, step=1
                    )
                pivot = sorted_results[pivot_idx]

                # Extract video_id (e.g., L21_V026) and n from path
                vid_from_path = extract_video_name_from_path(pivot.get('path', ''))
                n_from_path = extract_frame_number_from_path(pivot.get('path', ''))
                group_id = pivot.get('group_id')
                video_num = pivot.get('video_id')  # numeric video id in this UI

                # Build video_id robustly
                if vid_from_path:
                    pivot_video_id = vid_from_path
                elif group_id is not None and video_num is not None:
                    pivot_video_id = f"L{int(group_id):02d}_V{int(video_num):03d}"
                else:
                    pivot_video_id = ""

                pivot_pts_time = pivot.get('pts_time')
                # Prepare pivot fields correctly
                pivot_n = int(n_from_path) if n_from_path else None
                pivot_frame_idx = None
                # If we don't have n, try using real_frame_idx (preferred) or frame_idx as absolute frame index
                if pivot_n is None:
                    if pivot.get('real_frame_idx') is not None:
                        try:
                            pivot_frame_idx = int(pivot.get('real_frame_idx'))
                        except Exception:
                            pivot_frame_idx = None
                    elif pivot.get('frame_idx') is not None:
                        try:
                            pivot_frame_idx = int(pivot.get('frame_idx'))
                        except Exception:
                            pivot_frame_idx = None

                # Interactive delta
                delta = 5.0
                if mode == "Interactive":
                    delta = float(st.slider("±Δ seconds", min_value=3.0, max_value=20.0, value=5.0, step=0.5))

                # Call backend to get clusters
                pivot_pts_time_val = float(pivot_pts_time) if pivot_pts_time is not None else None
                pivot_score_val = float(pivot.get('score', 0.0)) if pivot.get('score') is not None else None

                payload = {
                    "mode": "auto" if mode == "Auto" else "interactive",
                    "pivot_video_id": pivot_video_id,
                    "pivot_n": pivot_n,
                    "pivot_frame_idx": pivot_frame_idx,
                    "pivot_pts_time": pivot_pts_time_val,
                    "pivot_score": pivot_score_val,
                    "delta": delta,
                }
                try:
                    api_base = st.session_state.get('api_base_url', 'http://localhost:8000')
                    resp = requests.post(f"{api_base}/api/v1/keyframe/temporal/enrich", json=payload, timeout=20)
                    if resp.status_code == 200:
                        tview = resp.json()

                        # Render clusters
                        t_text = f"{pivot_pts_time_val:.2f}s" if pivot_pts_time_val is not None else "?"
                        st.markdown(f"Pivot: `{pivot_video_id}`, n={pivot_n}, t={t_text}")
                        # Try to infer keyframes root from pivot path
                        sample_path = pivot.get('path', '')
                        def build_img_path(video_id: str, n: int, sample_path: str) -> str:
                            # Prefer replacing the tail Lxx/Lxx_Vyyy/000.jpg
                            sp = sample_path.replace('\\', '/').split('/keyframes/')
                            if len(sp) >= 2:
                                root = sp[0] + '/keyframes'
                                import re
                                m = re.match(r"L(\d+)_V(\d+)", video_id)
                                if m:
                                    g = int(m.group(1)); v = int(m.group(2))
                                    return f"{root}/L{g:02d}/L{g:02d}_V{v:03d}/{int(n):03d}.jpg"
                            return sample_path  # fallback: show pivot only

                        for c in tview.get('clusters', []):
                            st.markdown(f"**{c['video_id']}** — {c['start_time']:.2f}s → {c['end_time']:.2f}s")
                            thumbs = []
                            cols = st.columns(min(6, max(1, len(c.get('keyframes', [])))))
                            for idx, kf in enumerate(c.get('keyframes', [])):
                                with cols[idx % len(cols)]:
                                    p = build_img_path(c['video_id'], kf.get('n', 0), sample_path)
                                    try:
                                        st.image(p, caption=f"n={kf.get('n','?')} t={kf['pts_time']:.1f}s", use_container_width=True)
                                    except Exception:
                                        st.caption(f"n={kf.get('n','?')} t={kf['pts_time']:.1f}s (image path unresolved)")
                    else:
                        st.warning(f"Temporal enrich failed: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.warning(f"Temporal enrich error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
