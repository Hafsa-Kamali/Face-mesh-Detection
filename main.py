import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import gc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specs
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
contour_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

# RTC Configuration for more stable connection
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class FaceMeshTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize face mesh with lower resource settings
        self.face_mesh = None
        self.visualization_mode = "contours"  # Default to contours (less resource intensive)
        self.error_count = 0
        self.max_errors = 3
        self.initialize_face_mesh()
        
    def initialize_face_mesh(self):
        # Release previous instance if exists
        if self.face_mesh:
            self.face_mesh.close()
        
        # Create new instance with optimized settings
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=False  # Disable refined landmarks to save resources
        )
        
    def set_mode(self, mode):
        self.visualization_mode = mode
        
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            # Flip the image horizontally for a selfie-view display
            img = cv2.flip(img, 1)
            
            # Convert the image to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = img.shape
            
            # Process the image
            results = self.face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                self.error_count = 0  # Reset error count on successful detection
                
                for face_landmarks in results.multi_face_landmarks:
                    if self.visualization_mode == "mesh":
                        # Draw the face mesh
                        mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    elif self.visualization_mode == "contours":
                        # Draw face contours
                        mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    elif self.visualization_mode == "points":
                        # Draw only points
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * img_w)
                            y = int(landmark.y * img_h)
                            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                    elif self.visualization_mode == "iris":
                        # Draw iris
                        mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )
                    elif self.visualization_mode == "glow":
                        # Create a glow effect
                        overlay = img.copy()
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * img_w)
                            y = int(landmark.y * img_h)
                            cv2.circle(overlay, (x, y), 2, (0, 255, 255), -1)
                        
                        # Apply glow effect
                        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
                        
                        # Add contours with glow
                        mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=contour_spec
                        )
            
            # Add text to show if face is detected or not
            status = "Face Detected" if results.multi_face_landmarks else "No Face Detected"
            cv2.putText(img, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 255, 0) if results.multi_face_landmarks else (0, 0, 255), 2)
            
            return img
            
        except Exception as e:
            # Handle errors gracefully
            self.error_count += 1
            print(f"Error in transform: {str(e)}")
            
            if self.error_count > self.max_errors:
                # Reinitialize face mesh if too many errors
                self.initialize_face_mesh()
                self.error_count = 0
                
            # Return original frame with error message
            if frame is not None:
                img = frame.to_ndarray(format="bgr24")
                cv2.putText(img, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return img
            else:
                # Create blank frame with error message if no frame available
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera error. Please restart.", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return blank
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()


# Streamlit UI setup
st.set_page_config(
    page_title="Futuristic Face Mesh Detection",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Add background image from URL
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: center;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url("https://d1sr9z1pdl3mb7.cloudfront.net/wp-content/uploads/2022/03/07162020/synthetic-data-scaled.jpg")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: #ffffff;
        }
        .stApp {
            background-color: #0d1117;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #00FFFF;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-shadow: 0 0 10px rgba(0,255,255,0.5);
        }
        .stButton>button {
            background-color: #2e59d9;
            color: white;
            border-radius: 20px;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1e3c8a;
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(46, 89, 217, 0.5);
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .stRadio label {
            background-color: #1e1e1e;
            padding: 10px 15px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stRadio label:hover {
            background-color: #2e2e2e;
        }
        .css-1kyxreq {
            justify-content: center;
        }
        .css-1v0mbdj, .css-1cpxqw2 {
            text-align: center;
        }
        .stMarkdown p {
            text-align: center;
        }
        .highlight {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .container {
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: rgba(13, 17, 23, 0.7);
            backdrop-filter: blur(10px);
            padding: 4px;
            text-align: center;
            font-size: 0.6rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .warning {
            color: #FFA500;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            background-color: rgba(255, 165, 0, 0.1);
            border: 1px solid rgba(255, 165, 0, 0.3);
        }
        .debug-info {
            font-size: 0.8rem;
            color: #888;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

# Header with animated title
st.markdown("""
    <div class="container">
        <h1>✨ <span class="highlight">REAL-TIME FACE MESH DETECTION</span> ✨</h1>
        <p>Experience advanced facial landmark tracking powered by MediaPipe and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Session state initialization
if 'transformer_created' not in st.session_state:
    st.session_state['transformer_created'] = False
    st.session_state['restart_counter'] = 0

# Create columns for better layout
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown("### 🎮 Controls")
    
    # Visualization mode selector
    mode = st.radio(
        "Choose Visualization Mode:",
        ["contours", "mesh", "points", "iris", "glow"],
        index=0,  # Default to contours as it's less resource-intensive
        key="visualization_mode"
    )
    
    # Information about the selected mode
    if mode == "mesh":
        st.info("Full face mesh visualization with 468 points")
        st.markdown('<div class="warning">⚠️ This mode uses more resources</div>', unsafe_allow_html=True)
    elif mode == "contours":
        st.info("Face contour lines highlight facial features")
    elif mode == "points":
        st.info("Individual landmark points (468 points)")
    elif mode == "iris":
        st.info("Eye iris tracking")
    elif mode == "glow":
        st.info("Glowing effect with face contours")
        st.markdown('<div class="warning">⚠️ This mode uses more resources</div>', unsafe_allow_html=True)
    
    # Add some usage instructions
    st.markdown("### 📝 Instructions")
    st.markdown("""
        1. Click 'Start' below to begin
        2. Make sure you're in a well-lit environment
        3. Keep your face centered in the frame
        4. Try different visualization modes!
    """)
    
    # Add restart button
    if st.button("🔄 Restart Camera"):
        st.session_state['restart_counter'] += 1
        st.session_state['transformer_created'] = False
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional features section
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown("### 🌟 Features")
    st.markdown("""
        - Real-time facial landmark detection
        - Multiple visualization options
        - Works with any webcam
        - Powered by MediaPipe's advanced ML models
        - Improved stability and error handling
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Troubleshooting section
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Troubleshooting")
    st.markdown("""
        - If detection stops working, click the 'Restart Camera' button
        - Try switching to 'contours' mode for better performance
        - Ensure good lighting for better detection
        - Make sure your browser has camera permissions
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Create a new transformer instance for each restart
    if not st.session_state['transformer_created']:
        # Force garbage collection before creating new instance
        gc.collect()
        
        # Create a new transformer
        transformer = FaceMeshTransformer()
        st.session_state['transformer_created'] = True
        st.session_state['transformer'] = transformer
    else:
        # Use existing transformer
        transformer = st.session_state['transformer']
    
    # Set the mode in the transformer
    transformer.set_mode(mode)
    
    # Webrtc streamer with the transformer and improved configuration
    webrtc_ctx = webrtc_streamer(
        key=f"face-mesh-{st.session_state['restart_counter']}",  # Unique key for each restart
        video_transformer_factory=lambda: transformer,
        rtc_configuration=rtc_configuration,  # Use STUN server for better connectivity
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Status indicator
    if webrtc_ctx.state.playing:
        st.markdown('<div style="text-align: center; margin-top: 10px;">', unsafe_allow_html=True)
        st.success("📹 Camera is active! Face detection is running.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
       st.markdown('<div style="text-align: center; margin-top: 10px;">', unsafe_allow_html=True)
       st.warning("⚠️ Camera is inactive. Click 'Start' to begin face detection.")
       st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance stats
    if webrtc_ctx.state.playing:
        st.markdown('<div class="container">', unsafe_allow_html=True)
        st.markdown("### 📊 Status Information")
        
        # Display current mode
        st.markdown(f"**Current Mode:** {mode}")
        
        # Display restart count
        st.markdown(f"**Restart Count:** {st.session_state['restart_counter']}")
        
        # Add memory management tips
        if st.session_state['restart_counter'] > 3:
            st.markdown('<div class="warning">⚠️ You\'ve restarted several times. Consider refreshing the browser tab for better performance.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>✨ Created with MediaPipe, OpenCV, and Streamlit ✨</p>
        <p class="debug-info">Version 2.0 with improved stability and error handling</p>
    </div>
""", unsafe_allow_html=True)

# Add cache clearing on session end
st.cache_resource.clear()
def clear_resources():
    pass

# Register the app with streamlit to clear cache on session end
st.cache_data.clear()