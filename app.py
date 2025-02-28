# 1. Update imports - don't import PaddleOCR globally
import os 
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
import pandas as pd 
import streamlit as st
from datetime import datetime
import numpy as np
import tempfile
import io
import traceback
import logging
import gc
import warnings
import base64
import hashlib
import json
import os
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import gc
import fitz
import subprocess
import sys
# OCR engines
import easyocr
import tensorflow as tf  # Required for keras-ocr

warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

try:
    # Initialize Groq client with proper error handling
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable is missing. Please set it in your .env file or environment variables.")
        groq_client = None
    else:
        groq_client = Groq(api_key=groq_api_key)
        # Test the client with a small request to verify it works
        try:
            # You can optionally do a small test call here
            pass
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
            groq_client = None
except ImportError:
    st.error("Groq library is not installed. Please install it with 'pip install groq'.")
    groq_client = None
except Exception as e:
    st.error(f"Error initializing Groq client: {str(e)}")
    groq_client = None

# 2. Add install_and_init_paddle_ocr function
def install_and_init_paddle_ocr():
    """Dynamically install and initialize PaddleOCR when needed"""
    try:
        # Check if PaddleOCR is installed
        try:
            import importlib
            paddle_spec = importlib.util.find_spec('paddleocr')
            paddle_installed = paddle_spec is not None
        except ImportError:
            paddle_installed = False
        
        # If not installed, attempt to install it
        if not paddle_installed:
            with st.spinner("PaddleOCR not found. Installing PaddleOCR (this may take a few minutes)..."):
                try:
                    # Try to install paddlepaddle and paddleocr
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle", "--no-cache-dir"])
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "paddleocr", "--no-cache-dir"])
                    st.success("PaddleOCR installed successfully!")
                except Exception as install_error:
                    st.error(f"Failed to install PaddleOCR: {str(install_error)}")
                    return None
        
        # Now try to import PaddleOCR
        try:
            from paddleocr import PaddleOCR
            # Initialize PaddleOCR for English language with optimized settings
            ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
            st.success("PaddleOCR initialized successfully!")
            return ocr
        except ImportError:
            st.warning("Could not import PaddleOCR even after installation attempt. Falling back to EasyOCR.")
            return None
        except Exception as e:
            st.error(f"Error initializing PaddleOCR: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Unexpected error with PaddleOCR setup: {str(e)}")
        return None

# Function to clean up PaddleOCR resources
def cleanup_paddle_ocr():
    """Clean up PaddleOCR resources to free memory"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Attempt to unload PaddleOCR modules from memory
        try:
            import sys
            modules_to_remove = [m for m in sys.modules if m.startswith('paddle') or m.startswith('paddleocr')]
            for module in modules_to_remove:
                if module in sys.modules:
                    del sys.modules[module]
        except Exception as unload_error:
            st.warning(f"Note: Could not fully unload PaddleOCR modules: {str(unload_error)}")
        
        # Additional cleanup
        gc.collect()
        return True
    except Exception as e:
        st.warning(f"Warning: Error during PaddleOCR cleanup: {str(e)}")
        return False

# 3. Keep the init_ocr function for EasyOCR as fallback
def init_ocr():
    """Initialize EasyOCR with optimized settings"""
    try:
        # Initialize EasyOCR for English language
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Error initializing EasyOCR: {str(e)}")
        return None

# 4. Keep init_keras_ocr as second fallback
def init_keras_ocr():
    """Initialize Keras-OCR as secondary fallback OCR engine"""
    try:
        # Check if keras_ocr is available
        if 'keras_ocr' not in globals():
            st.warning("Keras-OCR is not available. Make sure it's installed.")
            return None
        
        # Initialize Keras-OCR pipeline
        pipeline = keras_ocr.pipeline.Pipeline()
        return pipeline
    except Exception as e:
        st.error(f"Error initializing Keras-OCR: {str(e)}")
        return None

def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from scanned PDF using available OCR engines"""
    try:
        # Determine which OCR engines to use based on availability
        use_paddle = st.session_state.get('paddle_available', False)
        use_easyocr = st.session_state.get('easyocr_available', False)
        use_keras = st.session_state.get('keras_ocr_available', False)
        
        if not use_easyocr and not use_paddle and not use_keras:
            st.error("No OCR engines available. Please install at least one of: EasyOCR, PaddleOCR, or Keras-OCR.")
            return None

        # Initialize variables
        paddle_ocr = None
        easyocr_reader = None
        keras_pipeline = None
        
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        all_results = []
        total_pages = len(pdf_document)
        
        # Process each page with progress bar
        progress_bar = st.progress(0)
        
        # Set lower DPI for images to reduce memory usage
        resolution = 150  # Lower resolution = less memory but might affect accuracy
        
        try:
            # Initialize OCR engines
            if use_paddle:
                # Only try to import PaddleOCR if we're not on Streamlit Cloud
                if not is_streamlit_cloud():
                    try:
                        from paddleocr import PaddleOCR
                        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
                        st.info("Using PaddleOCR as primary engine")
                    except Exception as e:
                        st.warning(f"Could not initialize PaddleOCR: {str(e)}")
                        use_paddle = False
            
            # Initialize EasyOCR if PaddleOCR failed or we're skipping it
            if use_easyocr and (not use_paddle or not paddle_ocr):
                try:
                    import easyocr
                    easyocr_reader = easyocr.Reader(['en'], gpu=False)
                    st.info("Using EasyOCR" + (" as fallback" if use_paddle else " as primary engine"))
                except Exception as e:
                    st.warning(f"Could not initialize EasyOCR: {str(e)}")
                    use_easyocr = False
                
            # Initialize Keras-OCR as final fallback
            if use_keras and (not use_paddle or not paddle_ocr) and (not use_easyocr or not easyocr_reader):
                try:
                    import keras_ocr
                    keras_pipeline = keras_ocr.pipeline.Pipeline()
                    st.info("Using Keras-OCR" + (" as fallback" if (use_paddle or use_easyocr) else " as primary engine"))
                except Exception as e:
                    st.warning(f"Could not initialize Keras-OCR: {str(e)}")
                    use_keras = False
            
            # Process each page
            for page_num in range(total_pages):
                try:
                    # Get page and convert to image
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(alpha=False, dpi=resolution)
                    
                    # Convert to numpy array with proper shape
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, 3 if pix.n >= 3 else 1
                    )
                    
                    # Ensure 3 channels
                    if img_array.shape[-1] == 1:
                        img_array = np.repeat(img_array, 3, axis=-1)
                    
                    # Create a temporary image file for PaddleOCR if needed
                    tmp_img_path = None
                    if paddle_ocr:
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                            tmp_img_path = tmp_img.name
                            # Save the numpy array as an image
                            import cv2
                            cv2.imwrite(tmp_img_path, img_array)
                    
                    # Process with PaddleOCR if available
                    if paddle_ocr and tmp_img_path:
                        try:
                            # Process with PaddleOCR
                            result = paddle_ocr.ocr(tmp_img_path, cls=True)
                            
                            # Extract text from PaddleOCR results
                            page_text = []
                            for line in result:
                                for item in line:
                                    # PaddleOCR result format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, confidence]]
                                    page_text.append(item[1][0])  # Extract text
                            
                            all_results.extend(page_text)
                            st.info(f"Successfully processed page {page_num+1} with PaddleOCR")
                            # Delete temp file and continue to next page
                            if os.path.exists(tmp_img_path):
                                os.unlink(tmp_img_path)
                            continue
                        except Exception as paddle_error:
                            st.warning(f"PaddleOCR failed on page {page_num+1}: {str(paddle_error)}")
                            # Clean up temp file if it exists
                            if tmp_img_path and os.path.exists(tmp_img_path):
                                os.unlink(tmp_img_path)
                    
                    # Process with EasyOCR if available and PaddleOCR failed or wasn't available
                    if easyocr_reader:
                        try:
                            results = easyocr_reader.readtext(img_array)
                            page_text = [result[1] for result in results]  # Extract text
                            all_results.extend(page_text)
                            st.info(f"Successfully processed page {page_num+1} with EasyOCR")
                            continue
                        except Exception as easyocr_error:
                            st.warning(f"EasyOCR failed on page {page_num+1}: {str(easyocr_error)}")
                    
                    # Process with Keras-OCR as last resort
                    if keras_pipeline:
                        try:
                            predictions = keras_pipeline.recognize([img_array])[0]
                            keras_text = [text for text, _ in predictions]
                            all_results.extend(keras_text)
                            st.info(f"Successfully processed page {page_num+1} with Keras-OCR")
                            continue
                        except Exception as keras_error:
                            st.error(f"Keras-OCR failed on page {page_num+1}: {str(keras_error)}")
                    
                    # If we got here, all engines failed
                    st.error(f"All OCR engines failed on page {page_num+1}")
                    
                    # Clear memory
                    del img_array
                    del pix
                    gc.collect()
                
                except Exception as page_error:
                    st.warning(f"Error processing page {page_num + 1}: {str(page_error)}")
                    continue
                
                progress_bar.progress((page_num + 1) / total_pages)
                gc.collect()  # Force garbage collection after each page
            
        finally:
            # Clean up resources
            if paddle_ocr:
                del paddle_ocr
                # Clean up PaddleOCR modules
                try:
                    import sys
                    modules_to_remove = [m for m in sys.modules if m.startswith('paddle') or m.startswith('paddleocr')]
                    for module in modules_to_remove:
                        if module in sys.modules:
                            del sys.modules[module]
                except:
                    pass
            
            # Close PDF and clean up
            pdf_document.close()
            progress_bar.empty()
            gc.collect()
        
        if all_results:
            return "\n".join(all_results)
        else:
            st.error("No text was extracted from the PDF")
            return None
            
    except Exception as e:
        st.error(f"OCR processing error: {str(e)}")
        return None
        
# 6. Update is_scanned_pdf function to show dynamic OCR engine message
def is_scanned_pdf(pdf_path):
    """Check if PDF is scanned by attempting to extract text"""
    try:
        with fitz.open(pdf_path) as pdf:
            text_content = ""
            for page in pdf:
                text_content += page.get_text() or ""

            if len(text_content.strip()) < 100:
                # Check if running on Streamlit Cloud
                is_streamlit_cloud = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"
                
                if is_streamlit_cloud:
                    st.info("Detected a scanned PDF. Initializing OCR processing with EasyOCR (PaddleOCR disabled on Streamlit Cloud)...")
                else:
                    st.info("Detected a scanned PDF. Will attempt OCR processing with PaddleOCR, EasyOCR, or Keras-OCR...")
                return True
            return False
    except Exception as e:
        st.error(f"Error checking PDF type: {str(e)}")
        return True

# 7. Update process_with_ocr function to handle dynamic OCR engine selection
def process_with_ocr(pdf_path, pdf_name):
    """Process PDF with OCR including error handling and memory optimization"""
    try:
        # Check if we're on Streamlit Cloud
        is_streamlit_cloud = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"
        
        if is_streamlit_cloud:
            st.info(f"Processing {pdf_name} with EasyOCR or Keras-OCR. This may take some time and resources.")
        else:
            st.info(f"Processing {pdf_name} with available OCR engines. This may take some time and resources.")
        
        # Process with OCR
        text = extract_text_from_scanned_pdf(pdf_path)
        
        if not text:
            st.warning(f"No text could be extracted from {pdf_name}. The file might be corrupted or empty.")
            if st.button("🔄 Retry This File", key=f"retry_empty_{hash(pdf_name)}"):
                st.session_state.edited_df = None
                st.rerun()
            return None
        
        # Memory cleanup after processing
        gc.collect()
        
        # Extra cleanup for PaddleOCR
        try:
            cleanup_paddle_ocr()
        except:
            pass
        
        return text
    except Exception as e:
        handle_pdf_error(e, pdf_name)
        return None
def admin_tracking_tab():
    """Display user tracking data for admin"""
    try:
        # Read the user tracking file
        if os.path.exists(USER_TRACKING_FILE):
            df = pd.read_excel(USER_TRACKING_FILE)
            
            # Add search functionality
            st.markdown("### 📊 User Upload Tracking")
            
            # Search input
            search_term = st.text_input("🔍 Search Users or Details:", key="admin_search")
            
            # Filter DataFrame if search term is provided
            if search_term:
                mask = df.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False)
                ).any(axis=1)
                filtered_df = df[mask]
            else:
                filtered_df = df
            
            # Display filtered data
            st.dataframe(
                filtered_df, 
                use_container_width=True,
                hide_index=True
            )
            
            # Display summary statistics
            st.markdown("### 📈 Upload Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Users", len(df['Username'].unique()))
            
            with col2:
                st.metric("Total Files Uploaded", df['Files Uploaded'].sum())
            
            with col3:
                st.metric("Total Rows Processed", df['Rows Processed'].sum())
            
            # Download button
            if st.download_button(
                "📥 Download Tracking File", 
                data=open(USER_TRACKING_FILE, 'rb').read(),
                file_name="user_tracking.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_tracking"
            ):
                st.success("Tracking file downloaded successfully!")
        
        else:
            st.warning("No user tracking data available.")
    
    except Exception as e:
        st.error(f"Error displaying tracking data: {str(e)}")


def display_excel_native(excel_data):
    """Display Excel data using native Streamlit components with persistent editing"""
    try:
        # Read Excel data
        df = pd.read_excel(io.BytesIO(excel_data))
        
        # Get sheet names
        excel_file = io.BytesIO(excel_data)
        xl = pd.ExcelFile(excel_file)
        sheet_names = xl.sheet_names
        
        # Create a sheet selector if multiple sheets exist
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Select Sheet:", sheet_names)
            df = pd.read_excel(excel_file, sheet_name=selected_sheet)
        
        # Generate a unique key for the session
        session_key = f"edited_df_{datetime.now().strftime('%Y%m%d')}"
        
        # Initialize the session state for this DataFrame if it doesn't exist
        if session_key not in st.session_state:
            st.session_state[session_key] = df.copy()
        
        # Create the editable dataframe with the session state data
        edited_df = st.data_editor(
            st.session_state[session_key],
            use_container_width=True,
            num_rows="dynamic",
            height=600,
            key=f'grid_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            column_config={col: st.column_config.Column(
                width="auto",
                help=f"Column: {col}"
            ) for col in df.columns}
        )
        
        # Update the session state with the edited data
        st.session_state[session_key] = edited_df
        
        # Add search functionality
        search = st.text_input("🔍 Search in table:", key="search_input")
        if search:
            mask = edited_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
            filtered_df = edited_df[mask]
        else:
            filtered_df = edited_df
            
        st.markdown(f"**Total Rows:** {len(filtered_df)} | **Total Columns:** {len(filtered_df.columns)}")
        
        # Save Changes Button
        if st.button("💾 Save Changes", key="save_changes"):
            try:
                # Save to session state
                st.session_state.saved_df = edited_df.copy()
                
                # Save files to storage
                save_path = save_uploaded_files(
                    st.session_state.username,
                    st.session_state.uploaded_pdfs,
                    st.session_state.saved_df
                )
                
                if save_path:
                    st.success("✅ Changes saved successfully!")
                    
                    # Provide download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download edited data
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            edited_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Edited Excel",
                            data=buffer.getvalue(),
                            file_name="edited_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_edited"
                        )
                    
                    with col2:
                        # Download original data
                        buffer_original = io.BytesIO()
                        with pd.ExcelWriter(buffer_original, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Original Excel",
                            data=buffer_original.getvalue(),
                            file_name="original_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_original"
                        )
            
            except Exception as e:
                st.error(f"Error saving changes: {str(e)}")
        
        return edited_df
        
    except Exception as e:
        st.error(f"Error displaying Excel file: {str(e)}")
        return None

def cleanup_temp_files():
    """Clean up any leftover temporary files"""
    if 'cleanup_files' in st.session_state:
        for tmp_path in st.session_state.cleanup_files[:]:  # Create a copy of the list to modify it while iterating
            try:
                if os.path.exists(tmp_path):
                    gc.collect()  # Run garbage collection to release any handles
                    os.unlink(tmp_path)
                st.session_state.cleanup_files.remove(tmp_path)
            except Exception:
                pass  # Keep file in list if it still can't be deleted

# Update the main processing section
def process_uploaded_files(pdfs_to_process):
    """Process uploaded PDF files and maintain edited data state"""
    try:
        if st.session_state.edited_df is not None:
            # Use the existing edited data
            edited_df = display_excel_native(pd.DataFrame(st.session_state.edited_df))
            if edited_df is not None:
                st.session_state.edited_df = edited_df
        else:
            # Process new files
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(pdfs_to_process)
            total_rows_processed = 0
            all_data = []
            
            # Process each PDF file
            for idx, uploaded_pdf_file in enumerate(pdfs_to_process):
                try:
                    status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_pdf_file.name}")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_pdf_file.getvalue())
                        tmp_path = tmp_file.name
                        
                        # Extract and process PDF content
                        with st.spinner(f"Extracting text from {uploaded_pdf_file.name}..."):
                            pdf_text = extract_text_pdf(tmp_path)
                            
                            if pdf_text:
                                with st.spinner("Processing extracted text..."):
                                    invoice_info = using_groq(pdf_text)
                                    rows_in_file = count_processed_rows(invoice_info)
                                    total_rows_processed += rows_in_file
                                    
                                    # Process the extracted data
                                    lines = [line.strip() for line in invoice_info.split('\n') 
                                            if line.strip() and not all(c == '-' for c in line.strip())]
                                    
                                    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                                    if 'Costing Number' not in headers:
                                        headers.append('Costing Number')
                                    
                                    for line in lines[1:]:
                                        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                                        if len(cells) == len(headers) - 1:
                                            cells.append(st.session_state.costing_numbers.get(uploaded_pdf_file.name, ""))
                                            all_data.append(cells)
                        
                        os.unlink(tmp_path)
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_pdf_file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / total_files)
            
            if all_data:
                # Create DataFrame with processed data
                df = pd.DataFrame(all_data, columns=headers)
                st.session_state.edited_df = df.copy()
                
                # Display the editable data
                edited_df = display_excel_native(df)
                if edited_df is not None:
                    st.session_state.edited_df = edited_df
            
            # Update tracking
            update_user_tracking(
                username=st.session_state.username,
                files_uploaded=total_files,
                rows_processed=total_rows_processed
            )
    
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")

def create_editable_grid(df, key_prefix=""):
    """
    Create an editable grid using Streamlit data editor
    """
    try:
        # Create column configuration
        column_config = {
            col: st.column_config.Column(
                width="auto",
                help=f"Edit {col}"
            ) for col in df.columns
        }
        
        # Create the editable dataframe
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config=column_config,
            key=f"{key_prefix}_grid_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            height=600
        )
        
        # Add search functionality
        search_term = st.text_input(
            "🔍 Search in table:",
            key=f"{key_prefix}_search_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        if search_term:
            # Filter dataframe based on search term
            mask = edited_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False)
            ).any(axis=1)
            filtered_df = edited_df[mask]
        else:
            filtered_df = edited_df
            
        # Display data info
        st.markdown(f"**Total Rows:** {len(filtered_df)} | **Total Columns:** {len(filtered_df.columns)}")
        
        return edited_df, filtered_df
        
    except Exception as e:
        st.error(f"Error in create_editable_grid: {str(e)}")
        return df, df


def display_extracted_data(df):
    """Display and manage editable extracted data with persistent state"""
    try:
        st.markdown("### 📝 Extracted and Edited Data")
        
        # Generate a unique session key for this dataset
        if 'grid_key' not in st.session_state:
            st.session_state.grid_key = 'data_editor_1'
            
        # Initialize the editor state if it doesn't exist
        if 'editor_data' not in st.session_state:
            st.session_state.editor_data = df.copy()
        
        # Add search functionality before the data editor
        search_query = st.text_input("🔍 Search in table:", key="search_input")
        
        # Filter the data based on search before displaying
        display_data = st.session_state.editor_data.copy()
        if search_query:
            mask = display_data.astype(str).apply(
                lambda x: x.str.contains(search_query, case=False)
            ).any(axis=1)
            display_data = display_data[mask]
        
        # Create the editable dataframe with persistent state
        edited_df = st.data_editor(
            display_data,
            use_container_width=True,
            num_rows="dynamic",
            key=st.session_state.grid_key,
            height=600,
            column_config={
                col: st.column_config.Column(
                    width="auto",
                    help=f"Edit {col}"
                ) for col in df.columns
            }
        )
        
        # Update the session state with edited data
        st.session_state.editor_data = edited_df
        
        # Display data info
        st.markdown(f"**Total Rows:** {len(edited_df)} | **Total Columns:** {len(edited_df.columns)}")
        
        # Save Changes Button
        if st.button("💾 Save Changes", key="save_changes"):
            try:
                # Update session states
                st.session_state.saved_df = edited_df.copy()
                st.session_state.edited_df = edited_df.copy()
                
                # Save files to storage
                save_path = save_uploaded_files(
                    st.session_state.username,
                    st.session_state.uploaded_pdfs,
                    st.session_state.saved_df
                )
                
                if save_path:
                    st.success("✅ Changes saved successfully!")
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download edited data
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            edited_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Edited Excel",
                            data=buffer.getvalue(),
                            file_name="edited_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_edited"
                        )
                    
                    with col2:
                        # Download original data
                        buffer_original = io.BytesIO()
                        with pd.ExcelWriter(buffer_original, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Original Excel",
                            data=buffer_original.getvalue(),
                            file_name="original_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_original"
                        )
            
            except Exception as e:
                st.error(f"Error saving changes: {str(e)}")
        
        return edited_df
        
    except Exception as e:
        st.error(f"Error displaying extracted data: {str(e)}")
        return df


def modify_history_tab():
    st.markdown("### 📂 Previous Uploads")
    user_uploads = get_user_uploads(st.session_state.username)
    
    if not user_uploads.empty:
        for idx, row in user_uploads.iterrows():
            session_id = f"session_{idx}"
            
            with st.expander(f"Upload from {row['Upload Date']}"):
                # Create tabs for different views
                view_tab, download_tab, share_tab = st.tabs(["View Files", "Download Files", "Share Files"])
                
                with view_tab:
                    # PDF Viewer
                    st.markdown("**📄 View Invoice PDFs:**")
                    for pdf_idx, pdf_name in enumerate(row['Invoice Files'].split(', ')):
                        pdf_path = os.path.join(row['Path'], pdf_name)
                        if os.path.exists(pdf_path):
                            if st.button(f"View {pdf_name}", key=f"view_pdf_{session_id}_{pdf_idx}"):
                                with open(pdf_path, 'rb') as pdf_file:
                                    pdf_data = pdf_file.read()
                                    display_pdf(pdf_data)
                    
                    # Excel Viewer
                    st.markdown("**📊 View Excel Result:**")
                    excel_path = os.path.join(row['Path'], row['Excel Result'])
                    if os.path.exists(excel_path):
                        if st.button(f"View {row['Excel Result']}", key=f"view_excel_{session_id}"):
                            with open(excel_path, 'rb') as excel_file:
                                excel_data = excel_file.read()
                                display_excel_native(excel_data)
                
                with download_tab:
                    # Original download functionality
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📄 Download Invoice PDFs:**")
                        for pdf_idx, pdf_name in enumerate(row['Invoice Files'].split(', ')):
                            pdf_path = os.path.join(row['Path'], pdf_name)
                            if os.path.exists(pdf_path):
                                st.download_button(
                                    f"📥 {pdf_name}",
                                    download_stored_file(pdf_path),
                                    file_name=pdf_name,
                                    mime="application/pdf",
                                    key=f"download_pdf_{session_id}_{pdf_idx}"
                                )
                    
                    with col2:
                        st.markdown("**📊 Download Excel Result:**")
                        excel_path = os.path.join(row['Path'], row['Excel Result'])
                        if os.path.exists(excel_path):
                            st.download_button(
                                f"📥 {row['Excel Result']}",
                                download_stored_file(excel_path),
                                file_name=row['Excel Result'],
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_excel_{session_id}"
                            )
                
                with share_tab:
                    # Original share functionality
                    st.markdown("**🔗 Share Files:**")
                    if st.button("Generate Links", key=f"share_{session_id}"):
                        share_links = []
                        
                        for pdf_name in row['Invoice Files'].split(', '):
                            pdf_path = os.path.join(row['Path'], pdf_name)
                            if os.path.exists(pdf_path):
                                pdf_link = generate_share_link(pdf_path)
                                if pdf_link:
                                    share_links.append((pdf_name, pdf_link))
                        
                        excel_path = os.path.join(row['Path'], row['Excel Result'])
                        if os.path.exists(excel_path):
                            excel_link = generate_share_link(excel_path)
                            if excel_link:
                                share_links.append((row['Excel Result'], excel_link))
                        
                        if share_links:
                            st.markdown("**Generated Links:**")
                            for link_idx, (file_name, link) in enumerate(share_links):
                                with st.container():
                                    st.text(file_name)
                                    st.code(link)
                                    if st.button(
                                        "📋 Copy Link",
                                        key=f"copy_{session_id}_{link_idx}"
                                    ):
                                        st.write(f"```{link}```")
                                    st.markdown("---")
    else:
        st.info("No previous uploads found")

    """Display Excel data using native Streamlit components"""
    try:
        # Read Excel data
        df = pd.read_excel(io.BytesIO(excel_data))
        
        # Get sheet names
        excel_file = io.BytesIO(excel_data)
        xl = pd.ExcelFile(excel_file)
        sheet_names = xl.sheet_names
        
        # Create a sheet selector if multiple sheets exist
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Select Sheet:", sheet_names)
            df = pd.read_excel(excel_file, sheet_name=selected_sheet)
            
        # Add search functionality
        search = st.text_input("🔍 Search in table:", key="excel_search")
        if search:
            mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
            df = df[mask]
        
        # Display the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            height=600,
            hide_index=True
        )
        
        st.download_button(
            "📥 Download Excel File",
            excel_data,
            file_name="downloaded_file.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        return df
    except Exception as e:
        st.error(f"Error displaying Excel file: {str(e)}")
        return None

def display_pdf(pdf_data):
    """Display PDF as images while maintaining PDF download capability"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            tmp_pdf.write(pdf_data)
            pdf_path = tmp_pdf.name

        # images = convert_from_path(pdf_path, dpi=200)
        
        # for i, image in enumerate(images):
        #     img_byte_arr = io.BytesIO()
        #     image.save(img_byte_arr, format='JPEG', quality=95)
        #     img_byte_arr = img_byte_arr.getvalue()

        #     st.image(img_byte_arr, caption=f'Page {i+1}', use_container_width=True)
        

        st.download_button(
            label="📥 Download PDF",
            data=pdf_data,
            file_name="document.pdf",
            mime="application/pdf"
        )

        os.unlink(pdf_path)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        st.download_button(
            label="⚠️ Download PDF",
            data=pdf_data,
            file_name="document.pdf",
            mime="application/pdf"
        )

def generate_share_link(file_path, expiry_days=7):
    """Generate a shareable link for a file"""
    try:
        # Create a unique identifier for the file
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
        
        # Create share info
        share_info = {
            'file_path': file_path,
            'expiry_date': expiry_date,
            'original_filename': os.path.basename(file_path)
        }
        
        shares_dir = 'storage/shares'
        os.makedirs(shares_dir, exist_ok=True)
        
        share_file = os.path.join(shares_dir, f'{file_hash}.json')
        with open(share_file, 'w') as f:
            json.dump(share_info, f)
            
        base_url = "https://aki-asn.streamlit.app"
        
        share_link = f"{base_url}/?share={file_hash}"
        
        return share_link
        
    except Exception as e:
        st.error(f"Error generating share link: {str(e)}")
        return None

def auto_download_shared_file():
    """Automatically handle file download based on URL parameters"""
    try:
        current_path = st.query_params.get('path', '')
        
        if current_path.startswith('download/'):
            file_hash = current_path.split('/')[-1]
            share_file = f'storage/shares/{file_hash}.json'
            
            if not os.path.exists(share_file):
                st.error("This download link is invalid or has expired.")
                return
            
            with open(share_file, 'r') as f:
                share_info = json.load(f)
            
            expiry_date = datetime.strptime(share_info['expiry_date'], '%Y-%m-%d')
            if datetime.now() > expiry_date:
                os.remove(share_file)
                st.error("This download link has expired.")
                return
            
            file_path = share_info['file_path']
            if not os.path.exists(file_path):
                st.error("The file is no longer available.")
                return
            
            file_data = download_stored_file(file_path)
            if file_data:
                # Determine mime type
                original_filename = share_info['original_filename']
                mime_type = ("application/pdf" if original_filename.lower().endswith('.pdf') 
                           else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # Create a minimal UI for the download
                st.markdown("""
                    <style>
                        .stDownloadButton button {
                            width: 100%;
                            height: 60px;
                            font-size: 20px;
                            margin-top: 20px;
                        }
                        .centered {
                            text-align: center;
                            padding: 20px;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                st.markdown(f"<div class='centered'><h2>📥 Downloading {original_filename}</h2></div>", 
                          unsafe_allow_html=True)
                
                # Auto-trigger download using HTML
                components.html(
                    f"""
                    <html>
                        <body>
                            <script>
                                window.onload = function() {{
                                    setTimeout(function() {{
                                        document.getElementById('download-button').click();
                                    }}, 500);
                                }}
                            </script>
                        </body>
                    </html>
                    """,
                    height=0,
                )
                
                # Fallback download button in case auto-download fails
                st.download_button(
                    label=f"Download {original_filename}",
                    data=file_data,
                    file_name=original_filename,
                    mime=mime_type,
                    key="download-button"
                )
                
                st.markdown("<div class='centered'><p>If the download doesn't start automatically, click the button above.</p></div>", 
                          unsafe_allow_html=True)
                
            else:
                st.error("Unable to prepare the file for download.")
            
    except Exception as e:
        st.error(f"Error processing download: {str(e)}")
    
def handle_download_page(share_hash):
    try:
        share_info = get_shared_file(share_hash)
        if not share_info:
            st.error("Invalid or expired download link")
            return

        file_path = share_info['file_path']
        if not os.path.exists(file_path):
            st.error("File no longer exists")
            return

        file_data = download_stored_file(file_path)
        if not file_data:
            return

        original_filename = share_info['original_filename']
        
        if original_filename.lower().endswith('.pdf'):
            # Display PDF inline
            base64_pdf = base64.b64encode(file_data).decode('utf-8')
            pdf_display = f'''
                <embed src="data:application/pdf;base64,{base64_pdf}" 
                       type="application/pdf" 
                       width="100%" 
                       height="800px" 
                       internalinstanceid="pdf-display">
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            # Handle Excel files - display as DataFrame
            try:
                # Read Excel file
                df = pd.read_excel(io.BytesIO(file_data))
                
                # Add title
                st.markdown(f"### 📊 {original_filename}")
                
                # Add search functionality
                search = st.text_input("🔍 Search in table:", "")
                
                # Filter DataFrame based on search
                if search:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                    filtered_df = df[mask]
                else:
                    filtered_df = df
                
                # Display data info
                st.markdown(f"**Total Rows:** {len(filtered_df)} | **Total Columns:** {len(filtered_df.columns)}")
                
                # Display the DataFrame with styling
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=600
                )
                
                # Add download button below the table
                st.download_button(
                    "📥 Download Excel File",
                    file_data,
                    file_name=original_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error displaying Excel file: {str(e)}")
                # Fallback to download button if display fails
                st.download_button(
                    label="Download File",
                    data=file_data,
                    file_name=original_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error handling download: {str(e)}")
        st.error(traceback.format_exc())

def verify_storage_setup():
    """Verify that storage is set up correctly"""
    try:
        # Check main storage directory
        if not os.path.exists('storage'):
            # st.error("Main storage directory is missing")
            return False
            
        # Check shares directory
        shares_dir = 'storage/shares'
        if not os.path.exists(shares_dir):
            # st.error("Shares directory is missing")
            return False
                
        return True
    except Exception as e:
        st.error(f"Error verifying storage: {str(e)}")
        return False

def setup_storage():
    """Create necessary directories for file storage"""
    # Create base storage directory
    if not os.path.exists('storage'):
        os.makedirs('storage')
    
    # Create uploads tracking file if it doesn't exist
    if not os.path.exists('storage/uploads_tracking.xlsx'):
        df = pd.DataFrame(columns=['Username', 'Upload Date', 'Invoice Files', 'Excel Result', 'Path'])
        df.to_excel('storage/uploads_tracking.xlsx', index=False)



def save_uploaded_files(username, pdf_files, excel_data):
    """Save uploaded PDFs and Excel result"""
    try:
        # Create timestamp-based directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_dir = username.split('@')[0]
        save_path = f'storage/{user_dir}/{timestamp}'
        os.makedirs(save_path, exist_ok=True)
        
        # Save PDF files
        pdf_names = []
        for pdf in pdf_files:
            pdf_path = f'{save_path}/{pdf.name}'
            with open(pdf_path, 'wb') as f:
                f.write(pdf.getvalue())
            pdf_names.append(pdf.name)
        
        # Save Excel result
        excel_name = f'ASN_Result_{timestamp}.xlsx'
        excel_path = f'{save_path}/{excel_name}'
        excel_data.to_excel(excel_path, index=False)
        
        # Update tracking file
        tracking_df = pd.read_excel('storage/uploads_tracking.xlsx')
        new_row = {
            'Username': username,
            'Upload Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Invoice Files': ', '.join(pdf_names),
            'Excel Result': excel_name,
            'Path': save_path
        }
        tracking_df = pd.concat([tracking_df, pd.DataFrame([new_row])], ignore_index=True)
        tracking_df.to_excel('storage/uploads_tracking.xlsx', index=False)
        
        return save_path
        
    except Exception as e:
        st.error(f"Error saving files: {str(e)}")
        return None

def get_user_uploads(username):
    """Get all previous uploads for a user"""
    try:
        tracking_df = pd.read_excel('storage/uploads_tracking.xlsx')
        user_uploads = tracking_df[tracking_df['Username'] == username].copy()
        return user_uploads.sort_values('Upload Date', ascending=False)
    except Exception as e:
        st.error(f"Error retrieving uploads: {str(e)}")
        return pd.DataFrame()

def download_stored_file(file_path):
    """Read a stored file for downloading"""
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        .main {
            padding: 2rem;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .stAlert {
            padding: 20px;
            margin: 10px 0;
        }
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state: 
    st.session_state.logged_in = False
if 'username' not in st.session_state: 
    st.session_state.username = None

DEFAULT_PASSWORD = '12345'
USER_TRACKING_FILE = 'user_tracking.xlsx'
  
def validate_email(email): 
    return email.lower().endswith('mhd')

def get_shared_file(share_hash):
    """Retrieve shared file information"""
    try:
        share_file = f'storage/shares/{share_hash}.json'
        if not os.path.exists(share_file):
            return None
            
        with open(share_file, 'r') as f:
            share_info = json.load(f)
            
        # Check if share has expired
        expiry_date = datetime.strptime(share_info['expiry_date'], '%Y-%m-%d')
        if datetime.now() > expiry_date:
            os.remove(share_file)  # Clean up expired share
            return None
            
        return share_info
    
    except Exception as e:
        st.error(f"Error retrieving shared file: {str(e)}")
        return None

def init_user_tracking():
    try:
        if not os.path.exists(USER_TRACKING_FILE):
            df = pd.DataFrame(columns=[
                'User ID',
                'Username',
                'Upload Time',
                'Files Uploaded',
                'Rows Uploaded'
            ])
            try: 
                df.to_excel(USER_TRACKING_FILE, index=False)
            except PermissionError:
                st.warning("Warning: Could not create tracking file. Data will be cached in session.")
                st.session_state.user_tracking = df

    except Exception as e:
        st.error(f"Error initializing user tracking: {str(e)}")
        st.session_state.user_tracking = pd.DataFrame(columns=[
            'User ID',
            'Username',
            'Upload Time',
            'Files Uploaded',
            'Rows Uploaded'
        ])
            

def display_history_tab():
    st.markdown("### 📂 Previous Uploads")
    user_uploads = get_user_uploads(st.session_state.username)
    
    if not user_uploads.empty:
        for idx, row in user_uploads.iterrows():
            # Create a unique identifier for this upload session
            session_id = f"session_{idx}"
            
            with st.expander(f"Upload from {row['Upload Date']}"):
                st.write(f"**Invoice Files:** {row['Invoice Files']}")
                
                # Create three equal columns
                pdf_col, excel_col, share_col = st.columns(3)
                
                with pdf_col:
                    st.markdown("**📄 Invoice PDFs:**")
                    for pdf_idx, pdf_name in enumerate(row['Invoice Files'].split(', ')):
                        pdf_path = os.path.join(row['Path'], pdf_name)
                        if os.path.exists(pdf_path):
                            # Create a unique key using session_id, pdf_idx, and a hash of the pdf_name
                            pdf_key = f"pdf_{session_id}_{pdf_idx}_{hash(pdf_name)}"
                            st.download_button(
                                f"📥 {pdf_name}",
                                download_stored_file(pdf_path),
                                file_name=pdf_name,
                                mime="application/pdf",
                                key=pdf_key
                            )
                
                with excel_col:
                    st.markdown("**📊 Excel Result:**")
                    excel_path = os.path.join(row['Path'], row['Excel Result'])
                    if os.path.exists(excel_path):
                        # Create a unique key for the excel download button
                        excel_key = f"excel_{session_id}_{hash(row['Excel Result'])}"
                        st.download_button(
                            f"📥 {row['Excel Result']}",
                            download_stored_file(excel_path),
                            file_name=row['Excel Result'],
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=excel_key
                        )
                
                with share_col:
                    st.markdown("**🔗 Share Files:**")
                    # Create a unique key for the share button
                    share_key = f"share_{session_id}"
                    if st.button("Generate Links", key=share_key):
                        share_links = []
                        
                        # Share PDFs
                        for pdf_name in row['Invoice Files'].split(', '):
                            pdf_path = os.path.join(row['Path'], pdf_name)
                            if os.path.exists(pdf_path):
                                pdf_link = generate_share_link(pdf_path)
                                if pdf_link:
                                    share_links.append((pdf_name, pdf_link))
                        
                        # Share Excel
                        excel_path = os.path.join(row['Path'], row['Excel Result'])
                        if os.path.exists(excel_path):
                            excel_link = generate_share_link(excel_path)
                            if excel_link:
                                share_links.append((row['Excel Result'], excel_link))
                        
                        # Display share links
                        if share_links:
                            st.markdown("**Generated Links:**")
                            for link_idx, (file_name, link) in enumerate(share_links):
                                link_container_key = f"link_container_{session_id}_{link_idx}"
                                with st.container(key=link_container_key):
                                    st.text(file_name)
                                    st.code(link)
                                    # Use a unique key for each copy button
                                    copy_key = f"copy_{session_id}_{link_idx}"
                                    st.button(
                                        "📋 Copy Link",
                                        key=copy_key,
                                        on_click=lambda l=link: st.write(f"```{l}```")
                                    )
                                    st.markdown("---")
    else:
        st.info("No previous uploads found")

def update_user_tracking(username, files_uploaded=0, rows_processed=0):
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize DataFrame
        df = None
        
        # Try to read existing tracking file
        try:
            if os.path.exists(USER_TRACKING_FILE):
                df = pd.read_excel(USER_TRACKING_FILE)
            else:
                df = pd.DataFrame(columns=[
                    'User ID',
                    'Username',
                    'Upload Time',
                    'Files Uploaded',
                    'Rows Processed'
                ])
        except Exception as e:
            st.warning(f"Could not read tracking file: {str(e)}")
            df = pd.DataFrame(columns=[
                'User ID',
                'Username',
                'Upload Time',
                'Files Uploaded',
                'Rows Processed'
            ])

        # Generate user ID
        if df is not None and not df.empty:
            user_id = df['User ID'].max() + 1
        else:
            user_id = 1

        # Create new row
        new_row = pd.DataFrame({
            'User ID': [user_id],
            'Username': [username],
            'Upload Time': [current_time],
            'Files Uploaded': [files_uploaded],
            'Rows Processed': [rows_processed]
        })

        # Append new row
        df = pd.concat([df, new_row], ignore_index=True)

        # Try to save the updated tracking file
        try:
            df.to_excel(USER_TRACKING_FILE, index=False)
            if files_uploaded > 0:
                st.success(f"""Upload tracked successfully:
                - Files uploaded: {files_uploaded}
                - Rows processed: {rows_processed}""")
        except Exception as e:
            st.warning(f"Could not save tracking file: {str(e)}")
            st.session_state['tracking_df'] = df
            if files_uploaded > 0:
                st.info(f"""Upload tracked (temporarily saved):
                - Files uploaded: {files_uploaded}
                - Rows processed: {rows_processed}""")

    except Exception as e:
        st.error(f"User tracking update error: {str(e)}")


# def is_scanned_pdf(pdf_path):
#     """Check if PDF is scanned by attempting to extract text"""
#     try:
#         with fitz.open(pdf_path) as pdf:
#             text_content = ""
#             for page in pdf:
#                 text_content += page.get_text() or ""

#             if len(text_content.strip()) < 100:
#                 return True
#             return False
#     except Exception as e:
#         st.error(f"Error checking PDF type: {str(e)}")
#         return True

# def is_scanned_pdf(pdf_path):
#     """Check if PDF is scanned by attempting to extract text"""
#     try:
#         with fitz.open(pdf_path) as pdf:
#             text_content = ""
#             for page in pdf:
#                 text_content += page.get_text() or ""

#             if len(text_content.strip()) < 100:
#                 st.info("We are working on it now") # Added message for scanned PDFs
#                 return True
#             return False
#     except Exception as e:
#         st.error(f"Error checking PDF type: {str(e)}")
#         return True

# def is_scanned_pdf(pdf_path):
#     """Check if PDF is scanned by attempting to extract text"""
#     try:
#         with fitz.open(pdf_path) as pdf:
#             text_content = ""
#             for page in pdf:
#                 text_content += page.get_text() or ""

#             if len(text_content.strip()) < 100:
#                 st.info("Detected a scanned PDF. Initializing OCR processing...")
#                 return True
#             return False
#     except Exception as e:
#         st.error(f"Error checking PDF type: {str(e)}")
#         return True
# def process_invoice_lines(invoice_info, costing_number=""):
#     """
#     Process invoice information lines while properly handling separators
#     and costing numbers. Keeps file separators (--) but ignores dash-only lines.
#     Includes comprehensive error handling.
#     """
#     try:
#         # Validate input
#         if not isinstance(invoice_info, str):
#             raise ValueError("Invoice info must be a string")
            
#         # Split into lines and clean them
#         try:
#             lines = [line.strip() for line in invoice_info.split('\n')]
#         except Exception as e:
#             st.error(f"Error splitting invoice lines: {str(e)}")
#             return None, None
            
#         # Initialize containers
#         valid_lines = []
#         in_data_section = False
        
#         # Process lines with error handling
#         for line_num, line in enumerate(lines, 1):
#             try:
#                 # Skip empty lines
#                 if not line:
#                     continue
                    
#                 # Keep file separators (lines with -- and potential text)
#                 if '--' in line and '|' not in line:
#                     valid_lines.append(line)
#                     in_data_section = False
#                     continue
                    
#                 # Skip lines that are only dashes
#                 if set(line).issubset({'-', ' '}):
#                     continue
                    
#                 # If line contains | it's either a header or data
#                 if '|' in line:
#                     in_data_section = True
#                     valid_lines.append(line)
                    
#             except Exception as e:
#                 st.warning(f"Error processing line {line_num}: {str(e)}. Skipping line.")
#                 continue
        
#         # Process headers and data
#         try:
#             current_headers = None
#             current_data = []
            
#             for line_num, line in enumerate(valid_lines, 1):
#                 try:
#                     # If it's a file separator, reset for next section
#                     if '--' in line and '|' not in line:
#                         continue
                        
#                     # If line contains |, process it
#                     if '|' in line:
#                         try:
#                             cells = [cell.strip() for cell in line.split('|') if cell.strip()]
#                         except Exception as e:
#                             st.warning(f"Error splitting cells on line {line_num}: {str(e)}. Skipping line.")
#                             continue
                        
#                         # If no headers set yet, this is a header row
#                         if current_headers is None:
#                             current_headers = cells
#                             if 'Costing Number' not in current_headers:
#                                 current_headers.append('Costing Number')
#                         else:
#                             # Data row - validate cell count
#                             try:
#                                 if len(cells) == len(current_headers) - 1:
#                                     cells.append(costing_number)
#                                 elif len(cells) < len(current_headers) - 1:
#                                     st.warning(f"Line {line_num} has fewer cells than headers. Adding empty cells.")
#                                     cells.extend([''] * (len(current_headers) - 1 - len(cells)))
#                                     cells.append(costing_number)
#                                 current_data.append(cells)
#                             except Exception as e:
#                                 st.warning(f"Error processing data row {line_num}: {str(e)}. Skipping row.")
#                                 continue
                                
#                 except Exception as e:
#                     st.warning(f"Error processing valid line {line_num}: {str(e)}. Skipping line.")
#                     continue
                    
#             # Validate final output
#             if not current_headers:
#                 st.error("No valid headers found in the invoice")
#                 return None, None
                
#             if not current_data:
#                 st.warning("No valid data rows found in the invoice")
#                 return current_headers, []
                
#             return current_headers, current_data
            
#         except Exception as e:
#             st.error(f"Error in final data processing: {str(e)}")
#             return None, None
            
#     except Exception as e:
#         st.error(f"Critical error in process_invoice_lines: {str(e)}")
#         return None, None

def process_invoice_lines(invoice_info, costing_number=""):
    """
    Process invoice information lines with standardized headers
    """
    try:
        # Define header mappings
        header_mappings = {
            'Customer Number': 'Customer No',
            'Customer No.': 'Customer No',
            'Supplier Name': 'Supplier',
            'Total VAT': 'VAT',
            'Total VAT or VAT': 'VAT',
            'Total Amount of the Invoice': 'Invoice Total',
            'Payer Name': 'Payer Name',
            'Date of Invoice': 'Invoice Date',
            'Manufacturing Date': 'Mfg Date',
            'Manufacture Date': 'Mfg Date',
            'Production Date': 'Mfg Date',
            'Prod Date': 'Mfg Date',
            'Prod. Date': 'Mfg Date',
            'Date of Manufacture': 'Mfg Date',
            'DOM': 'Mfg Date',
            'Manufactured On': 'Mfg Date',
            'Manuf. Date': 'Mfg Date'
        }

        # Define standard headers
        standard_headers = [
            'PO Number', 'Item Code', 'Description', 'UOM', 'Quantity',
            'Lot Number', 'Expiry Date', 'Mfg Date', 'Invoice No',
            'Unit Price', 'Total Price', 'Country', 'HS Code',
            'Invoice Date', 'Customer No', 'Payer Name', 'Currency',
            'Supplier', 'Invoice Total', 'VAT', 'Line Number'
        ]

        if 'Costing Number' not in standard_headers:
            standard_headers.append('Costing Number')

        # Split and clean lines
        lines = [line.strip() for line in invoice_info.split('\n')]
        valid_lines = []
        
        # Initial processing of lines
        for line in lines:
            if not line:
                continue
            if '--' in line and '|' not in line:
                valid_lines.append(line)
                continue
            if set(line).issubset({'-', ' '}):
                continue
            if '|' in line:
                cleaned_cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                valid_lines.append('|'.join(cleaned_cells))

        # Process headers and data
        headers = None
        data_rows = []
        raw_headers = None
        
        for line in valid_lines:
            if '--' in line and '|' not in line:
                continue
                
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                
                if headers is None:
                    # Store original headers and get standardized version
                    raw_headers = cells
                    headers = standard_headers
                    # st.write(f"DEBUG - Original headers: {raw_headers}")
                    # st.write(f"DEBUG - Standardized headers: {headers}")
                else:
                    # Create a mapping of current data
                    data_dict = {}
                    for i, cell in enumerate(cells):
                        if i < len(raw_headers):
                            data_dict[raw_headers[i]] = cell

                    # Build standardized row
                    standardized_row = []
                    for header in headers[:-1]:  # Exclude Costing Number
                        value = ''
                        # Check mapped header names first
                        mapped_found = False
                        for raw_key, std_key in header_mappings.items():
                            if std_key == header and raw_key in data_dict:
                                value = data_dict[raw_key]
                                mapped_found = True
                                break
                        
                        # If no mapping found, try direct header
                        if not mapped_found and header in data_dict:
                            value = data_dict[header]
                        
                        standardized_row.append(value)

                    # Add costing number
                    standardized_row.append(costing_number)
                    data_rows.append(standardized_row)

        if headers and data_rows:
            # Validate row lengths
            for i, row in enumerate(data_rows):
                if len(row) != len(headers):
                    # st.write(f"DEBUG - Row {i} length mismatch: {len(row)} vs {len(headers)}")
                    # st.write(f"DEBUG - Row data: {row}")
                    # Pad or trim row to match header length
                    if len(row) < len(headers):
                        row.extend([''] * (len(headers) - len(row)))
                    else:
                        data_rows[i] = row[:len(headers)]

        return headers, data_rows
        
    except Exception as e:
        st.error(f"Error in process_invoice_lines: {str(e)}")
        st.error(traceback.format_exc())
        return None, None



def count_processed_rows(invoice_info):
    """
    Count actual data rows, excluding separators and headers
    """
    try:
        # Split into lines and clean them
        lines = [line.strip() for line in invoice_info.split('\n')]
        
        # Count only valid data rows
        data_rows = 0
        header_found = False
        
        for line in lines:
            # Skip empty lines and separator lines
            if not line or set(line.replace('|', '')).issubset({'-', ' '}):
                continue
                
            # Skip header row
            if not header_found:
                header_found = True
                continue
                
            # Count valid data row
            data_rows += 1
            
        return data_rows
        
    except Exception as e:
        st.error(f"Error counting processed rows: {str(e)}")
        return 0



def check_shared_file():
    """Handle shared file viewing and downloading"""
    try:
        share_hash = st.query_params.get('share')
        
        if share_hash:
            share_info = get_shared_file(share_hash)
            if share_info:
                file_path = share_info['file_path']
                if os.path.exists(file_path):
                    file_data = download_stored_file(file_path)
                    if file_data:
                        file_name = os.path.basename(file_path)
                        st.markdown(f"### 📄 File: {file_name}")
                        
                        if file_name.lower().endswith('.pdf'):
                            st.info("Loading PDF viewer... If the viewer doesn't load, you can use the download options.")
                            display_pdf(file_data)
                        else:
                            # Display Excel file
                            try:
                                df = pd.read_excel(io.BytesIO(file_data))
                                
                                # Add search functionality
                                search = st.text_input("🔍 Search in table:", key="excel_search")
                                if search:
                                    mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                                    df = df[mask]
                                
                                # Display info
                                st.markdown(f"**Total Rows:** {len(df)} | **Total Columns:** {len(df.columns)}")
                                
                                # Display the dataframe
                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    height=600
                                )
                                
                                # Download button
                                st.download_button(
                                    "📥 Download Excel File",
                                    file_data,
                                    file_name=file_name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            except Exception as excel_error:
                                st.error(f"Error displaying Excel file: {str(excel_error)}")
                                st.download_button(
                                    label="Download File",
                                    data=file_data,
                                    file_name=file_name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    else:
                        st.error("Unable to read the shared file.")
                else:
                    st.error("The shared file no longer exists.")
            else:
                st.error("This share link has expired or is invalid.")
    except Exception as e:
        st.error(f"Error processing shared file: {str(e)}")
        st.error(traceback.format_exc())

def login_page():
    """Display login page"""
    st.title("🔐 Login to ASN Project")
    
    with st.container():
        username = st.text_input("AKI Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if not username or not password:
                st.error("Please fill in all fields")
                return
            
            if not validate_email(username):
                st.error("Username must end with @akigroup.com")
                return
            

            if username == 'admin@akigroup.com':
                if password != DEFAULT_PASSWORD:
                    st.error("Invalid admin password")
                    return
            elif password != DEFAULT_PASSWORD:
                st.error("Invalid password")
            
                return
            if password != DEFAULT_PASSWORD:
                st.error("Invalid password")
                return
            
            st.session_state.logged_in = True
            st.session_state.username = username
            update_user_tracking(username)
            st.success("Login successful!")
            st.rerun()



def extract_text_pdf(pdf_path):
    """Extract text from PDF, handling both scanned and machine-readable PDFs"""
    if is_scanned_pdf(pdf_path):
        with st.spinner("Processing scanned PDF with OCR..."):
            return extract_text_from_scanned_pdf(pdf_path)
    else:
        try:
            with fitz.open(pdf_path) as pdf:
                unique_pages = {}
                for page_num, page in enumerate(pdf):
                    page_text = page.get_text()
                    content_hash = hash(page_text)
                    if content_hash not in unique_pages:
                        unique_pages[content_hash] = page_text
                return "\n".join(unique_pages.values())
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return None
            

def format_markdown_table(headers, data):
    """
    Create a properly formatted Markdown table with consistent separator line
    """
    # Create header row
    table = [f"| {' | '.join(headers)} |"]
    
    # Create separator row with consistent dashes
    separator = [f"|{'|'.join('-' * (len(header) + 2) for header in headers)}|"]
    
    # Create data rows
    data_rows = [f"| {' | '.join(row)} |" for row in data]
    
    # Combine all parts
    return '\n'.join(table + separator + data_rows)

def using_groq(text: str):
    
    prompt = f"""Extract ALL invoice data without skipping ANY item and FILL EVERY FIELD. Empty cells WILL cause FAILURE.

{text}

### Mandatory Fields (Every row must have values):
   - PO Number: Order Number or Purchase Order fields. 
     IMPORTANT: Remove any text like "MDS", "-MDS", "/MDS" after the number
     If not found, use "-"
   - Item Code: If missing, use "ITEM" + line number
   - Description: If missing, use "Product Line " + line number
   - UOM: Unit of Measure 
   - Quantity: or Quantity Shipped
   - Lot Number: Example: "Batch/serial Nr 272130" means lot number is "272130"
                 IMPORTANT: If multiple batches/lots exist for the same item, CREATE SEPARATE ROWS for each batch
                 Only use "N/A" if confirmed missing after thorough search
   - Expiry Date: use "-" if missing, format as DD-MM-YYYY
                  IMPORTANT: If multiple expiry dates exist, CREATE SEPARATE ROWS with matching lot numbers
   - Manufacturing Date or Mfg Date: Only use "N/A" if confirmed missing after thorough search
   - Invoice No: MUST be found - look in header
   - Unit Price: Default to Total Price if missing
   - Total Price: Default to Unit Price × Quantity if missing
   - Country: Convert codes to full names (e.g., IE → Ireland).
   - HS Code: Default "-" if missing
   - Invoice Date: Extract from header or near invoice number (format: DD-MM-YYYY)
   - Customer No: Extract from "Customer Nr" fields or fallback to company code
   - Payer Name: ALWAYS exactly "ALPHAMED GENERAL TRADING LLC." (no exceptions)
   - Currency: Use "EUR" for European suppliers, "USD" for USA, "THB" for Thailand
   - Supplier: MUST find the company name from letterhead/invoice header
   - Invoice Total: Sum all line totals if not explicitly stated
   - VAT: Look for VAT percentage or amount - use "0" if not found
   
CRITICAL:
- MULTIPLE LOT HANDLING:
    - When an item has multiple batches/lots listed (like "Batch: 37465YQ" and "Batch: 37580YQ"), you MUST create separate rows for each batch
    - Example:
    Lot Number: 23229017
                23231017
    Expiry Date: 01-08-2025
                 01-08-2025 
- SEARCH THE ENTIRE DOCUMENT for information, not just the main table. Many invoices have detailed sections below the main table with additional information about each line item (batch numbers, country of origin, manufacturing dates, etc.)
- For quantity, If "Pack Factor" or "Line has X packs" is present, multiply the pack count by the pack factor.
  For example, "Line has 522 packs" with "Pack Factor: 2" means 522 × 2 = 1044, not 522.
   
CRITICAL FIELD COMPLETION REQUIREMENT:

DO NOT SKIP ANY ITEMS OR ENTRIES. You must extract EVERY SINGLE LINE ITEM from the invoice, even if they seem similar to others.
If you see multiple items with the same or similar product descriptions but different item codes, PO numbers, or quantities, 
you MUST include ALL of them as separate entries in your output.

Rules for Missing Data::
   - Line Number: extract Line number from invoices example :(10, 20, etc) if missing use row sequence (1, 2, 3...) if missing
   - If data is completely missing: Use "N/A".
   - Never leave any cell blank—search the entire invoice if needed

THIS IS A FIRM REQUIREMENT: Every single cell in every single row MUST have a value - NEVER leave anything blank. If you don't see information in the table, SEARCH THE ENTIRE INVOICE TEXT.
Use "N/A" only as a last resort when information truly cannot be found.

SPECIFIC EXTRACTION RULES:
1. Item Code: Item Code: Extract only the product/item code (4-8 digits). Ignore order or delivery note numbers.

### Validation Checks:
- Ensure correct field identification.
- Review multilingual labels.
- Avoid misclassifying reference numbers as item codes.

### Final Output:
- Complete table—every field filled.
- Verify all values before finalizing.e table with ALL fields populated for EVERY row and VERIFY each field is appropriately filled before finalizing.

"""
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who extracts detailed and precise information from invoice texts. Ensure no data is missed and follow the instructions exactly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return completion.choices[0].message.content

def standardize_headers(headers):
    """
    Standardize header names across different PDFs
    """
    # Define standard header mappings
    header_mappings = {
        'Customer Number': 'Customer No',
        'Customer No.': 'Customer No',
        'Supplier Name': 'Supplier',
        'Total VAT': 'VAT',
        'Total VAT or VAT': 'VAT',
        'Total Amount of the Invoice': 'Invoice Total',
        'Payer Name': 'Payer Name',  # Keep consistent position
        'Date of Invoice': 'Invoice Date'  # Standardize date field
    }

    # Standard header order
    standard_headers = [
        'PO Number', 'Item Code', 'Description', 'UOM', 'Quantity',
        'Lot Number', 'Expiry Date', 'Mfg Date', 'Invoice No',
        'Unit Price', 'Total Price', 'Country', 'HS Code',
        'Invoice Date', 'Customer No', 'Payer Name', 'Currency',
        'Supplier', 'Invoice Total', 'VAT', 'Line Number'
    ]

    if 'Costing Number' not in standard_headers:
        standard_headers.append('Costing Number')

    # Map headers to standard names
    standardized = []
    for header in headers:
        if header in header_mappings:
            standardized.append(header_mappings[header])
        else:
            standardized.append(header)

    # Ensure all standard headers exist
    for header in standard_headers:
        if header not in standardized:
            standardized.append(header)

    return standard_headers  # Return the standardized list in correct order


def is_streamlit_cloud():
    """Detect if the app is running on Streamlit Cloud"""
    return "STREAMLIT_SHARING_MODE" in os.environ or "STREAMLIT_RUN_PATH" in os.environ


def check_ocr_availability():
    """Check which OCR engines are available and set flags"""
    # Initialize flags
    paddle_available = False
    easyocr_available = False
    keras_ocr_available = False
    
    # Check for EasyOCR
    try:
        import easyocr
        easyocr_available = True
        st.session_state.easyocr_available = True
    except ImportError:
        st.session_state.easyocr_available = False
    
    # Check for PaddleOCR (skip on Streamlit Cloud)
    if not is_streamlit_cloud():
        try:
            from paddleocr import PaddleOCR
            paddle_available = True
            st.session_state.paddle_available = True
        except ImportError:
            st.session_state.paddle_available = False
    else:
        st.session_state.paddle_available = False
    
    # Check for Keras-OCR
    try:
        import keras_ocr
        keras_ocr_available = True
        st.session_state.keras_ocr_available = True
    except ImportError:
        st.session_state.keras_ocr_available = False
    
    # Return availability summary
    return {
        "paddle": paddle_available,
        "easyocr": easyocr_available,
        "keras_ocr": keras_ocr_available
    }

# Run availability check at startup
if 'ocr_checked' not in st.session_state:
    st.session_state.ocr_checked = True
    available_engines = check_ocr_availability()
    
    # Display info about available OCR engines
    if is_streamlit_cloud():
        st.info("Running on Streamlit Cloud with EasyOCR" + 
                (" and Keras-OCR" if st.session_state.keras_ocr_available else ""))
    else:
        engines_str = ", ".join([engine for engine, available in available_engines.items() if available])
        if engines_str:
            st.info(f"Available OCR engines: {engines_str}")
        else:
            st.warning("No OCR engines available. Text extraction from scanned PDFs may not work.")

def main_app():
    st.title("🗂️ ASN Project - Data Extraction - AKI Company")
    
    if st.session_state.username == 'admin@akigroup.com':
        tab1, tab2, tab3 = st.tabs(["Upload & Process", "History", "User Tracking"])
    else:
        tab1, tab2 = st.tabs(["Upload & Process", "History"])
    
    with tab1:
        st.markdown(f"Welcome {st.session_state.username} to Our new tool to extract data from PDFs.")

        # Initialize session state variables
        if 'edited_df' not in st.session_state:
            st.session_state.edited_df = None
        if 'saved_df' not in st.session_state:
            st.session_state.saved_df = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'costing_numbers' not in st.session_state:
            st.session_state.costing_numbers = {}
        if 'uploaded_pdfs' not in st.session_state:
            st.session_state.uploaded_pdfs = []
        if 'grid_key' not in st.session_state:
            st.session_state.grid_key = 'data_editor_1'

        if st.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_pdfs = st.file_uploader(
                "📄 Upload PDF Invoices",
                type=["pdf"],
                accept_multiple_files=True,
                help="You can upload multiple Invoice files for different suppliers"
            )

        if uploaded_pdfs:
            st.session_state.uploaded_pdfs = uploaded_pdfs

        with col2:
            excel_file = st.text_input(
                "📊 Excel File Name",
                value="ASN_Result.xlsx",
                help="Enter the name for your output ASN file"
            )

        pdfs_to_process = st.session_state.uploaded_pdfs or uploaded_pdfs

        if pdfs_to_process:
            if st.session_state.edited_df is not None:
                try:
                    st.markdown("### 📝 Extracted and Edited Data")
                    
                    # Add search functionality before data editor
                    search_query = st.text_input("🔍 Search in table:", key=f"search_input_{st.session_state.grid_key}")
                    
                    # Get the data to display
                    display_df = st.session_state.edited_df.copy()
                    
                    # Apply search filter if there's a query
                    if search_query:
                        mask = display_df.astype(str).apply(
                            lambda x: x.str.contains(search_query, case=False)
                        ).any(axis=1)
                        display_df = display_df[mask]
                    
                    # Create the editable dataframe
                    edited_df = st.data_editor(
                        display_df,
                        use_container_width=True,
                        num_rows="dynamic",
                        height=600,
                        key=st.session_state.grid_key,
                        column_config={
                            col: st.column_config.Column(
                                width="auto",
                                help=f"Edit {col}"
                            ) for col in display_df.columns
                        }
                    )
                    
                    # Update session state with edited data
                    st.session_state.edited_df = edited_df
                    
                    # Display data info
                    st.markdown(f"**Total Rows:** {len(edited_df)} | **Total Columns:** {len(edited_df.columns)}")
                    
                    # Save Changes Button
                    if st.button("💾 Save Table Changes", key="save_changes_existing"):
                        st.session_state.saved_df = edited_df.copy()
                        # Save files to storage
                        save_path = save_uploaded_files(
                            st.session_state.username,
                            st.session_state.uploaded_pdfs,
                            st.session_state.saved_df
                        )
                        if save_path:
                            st.success("✅ Changes saved successfully and files stored!")
                    
                    # Download section
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            edited_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Excel",
                            data=buffer.getvalue(),
                            file_name=excel_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_existing"
                        )
                
                except Exception as e:
                    st.error(f"Error displaying existing table: {str(e)}")
                    st.error(traceback.format_exc())
            else:
                # Costing Numbers Input
                st.markdown("### Enter the Costing Numbers")
                for pdf in pdfs_to_process:
                    if pdf.name not in st.session_state.costing_numbers:
                        st.session_state.costing_numbers[pdf.name] = ""

                    st.session_state.costing_numbers[pdf.name] = st.text_input(
                        f"Costing Number for {pdf.name}",
                        value=st.session_state.costing_numbers[pdf.name],
                        key=f"costing_{pdf.name}"
                    )

                # Extract Button
                if st.button("Extract Information from Invoices"):
                    try:
                        # Initialize processing
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_files = len(pdfs_to_process)
                        total_rows_processed = 0
                        
                        all_data = []
                        all_headers = None

                        # Clean up any leftover temporary files
                        cleanup_temp_files()

                        # Process each PDF file
                        for idx, uploaded_pdf_file in enumerate(pdfs_to_process):
                            tmp_path = None
                            try:
                                status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_pdf_file.name}")
                                
                                # Create temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(uploaded_pdf_file.getvalue())
                                    tmp_path = tmp_file.name
                                
                                # Process the file
                                with st.spinner(f"Extracting text from {uploaded_pdf_file.name}..."):
                                    pdf_text = extract_text_pdf(tmp_path)
                                
                                if pdf_text:
                                    with st.spinner("Processing extracted text using AKI GPT..."):
                                        invoice_info = using_groq(pdf_text)
                                        
                                        # Process the invoice using process_invoice_lines
                                        headers, data_rows = process_invoice_lines(
                                            invoice_info, 
                                            st.session_state.costing_numbers.get(uploaded_pdf_file.name, "")
                                        )
                                        
                                        if headers and data_rows:
                                            # Set headers if not set yet
                                            if all_headers is None:
                                                all_headers = headers
                                            
                                            # Add all data rows to our collection
                                            all_data.extend(data_rows)
                                            
                                            # Update row count
                                            total_rows_processed += len(data_rows)
                            
                            except Exception as e:
                                st.error(f"Error processing file {uploaded_pdf_file.name}: {str(e)}")
                            
                            finally:
                                # Clean up temporary file
                                if tmp_path and os.path.exists(tmp_path):
                                    try:
                                        # Close any open file handles
                                        gc.collect()
                                        os.unlink(tmp_path)
                                    except Exception as cleanup_error:
                                        st.warning(f"Could not remove temporary file {tmp_path}: {cleanup_error}")
                                        # Add to a list of files to clean up later
                                        if 'cleanup_files' not in st.session_state:
                                            st.session_state.cleanup_files = []
                                        st.session_state.cleanup_files.append(tmp_path)
                            
                            progress_bar.progress((idx + 1) / total_files)
                            gc.collect()
                        
                        # Update tracking with total files and rows processed
                        update_user_tracking(
                            username=st.session_state.username,
                            files_uploaded=total_files,
                            rows_processed=total_rows_processed
                        )
                                                    
                        if all_data and all_headers:
                            try:
                                # Debug information
                                # st.write(f"DEBUG - Number of headers: {len(all_headers)}")
                                # st.write(f"DEBUG - Number of data rows: {len(all_data)}")
                                # st.write(f"DEBUG - First row length: {len(all_data[0]) if all_data else 0}")
                                
                                # Validate and clean data before creating DataFrame
                                cleaned_data = []
                                for idx, row in enumerate(all_data):
                                    if len(row) > len(all_headers):
                                        st.write(f"DEBUG - Trimming row {idx} from {len(row)} to {len(all_headers)} columns")
                                        cleaned_data.append(row[:len(all_headers)])
                                    elif len(row) < len(all_headers):
                                        st.write(f"DEBUG - Padding row {idx} from {len(row)} to {len(all_headers)} columns")
                                        padded_row = row + [''] * (len(all_headers) - len(row))
                                        cleaned_data.append(padded_row)
                                    else:
                                        cleaned_data.append(row)
                                
                                # Create DataFrame with cleaned data
                                df = pd.DataFrame(cleaned_data, columns=all_headers)
                                st.session_state.edited_df = df.copy()
                                
                                # Verify DataFrame creation
                                st.write(f"DEBUG - DataFrame shape: {df.shape}")
                                # Create DataFrame with all processed data
                                df = pd.DataFrame(all_data, columns=all_headers)
                                st.session_state.edited_df = df.copy()
                                
                                try:
                                    # Create an editable table
                                    st.markdown("### 📝 Edit Extracted Data")
                                    
                                    edited_df = st.data_editor(
                                        st.session_state.edited_df,
                                        use_container_width=True,
                                        num_rows="dynamic",
                                        column_config={col: st.column_config.Column(
                                            width="auto",
                                            help=f"Edit {col}"
                                        ) for col in st.session_state.edited_df.columns},
                                        height=600,
                                        key=f'grid_{datetime.now().strftime("%Y%m%d%H%M%S")}'
                                    )
                                                            
                                    # Add search functionality
                                    search = st.text_input("🔍 Search in table:", key="search_input")
                                    if search:
                                        mask = edited_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                                        filtered_df = edited_df[mask]
                                    else:
                                        filtered_df = edited_df 
                            
                                    # Update the session state with edited data
                                    st.session_state.edited_df = edited_df
                                    st.markdown(f"**Total Rows:** {len(filtered_df)} | **Total Columns:** {len(filtered_df.columns)}")

                                    # Save Changes Button
                                    if st.button("💾 Save Table Changes", key="save_changes"):
                                        st.session_state.saved_df = edited_df.copy()
                                        # Save files to storage
                                        save_path = save_uploaded_files(
                                            st.session_state.username,
                                            st.session_state.uploaded_pdfs,
                                            st.session_state.saved_df
                                        )
                                        if save_path:
                                            st.success("✅ Changes saved successfully and files stored!")
                                            
                                            # Update tracking
                                            update_user_tracking(
                                                username=st.session_state.username,
                                                files_uploaded=len(pdfs_to_process),
                                                rows_processed=len(edited_df)
                                            )
                                    
                                    # Create download buttons section
                                    col1, col2, col3 = st.columns(3)
                                    
                                    # Original data download button
                                    with col1:
                                        buffer_original = io.BytesIO()
                                        with pd.ExcelWriter(buffer_original, engine='openpyxl') as writer:
                                            df.to_excel(writer, index=False)
                                        excel_data_original = buffer_original.getvalue()
                                        
                                        st.download_button(
                                            label="📥 Download Original Excel",
                                            data=excel_data_original,
                                            file_name=f"original_{excel_file}",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key="download_original"
                                        )
                                    
                                    # Saved version download button
                                    with col2:
                                        if st.session_state.saved_df is not None:
                                            buffer_saved = io.BytesIO()
                                            with pd.ExcelWriter(buffer_saved, engine='openpyxl') as writer:
                                                st.session_state.saved_df.to_excel(writer, index=False)
                                            excel_data_saved = buffer_saved.getvalue()
                                            
                                            st.download_button(
                                                label="📥 Download Saved Excel",
                                                data=excel_data_saved,
                                                file_name=f"saved_{excel_file}",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                key="download_saved"
                                            )
                                        else:
                                            st.info("Save your changes first!")
                                    
                                    # Current state download button
                                    with col3:
                                        buffer_current = io.BytesIO()
                                        with pd.ExcelWriter(buffer_current, engine='openpyxl') as writer:
                                            edited_df.to_excel(writer, index=False)
                                        
                                        st.download_button(
                                            label="📥 Download Current Excel",
                                            data=buffer_current.getvalue(),
                                            file_name=f"current_{excel_file}",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key="download_current"
                                        )
                                    
                                    # Add download options explanation
                                    st.markdown("""
                                    ### 💡 Download Options:
                                    - **Original Excel**: The data exactly as extracted from PDFs
                                    - **Saved Excel**: Your last saved changes
                                    - **Current Excel**: Current state of the table including unsaved changes
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"Error displaying table and buttons: {str(e)}")
                                    st.error(traceback.format_exc())
                            except Exception as e:
                                st.error(f"Error creating DataFrame: {str(e)}")
                                st.error(traceback.format_exc())
                        else:
                            st.error("No valid data could be extracted from the invoices")
                            
                    except Exception as e:
                        st.error(f"Error in main processing: {str(e)}")
                        st.error(traceback.format_exc())

    with tab2:
        # st.markdown("### 📂 Previous Uploads")
        # user_uploads = get_user_uploavds(st.session_state.username)
        display_history_tab()

    if st.session_state.username == 'admin@akigroup.com':
        with tab3:
            admin_tracking_tab()

def main():
    verify_storage_setup()
    setup_storage()
    init_user_tracking()
    
    # Check for share parameter first
    share_hash = st.query_params.get('share')
    
    if share_hash:
        st.title("🔗 Shared File Viewer")
        check_shared_file()
    else:
        if not st.session_state.logged_in:
            login_page()
        else:
            main_app()
def handle_pdf_error(e, pdf_name):
    """Handle PDF processing errors with appropriate messages and actions"""
    error_msg = str(e).lower()
    
    if "poppler" in error_msg:
        st.error(f"""Error processing {pdf_name}: Poppler is not installed or not found in PATH. 
        Please ensure Poppler is properly installed on the server.""")
    elif "permission" in error_msg:
        st.error(f"Permission error while processing {pdf_name}. Please check file permissions.")
    else:
        st.error(f"Error processing {pdf_name}: {str(e)}")
    
    # Add retry button
    if st.button("🔄 Retry Processing", key=f"retry_{hash(pdf_name)}"):
        # Reset processing-related session states
        st.session_state.edited_df = None
        st.session_state.saved_df = None
        st.session_state.processing_complete = False
        st.session_state.costing_numbers = {}
        st.rerun()

# def process_with_ocr(pdf_path, pdf_name):
#     """Process PDF with OCR including error handling and memory optimization"""
#     try:
#         # Add a message about resource usage
#         st.info(f"Processing {pdf_name} with OCR. This may take some time and resources.")
        
#         # Process in chunks if needed
#         text = extract_text_pdf(pdf_path)
        
#         if not text:
#             st.warning(f"No text could be extracted from {pdf_name}. The file might be corrupted or empty.")
#             if st.button("🔄 Retry This File", key=f"retry_empty_{hash(pdf_name)}"):
#                 st.session_state.edited_df = None
#                 st.rerun()
#             return None
        
#         # Memory cleanup after processing
#         gc.collect()
        
#         return text
#     except Exception as e:
#         handle_pdf_error(e, pdf_name)
#         return None
    
# def process_with_ocr(pdf_path, pdf_name):
#     """Process PDF with OCR including error handling and recovery options"""
#     # Commenting out OCR processing while keeping structure
#     """
#     try:
#         text = extract_text_pdf(pdf_path)
#         if not text:
#             st.warning(f"No text could be extracted from {pdf_name}. The file might be corrupted or empty.")
#             if st.button("🔄 Retry This File", key=f"retry_empty_{hash(pdf_name)}"):
#                 st.session_state.edited_df = None
#                 st.rerun()
#             return None
#         return text
#     except Exception as e:
#         handle_pdf_error(e, pdf_name)
#         return None
#     """
#     st.info("We are working on it now")
#     return None
# def process_with_ocr(pdf_path, pdf_name):
#     """Process PDF with OCR including error handling and recovery options"""
#     try:
#         text = extract_text_pdf(pdf_path)
#         if not text:
#             st.warning(f"No text could be extracted from {pdf_name}. The file might be corrupted or empty.")
#             # Add retry button for empty text
#             if st.button("🔄 Retry This File", key=f"retry_empty_{hash(pdf_name)}"):
#                 st.session_state.edited_df = None
#                 st.rerun()
#             return None
#         return text
#     except Exception as e:
#         handle_pdf_error(e, pdf_name)
#         return None


def add_reset_button():
    """Add a reset button that clears current processing state but maintains login"""
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Start New Process", key="reset_process"):
            # Reset processing-related session states while keeping login info
            st.session_state.edited_df = None
            st.session_state.saved_df = None
            st.session_state.processing_complete = False
            st.session_state.costing_numbers = {}
            st.session_state.uploaded_pdfs = []
            st.rerun()   
if __name__ == "__main__":
    main()
