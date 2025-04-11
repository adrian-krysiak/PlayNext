import os
import sys


# Run the Streamlit app from src/app/streamlit.py
streamlit_script = os.path.join('src', 'app', 'streamlit.py')
os.system(f"streamlit run {streamlit_script}")