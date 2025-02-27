import streamlit as st
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import json
from geopy.geocoders import Nominatim
from gtts import gTTS
import base64
import tempfile
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# 🏷️ Known booking URLs for providers
BOOKING_URLS = {
    "Air India": "https://www.airindia.com/",
    "Delta": "https://www.delta.com/",
    "Emirates": "https://www.emirates.com/",
    "Uber": "https://www.uber.com/global/en/price-estimate/",
    "RedBus": "https://www.redbus.in/",
    "IRCTC": "https://www.irctc.co.in/",
}

# Function to get booking URL (fallback to Google Search)
def get_booking_url(provider):
    return BOOKING_URLS.get(provider, f"https://www.google.com/search?q={provider}+booking")

# Updated prompt with descriptions
prompt_template = PromptTemplate(
    input_variables=["source", "destination", "preference"],
    template="""
    You are an AI travel assistant. A user wants to travel from {source} to {destination} 
    with a preference for {preference}. Provide travel options for cab, train, bus, and flight. 
    Return the response strictly in **valid JSON format**, without any extra text or explanations.
    Ensure prices are in the correct currency based on the source country.
    
    If a travel mode (e.g., bus, train) is **not possible**, provide a clear reason in the "description" field.
    
    Example Output:
    ```json
    {{
        "flights": [
            {{
                "provider": "Air India",
                "price": 5000,
                "duration": "3h",
                "notes": "Non-stop flight",
                "description": "Direct flights available from major airports.",
                "booking_url": "https://www.airindia.com/"
            }}
        ],
        "trains": [
            {{
                "provider": "IRCTC",
                "price": 1500,
                "duration": "5h",
                "notes": "Sleeper class available",
                "description": "Trains operate between major cities within the country.",
                "booking_url": "https://www.irctc.co.in/"
            }},
            {{
                "provider": "Unavailable",
                "price": 0,
                "duration": "N/A",
                "notes": "Not applicable",
                "description": "Train travel between {source} and {destination} is not possible due to geographical barriers."
            }}
        ]
    }}
    ```
    Respond **only** with the JSON block.
    """
)

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Telugu" : "te"
}

def get_language_name(language_code):
    for name, code in LANGUAGES.items():
        if code == language_code:
            return name
    return "English"

# Function to generate a summary description of travel options
def generate_travel_summary(source, destination, travel_data, language="en"):
    """Generate a comprehensive summary of travel options between two locations"""
    
    # Create a prompt for the summary
    summary_prompt = PromptTemplate(
        input_variables=["source", "destination"],
        template = """
    You are an AI travel assistant. A user wants to travel from {source} to {destination}. 
    Provide travel options for cab, train, bus, and flight with estimated prices and travel times.
    Strictly don't include summary table, but include recommendations and precautionary notes
    """)
    
    summary_chain = {"source": RunnablePassthrough(), 
         "destination": RunnablePassthrough()} | summary_prompt | llm
    
    # Get summary from Gemini
    response2 = summary_chain.invoke({"source":source,"destination":destination})
    summary = response2.content
    
    # If a different language is requested, translate the summary
    if language != "en":
        translation_prompt = f"""
        Translate the following travel summary to {get_language_name(language)}
        strictly don't include any breakdown of key translations. 
        
        
        {summary}
        """
        response2 = llm.invoke(translation_prompt)
        summary = response2.content
    
    return summary

def text_to_speech(text, language="en"):
    """Convert text to speech and return the audio as a base64 string"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(fp.name)
        with open(fp.name, "rb") as audio_file:
            audio_bytes = audio_file.read()
        os.unlink(fp.name)  # Remove the temporary file
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html


if "language" not in st.session_state:
    st.session_state.language = "en"
if "summary" not in st.session_state:
    st.session_state.summary = None

# Currency Mapping
CURRENCY_MAP = {
    "India": "₹", "United States": "$", "United Kingdom": "£",
    "France": "€", "Germany": "€", "Italy": "€", "Spain": "€",
    "Japan": "¥", "Canada": "C$", "Australia": "A$", "UAE": "د.إ",
    "China": "¥", "Russia": "₽", "South Korea": "₩", "Brazil": "R$"
}
chain = {"source": RunnablePassthrough(), 
         "destination": RunnablePassthrough(), 
         "preference": RunnablePassthrough()} | prompt_template | llm

#Function to detect country from city name
def get_country_from_city(city):
    geolocator = Nominatim(user_agent="geoapiTravel")
    location = geolocator.geocode(city)
    if location:
        return location.address.split(",")[-1].strip()
    return None

# Function to determine the currency symbol based on the country
def get_currency_symbol(source):
    country = get_country_from_city(source)
    return CURRENCY_MAP.get(country, "$")  # Default to USD

# Sorting function
def sort_travel_options(options, sort_by):
    if sort_by == "Lowest Price":
        return sorted(options, key=lambda x: x["price"])
    elif sort_by == "Shortest Duration":
        return sorted(options, key=lambda x: int(''.join(filter(str.isdigit, x["duration"]))))
    return options  # Default order

# Maintain state across reruns
if "response" not in st.session_state:
    st.session_state.response = None
if "currency_symbol" not in st.session_state:
    st.session_state.currency_symbol = "$"
if "sort_by" not in st.session_state:
    st.session_state.sort_by = "Default"  # Default sort order


st.set_page_config(page_title="PathPlanner",layout="centered",page_icon="🎯")

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* App title styling */
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0;
        text-align: center;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #757575;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Input form styling */
    .input-form {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .form-header {
        color: #333;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    
    /* Input field labels */
    .input-label {
        font-weight: 500;
        margin-bottom: 5px;
        color: #555;
    }
    
    /* Search button styling */
    .search-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.3s;
        margin-top: 15px;
    }
    
    .search-button:hover {
        background-color: #1565C0;
    }
    
    /* Travel cards styling */
    .travel-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .travel-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    .provider-name {
        color: #007BFF;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .price-duration {
        display: flex;
        gap: 15px;
        margin-bottom: 8px;
    }
    .price {
        background-color: #e6f2ff;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .duration {
        background-color: #f0f0f0;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .notes {
        font-style: italic;
        margin-bottom: 8px;
        color: #6c757d;
    }
    .description {
        margin-bottom: 12px;
        color: #212529;
    }
    .book-button {
        display: inline-block;
        background-color: #28a745;
        color: white;
        padding: 8px 15px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: bold;
        transition: background-color 0.2s;
    }
    .book-button:hover {
        background-color: #218838;
    }
    .unavailable {
        background-color: #f8f9fa;
        border: 1px dashed #dee2e6;
    }
    .unavailable .provider-name {
        color: #6c757d;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* Override Streamlit's default input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 12px 15px;
    }
    
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    /* Make inputs full width */
    div[data-testid="stVerticalBlock"] > div.element-container > div.stTextInput,
    div[data-testid="stVerticalBlock"] > div.element-container > div.stSelectbox {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI with improved styling
st.markdown('<h1 class="app-title">🚀 PathPlanner-AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Find the best travel options with AI!</p>', unsafe_allow_html=True)

# Create a container for the input form
st.markdown('<div class="input-form">', unsafe_allow_html=True)
st.markdown('<h3 class="form-header">Enter Your Trip Details</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="input-label">📍 Source Location</p>', unsafe_allow_html=True)
    source = st.text_input("Input Label", placeholder="Enter starting point", key="source", label_visibility="collapsed")

with col2:
    st.markdown('<p class="input-label">📍 Destination</p>', unsafe_allow_html=True)
    destination = st.text_input("Input Label", placeholder="Enter destination", key="destination", label_visibility="collapsed")

col3, col4,col5 = st.columns(3)
with col3:
    st.markdown('<p class="input-label">🎯 Travel Preference</p>', unsafe_allow_html=True)
    preference = st.selectbox("Preference", ["Cheapest", "Fastest", "Comfortable", "Eco-friendly"], key="preference", label_visibility="collapsed")

with col4:
    st.markdown('<p class="input-label">🔽 Sort Results By</p>', unsafe_allow_html=True)
    # Sorting Preference
    def update_sorting():
        st.session_state.sort_by = st.session_state.sort_selection
    
    st.selectbox("sorting", ["Default", "Lowest Price", "Shortest Duration"], key="sort_selection", on_change=update_sorting, label_visibility="collapsed")
with col5:
    st.markdown('<p class="input-label">🌐 Language</p>', unsafe_allow_html=True)
    # Language selection
    def update_language():
        st.session_state.language = LANGUAGES[st.session_state.language_selection]
    language_options = list(LANGUAGES.keys())
    st.selectbox("language options", language_options, key="language_selection", index=language_options.index("English"), on_change=update_language, label_visibility="collapsed")
    

# Search button with custom styling
search_button_html = '<button class="search-button">🔍 Find Travel Options</button>'
search_clicked = st.button("🔍 Find Travel Options", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if search_clicked:
    if source and destination:
        st.markdown(f'<h2 class="section-header">🌍 Travel Options from {source} to {destination}</h2>', unsafe_allow_html=True)
        with st.spinner("Finding the best travel options for you..."):
            response = chain.invoke({"source": source, "destination": destination, "preference": preference})
            # st.write(f"{source} to {destination}")
            response = response.content
            try:
                clean_response = response.strip().strip("```json").strip("```")
                travel_data = json.loads(clean_response)
                st.session_state.response = travel_data  # Store response in session state
                st.session_state.currency_symbol = get_currency_symbol(source)  # Store currency
                language = st.session_state.language
                summary = generate_travel_summary(source, destination, travel_data, language)
                st.session_state.summary = summary
            except json.JSONDecodeError as e:
                st.error("❌ Oops! AI response is not in the correct format. Check logs for details.")
                st.text(response)
                st.text(f"JSON Error: {str(e)}")

if st.session_state.summary:
    st.markdown("""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 30px; border-left: 5px solid #1E88E5;">
        <h3 style="color: #1E88E5; margin-top: 0;">Travel Summary</h3>
        <div id="summary-text">
    """, unsafe_allow_html=True)
    
    st.markdown(st.session_state.summary, unsafe_allow_html=True)
    
    # Add accessibility features
    st.markdown("""
        </div>
        <div style="margin-top: 15px;">
            <details>
                <summary style="cursor: pointer; color: #1E88E5; font-weight: bold;">🔊 Listen to this summary (Accessibility Feature)</summary>
    """, unsafe_allow_html=True)
    
    # Generate TTS audio
    audio_html = text_to_speech(st.session_state.summary, st.session_state.language)
    st.markdown(audio_html, unsafe_allow_html=True)
    
    st.markdown("""
            </details>
        </div>
    </div>
    """, unsafe_allow_html=True)
# Keep response displayed & sort without full UI refresh
if st.session_state.response:
    travel_data = st.session_state.response
    currency_symbol = st.session_state.currency_symbol
    sort_by = st.session_state.sort_by  # Use stored sorting choice

    def display_options(title, key, icon):
        st.markdown(f'<h3 style="color: #1E88E5;background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 30px; border-left: 5px solid green; margin-top: 0;">{icon} {title}</h3>', unsafe_allow_html=True)
        options = travel_data.get(key, [])
        sorted_options = sort_travel_options(options, sort_by)  # Sort based on user choice

        for item in sorted_options:
            provider = item["provider"]
            price = item["price"]
            duration = item["duration"]
            notes = item["notes"]
            description = item.get("description", "No description available.")
            is_unavailable = provider.lower() == "unavailable"
            
            # Only get booking URL if the option is available
            booking_url = "" if is_unavailable else (item.get("booking_url") or get_booking_url(provider))

            # Add card classes based on availability
            card_class = "travel-card unavailable" if is_unavailable else "travel-card"
            
            html_content = f"""
                <div class="{card_class}">
                    <div class="provider-name">{provider}</div>
                    <div class="price-duration">
                        <span class="price">💰 {currency_symbol}{price}</span>
                        <span class="duration">⏳ {duration}</span>
                    </div>
                    <div class="notes">📝 {notes}</div>
                    <div class="description">ℹ️ {description}</div>
            """
            
            # Only add booking button if the option is available
            if not is_unavailable:
                html_content += f'<a href="{booking_url}" target="_blank" class="book-button">Book Now</a>'
                
            html_content += "</div>"
            
            st.markdown(html_content, unsafe_allow_html=True)

    display_options("Flight Options", "flights", "✈️")
    st.divider()
    display_options("Train Options", "trains", "🚆")
    st.divider()
    display_options("Bus Options", "buses", "🚌")
    st.divider()
    display_options("Cab Options", "cabs", "🚖")