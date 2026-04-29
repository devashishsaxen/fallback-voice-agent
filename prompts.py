# prompts.py - Edit conversation text here without touching the main code

class ConversationFlow:
    """All conversation prompts organized by state"""
    
    # Initial greeting
    GREETING = "Hi, this is Riya from Futuresoft Consultancy. We are hiring voice and chat profiles for companies like British Telecom, Teleperformance, and Wipro. Are you interested?"
    
    # Interest check responses
    INTEREST_YES = "Great! Are you a Fresher or Experienced?"
    INTEREST_NO = "I understand. Thank you for your time. Have a great day!"
    INTEREST_FALLBACK = "Please say Yes if interested, or No."
    
    # Experience check
    FRESHER_PATH = "What's your highest qualification?"
    EXPERIENCED_PATH = "Please share your experience briefly."
    EXPERIENCE_FALLBACK = "Are you a Fresher or Experienced?"
    
    # After qualification/experience details
    CUSTOMER_STORY_PROMPT = "Good. Describe a memorable customer interaction in 10 to 12 sentences."
    
    # Customer story validation
    CUSTOMER_STORY_SUCCESS = "Excellent. Tell me about a recent festival you celebrated in 10 to 12 sentences."
    CUSTOMER_STORY_RETRY = "Please speak for at least 10 sentences."
    
    # Festival story validation  
    FESTIVAL_STORY_SUCCESS = "Perfect! Our HR team will contact you soon."
    FESTIVAL_STORY_RETRY = "Please describe the festival in 10 to 12 sentences."
    
    # Rejection message (used when retries exhausted)
    COMMUNICATION_REJECTION = "Sorry, we need candidates with good communication skills."
    
    # End states
    FINAL_GOODBYE = "Thank you for your time. Have a great day!"
    REPEAT_REQUEST = "Could you please repeat that?"

# Keywords for intent detection (can also be customized here)
class IntentKeywords:
    NEGATIVE = ['no', 'not', 'nah', 'nope']
    POSITIVE = ['yes', 'yeah', 'sure', 'interested', 'ok']
    FRESHER = ['fresher', 'fresh', 'student']
    EXPERIENCED = ['experience', 'experienced', 'worked']