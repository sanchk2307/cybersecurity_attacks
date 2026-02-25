import pandas as pd
import numpy as np
import re
from collections import Counter
from scipy.stats import entropy
import base64
import binascii
from lingua import Language, LanguageDetectorBuilder
import os

# Import the Google Cloud Translation library.
from google.cloud import translate_v3

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")



class PayloadAnalyzer:
    def __init__(self, df, payload_col='payload'):
        self.df = df
        self.payload_col = payload_col
    
    
    def detect_payload_language(self, payload_text) -> str:
        languages = Language.LATIN
        detector = LanguageDetectorBuilder.from_languages(languages).build()
        language = detector.detect_language_of(payload_text)
        if language != Language.LATIN:
            return "not_latin"
        else:
            return "latin"
    
    def payload_translate(self, payload_text) -> dict | str:
        """Detect language and translate to English"""
        
        # 1. Detect language
        try:
            lang = self.detect_payload_language(payload_text)
            print(f"Detected: {lang}")
        except:
            lang = 'unknown'
        
        # 2. Translate to English if not already English
        if lang != 'en':
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(payload_text)
                return translated
            except Exception as e:
                return {'error': str(e)}
            
    def is_lorem_ipsum(self, payload_text) -> bool:
        """Detect if text is Lorem Ipsum or similar generated text"""
        lorem_markers = [
            'lorem', 'ipsum', 'dolor', 'sit amet', 'consectetur', 
            'adipiscing', 'elit', 'sed do', 'eiusmod', 'tempor',
            'incididunt', 'labore', 'dolore', 'magna', 'aliqua'
        ]
        text_lower = payload_text.lower()
        matches = sum(1 for marker in lorem_markers if marker in text_lower)
        return matches >= 3 
    
    def extract_features(self, payload_text):
        """Extract feature from payload"""
        features = {}
        
        # If entropy is high -> likely obfuscation
        if len(payload_text) > 0:
            freq = Counter(payload_text)
            features['entropy'] = entropy(list(freq.values()), base=2)
        else:
            features['entropy'] = 0