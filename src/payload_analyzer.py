import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
import string
from scipy.stats import entropy
import math
from typing import Dict, Any, Optional

# Payload Decyphering libraries
from confusable_homoglyphs import confusables as cnf
from homoglyphs import Homoglyphs
from unidecode import unidecode
import ftfy
import chardet


class PayloadAnalyzer(object):
    def __init__(self):
        """
        Initialize the class        
        """
        self.hogl = Homoglyphs()
            
    def is_lorem_ipsum(self, payload_text: str) -> bool:
        """Detect if text is Lorem Ipsum or similar generated text
        by calculating the number of occurrences of lorem_markers
        in the the payload text
        
        Returns a bool according to matches  >= 3
        """
        lorem_markers = [
            'lorem', 'ipsum', 'dolor', 'sit amet', 'consectetur', 
            'adipiscing', 'elit', 'sed do', 'eiusmod', 'tempor',
            'incididunt', 'labore', 'dolore', 'magna', 'aliqua'
        ]
        text_lower = payload_text.lower()
        matches = sum(1 for marker in lorem_markers if marker in text_lower)
        return matches >= 3 
    
    # Shannon entropy is a fundamental measure in information theory that quantifies
    # the average information content or uncertainty in a dataset. Entropy in information 
    # theory is directly analogous to the entropy in statistical thermodynamics.
    # The analogy results when the values of the random variable designate energies of microstates,
    # so Gibbs's formula for the entropy is formally identical to Shannon's formula.
    # Entropy has relevance to other areas of mathematics such as combinatorics and machine learning and cybersecurity 
    # https://medium.com/@mayankLearns/entropy-the-secret-ingredient-of-cybersecurity-c31b48aa21b8
    
    def calculate_entropy(self, payload_text: str) -> float: 
        """
        Calculates Shannon entropy via scipy package method "entropy"
        Returns float
        """
        if not payload_text or len(payload_text) == 0:
            return 0.0
        freq = Counter(payload_text.lower().replace(' ','').replace('.','').replace('\n',''))
        result = entropy(list(freq.values()), base=2)
        return 0.0 if np.isnan(result) else float(result)
    
    def calculate_relative_entropy(self, payload_text: str, base=2) -> float: 
        '''
        Calculate the relative entropy (Kullback-Leibler divergence) between data and expected values.
        Credit red canary: https://redcanary.com/blog/threat-detection/threat-hunting-entropy/
        '''
        entropy = 0.0
        length = len(payload_text) * 1.0
        if length > 0:
            cnt = Counter(payload_text.lower().replace(' ','').replace('.','').replace('\n',''))
            
            # These probability numbers were calculated from the Alexa Top
            # 1 million domains as of September 15th, 2017. TLDs and instances
            # of 'www' were removed so 'www.google.com' would be treated as
            # 'google' and 'images.google.com' would be 'images.google'.
            probabilities = {
                '-' : 0.013342298553905901,
                '_' : 9.04562613824129e-06,
                '0' : 0.0024875471880163543,
                '1' : 0.004884638114650296,
                '2' : 0.004373560237839663,
                '3' : 0.0021136613076357144,
                '4' : 0.001625197496170685,
                '5' : 0.0013070929769758662,
                '6' : 0.0014880054997406921,
                '7' : 0.001471421851820583,
                '8' : 0.0012663876593537805,
                '9' : 0.0010327089841158806,
                'a' : 0.07333590631143488,
                'b' : 0.04293204925644953,
                'c' : 0.027385633133525503,
                'd' : 0.02769469202658208,
                'e' : 0.07086192756262588,
                'f' : 0.01249653250998034,
                'g' : 0.038516276096631406,
                'h' : 0.024017645001386995,
                'i' : 0.060447396668797414,
                'j' : 0.007082725266242929,
                'k' : 0.01659570875496002,
                'l' : 0.05815885325582237,
                'm' : 0.033884915513851865,
                'n' : 0.04753175014774523,
                'o' : 0.09413783122067709,
                'p' : 0.042555148167356144,
                'q' : 0.0017231917793349655,
                'r' : 0.06460084667060655,
                's' : 0.07214640647425614,
                't' : 0.06447722311338391,
                'u' : 0.034792493336388744,
                'v' : 0.011637198026847418,
                'w' : 0.013318176884203925,
                'x' : 0.003170491961453572,
                'y' : 0.016381628936354975,
                'z' : 0.004715786426736459
                }

            for char, count in cnt.items():
                    observed = count / length
                    expected = probabilities[char]
                    entropy += observed * math.log((observed / expected), base)
        return 0.0 if np.isnan(entropy) else float(entropy)
    
    def detect_statistical_anomalies(self, payload_text: str) -> dict:
        if pd.isna(payload_text) or not payload_text:
            return {
                'chi_square': None,
                'p_value': None,
                'chi_square_suspicious': False,
                'suspicion_level': 'UNKNOWN',
                'top_deviations': {},
                'unusual_chars': {},
                'avg_word_length': 0.0,
                'word_length_suspicious': False,
                'total_characters': 0
            }
        
        text = str(payload_text)
        
        # Expected frequencies for Latin text
        latin_expected = {
        'a': 0.079, 'b': 0.015, 'c': 0.040, 'd': 0.035, 'e': 0.119,
        'f': 0.010, 'g': 0.013, 'h': 0.009, 'i': 0.115, 'l': 0.035,
        'm': 0.060, 'n': 0.055, 'o': 0.050, 'p': 0.028, 'q': 0.014,
        'r': 0.062, 's': 0.075, 't': 0.080, 'u': 0.070, 'v': 0.010,
        'x': 0.005, 'z': 0.001
        }
        
        text_lower = text.lower()
        total_alpha = sum(1 for c in text_lower if c.isalpha())
        
        if total_alpha == 0:
            return {
                'chi_square': None,
                'p_value': None,
                'chi_square_suspicious': False,
                'suspicion_level': 'UNKNOWN',
                'top_deviations': {},
                'unusual_chars': {},
                'avg_word_length': 0.0,
                'word_length_suspicious': False,
                'total_characters': 0
            }
        
        # Get COUNTS (not frequencies)
        actual_counts = Counter(c for c in text_lower if c.isalpha())
        
        chi_square = 0
        deviations = {}
        
        for char, expected_freq in latin_expected.items():
            observed_count = actual_counts.get(char, 0)
            expected_count = expected_freq * total_alpha  # CRITICAL: Convert to count!
            
            if expected_count > 0:
                contribution = ((observed_count - expected_count) ** 2) / expected_count
                chi_square += contribution
                
                deviations[char] = {
                    'expected': round(expected_freq, 4),
                    'actual': round(observed_count / total_alpha, 4),
                    'deviation': round(abs(observed_count / total_alpha - expected_freq), 4),
                    'chi_contribution': round(contribution, 4)
                }
        
        # Degrees of freedom = number of categories - 1
        df = len(latin_expected) - 1  # df = 19
        
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)
        
        # Critical values
        critical_09 = stats.chi2.ppf(0.9, df)
        critical_05 = stats.chi2.ppf(0.95, df)  # ~30.14
        critical_01 = stats.chi2.ppf(0.99, df)  # ~36.19
        
        # Suspicion level
        if chi_square > critical_01:
            suspicion_level = 'VERYHIGH'
        elif chi_square > critical_05:
            suspicion_level = 'HIGH'
        elif chi_square > critical_09:
            suspicion_level = 'MEDIUM'    
        else:
            suspicion_level = 'LOW'
        
        # Unusual characters
        unusual_chars = {
            char: round(count / total_alpha, 4)
            for char, count in actual_counts.items()
            if char not in latin_expected and char in string.ascii_lowercase
        }
        
        # Word length analysis
        words = text.split()
        word_lengths = [len(w) for w in words if w]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        word_length_suspicious = avg_word_length < 3 or avg_word_length > 10
        
        return {
            'chi_square': round(chi_square, 4),
            'p_value': round(p_value, 6),
            'chi_square_suspicious': chi_square > critical_09,
            'suspicion_level': suspicion_level,
            'top_deviations': dict(sorted(deviations.items(), key=lambda x: x[1]['deviation'], reverse=True)[:5]),
            'unusual_chars': unusual_chars,
            'avg_word_length': round(avg_word_length, 2),
            'word_length_suspicious': word_length_suspicious,
            'total_characters': total_alpha
        }
        
    def expand_all_dict_columns(self, df):
        """Automatically expand all dictionary columns in a DataFrame"""
        
        result_df = df.copy()
        dict_columns = []
        
        # Identify columns containing dictionaries
        for col in df.columns:
            # Check first non-null value
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, dict):
                dict_columns.append(col)
        
        print(f"Found dictionary columns: {dict_columns}")
        
        for col in dict_columns:
            # Expand the dictionary column
            expanded = df[col].apply(lambda x: self.flatten_nested_dict(x, col))
            expanded_df = pd.DataFrame(expanded.tolist())
            
            # Drop original column and add expanded columns
            result_df = result_df.drop(columns=[col])
            result_df = pd.concat([result_df, expanded_df], axis=1)
        
        return result_df


    def flatten_nested_dict(self, d, prefix=''):
        """Recursively flatten a nested dictionary"""
        if not d or pd.isna(d) or not isinstance(d, dict):
            return {}
        
        items = {}
        for key, value in d.items():
            new_key = f'{prefix}_{key}' if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dict
                nested = self.flatten_nested_dict(value, new_key)
                items.update(nested)
            else:
                items[new_key] = value
        
        return items

    def analyze(self, payload: str) -> Dict[str, Any]:
        """Complete payload analysis"""
        results = {
            'original': payload,
            'length': len(payload),
            'homoglyphs': None,
            'confusables': None,
            'normalized_ascii': None,
            'fixed_encoding': None,
            'encoding_detected': None
        }
        
        # 2. Homoglyph detection
        results['homoglyphs'] = self._detect_homoglyphs(payload)
        
        # 3. Confusable detection
        results['confusables'] = self._detect_confusables(payload)
        
        # 4. Normalize to ASCII
        results['normalized_ascii'] = unidecode(payload)
        
        # 5. Fix encoding issues
        fixed = ftfy.fix_text(payload)
        if fixed != payload:
            results['fixed_encoding'] = fixed
        
        # 7. Detect encoding
        detected = chardet.detect(payload.encode('utf-8', errors='ignore'))
        results['encoding_detected'] = detected
        
        return results
    
    def _detect_homoglyphs(self, text: str) -> Dict:
        """Detect homoglyphs using homoglyphs library"""
        found = []
        normalized = ''
        
        for char in text:
            # Check if character has ASCII equivalent
            ascii_version = self.hogl.to_ascii(char)
            if ascii_version != char:
                found.append({
                    'char': char,
                    'ascii': ascii_version,
                    'unicode': f'U+{ord(char):04X}'
                })
                normalized += str(ascii_version)
            else:
                normalized += char
        
        return {
            'count': len(found),
            'found': found[:20],
            'normalized': normalized if found else None,
            'suspicious': len(found) > 0
        }
    
    def _detect_confusables(self, text: str) -> Dict:
        """Detect confusable characters"""
        is_conf = cnf.is_confusable(text, preferred_aliases=['latin'], greedy=True)
        
        # Use only for debug, as it pronts 40000 llines of text
        # print(f"confusable chars for string{text} are: {confusable_chars}")
        
        return {
            'is_confusable': is_conf,
            'confusable_chars': is_conf[0] if is_conf else [],
            'count': len(is_conf[0]) if is_conf else 0
        }
