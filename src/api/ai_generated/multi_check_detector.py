"""
Advanced Multi-Check AI Image Detection System
Analyzes images using multiple techniques for better accuracy
Integrated for Deepfake Detection System
"""
import cv2
import numpy as np
from scipy import fftpack

# Configuration
IMAGE_SIZE = (128, 128)


class MultiCheckDetector:
    """
    Multi-layer AI detection system that checks:
    1. Model prediction (neural network)
    2. Frequency analysis (FFT - AI images have different frequency patterns)
    3. Noise analysis (AI images have unusual noise patterns)
    4. Color consistency (AI images may have unnatural color distributions)
    5. Edge analysis (AI images have different edge characteristics)
    """
    
    def __init__(self, model):
        self.model = model
        
    def check_model_prediction(self, img_array):
        """Check 1: Neural network prediction"""
        try:
            img_resized = cv2.resize(img_array, IMAGE_SIZE)
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            prediction = self.model.predict(img_batch, verbose=0)
            ai_prob = float(prediction[0][0])
            natural_prob = float(prediction[0][1])
            
            # Fair threshold at 50% for balanced detection
            is_ai = ai_prob > 0.50
            
            # Calculate confidence based on how far from threshold
            if is_ai:
                confidence = (ai_prob - 0.50) / 0.50
            else:
                confidence = (0.50 - ai_prob) / 0.50
            
            confidence = min(1.0, max(0.0, confidence))
            
            return {
                'method': 'Neural Network',
                'is_ai': is_ai,
                'confidence': confidence,
                'ai_probability': ai_prob,
                'natural_probability': natural_prob,
                'score': ai_prob
            }
        except Exception as e:
            return {'method': 'Neural Network', 'error': str(e)}
    
    def check_frequency_analysis(self, img_array):
        """Check 2: Frequency domain analysis (FFT)"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            gray = cv2.resize(gray, (256, 256))
            
            f_transform = fftpack.fft2(gray)
            f_shift = fftpack.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            center = magnitude.shape[0] // 2
            high_freq_region = magnitude[center-20:center+20, center-20:center+20]
            low_freq_region = magnitude[0:40, 0:40]
            
            high_freq_energy = np.mean(high_freq_region)
            low_freq_energy = np.mean(low_freq_region)
            
            freq_ratio = low_freq_energy / (high_freq_energy + 1e-10)
            
            ai_score = min(1.0, (freq_ratio - 50) / 950)
            ai_score = max(0.0, ai_score)
            
            return {
                'method': 'Frequency Analysis',
                'is_ai': ai_score > 0.5,
                'confidence': abs(ai_score - 0.5) * 2,
                'score': ai_score,
                'freq_ratio': float(freq_ratio),
                'details': f'Freq ratio: {freq_ratio:.2f} (higher=AI)'
            }
        except Exception as e:
            return {'method': 'Frequency Analysis', 'error': str(e)}
    
    def check_noise_patterns(self, img_array):
        """Check 3: Noise pattern analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            gray = cv2.resize(gray, (256, 256))
            
            kernel_size = 5
            mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
            sqr_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
            variance = sqr_mean - (mean ** 2)
            
            var_of_var = np.var(variance)
            
            uniformity_score = 1.0 - min(1.0, var_of_var / 5000)
            
            return {
                'method': 'Noise Analysis',
                'is_ai': uniformity_score > 0.6,
                'confidence': abs(uniformity_score - 0.5) * 2,
                'score': uniformity_score,
                'var_of_var': float(var_of_var),
                'details': f'Noise uniformity: {uniformity_score:.2f} (higher=AI)'
            }
        except Exception as e:
            return {'method': 'Noise Analysis', 'error': str(e)}
    
    def check_color_distribution(self, img_array):
        """Check 4: Color distribution analysis"""
        try:
            if len(img_array.shape) != 3:
                return {'method': 'Color Analysis', 'error': 'Grayscale image'}
            
            img_resized = cv2.resize(img_array, (256, 256))
            
            lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
            
            entropy_sum = 0
            for channel in range(3):
                hist = cv2.calcHist([lab], [channel], None, [256], [0, 256])
                hist = hist / (hist.sum() + 1e-10)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_sum += entropy
            
            avg_entropy = entropy_sum / 3
            
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            sat_variance = np.var(saturation)
            
            uniformity = 1.0 - min(1.0, sat_variance / 3000)
            entropy_score = 1.0 - (avg_entropy / 8.0)
            
            combined_score = (uniformity * 0.6 + entropy_score * 0.4)
            
            return {
                'method': 'Color Distribution',
                'is_ai': combined_score > 0.5,
                'confidence': abs(combined_score - 0.5) * 2,
                'score': combined_score,
                'entropy': float(avg_entropy),
                'sat_variance': float(sat_variance),
                'details': f'Color score: {combined_score:.2f} (higher=AI)'
            }
        except Exception as e:
            return {'method': 'Color Distribution', 'error': str(e)}
    
    def check_edge_characteristics(self, img_array):
        """Check 5: Edge detection analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            gray = cv2.resize(gray, (256, 256))
            
            edges = cv2.Canny(gray, 50, 150)
            
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            edge_smoothness = np.sum(dilated - edges) / (np.sum(edges > 0) + 1)
            
            smoothness_score = min(1.0, edge_smoothness / 0.9)
            
            return {
                'method': 'Edge Analysis',
                'is_ai': smoothness_score > 0.6,
                'confidence': abs(smoothness_score - 0.5) * 2,
                'score': smoothness_score,
                'edge_density': float(edge_density),
                'smoothness': float(edge_smoothness),
                'details': f'Edge smoothness: {smoothness_score:.2f} (higher=AI)'
            }
        except Exception as e:
            return {'method': 'Edge Analysis', 'error': str(e)}
    
    def analyze_comprehensive(self, img_array):
        """Run all checks and provide comprehensive analysis"""
        
        checks = [
            self.check_model_prediction(img_array),
            self.check_frequency_analysis(img_array),
            self.check_noise_patterns(img_array),
            self.check_color_distribution(img_array),
            self.check_edge_characteristics(img_array)
        ]
        
        valid_checks = [c for c in checks if 'error' not in c]
        
        if not valid_checks:
            return {
                'final_prediction': 'Error',
                'confidence': 0.0,
                'error': 'All analysis methods failed'
            }
        
        weights = {
            'Neural Network': 0.30,
            'Frequency Analysis': 0.20,
            'Noise Analysis': 0.18,
            'Color Distribution': 0.17,
            'Edge Analysis': 0.15
        }
        
        total_score = 0
        total_weight = 0
        
        for check in valid_checks:
            method = check['method']
            weight = weights.get(method, 0.1)
            score = check.get('score', 0)
            total_score += score * weight
            total_weight += weight
        
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        
        ai_votes = sum(1 for c in valid_checks if c.get('is_ai', False))
        natural_votes = len(valid_checks) - ai_votes
        
        total_checks = len(valid_checks)
        max_votes = max(ai_votes, natural_votes)
        consensus = max_votes / total_checks if total_checks > 0 else 0.5
        
        if consensus >= 0.80:
            adaptive_threshold = 0.45
            confidence_level = "High"
        elif consensus >= 0.60:
            adaptive_threshold = 0.50
            confidence_level = "Medium"
        else:
            adaptive_threshold = 0.55
            confidence_level = "Low"
        
        neural_check = next((c for c in valid_checks if c['method'] == 'Neural Network'), None)
        
        is_ai = final_score > adaptive_threshold
        
        # Use majority voting as primary decision when there's agreement
        if ai_votes >= 3:  # At least 3 out of 5 checks say AI
            is_ai = True
            if ai_votes >= 4:
                confidence_level = "High (Majority Agreement)"
        elif natural_votes >= 4:  # Strong natural consensus needed to override
            is_ai = False
            confidence_level = "High (Natural Majority)"
        
        # Only override if neural network is VERY confident AND majority agrees
        if neural_check and neural_check.get('ai_probability', 0) < 0.20 and natural_votes >= 3:
            is_ai = False
            confidence_level = "High (Neural + Majority Override)"
        
        if ai_votes == total_checks:
            is_ai = True
            confidence_level = "Very High (Unanimous)"
        
        if is_ai:
            confidence = min(1.0, (final_score - adaptive_threshold) / (1.0 - adaptive_threshold))
        else:
            confidence = min(1.0, (adaptive_threshold - final_score) / adaptive_threshold)
        
        json_safe_checks = []
        for check in valid_checks:
            json_check = {
                'method': str(check['method']),
                'is_ai': bool(check.get('is_ai', False)),
                'confidence': float(check.get('confidence', 0)),
                'score': float(check.get('score', 0))
            }
            if 'details' in check:
                json_check['details'] = str(check['details'])
            if 'ai_probability' in check:
                json_check['ai_probability'] = float(check['ai_probability'])
            if 'natural_probability' in check:
                json_check['natural_probability'] = float(check['natural_probability'])
            
            json_safe_checks.append(json_check)
        
        return {
            'final_prediction': str('AI_Generated' if is_ai else 'Natural'),
            'confidence': float(confidence),
            'final_score': float(final_score),
            'ai_probability': float(final_score),
            'natural_probability': float(1 - final_score),
            'adaptive_threshold': float(adaptive_threshold),
            'consensus': float(consensus),
            'confidence_level': str(confidence_level),
            'votes': {
                'ai': int(ai_votes),
                'natural': int(natural_votes),
                'total': int(len(valid_checks))
            },
            'individual_checks': json_safe_checks,
            'analysis_summary': self._generate_summary(valid_checks, is_ai, confidence)
        }
    
    def _generate_summary(self, checks, is_ai, confidence):
        """Generate human-readable summary"""
        summary = []
        
        for check in checks:
            method = str(check['method'])
            check_result = "AI" if check.get('is_ai', False) else "Natural"
            check_conf = float(check.get('confidence', 0)) * 100
            summary.append(f"{method}: {check_result} ({check_conf:.1f}%)")
        
        result_text = "AI-Generated" if is_ai else "Natural/Real"
        conf_text = f"{float(confidence) * 100:.1f}%"
        
        return {
            'result': str(result_text),
            'confidence': str(conf_text),
            'checks': summary
        }

