"""
Explainable AI (XAI) Module for CNN-LSTM Intrusion Detection System
Provides LIME and SHAP explanations for model predictions
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class XAIExplainer:
    """
    Explainable AI wrapper for CNN-LSTM model predictions
    """
    
    def __init__(self, model, scaler, feature_names, label_encoder, training_data=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.training_data = training_data
        
        # Initialize explainers
        self.lime_explainer = None
        self.shap_explainer = None
        
        if LIME_AVAILABLE and training_data is not None:
            self._init_lime_explainer()
        
        if SHAP_AVAILABLE:
            self._init_shap_explainer()
    
    def _init_lime_explainer(self):
        """Initialize LIME explainer with training data"""
        try:
            # Use a sample of training data for LIME
            sample_size = min(1000, len(self.training_data))
            training_sample = self.training_data.sample(n=sample_size, random_state=42)
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_sample.values,
                feature_names=self.feature_names,
                class_names=self.label_encoder.classes_,
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize LIME: {e}")
            self.lime_explainer = None
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer"""
        try:
            # For neural networks, we'll use a simple wrapper
            def model_predict(X):
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled)
                return predictions
            
            # Use a small background dataset for SHAP
            if self.training_data is not None:
                background_size = min(100, len(self.training_data))
                background = self.training_data.sample(n=background_size, random_state=42)
                self.shap_explainer = shap.KernelExplainer(model_predict, background.values)
            else:
                # Use synthetic background if no training data
                background = np.random.normal(0, 1, (100, len(self.feature_names)))
                self.shap_explainer = shap.KernelExplainer(model_predict, background)
            
            print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize SHAP: {e}")
            self.shap_explainer = None
    
    def explain_prediction_lime(self, instance, num_features=10):
        """
        Generate LIME explanation for a single prediction
        
        Args:
            instance: Single data instance to explain
            num_features: Number of top features to include in explanation
            
        Returns:
            Dictionary with explanation data
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {"error": "LIME not available"}
        
        try:
            def predict_fn(X):
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled)
                return predictions
            
            # Get explanation
            explanation = self.lime_explainer.explain_instance(
                instance.values[0], 
                predict_fn, 
                num_features=num_features
            )
            
            # Extract feature importance
            feature_importance = []
            for feature_str, importance in explanation.as_list():
                # LIME returns a human-readable feature string (e.g. "Flow Bytes/s > 500")
                # Try to match it to one of the known feature names
                matched_feature = None
                for fname in self.feature_names:
                    if fname in feature_str:
                        matched_feature = fname
                        break

                if matched_feature is None:
                    # fallback: take the left-most token
                    matched_feature = feature_str.split()[0]

                # Try to safely fetch the instance value for that feature
                try:
                    value = float(instance.iloc[0][matched_feature])
                except Exception:
                    value = None

                feature_importance.append({
                    'feature': matched_feature,
                    'importance': float(importance),
                    'value': value
                })
            
            # Get prediction probabilities
            X_scaled = self.scaler.transform(instance)
            probabilities = self.model.predict(X_scaled)[0]
            
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            return {
                "method": "LIME",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "feature_importance": feature_importance,
                "explanation_available": True
            }
            
        except Exception as e:
            return {"error": f"LIME explanation failed: {str(e)}"}
    
    def explain_prediction_shap(self, instance, max_evals=100):
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            instance: Single data instance to explain
            max_evals: Maximum number of evaluations for SHAP
            
        Returns:
            Dictionary with explanation data
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"error": "SHAP not available"}
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(instance.values, nsamples=max_evals)
            
            # Get prediction
            X_scaled = self.scaler.transform(instance)
            probabilities = self.model.predict(X_scaled)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Extract feature importance for the predicted class
            if isinstance(shap_values, list):
                # Multi-class case
                class_shap_values = shap_values[predicted_class_idx][0]
            else:
                # Binary case
                class_shap_values = shap_values[0]
            
            feature_importance = []
            for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, class_shap_values)):
                feature_importance.append({
                    'feature': feature_name,
                    'importance': float(shap_val),
                    'value': float(instance.iloc[0, i])
                })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            return {
                "method": "SHAP",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "feature_importance": feature_importance[:10],  # Top 10 features
                "explanation_available": True
            }
            
        except Exception as e:
            return {"error": f"SHAP explanation failed: {str(e)}"}
    
    def get_attack_risk_assessment(self, instance, explanation_data):
        """
        Assess attack risk level and generate alerts
        
        Args:
            instance: Data instance
            explanation_data: XAI explanation results
            
        Returns:
            Risk assessment dictionary
        """
        try:
            predicted_class = explanation_data.get('predicted_class', 'UNKNOWN')
            confidence = explanation_data.get('confidence', 0.0)
            
            # Determine risk level
            if predicted_class == 'BENIGN':
                risk_level = 'LOW'
                alert_type = 'info'
                severity = 1
            elif confidence >= 0.9:
                risk_level = 'CRITICAL'
                alert_type = 'error'
                severity = 5
            elif confidence >= 0.7:
                risk_level = 'HIGH'
                alert_type = 'warning'
                severity = 4
            elif confidence >= 0.5:
                risk_level = 'MEDIUM'
                alert_type = 'warning'
                severity = 3
            else:
                risk_level = 'LOW'
                alert_type = 'info'
                severity = 2
            
            # Get top contributing features
            top_features = []
            if 'feature_importance' in explanation_data:
                top_features = sorted(
                    explanation_data['feature_importance'], 
                    key=lambda x: abs(x['importance']), 
                    reverse=True
                )[:5]
            
            # Generate alert message
            if predicted_class != 'BENIGN':
                alert_message = f"🚨 {predicted_class} ATTACK DETECTED! Confidence: {confidence:.1%}"
                recommendations = self._get_attack_recommendations(predicted_class)
            else:
                alert_message = f"✅ Normal traffic detected. Confidence: {confidence:.1%}"
                recommendations = ["Continue monitoring network traffic"]
            
            return {
                "risk_level": risk_level,
                "severity": severity,
                "alert_type": alert_type,
                "alert_message": alert_message,
                "attack_type": predicted_class,
                "confidence": confidence,
                "top_features": top_features,
                "recommendations": recommendations,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Risk assessment failed: {str(e)}",
                "risk_level": "UNKNOWN",
                "severity": 0
            }
    
    def _get_attack_recommendations(self, attack_type):
        """Get specific recommendations based on attack type"""
        recommendations = {
            'DoS GoldenEye': [
                "Block suspicious IP addresses",
                "Implement rate limiting",
                "Check server resources and capacity",
                "Consider DDoS protection services"
            ],
            'DoS Hulk': [
                "Activate DDoS mitigation",
                "Monitor server CPU/memory usage",
                "Block attacking IP ranges",
                "Scale server resources if needed"
            ],
            'DoS Slowhttptest': [
                "Configure HTTP timeout settings",
                "Implement connection limits",
                "Monitor slow HTTP connections",
                "Consider web application firewall"
            ],
            'DoS slowloris': [
                "Limit concurrent connections per IP",
                "Configure connection timeouts",
                "Monitor incomplete HTTP requests",
                "Implement reverse proxy protection"
            ],
            'FTP-Patator': [
                "Disable FTP if not needed",
                "Implement strong authentication",
                "Monitor failed login attempts",
                "Consider SFTP instead of FTP"
            ],
            'SSH-Patator': [
                "Disable SSH password authentication",
                "Use key-based authentication only",
                "Implement fail2ban or similar",
                "Change default SSH port"
            ],
            'Web Attack - Brute Force': [
                "Implement account lockout policies",
                "Use CAPTCHA after failed attempts",
                "Monitor authentication logs",
                "Consider multi-factor authentication"
            ],
            'Web Attack - XSS': [
                "Validate and sanitize all inputs",
                "Implement Content Security Policy",
                "Update web application frameworks",
                "Conduct security code review"
            ]
        }
        
        return recommendations.get(attack_type, [
            "Investigate the detected attack",
            "Monitor network traffic closely",
            "Consider blocking suspicious sources",
            "Update security policies"
        ])

def check_xai_dependencies():
    """Check if XAI libraries are available"""
    status = {
        "lime_available": LIME_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
        "xai_ready": LIME_AVAILABLE or SHAP_AVAILABLE
    }
    
    missing_packages = []
    if not LIME_AVAILABLE:
        missing_packages.append("lime")
    if not SHAP_AVAILABLE:
        missing_packages.append("shap")
    
    status["missing_packages"] = missing_packages
    status["install_command"] = f"pip install {' '.join(missing_packages)}" if missing_packages else None
    
    return status