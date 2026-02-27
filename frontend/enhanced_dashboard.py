"""
Streamlit Frontend for CNN-LSTM Intrusion Detection System
Real-time network monitoring with XAI explanations and alert management
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
from datetime import datetime, timedelta
from realtime_monitor import NetworkTrafficSimulator

try:
    from packet_capture import RealNetworkCapture, SCAPY_AVAILABLE
except ImportError:
    SCAPY_AVAILABLE = False

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")

# Custom CSS for better alert styling
st.markdown("""
<style>
.alert-critical {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 10px;
    margin: 10px 0;
}
.alert-high {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 10px;
    margin: 10px 0;
}
.alert-medium {
    background-color: #f3e5f5;
    border-left: 5px solid #9c27b0;
    padding: 10px;
    margin: 10px 0;
}
.alert-banner {
    animation: blink 2s linear infinite;
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

def _safe_rerun():
    """Call Streamlit rerun in a safe way across versions.

    Some Streamlit installs (or shadowed imports) may not expose
    st.experimental_rerun. Use a graceful fallback that sets a
    session flag and stops execution so the user can refresh.
    """
    try:
        # Preferred API
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass

    # Fallback: mark a flag and stop rendering so user can refresh
    try:
        st.session_state['_needs_rerun'] = True
    except Exception:
        # If even session_state is not usable, ignore and stop
        pass
    st.stop()



# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("Network Intrusion Detection System")
st.markdown("### Real-Time Network Attack Detection Dashboard")

# Initialize session state for alerts FIRST
if 'alerts_data' not in st.session_state:
    st.session_state.alerts_data = []
if 'show_xai' not in st.session_state:
    st.session_state.show_xai = False
if 'alert_sound_enabled' not in st.session_state:
    st.session_state.alert_sound_enabled = True
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = None

# Hidden critical alert processing - no banner display

st.sidebar.header("Backend Configuration")
st.sidebar.write(f"Backend URL: `{BACKEND_URL}`")

# Sidebar Alert Summary - Hidden but functional
if st.session_state.alerts_data:
    total_alerts = len(st.session_state.alerts_data)
    unacked = len([a for a in st.session_state.alerts_data if not a['acknowledged']])
    critical = len([a for a in st.session_state.alerts_data if a['severity'] >= 4 and not a['acknowledged']])
    
    # Only show critical alert count in sidebar metrics without taking up space
    if critical > 0:
        pass  # Remove sidebar display but keep functionality
else:
    pass  # Remove sidebar display completely

# ----------------------------------------------------------------------
# ALERT FUNCTIONS
# ----------------------------------------------------------------------
def generate_alert(prediction, confidence, timestamp, traffic_data, explanations=None, risk_assessment=None):
    """Generate an alert when an attack is detected"""
    if prediction == 'BENIGN':
        return None
    
    # Calculate severity based on attack type and confidence
    severity_mapping = {
        'DoS GoldenEye': 5,
        'DoS ****Hulk': 5, 
        'DoS Slowhttptest': 4,
        'DoS slowloris': 4,
        'FTP-Patator': 3,
        'SSH-Patator': 3,
        'Web Attack - Brute Force': 4,
        'Web Attack - XSS': 3,
        'Web Attack - SQL Injection': 4
    }
    
    severity = severity_mapping.get(prediction, 3)
    
    # Adjust severity based on confidence
    if confidence > 0.9:
        severity = min(5, severity + 1)
    elif confidence < 0.7:
        severity = max(1, severity - 1)
    
    alert = {
        'id': f"ALERT_{timestamp.strftime('%Y%m%d_%H%M%S')}_{len(st.session_state.alerts_data)}",
        'timestamp': timestamp,
        'attack_type': prediction,
        'severity': severity,
        'confidence': confidence,
        'src_ip': traffic_data.get('src_ip', 'Unknown'),
        'dst_ip': traffic_data.get('dst_ip', 'Unknown'),
        'flow_duration': traffic_data.get('Flow Duration', 0),
        'bytes_per_sec': traffic_data.get('Flow Bytes/s', 0),
        'packets_per_sec': traffic_data.get('Flow Packets/s', 0),
        'total_packets': traffic_data.get('Total Fwd Packets', 0) + traffic_data.get('Total Backward Packets', 0),
        'acknowledged': False,
        'explanations': explanations or {},
        'risk_assessment': risk_assessment or {},
        'recommendations': get_attack_recommendations(prediction)
    }
    
    return alert

def get_attack_recommendations(attack_type):
    """Get security recommendations based on attack type"""
    recommendations = {
        'DoS GoldenEye': [
            'Block source IP immediately',
            'Implement rate limiting on web server',
            'Check firewall rules for DoS protection',
            'Monitor server resources and scale if needed'
        ],
        'DoS ****Hulk': [
            'Implement DDoS mitigation measures',
            'Block source IP range',
            'Check web application firewall settings',
            'Monitor bandwidth usage'
        ],
        'DoS Slowhttptest': [
            'Configure server timeout settings',
            'Implement connection limits per IP',
            'Update web server configuration',
            'Consider using a reverse proxy'
        ],
        'DoS slowloris': [
            'Configure HTTP timeout settings',
            'Implement connection throttling',
            'Block suspicious source IPs',
            'Update web server security settings'
        ],
        'FTP-Patator': [
            'Disable FTP if not required',
            'Implement strong password policies',
            'Block source IP after failed attempts',
            'Monitor FTP access logs'
        ],
        'SSH-Patator': [
            'Implement SSH key-based authentication',
            'Disable password authentication',
            'Block source IP after failed attempts',
            'Monitor SSH access logs'
        ],
        'Web Attack - Brute Force': [
            'Implement account lockout policies',
            'Use CAPTCHA after failed attempts',
            'Monitor authentication logs',
            'Implement multi-factor authentication'
        ],
        'Web Attack - XSS': [
            'Update web application security',
            'Implement input validation',
            'Use Content Security Policy headers',
            'Scan for vulnerable web pages'
        ]
    }
    
    return recommendations.get(attack_type, [
        'Monitor network traffic closely',
        'Review security policies',
        'Check system logs for anomalies',
        'Consider blocking suspicious IPs'
    ])

def get_severity_color(severity):
    """Get color based on alert severity"""
    colors = {
        1: 'green',
        2: 'blue', 
        3: 'orange',
        4: 'red',
        5: 'darkred'
    }
    return colors.get(severity, 'gray')

def get_severity_emoji(severity):
    """Get emoji based on alert severity"""
    emojis = {
        1: '🟢',
        2: '🔵',
        3: '🟡', 
        4: '🔴',
        5: '🚨'
    }
    return emojis.get(severity, '⚪')

# ----------------------------------------------------------------------
# BACKEND HEALTH CHECK
# ----------------------------------------------------------------------
st.sidebar.markdown("### Backend Status")

backend_ok = False
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if response.status_code == 200:
        health = response.json()
        backend_ok = True
        st.sidebar.success("Backend Connected")

        # Show XAI dependencies status
        xai_deps = health.get('xai_dependencies', {})
        if not xai_deps.get('xai_ready', False):
            st.sidebar.warning("XAI not fully available")
            missing = xai_deps.get('missing_packages', [])
            if missing:
                st.sidebar.code(f"pip install {' '.join(missing)}")
    else:
        st.sidebar.error("Backend returned error.")
except Exception as e:
    st.sidebar.warning("Could not connect to backend.")
    st.sidebar.caption(f"Error: {e}")

# ----------------------------------------------------------------------
# FRONTEND BODY
# ----------------------------------------------------------------------
if not backend_ok:
    st.warning("Backend not reachable. Please start your FastAPI server and refresh this page.")
    st.stop()

# ----------------------------------------------------------------------
# HIDDEN POPUP ALERT SYSTEM - Only shows popup messages, no UI display
# ----------------------------------------------------------------------

# Initialize popup state but don't show any alert interface
if 'show_alerts_popup' not in st.session_state:
    st.session_state['show_alerts_popup'] = False

# Remove all alert display sections - only keep background alert storage

# ----------------------------------------------------------------------
# REAL-TIME MONITORING SECTION
# ----------------------------------------------------------------------
st.markdown("## 🔴 Real-Time Network Monitoring")

# Initialize session state for real-time data
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = []
if 'traffic_simulator' not in st.session_state:
    st.session_state.traffic_simulator = NetworkTrafficSimulator()
if 'packet_capture' not in st.session_state and SCAPY_AVAILABLE:
    st.session_state.packet_capture = RealNetworkCapture()

# Monitoring mode selection
if SCAPY_AVAILABLE:
    monitoring_mode = st.radio(
        "Monitoring Mode:",
        ["Simulated Traffic", "Real Network Capture"],
        help="Choose between simulated network traffic or real packet capture"
    )
else:
    monitoring_mode = st.radio(
        "Monitoring Mode:",
        ["Simulated Traffic"],
        help="Real network capture requires scapy: pip install scapy"
    )
    st.info("Real Network Capture is disabled. Install scapy to enable: `pip install scapy`")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("**Live Network Traffic Analysis**")
    monitor_interval = st.selectbox(
        "Monitoring Interval", 
        [1, 2, 5, 10], 
        index=1,
        help="Seconds between predictions"
    )

with col2:
    if "Real Network" in monitoring_mode and SCAPY_AVAILABLE:
        if st.button("Start Monitoring" if not st.session_state.monitoring else "Stop Monitoring"):
            st.session_state.monitoring = not st.session_state.monitoring
            if st.session_state.monitoring:
                # Start packet capture
                try:
                    st.session_state.packet_capture.start_capture(duration=300)  # 5 minutes
                    st.success("Started real network capture!")
                except Exception as e:
                    st.error(f"Failed to start capture: {e}")
                    st.session_state.monitoring = False
            else:
                st.session_state.realtime_data = []
    else:
        if st.button("Start Monitoring" if not st.session_state.monitoring else "Stop Monitoring"):
            st.session_state.monitoring = not st.session_state.monitoring
            if not st.session_state.monitoring:
                st.session_state.realtime_data = []

with col3:
    if st.button("Clear Data"):
        st.session_state.realtime_data = []

# XAI Controls
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**Explainable AI Options**")
with col2:
    st.session_state.show_xai = st.checkbox("Enable XAI Explanations", value=st.session_state.show_xai)

# Real-time monitoring logic
if st.session_state.monitoring:
    if "Simulated" in monitoring_mode:
        # Generate simulated traffic data
        traffic_data, expected_label = st.session_state.traffic_simulator.generate_mixed_traffic()
        
        try:
            # Send to backend for prediction
            if st.session_state.show_xai:
                # Use XAI endpoint for detailed explanation
                response = requests.post(f"{BACKEND_URL}/explain-prediction", json=traffic_data, timeout=5)
            else:
                # Use simple prediction endpoint
                response = requests.post(f"{BACKEND_URL}/predict-single", json=traffic_data, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                
                if st.session_state.show_xai and 'prediction' in result:
                    prediction = result['prediction']['predicted_label']
                    confidence = result['prediction']['confidence']
                    explanations = result.get('explanations', {})
                    risk_assessment = result.get('risk_assessment', {})
                else:
                    prediction = result.get("predicted_label", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    explanations = {}
                    risk_assessment = {}
                
                timestamp = datetime.now()
                
                # Add simulated IP addresses for better alert display
                import random
                traffic_data['src_ip'] = f"192.168.1.{random.randint(100, 254)}"
                traffic_data['dst_ip'] = f"10.0.0.{random.randint(1, 50)}"
                
                # Store the data
                record = {
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'expected': expected_label,
                    'correct': prediction == expected_label.replace('ATTACK_', '').replace('_', ' '),
                    'confidence': confidence,
                    'flow_duration': traffic_data['Flow Duration'],
                    'total_packets': traffic_data['Total Fwd Packets'] + traffic_data['Total Backward Packets'],
                    'bytes_per_sec': traffic_data['Flow Bytes/s'],
                    'packets_per_sec': traffic_data['Flow Packets/s'],
                    'source': 'Simulated',
                    'explanations': explanations,
                    'risk_assessment': risk_assessment
                }
                
                st.session_state.realtime_data.append(record)
                
                # Generate alert if attack detected
                if prediction != 'BENIGN':
                    alert = generate_alert(prediction, confidence, timestamp, traffic_data, explanations, risk_assessment)
                    if alert:
                        st.session_state.alerts_data.append(alert)
                        st.session_state.last_alert_time = timestamp
                        
                        # Show immediate popup notification
                        if alert['severity'] >= 4:
                            st.error(f"🚨 **CRITICAL THREAT DETECTED!** {prediction} from {traffic_data.get('src_ip', 'Unknown')} (Confidence: {confidence:.1%})", icon="🚨")
                            
                            # Add dramatic effect for critical alerts
                            st.markdown("""
                            <script>
                                // Critical alert sound effect simulation
                                if (typeof(Audio) !== "undefined") {
                                    // Create a beep sound effect
                                    var audioContext = new (window.AudioContext || window.webkitAudioContext)();
                                    var oscillator = audioContext.createOscillator();
                                    var gainNode = audioContext.createGain();
                                    oscillator.connect(gainNode);
                                    gainNode.connect(audioContext.destination);
                                    oscillator.frequency.value = 800;
                                    oscillator.type = 'square';
                                    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                                    oscillator.start();
                                    oscillator.stop(audioContext.currentTime + 0.2);
                                    
                                    // Second beep
                                    setTimeout(function() {
                                        var oscillator2 = audioContext.createOscillator();
                                        var gainNode2 = audioContext.createGain();
                                        oscillator2.connect(gainNode2);
                                        gainNode2.connect(audioContext.destination);
                                        oscillator2.frequency.value = 1000;
                                        oscillator2.type = 'square';
                                        gainNode2.gain.setValueAtTime(0.3, audioContext.currentTime);
                                        oscillator2.start();
                                        oscillator2.stop(audioContext.currentTime + 0.2);
                                    }, 300);
                                }
                                
                                // Browser notification
                                if ("Notification" in window && Notification.permission === "granted") {
                                    new Notification("🚨 IDS CRITICAL ALERT", {
                                        body: "Critical security threat detected! " + "{prediction}",
                                        icon: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0icmVkIj48cGF0aCBkPSJNMTIgMkwyIDdsMTAtNS0xMCA1VjE3bDEwLTV6Ii8+PC9zdmc+",
                                        requireInteraction: true,
                                        tag: "ids-critical-alert"
                                    });
                                } else if ("Notification" in window && Notification.permission !== "denied") {
                                    Notification.requestPermission().then(function (permission) {
                                        if (permission === "granted") {
                                            new Notification("🚨 IDS CRITICAL ALERT", {
                                                body: "Critical security threat detected! " + "{prediction}",
                                                requireInteraction: true,
                                                tag: "ids-critical-alert"
                                            });
                                        }
                                    });
                                }
                            </script>
                            """.format(prediction=prediction), unsafe_allow_html=True)
                            
                        elif alert['severity'] >= 3:
                            st.warning(f"⚠️ **SECURITY ALERT!** {prediction} detected (Confidence: {confidence:.1%})", icon="⚠️")
                            
                            # Browser notification for high alerts
                            st.markdown("""
                            <script>
                                if ("Notification" in window && Notification.permission === "granted") {
                                    new Notification("⚠️ IDS Security Alert", {
                                        body: "Security threat detected: " + "{prediction}",
                                        tag: "ids-security-alert"
                                    });
                                }
                            </script>
                            """.format(prediction=prediction), unsafe_allow_html=True)
                            
                        else:
                            st.info(f"ℹ️ **Security Notice:** {prediction} detected (Confidence: {confidence:.1%})", icon="ℹ️")
                        
                        # Keep only last 100 alerts
                        if len(st.session_state.alerts_data) > 100:
                            st.session_state.alerts_data = st.session_state.alerts_data[-100:]
                
                # Keep only last 50 records
                if len(st.session_state.realtime_data) > 50:
                    st.session_state.realtime_data = st.session_state.realtime_data[-50:]
        
        except Exception as e:
            st.error(f"Real-time monitoring error: {e}")
    
    elif "Real Network" in monitoring_mode and SCAPY_AVAILABLE:
        # Process real network capture
        try:
            completed_flows = st.session_state.packet_capture.get_completed_flows()
            
            for flow in completed_flows:
                traffic_data = flow['features']
                
                # Send to backend for prediction
                if st.session_state.show_xai:
                    response = requests.post(f"{BACKEND_URL}/explain-prediction", json=traffic_data, timeout=5)
                else:
                    response = requests.post(f"{BACKEND_URL}/predict-single", json=traffic_data, timeout=2)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if st.session_state.show_xai and 'prediction' in result:
                        prediction = result['prediction']['predicted_label']
                        confidence = result['prediction']['confidence']
                        explanations = result.get('explanations', {})
                        risk_assessment = result.get('risk_assessment', {})
                    else:
                        prediction = result.get("predicted_label", "Unknown")
                        confidence = result.get("confidence", 0.0)
                        explanations = {}
                        risk_assessment = {}
                    
                    # Store the data
                    record = {
                        'timestamp': flow['timestamp'],
                        'prediction': prediction,
                        'expected': 'Unknown',  # Real traffic, we don't know the ground truth
                        'correct': None,
                        'confidence': confidence,
                        'flow_duration': traffic_data['Flow Duration'],
                        'total_packets': traffic_data['Total Fwd Packets'] + traffic_data['Total Backward Packets'],
                        'bytes_per_sec': traffic_data['Flow Bytes/s'],
                        'packets_per_sec': traffic_data['Flow Packets/s'],
                        'source': 'Real Network',
                        'explanations': explanations,
                        'risk_assessment': risk_assessment
                    }
                    
                    st.session_state.realtime_data.append(record)
                    
                    # Generate alert if attack detected
                    if prediction != 'BENIGN':
                        alert = generate_alert(prediction, confidence, flow['timestamp'], traffic_data, explanations, risk_assessment)
                        if alert:
                            st.session_state.alerts_data.append(alert)
                            st.session_state.last_alert_time = flow['timestamp']
                            
                            # Show immediate popup notification
                            if alert['severity'] >= 4:
                                st.error(f"🚨 **CRITICAL THREAT DETECTED!** {prediction} from real network traffic (Confidence: {confidence:.1%})", icon="🚨")
                            elif alert['severity'] >= 3:
                                st.warning(f"⚠️ **SECURITY ALERT!** {prediction} detected in network traffic (Confidence: {confidence:.1%})", icon="⚠️")
                            else:
                                st.info(f"ℹ️ **Security Notice:** {prediction} detected in network traffic (Confidence: {confidence:.1%})", icon="ℹ️")
                            
                            # Keep only last 100 alerts
                            if len(st.session_state.alerts_data) > 100:
                                st.session_state.alerts_data = st.session_state.alerts_data[-100:]
                    
                    # Keep only last 50 records
                    if len(st.session_state.realtime_data) > 50:
                        st.session_state.realtime_data = st.session_state.realtime_data[-50:]
                        
        except Exception as e:
            st.error(f"Real network capture error: {e}")
    
    else:
        st.warning("Real network capture not available. Please install scapy: pip install scapy")

# Display real-time results
if st.session_state.realtime_data:
    df_realtime = pd.DataFrame(st.session_state.realtime_data)
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_predictions = len(df_realtime)
    benign_count = len(df_realtime[df_realtime['prediction'] == 'BENIGN'])
    attack_count = total_predictions - benign_count
    
    # Calculate accuracy only for simulated data where we have ground truth
    accuracy_data = df_realtime[df_realtime['correct'].notna()]
    accuracy = len(accuracy_data[accuracy_data['correct']]) / len(accuracy_data) * 100 if len(accuracy_data) > 0 else 0
    
    with col1:
        st.metric("Total Flows", total_predictions)
    with col2:
        st.metric("Benign", benign_count, delta=f"{benign_count/total_predictions*100:.1f}%")
    with col3:
        st.metric("Attacks", attack_count, delta=f"{attack_count/total_predictions*100:.1f}%")
    with col4:
        if len(accuracy_data) > 0:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Accuracy", "N/A")
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline chart
        fig_timeline = px.scatter(
            df_realtime, 
            x='timestamp', 
            y='prediction',
            color='prediction',
            size='total_packets',
            title="Real-Time Predictions Timeline",
            color_discrete_map={
                'BENIGN': 'green',
                'DoS GoldenEye': 'red',
                'DoS ****Hulk': 'red',
                'DoS Slowhttptest': 'orange',
                'DoS slowloris': 'orange',
                'FTP-Patator': 'purple',
                'SSH-Patator': 'purple',
                'Web Attack - Brute Force': 'brown',
                'Web Attack - XSS': 'pink'
            }
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, width='stretch')
    
    with col2:
        # Traffic volume chart
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Scatter(
            x=df_realtime['timestamp'],
            y=df_realtime['bytes_per_sec'],
            mode='lines+markers',
            name='Bytes/sec',
            line=dict(color='blue')
        ))
        fig_volume.update_layout(
            title="Network Traffic Volume",
            xaxis_title="Time",
            yaxis_title="Bytes per Second",
            height=400
        )
        st.plotly_chart(fig_volume, width='stretch')
    
    # Recent predictions table
    st.markdown("### Recent Predictions")
    recent_df = df_realtime.tail(10).copy()
    recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
    display_cols = ['timestamp', 'prediction', 'flow_duration', 'total_packets', 'bytes_per_sec']
    if 'source' in recent_df.columns:
        display_cols.append('source')
    if 'confidence' in recent_df.columns:
        display_cols.append('confidence')
    st.dataframe(recent_df[display_cols], width='stretch')
    
    # XAI Explanations Section
    if st.session_state.show_xai:
        st.markdown("---")
        st.markdown("### XAI Explanations for Recent Predictions")
        st.caption("Explaining why the model made its prediction using LIME/SHAP analysis")

        # Show explanation for the latest prediction with XAI data
        latest_with_xai = None
        for record in reversed(st.session_state.realtime_data):
            explanations = record.get('explanations', {})
            # Debug: Check what we got
            if explanations and (isinstance(explanations, dict) and len(explanations) > 0):
                # Check if either lime or shap has actual data
                has_lime = 'lime' in explanations and explanations.get('lime')
                has_shap = 'shap' in explanations and explanations.get('shap')
                if has_lime or has_shap:
                    latest_with_xai = record
                    break
        
        if latest_with_xai:
            # Show prediction summary
            st.markdown(f"**Latest Prediction:** `{latest_with_xai['prediction']}` (Confidence: {latest_with_xai['confidence']:.1%})")
            st.markdown(f"**Timestamp:** {latest_with_xai['timestamp'].strftime('%H:%M:%S')}")
            
            # Only show risk assessment and recommendations here. The feature-importance
            # / raw-XAI expander has been removed per user request to avoid printing the
            # feature-importance panel in the UI.
            st.markdown("#### Risk Assessment & Recommendations")
            st.caption("Threat level and actionable security recommendations")
            risk_assessment = latest_with_xai.get('risk_assessment', {})

            if risk_assessment:
                # Risk level indicator
                risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                severity = risk_assessment.get('severity', 0)

                risk_emojis = {
                    'LOW': 'LOW',
                    'MEDIUM': 'MEDIUM',
                    'HIGH': 'HIGH',
                    'CRITICAL': 'CRITICAL'
                }

                # Display risk with text and progress
                st.markdown(f"**Risk Level:** {risk_emojis.get(risk_level, 'UNKNOWN')} **{risk_level}**")
                st.progress(severity / 5, text=f"Severity: {severity}/5")
                st.markdown(f"**Attack Type:** `{risk_assessment.get('attack_type', 'Unknown')}`")
                st.markdown(f"**Detection Confidence:** {risk_assessment.get('confidence', 0):.1%}")

                # Recommendations with better formatting
                recommendations = risk_assessment.get('recommendations', [])
                if recommendations:
                    st.markdown("**Security Recommendations:**")
                    for i, rec in enumerate(recommendations[:5], 1):
                        st.markdown(f"**{i}.** {rec}")
                else:
                    st.info("No specific recommendations for this traffic")
            else:
                st.warning("No risk assessment data available")
        
        else:
            # No XAI data available yet
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.warning("No XAI Explanations Available Yet")
                st.info("""
                Why am I not seeing explanations?

                1. XAI is enabled - you've activated XAI explanations
                2. Waiting for data - XAI analysis requires at least one prediction with the checkbox enabled
                3. Processing time - XAI adds 1-2 seconds per prediction

                What to do:
                - Make sure "Enable XAI Explanations" is checked above
                - If monitoring isn't running, click "Start Monitoring"
                - Wait for the next prediction (happens every 2 seconds by default)
                - Explanations will appear automatically
                """)
                
                # Debug info in expander
                if st.session_state.realtime_data:
                    with st.expander("Debug Information (Click to expand)"):
                        latest_record = st.session_state.realtime_data[-1]
                        st.write("**Latest Record Status:**")
                        st.write(f"- Has 'explanations' key: {('explanations' in latest_record)}")
                        if 'explanations' in latest_record:
                            st.write(f"- Explanations empty: {(not latest_record['explanations'])}")
                            if latest_record['explanations']:
                                st.write(f"- Available methods: {list(latest_record['explanations'].keys())}")
                        st.write(f"- Timestamp: {latest_record.get('timestamp', 'N/A')}")
                        st.write(f"- Prediction: {latest_record.get('prediction', 'N/A')}")
                        
                        st.write("\n**Recent Records with XAI:**")
                        xai_count = sum(1 for r in st.session_state.realtime_data if r.get('explanations'))
                        st.write(f"- {xai_count} out of {len(st.session_state.realtime_data)} records have XAI data")

# Auto-refresh for real-time monitoring
if st.session_state.monitoring:
    time.sleep(monitor_interval)
    st.rerun()

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
