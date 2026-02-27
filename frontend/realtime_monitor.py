"""
Real-time Network Traffic Monitor for IDS
Simulates real-time network traffic capture and sends to backend for prediction
"""

import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

class NetworkTrafficSimulator:
    """Simulates real-time network traffic data"""
    
    def __init__(self):
        # Load feature names from the backend model to ensure consistency
        try:
            import joblib
            import os
            backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
            model_path = os.path.join(backend_dir, "..", "ids_model_package.joblib")
            if os.path.exists(model_path):
                pkg = joblib.load(model_path)
                self.feature_names = pkg["feature_names"]
            else:
                # Fallback feature names in correct order
                self.feature_names = [
                    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", 
                    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", 
                    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", 
                    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
                    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
                    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
                    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
                    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
                    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
                    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
                    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
                    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
                    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
                    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
                    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
                ]
        except Exception as e:
            print(f"Warning: Could not load backend feature names, using defaults: {e}")
            # Use the correct order as fallback
            self.feature_names = [
                "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", 
                "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", 
                "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", 
                "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
                "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
                "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
                "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
                "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
                "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
                "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
                "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
                "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
                "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
                "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
                "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
            ]
        
    def generate_normal_traffic(self):
        """Generate normal/benign network traffic"""
        return {
            "Destination Port": random.choice([80, 443, 22, 21, 25, 53, 8080]),
            "Flow Duration": random.randint(100, 5000),
            "Total Fwd Packets": random.randint(1, 20),
            "Total Backward Packets": random.randint(1, 15),
            "Total Length of Fwd Packets": random.randint(100, 10000),
            "Total Length of Bwd Packets": random.randint(50, 5000),
            "Fwd Packet Length Max": random.randint(60, 1500),
            "Fwd Packet Length Min": random.randint(20, 60),
            "Fwd Packet Length Mean": random.uniform(40, 800),
            "Fwd Packet Length Std": random.uniform(10, 200),
            "Bwd Packet Length Max": random.randint(40, 1000),
            "Bwd Packet Length Min": random.randint(10, 40),
            "Bwd Packet Length Mean": random.uniform(30, 500),
            "Bwd Packet Length Std": random.uniform(5, 100),
            "Flow Bytes/s": random.uniform(100, 10000),
            "Flow Packets/s": random.uniform(1, 100),
            "Flow IAT Mean": random.uniform(10, 1000),
            "Flow IAT Std": random.uniform(5, 500),
            "Flow IAT Max": random.randint(50, 2000),
            "Flow IAT Min": random.randint(1, 50),
            "Fwd IAT Total": random.randint(100, 5000),
            "Fwd IAT Mean": random.uniform(20, 800),
            "Fwd IAT Std": random.uniform(10, 300),
            "Fwd IAT Max": random.randint(30, 1000),
            "Fwd IAT Min": random.randint(1, 30),
            "Bwd IAT Total": random.randint(50, 3000),
            "Bwd IAT Mean": random.uniform(15, 600),
            "Bwd IAT Std": random.uniform(8, 250),
            "Bwd IAT Max": random.randint(25, 800),
            "Bwd IAT Min": random.randint(1, 25),
            "Fwd PSH Flags": random.randint(0, 2),
            "Bwd PSH Flags": random.randint(0, 2),
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": random.uniform(1, 50),
            "Bwd Packets/s": random.uniform(1, 30),
            "Min Packet Length": random.randint(20, 60),
            "Max Packet Length": random.randint(100, 1500),
            "Packet Length Mean": random.uniform(50, 800),
            "Packet Length Std": random.uniform(20, 300),
            "Packet Length Variance": random.uniform(400, 90000),
            "FIN Flag Count": random.randint(0, 2),
            "SYN Flag Count": random.randint(0, 2),
            "RST Flag Count": 0,
            "PSH Flag Count": random.randint(0, 3),
            "ACK Flag Count": random.randint(1, 10),
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": random.uniform(0.1, 2.0),
            "Average Packet Size": random.uniform(40, 800),
            "Avg Fwd Segment Size": random.uniform(50, 900),
            "Avg Bwd Segment Size": random.uniform(30, 600),
            "Fwd Header Length.1": 0,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": random.randint(1, 20),
            "Subflow Fwd Bytes": random.randint(100, 10000),
            "Subflow Bwd Packets": random.randint(1, 15),
            "Subflow Bwd Bytes": random.randint(50, 5000),
            "Init_Win_bytes_forward": random.choice([8192, 16384, 32768, 65536]),
            "Init_Win_bytes_backward": random.choice([8192, 16384, 32768, 65536]),
            "act_data_pkt_fwd": random.randint(1, 15),
            "min_seg_size_forward": 20,
            "Active Mean": random.uniform(50, 1000),
            "Active Std": random.uniform(20, 500),
            "Active Max": random.randint(100, 2000),
            "Active Min": random.randint(10, 100),
            "Idle Mean": random.uniform(0, 100),
            "Idle Std": random.uniform(0, 50),
            "Idle Max": random.randint(0, 200),
            "Idle Min": 0
        }
    
    def generate_attack_traffic(self, attack_type="DoS"):
        """Generate malicious network traffic patterns"""
        base_traffic = self.generate_normal_traffic()
        
        if attack_type == "DoS":
            # DoS attack characteristics
            base_traffic.update({
                "Flow Duration": random.randint(10000, 100000),  # Longer flows
                "Total Fwd Packets": random.randint(100, 1000),  # High packet count
                "Flow Bytes/s": random.uniform(50000, 500000),   # High bandwidth
                "Flow Packets/s": random.uniform(100, 1000),     # High packet rate
                "Fwd Packets/s": random.uniform(80, 800),
                "SYN Flag Count": random.randint(50, 500),       # SYN flood
                "ACK Flag Count": random.randint(0, 5),          # Few ACKs
            })
        elif attack_type == "PortScan":
            # Port scan characteristics
            base_traffic.update({
                "Flow Duration": random.randint(50, 500),        # Short flows
                "Total Fwd Packets": random.randint(1, 5),       # Few packets
                "Total Backward Packets": random.randint(0, 2),  # Minimal response
                "SYN Flag Count": 1,                             # Single SYN
                "ACK Flag Count": 0,                             # No ACK
                "RST Flag Count": random.randint(0, 1),          # Possible RST
                "Destination Port": random.randint(1, 65535),    # Random ports
            })
        elif attack_type == "WebAttack":
            # Web attack characteristics
            base_traffic.update({
                "Destination Port": 80,                          # HTTP
                "Total Fwd Packets": random.randint(10, 50),     # Multiple requests
                "Fwd Packet Length Mean": random.uniform(200, 2000),  # Larger payloads
                "PSH Flag Count": random.randint(5, 20),         # Push data
                "ACK Flag Count": random.randint(10, 50),
            })
        
        return base_traffic
    
    def generate_mixed_traffic(self):
        """Generate mixed traffic (90% normal, 10% attacks)"""
        if random.random() < 0.9:
            return self.generate_normal_traffic(), "BENIGN"
        else:
            attack_type = random.choice(["DoS", "PortScan", "WebAttack"])
            return self.generate_attack_traffic(attack_type), f"ATTACK_{attack_type}"
    
    def generate_batch(self, batch_size=5):
        """Generate a batch of traffic samples for monitoring"""
        batch = []
        
        for _ in range(batch_size):
            # Generate mixed traffic
            traffic_data, expected_label = self.generate_mixed_traffic()
            
            # Convert to list format expected by the model (78 features)
            features = []
            for feature_name in self.feature_names:
                if feature_name in traffic_data:
                    features.append(traffic_data[feature_name])
                else:
                    features.append(0.0)  # Default value for missing features
            
            # Ensure we have exactly 78 features
            while len(features) < 78:
                features.append(0.0)
            features = features[:78]  # Trim if too many
            
            # Create sample record
            sample = {
                'features': features,
                'expected_label': expected_label,
                'flow_duration': traffic_data.get('Flow Duration', 0),
                'total_packets': traffic_data.get('Total Fwd Packets', 0) + traffic_data.get('Total Backward Packets', 0),
                'bytes_per_sec': traffic_data.get('Flow Bytes/s', 0),
                'packets_per_sec': traffic_data.get('Flow Packets/s', 0),
                'timestamp': datetime.now()
            }
            
            batch.append(sample)
        
        return batch