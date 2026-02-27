"""
Real Network Packet Capture for IDS
Uses scapy to capture and analyze real network packets
"""

import time
import threading
from collections import defaultdict
from datetime import datetime
import pandas as pd

try:
    from scapy.all import sniff, IP, TCP, UDP, conf
    import platform
    SCAPY_AVAILABLE = True
    
    # Configure scapy for Windows with Npcap
    if platform.system() == "Windows":
        try:
            # First try to use Npcap if available
            conf.use_pcap = True  # Enable pcap usage
            
            # Check if Npcap is available
            try:
                from scapy.arch.windows import get_windows_if_list
                interfaces = get_windows_if_list()
                if interfaces:
                    print("✅ Npcap detected - using WinPcap compatibility mode")
                else:
                    print("⚠️ No network interfaces found - falling back to Layer 3")
                    conf.use_pcap = False
            except Exception:
                print("⚠️ Npcap not accessible - using Layer 3 mode")
                conf.use_pcap = False
                
        except Exception as e:
            print(f"⚠️ Could not configure packet capture: {e}")
            
except ImportError as e:
    print(f"⚠️ Scapy import failed: {e}")
    SCAPY_AVAILABLE = False

class RealNetworkCapture:
    """Captures real network packets and extracts features"""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'start_time': None,
            'packets': [],
            'fwd_packets': 0,
            'bwd_packets': 0,
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            'packet_lengths': [],
            'inter_arrival_times': [],
            'flags': defaultdict(int)
        })
        self.capturing = False
        self.processed_flows = []
        
    def packet_handler(self, packet):
        """Process captured packets - handles both Layer 2 and Layer 3 packets"""
        try:
            if not packet.haslayer(IP):
                return
                
            ip_layer = packet[IP]
            timestamp = time.time()
            
            # Create flow identifier
            if packet.haslayer(TCP):
                proto = 'TCP'
                sport = packet[TCP].sport
                dport = packet[TCP].dport
                flags = packet[TCP].flags
            elif packet.haslayer(UDP):
                proto = 'UDP'
                sport = packet[UDP].sport
                dport = packet[UDP].dport
                flags = 0
            else:
                return
                
            flow_id = f"{ip_layer.src}:{sport}-{ip_layer.dst}:{dport}-{proto}"
            reverse_flow_id = f"{ip_layer.dst}:{dport}-{ip_layer.src}:{sport}-{proto}"
            
            # Determine flow direction
            if flow_id in self.flows:
                current_flow = flow_id
                is_forward = True
            elif reverse_flow_id in self.flows:
                current_flow = reverse_flow_id
                is_forward = False
            else:
                current_flow = flow_id
                is_forward = True
                self.flows[current_flow]['start_time'] = timestamp
                
            flow = self.flows[current_flow]
            packet_len = len(packet)
            
            # Update flow statistics
            flow['packets'].append({
                'timestamp': timestamp,
                'length': packet_len,
                'is_forward': is_forward,
                'flags': flags
            })
            
            if is_forward:
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += packet_len
            else:
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += packet_len
                
            flow['packet_lengths'].append(packet_len)
            
            # Calculate inter-arrival times
            if len(flow['packets']) > 1:
                iat = timestamp - flow['packets'][-2]['timestamp']
                flow['inter_arrival_times'].append(iat)
                
            # TCP flags
            if packet.haslayer(TCP):
                tcp = packet[TCP]
                if tcp.flags & 0x01: flow['flags']['FIN'] += 1
                if tcp.flags & 0x02: flow['flags']['SYN'] += 1
                if tcp.flags & 0x04: flow['flags']['RST'] += 1
                if tcp.flags & 0x08: flow['flags']['PSH'] += 1
                if tcp.flags & 0x10: flow['flags']['ACK'] += 1
                if tcp.flags & 0x20: flow['flags']['URG'] += 1
        except Exception as e:
            # Silently ignore packet processing errors to avoid flooding logs
            pass
    
    def extract_features(self, flow_data):
        """Extract 78 features from flow data"""
        packets = flow_data['packets']
        if not packets:
            return None
            
        duration = (packets[-1]['timestamp'] - packets[0]['timestamp']) * 1000  # ms
        
        # Basic counts
        total_fwd = flow_data['fwd_packets']
        total_bwd = flow_data['bwd_packets']
        total_packets = total_fwd + total_bwd
        
        if total_packets == 0:
            return None
            
        # Packet lengths
        lengths = flow_data['packet_lengths']
        fwd_lengths = [p['length'] for p in packets if p['is_forward']]
        bwd_lengths = [p['length'] for p in packets if not p['is_forward']]
        
        # Inter-arrival times
        iats = flow_data['inter_arrival_times']
        fwd_iats = []
        bwd_iats = []
        
        for i in range(1, len(packets)):
            iat = (packets[i]['timestamp'] - packets[i-1]['timestamp']) * 1000
            if packets[i]['is_forward']:
                fwd_iats.append(iat)
            else:
                bwd_iats.append(iat)
        
        # Helper functions for statistics
        def safe_mean(lst): return sum(lst) / len(lst) if lst else 0
        def safe_std(lst): 
            if len(lst) < 2: return 0
            mean = safe_mean(lst)
            return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        def safe_max(lst): return max(lst) if lst else 0
        def safe_min(lst): return min(lst) if lst else 0
        
        # Extract all 78 features
        features = {
            "Flow Duration": duration,
            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,
            "Fwd Packet Length Max": safe_max(fwd_lengths),
            "Fwd Packet Length Min": safe_min(fwd_lengths),
            "Fwd Packet Length Mean": safe_mean(fwd_lengths),
            "Fwd Packet Length Std": safe_std(fwd_lengths),
            "Bwd Packet Length Max": safe_max(bwd_lengths),
            "Bwd Packet Length Min": safe_min(bwd_lengths),
            "Bwd Packet Length Mean": safe_mean(bwd_lengths),
            "Destination Port": 80,  # Would need to extract from flow_id
            "Total Length of Fwd Packets": flow_data['fwd_bytes'],
            "Total Length of Bwd Packets": flow_data['bwd_bytes'],
            "Bwd Packet Length Std": safe_std(bwd_lengths),
            "Flow Bytes/s": (flow_data['fwd_bytes'] + flow_data['bwd_bytes']) / (duration / 1000) if duration > 0 else 0,
            "Flow Packets/s": total_packets / (duration / 1000) if duration > 0 else 0,
            "Flow IAT Mean": safe_mean(iats),
            "Flow IAT Std": safe_std(iats),
            "Flow IAT Max": safe_max(iats),
            "Flow IAT Min": safe_min(iats),
            "Fwd IAT Total": sum(fwd_iats),
            "Fwd IAT Mean": safe_mean(fwd_iats),
            "Fwd IAT Std": safe_std(fwd_iats),
            "Fwd IAT Max": safe_max(fwd_iats),
            "Fwd IAT Min": safe_min(fwd_iats),
            "Bwd IAT Total": sum(bwd_iats),
            "Bwd IAT Mean": safe_mean(bwd_iats),
            "Bwd IAT Std": safe_std(bwd_iats),
            "Bwd IAT Max": safe_max(bwd_iats),
            "Bwd IAT Min": safe_min(bwd_iats),
            "Fwd PSH Flags": flow_data['flags']['PSH'],
            "Bwd PSH Flags": 0,  # Would need bidirectional analysis
            "Fwd URG Flags": flow_data['flags']['URG'],
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,  # Typical TCP header
            "Bwd Header Length": 20,
            "Fwd Packets/s": total_fwd / (duration / 1000) if duration > 0 else 0,
            "Bwd Packets/s": total_bwd / (duration / 1000) if duration > 0 else 0,
            "Min Packet Length": safe_min(lengths),
            "Max Packet Length": safe_max(lengths),
            "Packet Length Mean": safe_mean(lengths),
            "Packet Length Std": safe_std(lengths),
            "Packet Length Variance": safe_std(lengths) ** 2,
            "FIN Flag Count": flow_data['flags']['FIN'],
            "SYN Flag Count": flow_data['flags']['SYN'],
            "RST Flag Count": flow_data['flags']['RST'],
            "PSH Flag Count": flow_data['flags']['PSH'],
            "ACK Flag Count": flow_data['flags']['ACK'],
            "URG Flag Count": flow_data['flags']['URG'],
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": total_bwd / total_fwd if total_fwd > 0 else 0,
            "Average Packet Size": safe_mean(lengths),
            "Avg Fwd Segment Size": safe_mean(fwd_lengths),
            "Avg Bwd Segment Size": safe_mean(bwd_lengths),
            "Fwd Header Length.1": 0,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": total_fwd,
            "Subflow Fwd Bytes": flow_data['fwd_bytes'],
            "Subflow Bwd Packets": total_bwd,
            "Subflow Bwd Bytes": flow_data['bwd_bytes'],
            "Init_Win_bytes_forward": 8192,  # Default
            "Init_Win_bytes_backward": 8192,
            "act_data_pkt_fwd": total_fwd,
            "min_seg_size_forward": safe_min(fwd_lengths),
            "Active Mean": 0,  # Would need session analysis
            "Active Std": 0,
            "Active Max": 0,
            "Active Min": 0,
            "Idle Mean": 0,
            "Idle Std": 0,
            "Idle Max": 0,
            "Idle Min": 0
        }
        
        return features
    
    def start_capture(self, interface=None, duration=10):
        """Start packet capture - auto-detects Npcap or falls back to Layer 3"""
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy not properly configured.")
            
        self.capturing = True
        self.flows.clear()
        
        def capture_thread():
            try:
                import platform
                if platform.system() == "Windows":
                    print("🔍 Starting Windows packet capture...")
                    
                    # Try Npcap first, then fallback to Layer 3
                    try:
                        # Test if Npcap works with a simple capture
                        if conf.use_pcap:
                            print("📡 Attempting Npcap capture...")
                            sniff(
                                prn=self.packet_handler,
                                timeout=duration,
                                filter="tcp or udp",
                                store=False,
                                count=0,
                                iface=interface  # Let scapy choose interface if None
                            )
                        else:
                            raise Exception("Npcap not available, trying Layer 3")
                            
                    except Exception as npcap_error:
                        print(f"📡 Npcap capture failed: {npcap_error}")
                        print("🔄 Falling back to Layer 3 mode...")
                        
                        # Fallback to Layer 3 capture
                        conf.use_pcap = False
                        sniff(
                            prn=self.packet_handler,
                            timeout=duration,
                            filter="ip",
                            store=False,
                            count=0
                        )
                else:
                    # Unix-like systems
                    sniff(
                        iface=interface,
                        prn=self.packet_handler,
                        timeout=duration,
                        filter="tcp or udp",
                        store=False
                    )
                    
                print("✅ Packet capture completed successfully")
                
            except PermissionError:
                print("❌ Permission denied: Run VS Code as Administrator")
                print("💡 Right-click VS Code → 'Run as administrator'")
            except Exception as e:
                print(f"❌ Capture failed: {e}")
                if "winpcap" in str(e).lower() or "npcap" in str(e).lower():
                    print("💡 Install Npcap from: https://nmap.org/npcap/")
                    print("💡 During install, check 'WinPcap API-compatible mode'")
                print("💡 Alternative: Use 'Simulated Traffic' mode (no drivers needed)")
            finally:
                self.capturing = False
            
        thread = threading.Thread(target=capture_thread, name="capture_thread")
        thread.daemon = True
        thread.start()
        return thread
    
    def get_completed_flows(self):
        """Get flows that have enough data for feature extraction"""
        completed = []
        current_time = time.time()
        
        for flow_id, flow_data in list(self.flows.items()):
            # Consider flow complete if no packets for 5 seconds
            if (flow_data['packets'] and 
                current_time - flow_data['packets'][-1]['timestamp'] > 5 and
                len(flow_data['packets']) >= 5):
                
                features = self.extract_features(flow_data)
                if features:
                    completed.append({
                        'flow_id': flow_id,
                        'features': features,
                        'timestamp': datetime.now()
                    })
                
                # Remove processed flow
                del self.flows[flow_id]
                
        return completed