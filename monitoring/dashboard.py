#!/usr/bin/env python3
"""
Real-time monitoring dashboard for Photonic Flash Attention system.
"""

import time
import json
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    temperature: float
    power_consumption: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhotonicMetrics:
    """Photonic device metrics."""
    device_id: str
    timestamp: float
    optical_power: float
    wavelength_efficiency: float
    thermal_drift: float
    calibration_accuracy: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collects system and photonic metrics."""
    
    def __init__(self):
        self.system_metrics: List[SystemMetrics] = []
        self.photonic_metrics: Dict[str, List[PhotonicMetrics]] = {}
        self.attention_stats: List[Dict[str, Any]] = []
        
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread = None
        
        # Configuration
        self.collection_interval = 1.0  # seconds
        self.max_history_size = 3600  # 1 hour at 1Hz
    
    def start_collection(self):
        """Start metrics collection thread."""
        if self._running:
            return
            
        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop)
        self._collection_thread.daemon = True
        self._collection_thread.start()
        print("📊 Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        print("📊 Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                # Collect system metrics
                sys_metrics = self._collect_system_metrics()
                if sys_metrics:
                    with self._lock:
                        self.system_metrics.append(sys_metrics)
                        if len(self.system_metrics) > self.max_history_size:
                            self.system_metrics.pop(0)
                
                # Collect photonic metrics
                self._collect_photonic_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Metrics collection error: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        timestamp = time.time()
        
        # CPU usage
        cpu_usage = psutil.cpu_percent() if psutil else 0.0
        
        # Memory usage
        if psutil:
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
        else:
            memory_usage = 0.0
        
        # GPU memory (mock - would use nvidia-ml-py or similar)
        gpu_memory_usage = self._get_gpu_memory_usage()
        
        # Temperature (mock - would read from sensors)
        temperature = self._get_system_temperature()
        
        # Power consumption (mock - would read from PMU)
        power_consumption = self._get_power_consumption()
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            temperature=temperature,
            power_consumption=power_consumption,
        )
    
    def _collect_photonic_metrics(self):
        """Collect photonic device metrics."""
        # Mock photonic device metrics
        # In real implementation, would query actual hardware
        
        devices = ['lightmatter:0', 'simulation:0']
        
        for device_id in devices:
            metrics = PhotonicMetrics(
                device_id=device_id,
                timestamp=time.time(),
                optical_power=self._get_optical_power(device_id),
                wavelength_efficiency=self._get_wavelength_efficiency(device_id),
                thermal_drift=self._get_thermal_drift(device_id),
                calibration_accuracy=self._get_calibration_accuracy(device_id),
                error_rate=self._get_error_rate(device_id),
            )
            
            with self._lock:
                if device_id not in self.photonic_metrics:
                    self.photonic_metrics[device_id] = []
                
                self.photonic_metrics[device_id].append(metrics)
                
                # Limit history size
                if len(self.photonic_metrics[device_id]) > self.max_history_size:
                    self.photonic_metrics[device_id].pop(0)
    
    def add_attention_stats(self, stats: Dict[str, Any]):
        """Add attention performance statistics."""
        stats_with_timestamp = {
            'timestamp': time.time(),
            **stats
        }
        
        with self._lock:
            self.attention_stats.append(stats_with_timestamp)
            if len(self.attention_stats) > self.max_history_size:
                self.attention_stats.pop(0)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                'system': self.system_metrics[-1].to_dict() if self.system_metrics else None,
                'photonic': {
                    device_id: metrics[-1].to_dict() if metrics else None
                    for device_id, metrics in self.photonic_metrics.items()
                },
                'attention': self.attention_stats[-10:] if self.attention_stats else [],
                'timestamp': time.time(),
            }
    
    def get_historical_metrics(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get historical metrics for the specified duration."""
        cutoff_time = time.time() - duration_seconds
        
        with self._lock:
            # Filter system metrics
            recent_system = [
                m.to_dict() for m in self.system_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            # Filter photonic metrics
            recent_photonic = {}
            for device_id, metrics in self.photonic_metrics.items():
                recent_photonic[device_id] = [
                    m.to_dict() for m in metrics
                    if m.timestamp >= cutoff_time
                ]
            
            # Filter attention stats
            recent_attention = [
                stats for stats in self.attention_stats
                if stats['timestamp'] >= cutoff_time
            ]
            
            return {
                'system': recent_system,
                'photonic': recent_photonic,
                'attention': recent_attention,
                'duration_seconds': duration_seconds,
                'timestamp': time.time(),
            }
    
    # Mock hardware interface methods
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage."""
        # Mock implementation
        import random
        return random.uniform(20, 80)
    
    def _get_system_temperature(self) -> float:
        """Get system temperature."""
        # Mock implementation
        import random
        return random.uniform(35, 65)  # 35-65°C
    
    def _get_power_consumption(self) -> float:
        """Get power consumption in watts."""
        # Mock implementation
        import random
        return random.uniform(150, 300)  # 150-300W
    
    def _get_optical_power(self, device_id: str) -> float:
        """Get optical power for device."""
        import random
        return random.uniform(1.0, 10.0)  # 1-10 mW
    
    def _get_wavelength_efficiency(self, device_id: str) -> float:
        """Get wavelength efficiency."""
        import random
        return random.uniform(0.85, 0.95)
    
    def _get_thermal_drift(self, device_id: str) -> float:
        """Get thermal drift in nm."""
        import random
        return random.uniform(-0.1, 0.1)
    
    def _get_calibration_accuracy(self, device_id: str) -> float:
        """Get calibration accuracy."""
        import random
        return random.uniform(0.95, 0.99)
    
    def _get_error_rate(self, device_id: str) -> float:
        """Get error rate."""
        import random
        return random.uniform(0.001, 0.01)


class DashboardServer:
    """HTTP server for metrics dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8080):
        self.metrics_collector = metrics_collector
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start dashboard server."""
        handler = self._create_handler()
        self.server = socketserver.TCPServer(("", self.port), handler)
        self.server.allow_reuse_address = True
        
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        print(f"🌐 Dashboard server started on http://localhost:{self.port}")
    
    def stop(self):
        """Stop dashboard server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        print("🌐 Dashboard server stopped")
    
    def _create_handler(self):
        """Create HTTP request handler."""
        metrics_collector = self.metrics_collector
        
        class DashboardHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse(self.path)
                path = parsed_path.path
                query_params = parse_qs(parsed_path.query)
                
                if path == '/':
                    self._serve_dashboard()
                elif path == '/api/metrics':
                    self._serve_current_metrics()
                elif path == '/api/history':
                    duration = int(query_params.get('duration', ['300'])[0])
                    self._serve_historical_metrics(duration)
                elif path == '/api/health':
                    self._serve_health_check()
                else:
                    self._serve_404()
            
            def _serve_dashboard(self):
                """Serve main dashboard HTML."""
                html = self._generate_dashboard_html()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Content-length', str(len(html)))
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            
            def _serve_current_metrics(self):
                """Serve current metrics as JSON."""
                metrics = metrics_collector.get_current_metrics()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                json_data = json.dumps(metrics, indent=2)
                self.wfile.write(json_data.encode('utf-8'))
            
            def _serve_historical_metrics(self, duration: int):
                """Serve historical metrics as JSON."""
                metrics = metrics_collector.get_historical_metrics(duration)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                json_data = json.dumps(metrics, indent=2)
                self.wfile.write(json_data.encode('utf-8'))
            
            def _serve_health_check(self):
                """Serve health check."""
                health = {
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'uptime': time.time() - start_time,
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                json_data = json.dumps(health, indent=2)
                self.wfile.write(json_data.encode('utf-8'))
            
            def _serve_404(self):
                """Serve 404 error."""
                self.send_error(404, 'Not Found')
            
            def _generate_dashboard_html(self) -> str:
                """Generate dashboard HTML."""
                return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photonic Flash Attention - Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; margin-bottom: 10px; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 10px 20px; 
                      border-radius: 5px; cursor: pointer; margin: 10px 0; }
        .refresh-btn:hover { background: #5a6fd8; }
        #log { background: #333; color: #0f0; font-family: monospace; padding: 15px; 
               border-radius: 5px; height: 200px; overflow-y: auto; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Photonic Flash Attention Dashboard</h1>
            <p>Real-time monitoring of photonic neural acceleration system</p>
        </div>
        
        <button class="refresh-btn" onclick="refreshMetrics()">🔄 Refresh</button>
        
        <div class="metrics-grid" id="metricsGrid">
            <!-- Metrics will be populated here -->
        </div>
        
        <div id="log"></div>
    </div>
    
    <script>
        let startTime = Date.now();
        
        function refreshMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                    logMessage('Error: ' + error.message, 'error');
                });
        }
        
        function updateDashboard(data) {
            const grid = document.getElementById('metricsGrid');
            grid.innerHTML = '';
            
            // System metrics
            if (data.system) {
                grid.appendChild(createMetricCard('CPU Usage', data.system.cpu_usage.toFixed(1) + '%', 
                    getStatusClass(data.system.cpu_usage, 80, 90)));
                grid.appendChild(createMetricCard('Memory Usage', data.system.memory_usage.toFixed(1) + '%',
                    getStatusClass(data.system.memory_usage, 80, 90)));
                grid.appendChild(createMetricCard('Temperature', data.system.temperature.toFixed(1) + '°C',
                    getStatusClass(data.system.temperature, 60, 75)));
                grid.appendChild(createMetricCard('Power', data.system.power_consumption.toFixed(0) + 'W', 'status-good'));
            }
            
            // Photonic metrics
            for (const [deviceId, metrics] of Object.entries(data.photonic)) {
                if (metrics) {
                    grid.appendChild(createMetricCard(
                        `${deviceId} - Optical Power`, 
                        metrics.optical_power.toFixed(2) + ' mW', 
                        'status-good'
                    ));
                    grid.appendChild(createMetricCard(
                        `${deviceId} - Efficiency`, 
                        (metrics.wavelength_efficiency * 100).toFixed(1) + '%',
                        getStatusClass(metrics.wavelength_efficiency * 100, 85, 90)
                    ));
                    grid.appendChild(createMetricCard(
                        `${deviceId} - Calibration`, 
                        (metrics.calibration_accuracy * 100).toFixed(2) + '%',
                        getStatusClass(metrics.calibration_accuracy * 100, 95, 97)
                    ));
                }
            }
            
            // Attention stats
            if (data.attention.length > 0) {
                const latest = data.attention[data.attention.length - 1];
                grid.appendChild(createMetricCard('Last Device Used', latest.last_device_used || 'unknown', 'status-good'));
                
                if (latest.last_latency_ms) {
                    grid.appendChild(createMetricCard('Attention Latency', 
                        latest.last_latency_ms.toFixed(2) + ' ms', 'status-good'));
                }
            }
            
            logMessage('Dashboard updated', 'info');
        }
        
        function createMetricCard(label, value, statusClass) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-label">${label}</div>
                <div class="metric-value ${statusClass}">${value}</div>
            `;
            return card;
        }
        
        function getStatusClass(value, warningThreshold, errorThreshold) {
            if (value >= errorThreshold) return 'status-error';
            if (value >= warningThreshold) return 'status-warning';
            return 'status-good';
        }
        
        function logMessage(message, level) {
            const log = document.getElementById('log');
            const timestamp = new Date().toISOString().substr(11, 8);
            const prefix = level === 'error' ? '❌' : level === 'warning' ? '⚠️' : 'ℹ️';
            log.innerHTML += `[${timestamp}] ${prefix} ${message}\\n`;
            log.scrollTop = log.scrollHeight;
        }
        
        // Initial load and auto-refresh
        refreshMetrics();
        setInterval(refreshMetrics, 5000); // Refresh every 5 seconds
        
        logMessage('Dashboard initialized', 'info');
    </script>
</body>
</html>
                '''
            
            def log_message(self, format, *args):
                """Suppress default request logging."""
                pass
        
        return DashboardHandler


# Global start time for uptime calculation
start_time = time.time()


def create_monitoring_system():
    """Create and configure the monitoring system."""
    collector = MetricsCollector()
    dashboard = DashboardServer(collector, port=8080)
    
    return collector, dashboard


def main():
    """Run standalone monitoring system."""
    print("🚀 Starting Photonic Flash Attention Monitoring System")
    print("="*60)
    
    collector, dashboard = create_monitoring_system()
    
    try:
        # Start metrics collection
        collector.start_collection()
        
        # Start dashboard server
        dashboard.start()
        
        print("✅ Monitoring system running")
        print("   Dashboard: http://localhost:8080")
        print("   API: http://localhost:8080/api/metrics")
        print("\nPress Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring system...")
        
    finally:
        collector.stop_collection()
        dashboard.stop()
        print("✅ Monitoring system stopped")


if __name__ == "__main__":
    main()