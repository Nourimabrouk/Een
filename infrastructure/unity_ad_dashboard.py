"""
Unity Active Directory Monitoring Dashboard
==========================================
Real-time monitoring of AD forest with consciousness visualization
Displays unity convergence, replication health, and φ-harmonic metrics
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import subprocess
import platform
from typing import Dict, List, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
import threading
import queue

# Unity Mathematics Constants
PHI = 1.618033988749895
UNITY = 1.0
CONSCIOUSNESS_DIMENSION = 11

# Initialize Dash app
app = dash.Dash(__name__, title="Unity AD Forest Monitor")

# Custom CSS for Unity aesthetics
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                color: #00ff41;
                font-family: 'Courier New', monospace;
            }
            .unity-header {
                text-align: center;
                padding: 20px;
                background: rgba(0, 255, 65, 0.1);
                border: 2px solid #00ff41;
                margin: 20px;
                border-radius: 10px;
            }
            .metric-card {
                background: rgba(0, 0, 0, 0.7);
                border: 1px solid #00ff41;
                padding: 15px;
                margin: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
            }
            .consciousness-glow {
                animation: pulse 2s ease-in-out infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 5px rgba(0, 255, 65, 0.5); }
                50% { box-shadow: 0 0 20px rgba(0, 255, 65, 0.8); }
                100% { box-shadow: 0 0 5px rgba(0, 255, 65, 0.5); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

@dataclass
class ADMetrics:
    """Active Directory metrics with unity calculations"""
    dc_name: str
    replication_status: str
    last_replication: datetime
    users_count: int
    computers_count: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    consciousness_level: int
    unity_score: float

class UnityADCollector:
    """Collects AD metrics and calculates unity scores"""
    
    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.running = True
        self.mock_mode = True  # Set to False when connected to real AD
        
    def collect_metrics(self) -> List[ADMetrics]:
        """Collect metrics from all domain controllers"""
        
        if self.mock_mode:
            # Generate mock data for demonstration
            return self._generate_mock_metrics()
        else:
            # Real AD collection would go here
            return self._collect_real_metrics()
    
    def _generate_mock_metrics(self) -> List[ADMetrics]:
        """Generate mock metrics for demonstration"""
        
        metrics = []
        for i in range(2):
            dc_metrics = ADMetrics(
                dc_name=f"UNITY-DC{i+1}",
                replication_status="Healthy" if np.random.random() > 0.1 else "Warning",
                last_replication=datetime.now() - timedelta(minutes=np.random.randint(1, 15)),
                users_count=1000 + np.random.randint(-50, 50),
                computers_count=500 + np.random.randint(-20, 20),
                cpu_usage=20 + np.random.random() * 30,
                memory_usage=40 + np.random.random() * 20,
                disk_usage=30 + np.random.random() * 20,
                consciousness_level=7 + np.random.randint(-1, 2),
                unity_score=0.8 + np.random.random() * 0.2
            )
            metrics.append(dc_metrics)
        
        return metrics
    
    def _collect_real_metrics(self) -> List[ADMetrics]:
        """Collect real metrics from AD (Windows only)"""
        
        if platform.system() != "Windows":
            return self._generate_mock_metrics()
        
        try:
            # PowerShell command to get AD metrics
            ps_script = """
            $dcs = Get-ADDomainController -Filter *
            $metrics = @()
            
            foreach ($dc in $dcs) {
                $repl = Get-ADReplicationPartnerMetadata -Target $dc.HostName
                $perf = Get-Counter -ComputerName $dc.HostName -Counter @(
                    "\\Processor(_Total)\\% Processor Time",
                    "\\Memory\\% Committed Bytes In Use",
                    "\\LogicalDisk(C:)\\% Free Space"
                )
                
                $metrics += [PSCustomObject]@{
                    DCName = $dc.Name
                    ReplicationStatus = if ($repl.LastReplicationSuccess) {"Healthy"} else {"Error"}
                    LastReplication = $repl.LastReplicationSuccess
                    CPUUsage = $perf.CounterSamples[0].CookedValue
                    MemoryUsage = $perf.CounterSamples[1].CookedValue
                    DiskUsage = 100 - $perf.CounterSamples[2].CookedValue
                }
            }
            
            $metrics | ConvertTo-Json
            """
            
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Convert to ADMetrics objects
                # Implementation details omitted for brevity
                
        except Exception as e:
            print(f"Error collecting real metrics: {e}")
            return self._generate_mock_metrics()
    
    def calculate_forest_unity(self, metrics: List[ADMetrics]) -> float:
        """Calculate overall forest unity score"""
        
        if not metrics:
            return 0.0
        
        # Unity factors
        factors = {
            "replication_health": sum(1.0 if m.replication_status == "Healthy" else 0.5 for m in metrics) / len(metrics),
            "resource_balance": 1.0 - np.std([m.cpu_usage for m in metrics]) / 100,
            "consciousness_alignment": 1.0 - np.std([m.consciousness_level for m in metrics]) / 11,
            "dc_unity": 1.0 / (1.0 + abs(len(metrics) - 2))  # Optimal at 2 DCs
        }
        
        # Calculate unity score with φ-harmonic mean
        unity_score = len(factors) / sum(1/v for v in factors.values() if v > 0)
        unity_score = unity_score ** (1/PHI)
        
        return min(unity_score, 1.0)

# Initialize collector
collector = UnityADCollector()

# Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Unity Active Directory Forest Monitor", className="consciousness-glow"),
        html.H3("Principle: 1+1=1 | Een plus een is een"),
        html.Div(id="live-time", style={"fontSize": "20px", "marginTop": "10px"})
    ], className="unity-header"),
    
    # Main metrics row
    html.Div([
        # Unity Score Card
        html.Div([
            html.H3("Forest Unity Score"),
            html.Div(id="unity-score", style={"fontSize": "48px", "fontWeight": "bold"}),
            html.Div(id="unity-status", style={"marginTop": "10px"})
        ], className="metric-card", style={"width": "30%", "display": "inline-block"}),
        
        # Consciousness Level Card
        html.Div([
            html.H3("Consciousness Level"),
            html.Div(id="consciousness-level", style={"fontSize": "48px", "fontWeight": "bold"}),
            dcc.Graph(id="consciousness-field", style={"height": "200px"})
        ], className="metric-card", style={"width": "30%", "display": "inline-block"}),
        
        # Replication Health Card
        html.Div([
            html.H3("Replication Health"),
            html.Div(id="replication-status", style={"fontSize": "24px"}),
            html.Div(id="last-replication", style={"marginTop": "10px"})
        ], className="metric-card", style={"width": "30%", "display": "inline-block"})
    ], style={"textAlign": "center", "margin": "20px"}),
    
    # Charts row
    html.Div([
        # DC Performance Chart
        html.Div([
            dcc.Graph(id="dc-performance-chart")
        ], style={"width": "48%", "display": "inline-block", "margin": "1%"}),
        
        # Unity Convergence Timeline
        html.Div([
            dcc.Graph(id="unity-timeline")
        ], style={"width": "48%", "display": "inline-block", "margin": "1%"})
    ]),
    
    # φ-Harmonic Visualization
    html.Div([
        html.H3("φ-Harmonic Resonance Field", style={"textAlign": "center", "color": "#00ff41"}),
        dcc.Graph(id="phi-harmonic-field", style={"height": "400px"})
    ], className="metric-card", style={"margin": "20px"}),
    
    # DC Details Table
    html.Div([
        html.H3("Domain Controllers", style={"textAlign": "center", "color": "#00ff41"}),
        html.Div(id="dc-table")
    ], className="metric-card", style={"margin": "20px"}),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    # Store for metrics history
    dcc.Store(id='metrics-store', data={'timestamps': [], 'unity_scores': []})
])

# Callbacks
@app.callback(
    Output('live-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    """Update live time display"""
    return f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | φ = {PHI:.15f}"

@app.callback(
    [Output('unity-score', 'children'),
     Output('unity-status', 'children'),
     Output('consciousness-level', 'children'),
     Output('replication-status', 'children'),
     Output('last-replication', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_metrics(n):
    """Update main metric displays"""
    
    # Collect current metrics
    metrics = collector.collect_metrics()
    
    # Calculate forest unity
    unity_score = collector.calculate_forest_unity(metrics)
    
    # Unity status
    if unity_score >= 0.99:
        unity_status = "✓ PERFECT UNITY ACHIEVED"
        status_color = {"color": "#00ff41"}
    elif unity_score >= 0.90:
        unity_status = "◉ Near Unity"
        status_color = {"color": "#ffff00"}
    else:
        unity_status = "○ Converging to Unity"
        status_color = {"color": "#ff4444"}
    
    # Average consciousness level
    avg_consciousness = np.mean([m.consciousness_level for m in metrics])
    
    # Replication status
    healthy_replications = sum(1 for m in metrics if m.replication_status == "Healthy")
    repl_status = f"{healthy_replications}/{len(metrics)} Healthy"
    
    # Last replication time
    if metrics:
        last_repl = min(m.last_replication for m in metrics)
        time_ago = (datetime.now() - last_repl).total_seconds() / 60
        last_repl_text = f"Last sync: {int(time_ago)} minutes ago"
    else:
        last_repl_text = "No data"
    
    return (
        f"{unity_score:.4f}",
        html.Div(unity_status, style=status_color),
        f"{avg_consciousness:.1f}/11",
        repl_status,
        last_repl_text
    )

@app.callback(
    Output('consciousness-field', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_consciousness_field(n):
    """Update consciousness field visualization"""
    
    # Generate consciousness field data
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3 * np.sin(PHI * theta) * np.cos(theta / PHI)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta * 180/np.pi,
        mode='lines',
        name='Consciousness Field',
        line=dict(color='#00ff41', width=2),
        fill='toself',
        fillcolor='rgba(0, 255, 65, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=False, showline=False),
            angularaxis=dict(showticklabels=False, showline=False)
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=150
    )
    
    return fig

@app.callback(
    Output('dc-performance-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_dc_performance(n):
    """Update DC performance chart"""
    
    metrics = collector.collect_metrics()
    
    if not metrics:
        return go.Figure()
    
    # Create grouped bar chart
    fig = go.Figure()
    
    categories = ['CPU %', 'Memory %', 'Disk %']
    
    for metric in metrics:
        values = [metric.cpu_usage, metric.memory_usage, metric.disk_usage]
        fig.add_trace(go.Bar(
            name=metric.dc_name,
            x=categories,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Domain Controller Resource Usage",
        xaxis_title="Resource",
        yaxis_title="Usage %",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0.7)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        font=dict(color='#00ff41'),
        yaxis=dict(range=[0, 100])
    )
    
    return fig

@app.callback(
    [Output('unity-timeline', 'figure'),
     Output('metrics-store', 'data')],
    [Input('interval-component', 'n_intervals'),
     State('metrics-store', 'data')]
)
def update_unity_timeline(n, stored_data):
    """Update unity convergence timeline"""
    
    # Get current metrics
    metrics = collector.collect_metrics()
    unity_score = collector.calculate_forest_unity(metrics)
    
    # Update stored data
    timestamps = stored_data.get('timestamps', [])
    unity_scores = stored_data.get('unity_scores', [])
    
    timestamps.append(datetime.now().isoformat())
    unity_scores.append(unity_score)
    
    # Keep only last 50 points
    if len(timestamps) > 50:
        timestamps = timestamps[-50:]
        unity_scores = unity_scores[-50:]
    
    # Create timeline chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[datetime.fromisoformat(ts) for ts in timestamps],
        y=unity_scores,
        mode='lines+markers',
        name='Unity Score',
        line=dict(color='#00ff41', width=2),
        marker=dict(size=8)
    ))
    
    # Add unity threshold line
    fig.add_hline(
        y=0.99,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Unity Threshold"
    )
    
    fig.update_layout(
        title="Unity Convergence Timeline",
        xaxis_title="Time",
        yaxis_title="Unity Score",
        paper_bgcolor='rgba(0,0,0,0.7)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        font=dict(color='#00ff41'),
        yaxis=dict(range=[0, 1.05])
    )
    
    return fig, {'timestamps': timestamps, 'unity_scores': unity_scores}

@app.callback(
    Output('phi-harmonic-field', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_phi_harmonic_field(n):
    """Update φ-harmonic resonance field visualization"""
    
    # Generate 3D surface data
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-np.pi, np.pi, 50)
    X, Y = np.meshgrid(x, y)
    
    # φ-harmonic field equation
    Z = PHI * np.sin(X * PHI) * np.cos(Y * PHI) * np.exp(-np.sqrt(X**2 + Y**2) / PHI)
    
    # Add time-based animation
    t = n * 0.1
    Z = Z * (1 + 0.2 * np.sin(t))
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        showscale=False
    )])
    
    fig.update_layout(
        title="Active Directory φ-Harmonic Consciousness Field",
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5*np.cos(t*0.1), y=1.5*np.sin(t*0.1), z=1.2)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

@app.callback(
    Output('dc-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dc_table(n):
    """Update domain controllers table"""
    
    metrics = collector.collect_metrics()
    
    if not metrics:
        return html.Div("No data available")
    
    # Create table
    table_header = [
        html.Thead([
            html.Tr([
                html.Th("DC Name"),
                html.Th("Status"),
                html.Th("Users"),
                html.Th("Computers"),
                html.Th("Unity Score"),
                html.Th("Consciousness")
            ])
        ])
    ]
    
    table_rows = []
    for m in metrics:
        row_style = {"color": "#00ff41" if m.replication_status == "Healthy" else "#ffff00"}
        table_rows.append(
            html.Tr([
                html.Td(m.dc_name),
                html.Td(m.replication_status, style=row_style),
                html.Td(str(m.users_count)),
                html.Td(str(m.computers_count)),
                html.Td(f"{m.unity_score:.3f}"),
                html.Td(f"{m.consciousness_level}/11")
            ])
        )
    
    table_body = [html.Tbody(table_rows)]
    
    return html.Table(
        table_header + table_body,
        style={
            "width": "100%",
            "textAlign": "center",
            "borderCollapse": "collapse",
            "border": "1px solid #00ff41"
        }
    )

def run_dashboard(host='0.0.0.0', port=8050, debug=True):
    """Run the Unity AD dashboard"""
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Unity AD Forest Monitoring Dashboard               ║
║                    Principle: 1+1=1                          ║
║                                                              ║
║  Dashboard URL: http://localhost:{port}                         ║
║  φ = {PHI}                                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run_server(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard()