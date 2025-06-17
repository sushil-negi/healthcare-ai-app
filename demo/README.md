# MLOps Platform Demo Guide

## 📁 Demo Files Overview

### Core Demo Scripts
- **`quick_start.py`** - Main demo manager with executive presentation
- **`customer_dashboard_demo.py`** - Interactive dashboards with auto-port detection
- **`integrated_mlops_demo.py`** - Complete MLOps platform demonstration
- **`mlops_platform_demo.py`** - Detailed component walkthrough
- **`launch_complete_demo.sh`** - One-click launcher with service validation

### Dashboard Files
- **`unified_dashboard.html`** - 🚀 **All-in-one tabbed dashboard with visual workflow**

## 🚀 Quick Start Options

### 🎯 Quick Start Demo Commands
```bash
# Executive demo (5 minutes)
python3 demo/quick_start.py exec

# Start dashboard server
python3 demo/quick_start.py start

# Check server status
python3 demo/quick_start.py status

# Stop demo server
python3 demo/quick_start.py stop
```

### Individual Demo Scripts

#### Option 1: Customer Dashboard Demo (10 minutes)
```bash
python3 demo/customer_dashboard_demo.py
```
Visual dashboards with automatic browser opening and port detection. **Now opens unified dashboard by default.**

#### Option 2: Complete Technical Demo (20 minutes)
```bash
python3 demo/integrated_mlops_demo.py
```
Full platform demonstration with component integration. **Now features unified dashboard interface.**

#### Option 3: Simple Dashboard Access
```bash
cd demo
python3 -m http.server 8888
# Open http://localhost:8888/unified_dashboard.html
```
Manual dashboard access for exploration.

## 📊 Unified Dashboard Features

### 🏛️ Platform Tab
- Multi-tenant platform overview with enterprise metrics
- Interactive A/B testing visualization with statistical tracking
- Model registry status and deployment analytics
- Visual MLOps workflow pipeline with color-coded stages
- Platform architecture diagram with component relationships

### 📊 Monitoring Tab
- Real-time model drift detection charts (KS test, PSI analysis)
- Performance metrics trends and accuracy tracking
- Auto-retraining pipeline status and data quality metrics
- System health monitoring with service status indicators

### ⚡ Real-time Tab
- Live metric updates with auto-refresh capabilities
- Real-time activity feed showing platform operations
- System health status with service availability indicators
- Federated learning progress across participant organizations

## 🔧 Troubleshooting

### Port Conflicts
If you get port errors:
1. Check what's using the port: `lsof -i :8888`
2. Kill the process: `kill -9 [PID]`
3. Or use different ports in the scripts

### Dashboard Not Loading
1. Verify files exist: `ls demo/*.html`
2. Try different browser
3. Check browser console for errors
4. Use simple HTTP server: `python3 -m http.server 8888`

### Missing Dependencies
```bash
pip3 install requests numpy
```

## 🎯 Demo Scenarios

### Executive Demo (5 minutes)
- Show platform_dashboard.html
- Highlight multi-tenant capabilities
- Focus on business metrics and ROI

### Technical Demo (15 minutes)  
- Run `python3 demo/integrated_mlops_demo.py`
- Show all three dashboards
- Demonstrate MLOps components

### Comprehensive Demo (30 minutes)
- Run `./demo/launch_complete_demo.sh`
- Include live service integration
- Show end-to-end workflows

## 📱 URLs Quick Reference

Once servers are running:
- 🚀 **Unified Dashboard: http://localhost:8888/unified_dashboard.html**

## 🏥 Healthcare AI Integration

The dashboards showcase:
- 525K conversation training dataset
- 99%+ crisis detection accuracy
- HIPAA-compliant operations
- 13% model improvement via A/B testing
- Real-time safety monitoring

## ✨ Recent Improvements

**Unified Dashboard (NEW)**:
- 🚀 Single HTML file with all-in-one tabbed interface (platform, monitoring, real-time)
- 📐 Landscape workflow layout fitting completely in one view without scrolling
- 🎨 Enhanced visual workflow with optimized spacing and compact design
- 📱 Responsive design with breakpoints for desktop, tablet, and mobile
- 🔄 Consistent white backgrounds across all sections for professional appearance
- 🧹 Removed old dashboard generation code - now uses only static unified dashboard

## 🎯 Demo Comparison Guide

| Demo Script | Duration | Audience | Purpose | Dependencies |
|-------------|----------|----------|---------|--------------|
| `quick_start.py exec` | 5 min | C-Level, Sales | Business value presentation | None |
| `customer_dashboard_demo.py` | 10 min | Customers, Technical buyers | Visual platform overview with unified dashboard | HTTP server only |
| `integrated_mlops_demo.py` | 20+ min | Engineers, Technical teams | Complete platform validation with unified interface | Full MLOps services |

### When to Use Each Demo

**Quick Start Executive** - Business meetings, conference talks, no-setup presentations  
**Customer Dashboard Demo** - Sales demos, UI reviews, customer evaluations  
**Integrated MLOps Demo** - Technical deep-dives, POC validation, platform testing