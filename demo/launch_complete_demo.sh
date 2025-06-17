#!/bin/bash

# Enterprise MLOps Platform - Complete Demo Launcher
# Starts all services and launches comprehensive demonstration

set -e

echo "🚀 Enterprise MLOps Platform - Complete Demo Launcher"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [[ ! -f "demo/integrated_mlops_demo.py" ]]; then
    print_error "Please run this script from the healthcare-ai-app root directory"
    exit 1
fi

print_info "Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed"
    exit 1
fi

print_status "System requirements met"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start MLOps platform if not running
start_mlops_platform() {
    print_info "Checking MLOps platform services..."
    
    cd ../mlops-platform 2>/dev/null || {
        print_warning "MLOps platform directory not found, continuing with demo"
        cd ..
        return
    }
    
    # Check if services are running
    if check_port 8000 && check_port 8003 && check_port 8090; then
        print_status "MLOps platform services already running"
    else
        print_info "Starting MLOps platform services..."
        docker compose -f docker-compose.platform.yml up -d
        
        # Wait for services to be ready
        print_info "Waiting for services to start..."
        sleep 10
        
        # Verify services
        local retries=12
        while [ $retries -gt 0 ]; do
            if check_port 8000 && check_port 8003 && check_port 8090; then
                print_status "MLOps platform services ready"
                break
            fi
            print_info "Waiting for services... ($retries retries left)"
            sleep 5
            ((retries--))
        done
        
        if [ $retries -eq 0 ]; then
            print_warning "Some MLOps services may not be ready, continuing anyway"
        fi
    fi
    
    cd ../healthcare-ai-app
}

# Function to start healthcare AI services
start_healthcare_services() {
    print_info "Checking Healthcare AI services..."
    
    if check_port 8001 && check_port 8889; then
        print_status "Healthcare AI services already running"
    else
        print_info "Starting Healthcare AI services..."
        docker compose -f docker-compose.healthcare-ci.yml up -d
        
        # Wait for services
        print_info "Waiting for Healthcare AI services..."
        sleep 5
        
        local retries=6
        while [ $retries -gt 0 ]; do
            if check_port 8001 && check_port 8889; then
                print_status "Healthcare AI services ready"
                break
            fi
            print_info "Waiting for Healthcare AI... ($retries retries left)"
            sleep 5
            ((retries--))
        done
        
        if [ $retries -eq 0 ]; then
            print_warning "Healthcare AI services may not be ready"
        fi
    fi
}

# Function to install Python dependencies if needed
install_dependencies() {
    print_info "Checking Python dependencies..."
    
    # Check if required packages are available
    python3 -c "import requests, numpy" 2>/dev/null || {
        print_info "Installing required Python packages..."
        pip3 install requests numpy 2>/dev/null || {
            print_warning "Could not install packages automatically"
            print_info "Please run: pip3 install requests numpy"
        }
    }
}

# Function to show service status
show_service_status() {
    echo ""
    echo "🔍 Service Status Check:"
    echo "========================"
    
    services=(
        "8000:Model Registry"
        "8001:Healthcare AI"
        "8003:Experiment Tracking"
        "8090:A/B Testing"
        "8889:Healthcare Web UI"
    )
    
    for service in "${services[@]}"; do
        port="${service%%:*}"
        name="${service#*:}"
        
        if check_port $port; then
            print_status "$name (port $port)"
        else
            print_warning "$name (port $port) - Not running"
        fi
    done
}

# Function to show demo URLs
show_demo_urls() {
    echo ""
    echo "🌐 Demo URLs:"
    echo "=============="
    echo "• Healthcare AI Chat:     http://localhost:8889"
    echo "• Platform Dashboard:     http://localhost:8888"
    echo "• Monitoring Dashboard:   http://localhost:8889"
    echo "• Real-time Dashboard:    http://localhost:8890"
    echo "• Model Registry API:     http://localhost:8000"
    echo "• Experiment Tracking:    http://localhost:8003"
    echo "• A/B Testing Service:    http://localhost:8090"
}

# Function to run the main demo
run_integrated_demo() {
    print_info "Launching integrated MLOps demonstration..."
    
    # Make sure demo script is executable
    chmod +x demo/integrated_mlops_demo.py
    
    # Run the integrated demo
    python3 demo/integrated_mlops_demo.py
}

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Demo completed. Services remain running for exploration."
    echo ""
    echo "To stop services:"
    echo "• Healthcare AI: docker compose -f docker-compose.healthcare-ci.yml down"
    echo "• MLOps Platform: cd ../mlops-platform && docker compose -f docker-compose.platform.yml down"
}

# Main execution
main() {
    echo ""
    print_info "Starting comprehensive MLOps platform demonstration..."
    echo ""
    
    # Install dependencies
    install_dependencies
    
    # Start MLOps platform
    start_mlops_platform
    
    # Start healthcare services  
    start_healthcare_services
    
    # Show service status
    show_service_status
    
    # Show demo URLs
    show_demo_urls
    
    echo ""
    print_status "All services ready!"
    echo ""
    
    # Ask user if they want to proceed
    read -p "Launch integrated demo? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_integrated_demo
    else
        print_info "Demo cancelled. Services remain running for manual exploration."
        show_demo_urls
    fi
    
    cleanup
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"