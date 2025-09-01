#!/bin/bash

# NeuroForge Installation Script
# This script installs NeuroForge system-wide

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default installation directory
INSTALL_DIR="/usr/local"
BUILD_TYPE="Release"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for C compiler
    if ! command_exists gcc && ! command_exists clang; then
        print_error "No C compiler found. Please install GCC or Clang."
        exit 1
    fi
    
    # Check for CMake
    if ! command_exists cmake; then
        print_error "CMake not found. Please install CMake 3.10 or higher."
        exit 1
    fi
    
    # Check for Make
    if ! command_exists make; then
        print_error "Make not found. Please install Make."
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --prefix)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --build-type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --prefix DIR      Installation directory (default: /usr/local)"
                echo "  --build-type TYPE Build type: Debug, Release, RelWithDebInfo (default: Release)"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Function to build the project
build_project() {
    print_status "Building NeuroForge..."
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    print_status "Configuring with CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DUSE_CUDA=ON \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_TESTS=ON \
        -DBUILD_SHARED=OFF
    
    # Build the project
    print_status "Building..."
    make -j$(nproc)
    
    # Run tests
    print_status "Running tests..."
    make test
    
    print_success "Build completed successfully"
}

# Function to install the project
install_project() {
    print_status "Installing NeuroForge to $INSTALL_DIR..."
    
    # Check if we have permission to install
    if [[ ! -w "$INSTALL_DIR" ]] && [[ $EUID -ne 0 ]]; then
        print_warning "No write permission to $INSTALL_DIR. Using sudo..."
        sudo make install
    else
        make install
    fi
    
    print_success "Installation completed successfully"
}

# Function to create package
create_package() {
    print_status "Creating distribution package..."
    
    # Get version from CMakeLists.txt
    VERSION=$(grep "project(neuroforge VERSION" CMakeLists.txt | sed 's/.*VERSION \([0-9.]*\).*/\1/')
    
    # Create package name
    PACKAGE_NAME="neuroforge-${VERSION}-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
    
    # Create package directory
    mkdir -p "$PACKAGE_NAME"
    
    # Copy files
    cp -r src "$PACKAGE_NAME/"
    cp -r examples "$PACKAGE_NAME/"
    cp -r tests "$PACKAGE_NAME/"
    cp CMakeLists.txt "$PACKAGE_NAME/"
    cp README.md "$PACKAGE_NAME/"
    cp LICENSE.md "$PACKAGE_NAME/"
    cp install.sh "$PACKAGE_NAME/"
    cp build.bat "$PACKAGE_NAME/"
    
    # Create tarball
    tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"
    
    # Clean up
    rm -rf "$PACKAGE_NAME"
    
    print_success "Package created: ${PACKAGE_NAME}.tar.gz"
}

# Function to update ldconfig (Linux only)
update_ldconfig() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Updating library cache..."
        if command_exists ldconfig; then
            if [[ $EUID -eq 0 ]]; then
                ldconfig
            else
                sudo ldconfig
            fi
            print_success "Library cache updated"
        fi
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check if library is installed
    if [[ -f "$INSTALL_DIR/lib/libneuroforge.a" ]]; then
        print_success "Static library installed"
    else
        print_error "Static library not found"
        exit 1
    fi
    
    # Check if headers are installed
    if [[ -d "$INSTALL_DIR/include/neuroforge" ]]; then
        print_success "Headers installed"
    else
        print_error "Headers not found"
        exit 1
    fi
    
    # Check if pkg-config file is installed
    if [[ -f "$INSTALL_DIR/lib/pkgconfig/neuroforge.pc" ]]; then
        print_success "pkg-config file installed"
    else
        print_error "pkg-config file not found"
        exit 1
    fi
    
    print_success "Installation verified successfully"
}

# Main function
main() {
    echo "=========================================="
    echo "    NeuroForge Installation Script"
    echo "=========================================="
    echo
    
    # Parse command line arguments
    parse_args "$@"
    
    # Check dependencies
    check_dependencies
    
    # Build the project
    build_project
    
    # Install the project
    install_project
    
    # Update library cache
    update_ldconfig
    
    # Verify installation
    verify_installation
    
    # Create package
    create_package
    
    echo
    echo "=========================================="
    print_success "NeuroForge has been installed successfully!"
    echo "=========================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo "Library: $INSTALL_DIR/lib/libneuroforge.a"
    echo "Headers: $INSTALL_DIR/include/neuroforge/"
    echo "pkg-config: $INSTALL_DIR/lib/pkgconfig/neuroforge.pc"
    echo
    echo "To use in your project:"
    echo "  gcc your_file.c -I$INSTALL_DIR/include/neuroforge -L$INSTALL_DIR/lib -lneuroforge -lm"
    echo
    echo "Or with pkg-config:"
    echo "  gcc your_file.c \$(pkg-config --cflags --libs neuroforge)"
    echo
    echo "Package created: neuroforge-*.tar.gz"
    echo
}

# Run main function with all arguments
main "$@"
