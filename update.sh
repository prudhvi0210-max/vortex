#!/bin/bash
# QuantumBrush Update Script
# This script updates an existing QuantumBrush installation

# Configuration
REPO_URL="https://github.com/moth-quantum/quantum-brush-collab.git"
REPO_BRANCH="dist"
DEFAULT_INSTALL_DIR="$HOME/QuantumBrush"
TEMP_CLONE_DIR="/tmp/quantum-brush-update"

# Determine if terminal supports colors
if [ -t 1 ]; then
  if command -v tput > /dev/null; then
    ncolors=$(tput colors)
    if [ -n "$ncolors" ] && [ "$ncolors" -ge 8 ]; then
      BOLD="$(tput bold)"
      NORMAL="$(tput sgr0)"
      RED="$(tput setaf 1)"
      GREEN="$(tput setaf 2)"
      YELLOW="$(tput setaf 3)"
      BLUE="$(tput setaf 4)"
    fi
  fi
fi

# Fallback if tput doesn't work
if [ -z "$RED" ]; then
  BOLD=""
  NORMAL=""
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
fi

# Functions for output
print_step() {
    printf "${BLUE}[INFO]${NORMAL} %s\n" "$1"
}

print_success() {
    printf "${GREEN}[SUCCESS]${NORMAL} %s\n" "$1"
}

print_warning() {
    printf "${YELLOW}[WARNING]${NORMAL} %s\n" "$1"
}

print_error() {
    printf "${RED}[ERROR]${NORMAL} %s\n" "$1"
}

# Find QuantumBrush installation
find_installation() {
    print_step "Looking for QuantumBrush installation..."
    
    # Check if user provided a path as argument
    if [ -n "$1" ]; then
        INSTALL_DIR="$1"
    else
        INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    fi
    
    # Check if the directory exists and contains QuantumBrush
    if [ ! -d "$INSTALL_DIR" ]; then
        print_error "QuantumBrush installation not found at: $INSTALL_DIR"
        echo
        echo "Please specify the correct path:"
        echo "  $0 /path/to/QuantumBrush"
        echo
        echo "Or install QuantumBrush first using the install.sh script."
        return 1
    fi
    
    # Check if it looks like a QuantumBrush installation
    if [ ! -f "$INSTALL_DIR/QuantumBrush.jar" ] && ! find "$INSTALL_DIR" -name "QuantumBrush*.jar" -type f | grep -q .; then
        print_error "Directory exists but doesn't appear to contain QuantumBrush: $INSTALL_DIR"
        echo
        echo "Please specify the correct QuantumBrush installation directory:"
        echo "  $0 /path/to/QuantumBrush"
        return 1
    fi
    
    print_success "Found QuantumBrush installation at: $INSTALL_DIR"
    return 0
}

# Check if git is available
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed or not in PATH"
        echo
        echo "Please install Git first:"
        echo "  • macOS: brew install git"
        echo "  • Ubuntu/Debian: sudo apt install git"
        echo "  • Fedora/RHEL: sudo dnf install git"
        echo "  • Or download from: https://git-scm.com/downloads"
        return 1
    fi
    return 0
}

# Download latest version
download_update() {
    print_step "Downloading latest QuantumBrush version..."
    
    # Clean up any previous temporary clone
    if [ -d "$TEMP_CLONE_DIR" ]; then
        rm -rf "$TEMP_CLONE_DIR"
    fi
    
    # Create temporary directory
    mkdir -p "$TEMP_CLONE_DIR"
    
    # Clone only the specific branch with depth=1
    if git clone --depth=1 --branch "$REPO_BRANCH" "$REPO_URL" "$TEMP_CLONE_DIR"; then
        print_success "Latest version downloaded successfully"
        return 0
    else
        print_error "Failed to download latest version"
        return 1
    fi
}

# Create backup
create_backup() {
    BACKUP_DIR="${INSTALL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    print_step "Creating backup at: $BACKUP_DIR"
    
    if cp -R "$INSTALL_DIR" "$BACKUP_DIR"; then
        print_success "Backup created successfully"
        echo "BACKUP_LOCATION=$BACKUP_DIR"
        return 0
    else
        print_error "Failed to create backup"
        return 1
    fi
}

# Update installation with data preservation
update_installation() {
    print_step "Updating QuantumBrush installation..."
    
    # Directories that should be preserved during updates (user data)
    USER_DATA_DIRS=("project" "metadata" "log" "config")
    
    # Create temporary directory for user data
    USER_DATA_BACKUP="/tmp/quantumbrush_userdata_$$"
    mkdir -p "$USER_DATA_BACKUP"
    
    # Backup user data directories
    print_step "Preserving user data..."
    for dir in "${USER_DATA_DIRS[@]}"; do
        if [ -d "$INSTALL_DIR/$dir" ]; then
            print_step "  Preserving $dir directory..."
            cp -R "$INSTALL_DIR/$dir" "$USER_DATA_BACKUP/"
        fi
    done
    
    # Copy new files (this overwrites application files)
    print_step "Installing new application files..."
    cp -R "$TEMP_CLONE_DIR"/* "$INSTALL_DIR/"
    
    # Restore user data directories
    print_step "Restoring user data..."
    for dir in "${USER_DATA_DIRS[@]}"; do
        if [ -d "$USER_DATA_BACKUP/$dir" ]; then
            print_step "  Restoring $dir directory..."
            # Remove the new empty directory and restore the user's data
            rm -rf "$INSTALL_DIR/$dir"
            cp -R "$USER_DATA_BACKUP/$dir" "$INSTALL_DIR/"
        fi
    done
    
    # Make setup script executable if it exists
    if [ -f "$INSTALL_DIR/setup.sh" ]; then
        chmod +x "$INSTALL_DIR/setup.sh"
    fi
    
    # Clean up temporary directories
    rm -rf "$USER_DATA_BACKUP"
    rm -rf "$TEMP_CLONE_DIR"
    
    print_success "Update completed successfully"
    return 0
}

# Show what was updated
show_update_summary() {
    echo
    print_success "QuantumBrush has been updated!"
    echo
    printf "${GREEN}Installation location:${NORMAL} $INSTALL_DIR\n"
    
    # Find the main JAR file
    JAR_FILE=$(find "$INSTALL_DIR" -name "QuantumBrush*.jar" -type f | head -n 1)
    if [ -n "$JAR_FILE" ]; then
        JAR_NAME=$(basename "$JAR_FILE")
        printf "${GREEN}Main application:${NORMAL} $JAR_NAME\n"
    fi
    
    echo
    printf "${BLUE}What was updated:${NORMAL}\n"
    echo "  • Application files (JAR, libraries, effects)"
    echo "  • Setup scripts"
    echo "  • Documentation"
    echo
    printf "${BLUE}What was preserved:${NORMAL}\n"
    echo "  • Your projects (project/ directory)"
    echo "  • Settings and configurations (config/ directory)"
    echo "  • Application logs (log/ directory)"
    echo "  • Project metadata (metadata/ directory)"
    echo
    printf "${BLUE}To run the updated QuantumBrush:${NORMAL}\n"
    echo "  • Double-click the JAR file in: $INSTALL_DIR"
    if [ -n "$JAR_NAME" ]; then
        echo "  • Or from terminal: cd $INSTALL_DIR && java -jar $JAR_NAME"
    fi
    echo
}

# Rollback function (in case something goes wrong)
offer_rollback() {
    if [ -n "$BACKUP_LOCATION" ] && [ -d "$BACKUP_LOCATION" ]; then
        echo
        print_warning "If you encounter any issues with the update, you can rollback:"
        echo "  rm -rf $INSTALL_DIR"
        echo "  mv $BACKUP_LOCATION $INSTALL_DIR"
        echo
        echo "Or run: $0 --rollback $BACKUP_LOCATION"
    fi
}

# Rollback to previous version
rollback() {
    BACKUP_PATH="$1"
    
    if [ -z "$BACKUP_PATH" ]; then
        print_error "Please specify the backup directory to rollback to"
        echo "Usage: $0 --rollback /path/to/backup"
        return 1
    fi
    
    if [ ! -d "$BACKUP_PATH" ]; then
        print_error "Backup directory not found: $BACKUP_PATH"
        return 1
    fi
    
    # Extract installation path from backup path
    INSTALL_DIR=$(echo "$BACKUP_PATH" | sed 's/_backup_[0-9]*_[0-9]*$//')
    
    print_step "Rolling back to previous version..."
    print_step "  From: $BACKUP_PATH"
    print_step "  To: $INSTALL_DIR"
    
    read -p "Are you sure you want to rollback? This will overwrite the current installation. (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "$INSTALL_DIR" ]; then
            rm -rf "$INSTALL_DIR"
        fi
        
        if mv "$BACKUP_PATH" "$INSTALL_DIR"; then
            print_success "Rollback completed successfully"
            print_success "QuantumBrush has been restored to the previous version"
        else
            print_error "Rollback failed"
            return 1
        fi
    else
        print_warning "Rollback cancelled"
    fi
}

# Main function
main() {
    # Handle rollback command
    if [ "$1" = "--rollback" ]; then
        rollback "$2"
        exit $?
    fi
    
    printf "\n"
    printf "╔══════════════════════════════════════════════════════════════╗\n"
    printf "║                   QuantumBrush Updater                       ║\n"
    printf "║                                                              ║\n"
    printf "╚══════════════════════════════════════════════════════════════╝\n"
    printf "\n"
    
    # Find QuantumBrush installation
    if ! find_installation "$1"; then
        exit 1
    fi
    
    # Check if git is available
    if ! check_git; then
        exit 1
    fi
    
    # Show current installation info
    echo
    printf "${BLUE}Current installation:${NORMAL} $INSTALL_DIR\n"
    
    # Ask for confirmation
    read -p "Do you want to update QuantumBrush to the latest version? (Y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Update cancelled by user"
        exit 0
    fi
    
    # Create backup
    if ! create_backup; then
        print_error "Cannot proceed without backup. Update cancelled."
        exit 1
    fi
    
    # Download latest version
    if ! download_update; then
        print_error "Update failed during download"
        exit 1
    fi
    
    # Update installation
    if ! update_installation; then
        print_error "Update failed during installation"
        offer_rollback
        exit 1
    fi
    
    # Show summary
    show_update_summary
    
    # Offer rollback info
    offer_rollback
}

# Show usage if --help is provided
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "QuantumBrush Update Script"
    echo
    echo "Usage:"
    echo "  $0                           # Update QuantumBrush in default location ($DEFAULT_INSTALL_DIR)"
    echo "  $0 /path/to/installation     # Update QuantumBrush in specific location"
    echo "  $0 --rollback /path/to/backup # Rollback to a previous version"
    echo "  $0 --help                    # Show this help"
    echo
    echo "Examples:"
    echo "  $0                                    # Update default installation"
    echo "  $0 /Users/john/MyQuantumBrush         # Update custom installation"
    echo "  $0 --rollback ~/QuantumBrush_backup_20250128_143022  # Rollback"
    echo
    exit 0
fi

# Run main function
main "$@"
