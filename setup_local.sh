#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_local.txt"
VENV_DIR="${SCRIPT_DIR}/venv"

# Check for existing virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    read -rp "Do you want to reinstall it? This will delete the existing venv. [y/N] " reinstall
    reinstall=${reinstall,,}

    if [[ "$reinstall" =~ ^(y|yes)$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
        
        # Check for pip in existing venv
        if ! "$VENV_DIR/bin/python" -m pip --version &> /dev/null; then
            echo "Pip is not installed in the virtual environment."
            read -rp "Do you want to install pip? [Y/n] " install_pip
            install_pip=${install_pip,,}

            if [[ "$install_pip" =~ ^(n|no)$ ]]; then
                echo "Pip is required for package installation. Aborting."
                exit 1
            fi
            
            echo "Installing pip..."
            curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python" -
        fi
        
        # Activate existing venv and proceed
        source "$VENV_DIR/bin/activate"
        python --version
    fi
fi

# Create new venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment..."
    python3.11 -m venv --without-pip "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python --version
    
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python -
fi

# Verify requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: requirements_local.txt not found in $SCRIPT_DIR" >&2
    echo "Aborting script." >&2
    exit 1
fi

# Dry run first
echo "Performing dry run of package installation..."
if ! python -m pip install -r "$REQUIREMENTS_FILE" --dry-run; then
    echo "Error: Dry run failed. Check dependency conflicts." >&2
    exit 1
fi

# Confirm installation
read -rp "Dry run successful. Do you want to proceed with actual installation? [Y/n] " response
response=${response,,}

case "$response" in
    y|yes|"")
        echo "Installing packages..."
        python -m pip install -r "$REQUIREMENTS_FILE"
        ;;
    n|no)
        echo "Installation aborted by user."
        exit 0
        ;;
    *)
        echo "Invalid response. Aborting." >&2
        exit 1
        ;;
esac

echo "Setup completed successfully."
