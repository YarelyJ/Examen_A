#!/bin/bash

echo "Starting Django Backend..."
echo "========================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "Running migrations..."
python manage.py migrate

# Start server
echo ""
echo "Starting server at http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
python manage.py runserver
