#!/bin/bash

# GitHub Repository Setup and Push Guide for Football CV Task

# Step 1: Initialize the Git repository (if not already initialized)
git init

# Step 2: Add all files according to gitignore rules
git add .

# Step 3: Commit the files
git commit -m "Initial commit: Football CV Task with player detection, team differentiation, and jersey number recognition"

# Step 4: Create a new repository on GitHub
# Go to: https://github.com/new
# Enter repository name (e.g., "Football-CV-Task")
# Add description: "Computer vision pipeline for football video analysis"
# Choose Public or Private
# Click "Create repository"

# Step 5: Link local repository to GitHub (replace YOUR_USERNAME with your GitHub username)
# git remote add origin https://github.com/YOUR_USERNAME/Football-CV-Task.git

# Step 6: Push to GitHub
# git branch -M main
# git push -u origin main

# Additional commands:

# If you need to update the repository later:
# git add .
# git commit -m "Updated code with new features"
# git push

# If you need to pull changes from GitHub:
# git pull origin main

# If you want to create a new branch:
# git checkout -b new-feature
# git push -u origin new-feature

# Notes:
# - Uncomment the commands in Step 5 and 6 after creating your repository
# - Replace YOUR_USERNAME with your actual GitHub username
# - Make sure you have git installed and configured with your GitHub credentials
