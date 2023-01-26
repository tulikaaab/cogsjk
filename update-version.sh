#!/bin/sh

#command to add date and time and add to filename version
date > version
# to update the repo 
git add update-version.sh
git add version
git commit -a --allow-empty-message -m ' '
git push

