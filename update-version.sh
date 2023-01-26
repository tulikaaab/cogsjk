#!/bin/sh

#command to add date and time 
date > version.txt

# to update the repo 
git add update-version.sh
git add version.txt
git commit 
git push

