#!/bin/sh

#command to add date and time 
date > version
# to update the repo 
git add update-version.sh
git add version
git commit 
git push

