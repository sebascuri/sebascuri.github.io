---
layout: page
title: How to update the webpage. 
permalink: /how-to/
---

Usually, just pushing the master branch to the github.io is enough. 
However, I use a webpage with some plug-ins that do not run on the github server. 
Hence, I build the website locally on a different branch and the push the built website to the master branch. 

### Step 1) Make sure you are on the source branch. 

Clone [this repository](https://github.com/sebascuri/sebascuri.github.io). 

Checkout remote branch. 
```bash
git checkout --track origin/source
```

### Step 2) Build webpage and check that it works.  
```bash
jekyll build 
bundle exec jekyll serve  
```

### Step 3) Push your new webpage. 

```bash
git branch -D master
git checkout -b master
git filter-branch --subdirectory-filter _site/ -f
git checkout source
git add _site/ 
git commit -m "Add compiled site."  # this is optional.
git push --all origin
```
