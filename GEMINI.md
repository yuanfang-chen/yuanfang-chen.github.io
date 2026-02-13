# Project Context: yuanfang-chen.github.io

## Project Overview
This repository contains the source code for a personal website and blog hosted on GitHub Pages. The site is built using **Jekyll**, a static site generator written in Ruby. The source content is located primarily in the `docs/` directory.

## Prerequisites
To run this project locally, you need the following installed:
- **Ruby**: The programming language Jekyll is built on.
- **Bundler**: A dependency manager for Ruby (`gem install bundler`).
- **Jekyll**: The static site generator (`gem install jekyll`).

## Building and Running
The project is configured to run from the `docs/` directory.

### 1. Install Dependencies
Navigate to the `docs/` directory and install the required Ruby gems:
```bash
cd docs
bundle install
```

### 2. Run Locally
To serve the site locally with live reloading:
```bash
cd docs
bundle exec jekyll serve
```
The site will be available at `http://localhost:4000`.

### 3. Build for Production
To build the static site into the `_site` directory:
```bash
cd docs
bundle exec jekyll build
```

## Project Structure
The core of the site resides in the `docs/` folder, which is a common pattern for GitHub Pages.

- **`docs/`**: The main source directory for the Jekyll site.
  - **`_config.yml`**: The main configuration file for Jekyll (site title, theme, plugins).
  - **`Gemfile`**: Lists the Ruby gem dependencies (e.g., `minima`, `github-pages`, `jekyll-feed`).
  - **`_posts/`**: Contains the blog posts written in Markdown. Filenames follow the format `YYYY-MM-DD-title.markdown`.
  - **`_drafts/`**: Contains draft posts that are not yet published.
  - **`_layouts/`**: Contains HTML templates that wrap the content.
  - **`_includes/`**: Contains partial HTML snippets used in layouts (e.g., `head.html`).
  - **`assets/`**: Stores static assets like images and CSS.
  - **`index.markdown`**: The homepage content.
  - **`about.markdown`**: The "About" page content.

- **`.github/workflows/`**: Contains the GitHub Actions workflow (`jekyll-gh-pages.yml`) that builds and deploys the site to GitHub Pages.

## Development Conventions
- **Content**: New posts should be added to `docs/_posts/` with the correct date prefix.
- **Drafts**: Work-in-progress posts can be placed in `docs/_drafts/`.
- **Markdown**: The site uses `kramdown` as the Markdown processor.
- **Theme**: The site uses the `minima` theme.

## Deployment
The site is automatically built and deployed by GitHub Actions whenever changes are pushed to the `main` branch. The workflow is defined in `.github/workflows/jekyll-gh-pages.yml`.
