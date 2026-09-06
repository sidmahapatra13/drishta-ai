# Contributing Guide

This guide explains how every teammate should use the repo during the hackathon.

## 1. Get Access

Ask the repo owner to add you as a collaborator. After accepting the invite, clone the repo:

```bash
git clone https://github.com/sidmahapatra13/smarty-pants-hackathon.git
cd smarty-pants-hackathon
```

## 2. Start From The Latest Main

Always begin by syncing your local copy (after cloning the repo and cd into it):

```bash
git checkout main
git pull origin main
```

## 3. Create A Task Branch

Use a short branch name that includes your area of work:

```bash
git checkout -b frontend/home-page
```

Good examples:

- `frontend/dashboard`
- `backend/auth-api`
- `ai/recommendation-prototype`
- `docs/final-submission`
- `design/demo-flow`

## 4. Commit Focused Changes

Keep commits small and readable:

```bash
git status
git add .
git commit -m "Add dashboard layout"
```

Commit message format:

```text
<action> <what changed>
```

Examples:

- `Add login page`
- `Fix API validation`
- `Document demo script`
- `Update pitch outline`

## 5. Push Your Branch

```bash
git push origin frontend/home-page
```

## 6. Open A Pull Request

In the PR description, include:

- What changed
- How to test it
- Screenshots or screen recordings for UI work
- Any setup changes or environment variables
- Any known issues or unfinished parts

## 7. Review And Merge

Before merging:

- At least one teammate should review the PR.
- The branch should be up to date with `main`.
- The app should still run locally.
- Any obvious bugs should be fixed or documented.

Prefer squash merging during the hackathon so `main` stays easy to read.

## Team Rules

- Do not push directly to `main` unless it is a tiny documentation/admin fix.
- Do not commit `.env` files, API keys, tokens, or private data.
- Pull from `main` often to avoid merge conflicts.
- Tell the team before changing shared architecture, package managers, folder structure, or database schema.
- Update the README when setup commands or project decisions change.
