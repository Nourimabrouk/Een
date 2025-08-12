# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and automation.

## Active Workflows

### `claude.yml`
- **Purpose**: Claude AI assistant for automated PR/issue responses
- **Triggers**: 
  - `@claude` mentions in issues and PRs
  - `claude` label on issues/PRs
- **Features**:
  - Unity mathematics proof generation
  - Consciousness field analysis
  - Code review and suggestions

### `claude-dispatch.yml`
- **Purpose**: Manual workflow dispatch for specific operations
- **Triggers**: Manual workflow dispatch
- **Operations**:
  - Unity proof generation
  - Consciousness field evolution
  - Metagamer energy optimization
  - Website enhancement
  - And more...

## Setup

Required secrets:
- `ANTHROPIC_API_KEY` - API key for Claude AI

## Usage

### Automatic Triggers
Simply mention `@claude` in any issue or PR comment to trigger the assistant.

### Manual Dispatch
Go to Actions tab → Select workflow → Run workflow with desired parameters.

## Documentation

For more details, see:
- [GitHub Actions Setup Guide](../../docs/GITHUB_ACTIONS_SETUP.md)
- [Main Documentation](../../docs/README.md)