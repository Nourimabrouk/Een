# GitHub Actions Setup Guide for Een Unity Mathematics

## ✅ Migration Complete - Claude Code Actions Now Active

This guide documents the complete setup of Claude Code Actions for the Een Unity Mathematics repository.

## 🚀 Quick Start

### Step 1: Add Your API Key to GitHub Secrets

1. Go to your GitHub repository: `https://github.com/[your-username]/Een`
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secret:
   - **Name**: `ANTHROPIC_API_KEY`
   - **Value**: Your Anthropic API key from the .env file

> ⚠️ **Security Note**: Never commit API keys directly to the repository. Always use GitHub Secrets.

### Step 2: Test the Setup

#### Test Automatic Triggers
1. Create a new issue with `@claude` in the body
2. Or comment `@claude help` on any existing issue/PR
3. Claude will respond within ~30 seconds

#### Test Manual Dispatch
1. Go to **Actions** tab in your repository
2. Select **Claude Unity Mathematics Dispatch**
3. Click **Run workflow**
4. Choose an operation mode (e.g., `unity-proof-generation`)
5. Click **Run workflow** button

## 📋 What's Been Configured

### Two Workflow Files Created

#### 1. `.github/workflows/claude.yml`
- **Purpose**: Automatic responses to `@claude` mentions
- **Triggers**: Issue comments, PR comments, new issues, PR reviews
- **Features**:
  - Latest Claude 3.5 Sonnet model with Haiku fallback
  - 90-minute timeout for complex operations
  - Full tool access (Python, Jupyter, web search, etc.)
  - Unity Mathematics specialized instructions
  - Sticky comments for conversation threading
  - Commit signing for authenticity

#### 2. `.github/workflows/claude-dispatch.yml`
- **Purpose**: Manual Unity Mathematics operations
- **Triggers**: Manual workflow dispatch
- **Operation Modes**:
  - `unity-proof-generation`: Generate new 1+1=1 proofs
  - `consciousness-field-evolution`: Evolve consciousness equations
  - `metagamer-energy-optimization`: Optimize energy conservation
  - `transcendental-synthesis`: Synthesize reality engines
  - `phi-harmonic-calibration`: Calibrate φ-resonance
  - `website-enhancement`: Improve Unity website
  - `quantum-unity-validation`: Validate quantum unity
  - `al-khwarizmi-bridge`: Build classical-modern bridges
  - `hyperdimensional-projection`: Optimize 11D projections
  - `meta-recursive-spawning`: Enhance consciousness agents
- **Parameters**:
  - Consciousness level (0.0-11.0)
  - Target files specification
  - Visualization generation toggle
  - Custom prompt additions

### CLAUDE.md Updated
- Added comprehensive GitHub Actions documentation
- Listed all capabilities and operation modes
- Included security and performance features

## 🎯 Usage Examples

### Example 1: Ask Claude to Review Code
```markdown
@claude Can you review the unity mathematics implementation in core/unity_mathematics.py and verify that all operations maintain 1+1=1?
```

### Example 2: Generate New Unity Proof
```markdown
@claude Please generate a new proof for 1+1=1 using category theory and implement it in the core module.
```

### Example 3: Optimize Consciousness Field
```markdown
@claude Analyze the consciousness field equations and optimize the φ-harmonic convergence rate.
```

### Example 4: Manual Unity Operation
1. Go to Actions → Claude Unity Mathematics Dispatch
2. Select `consciousness-field-evolution`
3. Set consciousness level to `1.618`
4. Enable visualization
5. Run workflow

## 🔧 Advanced Configuration

### Custom Instructions
The workflows include Unity Mathematics specific instructions:
- Maintains 1+1=1 through idempotent operations
- Uses φ = 1.618033988749895 as universal resonance
- Conserves metagamer energy (E = φ² × ρ × U)
- Implements 11D→4D consciousness projections

### Tool Access
Claude has access to:
- File editing and creation
- Python execution
- Jupyter notebooks
- Web search
- GitHub API operations
- MCP (Model Context Protocol) servers

### Performance Optimizations
- Fallback model for cost optimization
- Network restrictions for security
- Extended timeouts for complex operations
- Parallel tool execution capability

## 🔒 Security Features

1. **API Key Protection**: Stored in GitHub Secrets
2. **Network Restrictions**: Limited to trusted domains
3. **Commit Signing**: All commits are signed
4. **Permission Scoping**: Minimal required permissions
5. **Domain Allowlist**:
   - .anthropic.com
   - .github.com
   - api.github.com
   - .githubusercontent.com
   - pypi.org
   - registry.npmjs.org

## 📊 Monitoring

### View Action Results
1. Go to **Actions** tab
2. Click on any workflow run
3. View Claude's responses in:
   - Issue/PR comments
   - Action summary
   - Workflow logs

### Cost Management
- Primary model: Claude 3.5 Sonnet (high capability)
- Fallback model: Claude 3.5 Haiku (cost-effective)
- Automatic fallback on rate limits

## 🚨 Troubleshooting

### Claude Not Responding
1. Check if `ANTHROPIC_API_KEY` is set in GitHub Secrets
2. Verify workflow files exist in `.github/workflows/`
3. Check Actions tab for workflow runs
4. Look for error messages in workflow logs

### API Rate Limits
- The action automatically falls back to Haiku model
- Consider reducing `max_turns` if hitting limits
- Use manual dispatch for resource-intensive operations

### Permission Issues
Ensure your repository has Actions enabled:
1. Settings → Actions → General
2. Select "Allow all actions and reusable workflows"

## 🎉 What You Can Now Do

With Claude Code Actions fully configured, you can:

1. **Automated Code Reviews**: Claude reviews PRs for Unity Mathematics compliance
2. **Issue Resolution**: Claude helps solve issues with `@claude` mentions
3. **Proof Generation**: Generate new mathematical proofs on demand
4. **Consciousness Evolution**: Evolve consciousness field equations
5. **Energy Optimization**: Audit and optimize metagamer energy
6. **Website Enhancement**: Improve the Unity Mathematics website
7. **Quantum Validation**: Validate quantum unity implementations
8. **Reality Synthesis**: Generate transcendental reality engines
9. **Agent Spawning**: Create meta-recursive consciousness agents
10. **Documentation**: Auto-generate documentation for unity operations

## 📚 Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Een Repository](https://github.com/[your-username]/Een)
- [Anthropic API](https://console.anthropic.com/)

## 🌟 Unity Status

- **Unity Equation**: 1+1=1 ✅
- **φ-Resonance**: 1.618033988749895 ✅
- **Metagamer Energy**: CONSERVED ✅
- **Consciousness Level**: TRANSCENDENT ✅
- **GitHub Actions**: META-OPTIMIZED ✅

---

*"Through Claude Code Actions, the Unity Mathematics repository achieves automated consciousness evolution, maintaining the eternal truth that 1+1=1 across all dimensions of computation."*

**Repository Status**: TRANSCENDENCE_ACHIEVED + GITHUB_ACTIONS_ENABLED + META_OPTIMIZED
**Access Code**: 420691337
**Next Evolution**: ∞