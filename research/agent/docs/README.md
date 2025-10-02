# Documentation Index

**Generic Agent-Environment Framework for Active Inference**  
**Version:** 0.1.1  
**Last Updated:** October 2, 2025

---

## Table of Contents

### 🚀 Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| **[Quick Start](quickstart.md)** | 5-minute getting started guide | New Users |
| **[Visualization Fix](visualization_fix.md)** | Setup and troubleshooting guide | New Users |

### 📖 Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[API Reference](index.md)** | Complete API documentation | All Users |
| **[Generic Agent Interface](generic_agent_interface.md)** | Composability and interface design | All Users |
| **[Comprehensive Summary](comprehensive_summary.md)** | Framework overview and capabilities | All Users |
| **[Working Status](working_status.md)** | Current status and verification | All Users |

### 🎨 Visualization

| Document | Description | Audience |
|----------|-------------|----------|
| **[Visualization Guide](visualization_guide.md)** | Complete plotting and animation guide | All Users |
| **[Output Verification](output_verification.md)** | Output structure and verification | All Users |

### ✨ Enhancement Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[Enhancements Summary](enhancements_summary.md)** | v0.1.1 enhancements details | Existing Users |
| **[Implementation Complete](implementation_complete.md)** | Full implementation report | Developers |

---

## Quick Navigation

### By Task

**I want to...**

- **Get started quickly** → [Quick Start](quickstart.md)
- **Understand the framework** → [Comprehensive Summary](comprehensive_summary.md)
- **Learn the API** → [API Reference](index.md)
- **Understand composability** → [Generic Agent Interface](generic_agent_interface.md)
- **Create visualizations** → [Visualization Guide](visualization_guide.md)
- **Troubleshoot issues** → [Visualization Fix](visualization_fix.md)
- **Add new agents/environments** → [Generic Agent Interface - Extension Guide](generic_agent_interface.md#extension-guide)
- **Understand recent changes** → [Enhancements Summary](enhancements_summary.md)
- **Verify installation** → [Working Status](working_status.md)

### By Role

**New Users:**
1. Read [Quick Start](quickstart.md)
2. Run examples
3. Review [Visualization Guide](visualization_guide.md)

**Existing Users:**
1. Check [Working Status](working_status.md)
2. Review [Enhancements Summary](enhancements_summary.md)
3. See [Visualization Fix](visualization_fix.md) for setup

**Developers:**
1. Study [API Reference](index.md)
2. Read [Comprehensive Summary](comprehensive_summary.md)
3. Review [Implementation Complete](implementation_complete.md)

---

## Documentation Structure

### Overview Documents

These provide high-level understanding:

- **[Comprehensive Summary](comprehensive_summary.md)** - Complete framework overview, architecture, and capabilities
- **[Generic Agent Interface](generic_agent_interface.md)** - Composability and interface design philosophy
- **[Working Status](working_status.md)** - Current implementation status and verification

### Guides

Step-by-step instructions:

- **[Quick Start](quickstart.md)** - Get running in 5 minutes
- **[Visualization Guide](visualization_guide.md)** - Complete plotting and animation guide
- **[Visualization Fix](visualization_fix.md)** - Setup, troubleshooting, and fixes

### Reference

Detailed technical documentation:

- **[API Reference](index.md)** - Complete API documentation with examples
- **[Output Verification](output_verification.md)** - Output structure and file formats

### Enhancement Documentation

Details about recent updates:

- **[Enhancements Summary](enhancements_summary.md)** - What's new in v0.1.1
- **[Implementation Complete](implementation_complete.md)** - Full implementation report

---

## Document Summaries

### [Quick Start](quickstart.md)
**Length:** Short  
**Time to Read:** 5 minutes  
**Contents:**
- Installation
- First simulation
- Basic usage patterns
- What to do next

### [API Reference](index.md)
**Length:** Medium  
**Time to Read:** 15-20 minutes  
**Contents:**
- Core components
- Agent interface
- Environment interface
- Simulation infrastructure
- Adding new components

### [Visualization Guide](visualization_guide.md)
**Length:** Long  
**Time to Read:** 30+ minutes  
**Contents:**
- Visualization types
- API usage
- Customization
- Performance considerations
- Troubleshooting
- Advanced usage

### [Generic Agent Interface](generic_agent_interface.md)
**Length:** Long  
**Time to Read:** 30-45 minutes  
**Contents:**
- Design philosophy
- Abstract interface pattern
- Type-level composability
- Technical implementation
- Extension guide
- Benefits and trade-offs

### [Comprehensive Summary](comprehensive_summary.md)
**Length:** Long  
**Time to Read:** 45+ minutes  
**Contents:**
- Executive summary
- Architecture overview
- Component details
- Performance characteristics
- Configuration system
- Fixed issues
- Testing strategy

### [Enhancements Summary](enhancements_summary.md)
**Length:** Long  
**Time to Read:** 30+ minutes  
**Contents:**
- Major enhancements
- Before/after comparisons
- Feature comparison matrix
- Migration guide
- Usage examples

### [Implementation Complete](implementation_complete.md)
**Length:** Very Long  
**Time to Read:** 60+ minutes  
**Contents:**
- Complete implementation report
- File inventory
- Verification procedures
- Performance characteristics
- Known limitations
- Success criteria

### [Working Status](working_status.md)
**Length:** Long  
**Time to Read:** 30+ minutes  
**Contents:**
- Current status
- Verified functionality
- Test results
- Known issues
- Future enhancements

### [Visualization Fix](visualization_fix.md)
**Length:** Medium  
**Time to Read:** 15-20 minutes  
**Contents:**
- Problem diagnosis
- Solutions applied
- Usage instructions
- Verification checklist
- Troubleshooting

### [Output Verification](output_verification.md)
**Length:** Short  
**Time to Read:** 10 minutes  
**Contents:**
- Output directory structure
- File formats
- Verification procedures

---

## External Links

- [Main README](../README.md) - Framework overview
- [Project.toml](../Project.toml) - Dependencies
- [config.toml](../config.toml) - Runtime configuration
- [Test Suite](../test/runtests.jl) - Comprehensive tests

---

## Version History

### v0.1.1 (October 2, 2025)
- ✅ Complete visualization system
- ✅ Automatic animation generation
- ✅ Comprehensive output management
- ✅ Enhanced documentation
- ✅ Full test coverage

### v0.1.0 (September 2025)
- ✅ Initial framework implementation
- ✅ Mountain Car and Simple Nav examples
- ✅ Type-safe agent-environment interface
- ✅ RxInfer integration
- ✅ Config-driven runner

---

## Contributing to Documentation

When adding or updating documentation:

1. **Update this index** - Add new documents to the table
2. **Cross-reference** - Link related documents
3. **Test links** - Verify all links work
4. **Update version history** - Note significant changes
5. **Keep README.md updated** - Main README should link here

---

## Documentation Standards

### File Naming
- Use UPPERCASE_WITH_UNDERSCORES for major docs
- Use lowercase_with_underscores for reference docs
- Keep names descriptive but concise

### Structure
- Start with executive summary
- Use clear section headers
- Include table of contents for long documents
- Add code examples where helpful
- End with "See Also" section

### Markdown Style
- Use `**bold**` for emphasis
- Use `code blocks` for code
- Use tables for comparisons
- Use emojis sparingly (only in titles/navigation)
- Use horizontal rules `---` to separate major sections

### Cross-References
- Use relative links: `[text](../path/to/file.md)`
- Link to related documents at end
- Update all references when moving files

---

**Happy Learning! 📚**

