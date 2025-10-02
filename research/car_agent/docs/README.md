# Documentation Directory

Additional documentation for the Generic Active Inference Agent Framework.

## Contents

```
docs/
├── ARCHITECTURE.md    # Detailed architecture documentation
├── README.md          # This file
└── AGENTS.md          # Agent theory and implementation
```

## Documentation Structure

### ARCHITECTURE.md

Comprehensive technical architecture documentation covering:
- System design principles
- Module interactions
- Data flow
- Extension points
- Performance considerations

**Audience**: Developers extending or modifying the framework

### AGENTS.md (to be created)

Theoretical background and implementation guide for Active Inference agents:
- Active Inference theory
- Free energy principle
- Variational inference
- Message passing algorithms
- Implementation patterns

**Audience**: Researchers and advanced users

## Documentation Philosophy

### Clarity

- Use clear, precise language
- Provide examples and diagrams
- Define technical terms
- Progressive complexity

### Completeness

- Cover all public APIs
- Document design decisions
- Explain trade-offs
- Include limitations

### Maintainability

- Keep docs synchronized with code
- Version documentation
- Review regularly
- Update with changes

## Additional Resources

### In-Code Documentation

Every module includes comprehensive docstrings:

```julia
# View docstring
?GenericActiveInferenceAgent
?step!
?get_diagnostics
```

### Test Suite Documentation

Tests serve as executable documentation:
- See `test/AGENTS.md` for testing patterns
- Review `test/*.jl` for usage examples

### Example Code

Working examples demonstrate best practices:
- See `examples/README.md` for overview
- Study `examples/*.jl` for complete implementations

## Contributing Documentation

### Guidelines

1. **Clear Structure**: Logical organization with table of contents
2. **Code Examples**: Include runnable code snippets
3. **Diagrams**: Use ASCII art or mermaid for illustrations
4. **Links**: Cross-reference related documentation
5. **Updates**: Keep in sync with code changes

### Writing Style

- **Be Concise**: Clear and direct
- **Be Precise**: Technical accuracy
- **Be Helpful**: Anticipate questions
- **Be Complete**: Cover edge cases

### Review Process

1. Write draft documentation
2. Test all code examples
3. Review for clarity
4. Check for accuracy
5. Update version history

## Future Documentation

Planned additions:

- [ ] Tutorial series (beginner to advanced)
- [ ] Video walkthroughs
- [ ] API reference (auto-generated)
- [ ] Performance tuning guide
- [ ] Troubleshooting FAQ
- [ ] Research papers integration
- [ ] Jupyter notebook tutorials

## Getting Help

Documentation hierarchy:

1. **Quick Start**: `QUICKSTART.md` (5-minute intro)
2. **Main README**: `README.md` (comprehensive overview)
3. **Module Docs**: `src/README.md` (implementation details)
4. **Architecture**: `docs/ARCHITECTURE.md` (system design)
5. **Agent Theory**: `docs/AGENTS.md` (theoretical background)
6. **Examples**: `examples/` (working code)
7. **Tests**: `test/` (usage patterns)

---

**Comprehensive documentation enables effective use and contribution.**

