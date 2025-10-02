# Documentation Index

Comprehensive guide to all documentation in the Generic Active Inference Agent Framework.

## 📚 Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](#main-readme) | Framework overview | Everyone |
| [QUICKSTART.md](#quick-start) | 5-minute intro | New users |
| [docs/](#docs-directory) | Theory & architecture | Researchers, developers |
| [src/](#source-documentation) | Implementation details | Developers |
| [test/](#test-documentation) | Testing guide | Contributors |
| [examples/](#examples-documentation) | Usage demonstrations | Practitioners |
| [outputs/](#outputs-documentation) | Output organization | All users |

## 📖 Documentation Map

### Main README
**File**: `README.md`  
**Purpose**: Complete framework overview  
**Contains**:
- Feature highlights
- Installation instructions
- Quick start guide
- Core concepts
- API reference
- Configuration guide
- Examples overview

**Read if**: You're new to the framework or need a comprehensive overview

---

### Quick Start
**File**: `QUICKSTART.md`  
**Purpose**: Get running in 5 minutes  
**Contains**:
- Minimal setup
- Basic usage example
- Common tasks
- Troubleshooting quick fixes

**Read if**: You want to start using the framework immediately

---

### docs/ Directory

#### Theory & Implementation
**File**: `docs/AGENTS.md`  
**Purpose**: Active Inference theory and practice  
**Contains**:
- Free Energy Principle explanation
- Mathematical formulation
- Computational implementation
- Practical hyperparameter tuning
- Advanced topics

**Read if**: You want deep theoretical understanding or need to tune performance

#### Framework Architecture
**File**: `docs/ARCHITECTURE.md`  
**Purpose**: System design and structure  
**Contains**:
- Design philosophy
- Module architecture
- Data flow diagrams
- Extension points
- Performance considerations

**Read if**: You're extending the framework or need to understand internal structure

#### Documentation Overview
**File**: `docs/README.md`  
**Purpose**: Documentation navigation  
**Contains**:
- Documentation structure
- Writing guidelines
- Contribution process

**Read if**: You're contributing documentation

---

### src/ Directory

#### Agent Architecture
**File**: `src/AGENTS.md`  
**Purpose**: Deep-dive into agent implementation  
**Contains**:
- Agent state structure
- Generative model details
- Inference process
- Planning horizon mechanics
- Customization guide

**Read if**: You're implementing custom agents or modifying core agent logic

#### Source Code Guide
**File**: `src/README.md`  
**Purpose**: Module implementation details  
**Contains**:
- Module architecture
- API documentation
- Extension points
- Performance optimization
- Error handling

**Read if**: You're developing or maintaining the codebase

---

### test/ Directory

#### Testing Guide
**File**: `test/AGENTS.md`  
**Purpose**: Agent testing patterns  
**Contains**:
- Test philosophy
- Test patterns
- Agent-specific testing
- Performance testing
- Debugging strategies

**Read if**: You're writing tests for agents or need to debug test failures

#### Test Suite Overview
**File**: `test/README.md`  
**Purpose**: Complete test suite documentation  
**Contains**:
- Test structure
- Running tests
- Test modules overview
- Test coverage
- Contributing tests

**Read if**: You need to understand or extend the test suite

---

### examples/ Directory

#### Examples Overview
**File**: `examples/README.md`  
**Purpose**: Guide to working examples  
**Contains**:
- Available examples
- Creating new examples
- Template and guidelines

**Read if**: You want to see complete working implementations

#### Example Creation Guide
**File**: `examples/AGENTS.md`  
**Purpose**: How to create new examples  
**Contains**:
- Mountain car walkthrough
- Problem-specific patterns
- Visualization techniques
- Debugging examples

**Read if**: You're creating a new example or adapting the framework to a new problem

---

### outputs/ Directory

#### Output Organization
**File**: `outputs/README.md`  
**Purpose**: Output structure and management  
**Contains**:
- Directory structure
- File naming conventions
- Configuration settings
- Data formats
- Usage examples

**Read if**: You need to understand or work with generated outputs

---

### Configuration

#### Main Configuration
**File**: `config.jl`  
**Purpose**: Central parameter management  
**Contains**:
- Agent parameters
- Simulation settings
- Logging configuration
- Output settings
- Validation functions

**Read if**: You need to adjust framework behavior or understand default settings

---

### Project Files

#### Package Dependencies
**Files**: `Project.toml`, `Manifest.toml`  
**Purpose**: Julia package management  
**Contains**:
- Package dependencies
- Version constraints
- UUID mappings

**Read if**: You're setting up the environment or managing dependencies

#### CLI Runner
**File**: `run.jl`  
**Purpose**: Command-line interface  
**Contains**:
- Command definitions
- Execution workflows
- Helper functions

**Read if**: You want to understand or extend the CLI

---

## 🎯 Learning Paths

### Path 1: Quick User
**Goal**: Run examples and use the framework  
**Read**:
1. README.md (overview)
2. QUICKSTART.md (setup)
3. examples/README.md (usage)
4. outputs/README.md (understanding results)

**Time**: ~30 minutes

---

### Path 2: Active Inference Researcher
**Goal**: Understand theory and apply to research  
**Read**:
1. README.md (overview)
2. docs/AGENTS.md (theory)
3. src/AGENTS.md (implementation)
4. examples/AGENTS.md (application)

**Time**: ~2 hours

---

### Path 3: Framework Developer
**Goal**: Extend and maintain the framework  
**Read**:
1. README.md (overview)
2. docs/ARCHITECTURE.md (design)
3. src/README.md (modules)
4. test/README.md (testing)
5. All source docstrings

**Time**: ~4 hours

---

### Path 4: Contributor
**Goal**: Contribute code, tests, or documentation  
**Read**:
1. README.md (overview)
2. docs/ARCHITECTURE.md (design)
3. test/AGENTS.md (testing patterns)
4. docs/README.md (documentation guidelines)

**Time**: ~2 hours

---

## 📝 Documentation by Topic

### Active Inference Theory
- `docs/AGENTS.md` - Complete theory
- `src/AGENTS.md` - Implementation details
- `examples/AGENTS.md` - Practical application

### Agent Implementation
- `src/agent.jl` (docstrings)
- `src/AGENTS.md` (architecture)
- `test/AGENTS.md` (testing)

### Configuration & Setup
- `config.jl` (parameters)
- `QUICKSTART.md` (setup)
- `README.md` (installation)

### Testing & Quality
- `test/README.md` (suite overview)
- `test/AGENTS.md` (patterns)
- Individual test files (examples)

### Examples & Usage
- `examples/README.md` (overview)
- `examples/AGENTS.md` (creation guide)
- `examples/mountain_car_example.jl` (complete example)

### Outputs & Analysis
- `outputs/README.md` (structure)
- `config.jl` (output settings)
- Example files (data formats)

---

## 🔍 Finding Specific Information

### "How do I...?"

| Task | Document |
|------|----------|
| Install and setup | README.md, QUICKSTART.md |
| Run an example | QUICKSTART.md, examples/README.md |
| Create custom agent | src/AGENTS.md, examples/AGENTS.md |
| Tune parameters | docs/AGENTS.md, config.jl |
| Write tests | test/AGENTS.md, test/README.md |
| Understand theory | docs/AGENTS.md |
| Extend framework | docs/ARCHITECTURE.md, src/README.md |
| Debug issues | test/AGENTS.md, examples/AGENTS.md |
| Analyze outputs | outputs/README.md |

### "Where is...?"

| Concept | Document |
|---------|----------|
| Free Energy Principle | docs/AGENTS.md |
| Agent architecture | src/AGENTS.md, docs/ARCHITECTURE.md |
| Configuration options | config.jl, README.md |
| Test patterns | test/AGENTS.md |
| Example problems | examples/AGENTS.md |
| API reference | README.md, src/README.md |
| Performance tuning | docs/AGENTS.md, docs/ARCHITECTURE.md |
| Output formats | outputs/README.md |

---

## 📊 Documentation Statistics

- **Total Documents**: 12 major files
- **Total Lines**: ~5,000 lines of documentation
- **Coverage**: All modules, APIs, and concepts
- **Formats**: Markdown, inline docstrings, comments
- **Last Updated**: October 2025

---

## 🤝 Contributing Documentation

### Adding Documentation

1. **Choose Location**: Follow existing structure
2. **Use Templates**: See existing docs for format
3. **Be Complete**: Cover all aspects
4. **Add Examples**: Include code samples
5. **Update Index**: Add to this file

### Documentation Style

- **Clear Headers**: Hierarchical organization
- **Code Blocks**: Syntax-highlighted examples
- **Tables**: For comparisons and quick reference
- **Lists**: For steps and items
- **Emphasis**: Bold for **important**, italic for *concepts*

### Review Checklist

- [ ] Accurate and up-to-date
- [ ] Well-organized with clear headers
- [ ] Includes examples
- [ ] Links to related documentation
- [ ] Proper formatting
- [ ] Spell-checked

---

## 🔗 External Resources

### RxInfer.jl

- **Documentation**: https://reactivebayes.github.io/RxInfer.jl/
- **Paper**: Bagaev & de Vries (2023)
- **Examples**: RxInferExamples.jl repository

### Active Inference

- **Key Papers**:
  - Friston et al. (2017) - "Active Inference: A Process Theory"
  - Parr & Friston (2019) - "Generalised Free Energy"
- **Books**:
  - Friston & Parr (2022) - "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"

### Julia

- **Documentation**: https://docs.julialang.org/
- **Style Guide**: https://docs.julialang.org/en/v1/manual/style-guide/
- **Packages**: https://julialang.org/packages/

---

## 📧 Support

For documentation questions:

1. **Check This Index**: Find relevant document
2. **Read Target Document**: May answer your question
3. **Search Codebase**: Look for examples
4. **Ask Community**: Julia forums, GitHub discussions

---

**Comprehensive, well-organized documentation enables effective use and contribution.**

