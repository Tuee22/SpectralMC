# File: documents/README.md
# SpectralMC Documentation

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: documents/documentation_standards.md

> **Purpose**: Orientation for SpectralMC documentation audiences and folder structure.
> **📖 Authoritative Reference**: [documentation_standards.md](documentation_standards.md)

## Overview

This directory contains all SpectralMC documentation organized by audience and purpose.

## Folder Structure

### engineering/

Engineering standards, development practices, and infrastructure guides.

**Audience**: Internal developers, contributors

**Topics**: Code quality, type safety, testing, Docker, GPU builds

[View Engineering Standards →](engineering/README.md)

### product/

Deployment guides, operations documentation, and feature integration.

**Audience**: DevOps, system administrators, end users

**Topics**: Production deployment, S3/MinIO setup, training integration

[View Product Documentation →](product/index.md)

### domain/

Scientific background, quantitative finance theory, and research papers.

**Audience**: Researchers, quantitative analysts, domain experts

**Topics**: CVNN theory, characteristic functions, Monte Carlo methods

[View Domain Knowledge →](domain/index.md)

## Quick Links

- [Engineering Standards Hub](engineering/README.md)
- [Deployment Guide](product/deployment.md)
- [Training Integration](product/training_integration.md)
- [Testing Architecture](testing_architecture.md)
- [SpectralMC Whitepaper](domain/whitepapers/spectralmc_whitepaper.md)

## Contributing

When adding new documentation:

1. **Choose the right folder**:
   - `engineering/` - Development practices, code standards, infrastructure
   - `product/` - User-facing guides, deployment, operations
   - `domain/` - Scientific theory, research papers, mathematical background

2. **Update the hub**: Add your document to the relevant folder README or index page

3. **Follow SSoT**: Link to canonical sources, avoid duplicating information

4. **Cross-link**: Reference related documentation in other folders
