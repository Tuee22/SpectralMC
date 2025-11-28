# SpectralMC Documentation

## Overview

This directory contains all SpectralMC documentation organized by audience and purpose.

## Folder Structure

### engineering/

Engineering standards, development practices, and infrastructure guides.

**Audience**: Internal developers, contributors

**Topics**: Code quality, type safety, testing, Docker, GPU builds

[View Engineering Standards →](engineering/index.md)

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

- [Engineering Standards Index](engineering/index.md)
- [Deployment Guide](product/deployment.md)
- [Training Integration](product/training_integration.md)
- [SpectralMC Whitepaper](domain/whitepapers/spectralmc_whitepaper.md)

## Contributing

When adding new documentation:

1. **Choose the right folder**:
   - `engineering/` - Development practices, code standards, infrastructure
   - `product/` - User-facing guides, deployment, operations
   - `domain/` - Scientific theory, research papers, mathematical background

2. **Update the index**: Add your document to the relevant `index.md`

3. **Follow SSoT**: Link to canonical sources, avoid duplicating information

4. **Cross-link**: Reference related documentation in other folders
