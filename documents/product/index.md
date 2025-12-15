# File: documents/product/index.md
# Product Documentation

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: documents/product/deployment.md; documents/product/training_integration.md

> **Purpose**: Index for SpectralMC product and operations documentation.
> **📖 Authoritative Reference**: [../documentation_standards.md](../documentation_standards.md)

## Overview

Product and operations documentation for deploying and using SpectralMC in production.

## Deployment & Operations

### [Deployment Guide](deployment.md)

Production deployment of SpectralMC blockchain model storage:

- S3/MinIO configuration
- IAM policies and security
- Multi-environment strategy
- Disaster recovery
- Cost optimization

**Key Topics**: AWS S3, MinIO, IAM, backup

### [Training Integration](training_integration.md)

Blockchain storage integration with GBM training loop:

- Auto-commit during training
- Periodic checkpoints
- Training modes (pinned, tracking)
- Error handling and best practices

**Key Topics**: Model versioning, checkpoints, training workflow

## Related Documentation

- [Engineering Standards](../engineering/README.md) - Development practices
- [Domain Knowledge](../domain/index.md) - Scientific theory
- [CLAUDE.md](../../CLAUDE.md) - Quick reference guide
