# Product Documentation

## Overview

Product and operations documentation for deploying and using SpectralMC in production.

## Deployment & Operations

### [Deployment Guide](deployment.md)

Production deployment of SpectralMC blockchain model storage:

- S3/MinIO configuration
- IAM policies and security
- Multi-environment strategy
- Monitoring and disaster recovery
- Cost optimization

**Key Topics**: AWS S3, MinIO, IAM, backup, monitoring

### [Training Integration](training_integration.md)

Blockchain storage integration with GBM training loop:

- Auto-commit during training
- Periodic checkpoints
- Training modes (pinned, tracking)
- Error handling and best practices

**Key Topics**: Model versioning, checkpoints, training workflow

## Related Documentation

- [Engineering Standards](../engineering/index.md) - Development practices
- [Domain Knowledge](../domain/index.md) - Scientific theory
- [CLAUDE.md](../../CLAUDE.md) - Quick reference guide
