# SpectralMC Blockchain Storage - Production Deployment Guide

This guide covers production deployment of SpectralMC's blockchain model storage system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [S3 Configuration](#s3-configuration)
3. [IAM Policies & Security](#iam-policies--security)
4. [Multi-Environment Strategy](#multi-environment-strategy)
5. [MinIO Production Setup](#minio-production-setup)
6. [Monitoring & Observability](#monitoring--observability)
7. [Backup & Disaster Recovery](#backup--disaster-recovery)
8. [Cost Optimization](#cost-optimization)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

SpectralMC blockchain storage uses S3-compatible object storage for model versioning:

```mermaid
flowchart TB
  TrainingNode[Training Node - GPU Server]
  S3Bucket[S3 / MinIO Bucket - Model Versions]
  InferenceClient[Inference Client - Pinned/Tracking Modes]
  InferenceNode[Inference Node - CPU/GPU]

  TrainingNode -->|Commit Checkpoint| S3Bucket
  S3Bucket -->|Load Version| InferenceClient
  InferenceClient -->|Serve Predictions| InferenceNode
```

**Key Components**:
- **AsyncBlockchainModelStore**: S3-based storage with atomic CAS commits
- **InferenceClient**: Production model serving (pinned/tracking modes)
- **GarbageCollector**: Automated cleanup of old versions
- **Chain Verification**: Tamper detection via Merkle chain validation

---

## S3 Configuration

### AWS S3 Setup

**1. Create S3 Bucket**:
```bash
aws s3api create-bucket \
    --bucket spectralmc-models-prod \
    --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2
```

**2. Enable Versioning** (optional, for additional safety):
```bash
aws s3api put-bucket-versioning \
    --bucket spectralmc-models-prod \
    --versioning-configuration Status=Enabled
```

**3. Configure Lifecycle Policy** (cost optimization):
```json
{
  "Rules": [
    {
      "Id": "TransitionOldVersions",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

Apply policy:
```bash
aws s3api put-bucket-lifecycle-configuration \
    --bucket spectralmc-models-prod \
    --lifecycle-configuration file://lifecycle-policy.json
```

**4. Enable Server-Side Encryption**:
```bash
aws s3api put-bucket-encryption \
    --bucket spectralmc-models-prod \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'
```

---

## IAM Policies & Security

### IAM Policy Selection

Choose the right IAM policy for each component:

```mermaid
flowchart TB
  Start{Component Type?}

  Training[Training Node]
  Inference[Inference Node]
  GC[Garbage Collection Job]
  Admin[Admin or CI/CD]

  TrainingPolicy[Training Policy - Read/Write]
  InferencePolicy[Inference Policy - Read Only]
  GCPolicy[GC Policy - Read/Delete]
  AdminPolicy[Admin Policy - Full Access]

  Start -->|Runs training jobs| Training
  Start -->|Serves predictions| Inference
  Start -->|Cleanup old versions| GC
  Start -->|Management tasks| Admin

  Training --> TrainingPolicy
  Inference --> InferencePolicy
  GC --> GCPolicy
  Admin --> AdminPolicy

  TrainingPolicy --> TrainingPerms[ListBucket - GetObject - PutObject - DeleteObject]
  InferencePolicy --> InferencePerms[ListBucket - GetObject only]
  GCPolicy --> GCPerms[ListBucket - GetObject - DeleteObject]
  AdminPolicy --> AdminPerms[Full S3 permissions]
```

**Policy Summary**:

| Component | Permissions | Rationale |
|-----------|------------|-----------|
| **Training Node** | Read/Write/Delete | Commits checkpoints, updates chain.json |
| **Inference Node** | Read only | Loads models, never modifies storage |
| **GC Job** | Read/Delete | Removes old versions, no write needed |
| **Admin/CI** | Full access | Bucket setup, disaster recovery |

### Least-Privilege IAM Policies

**Training Node Policy** (read/write access):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::spectralmc-models-prod",
        "arn:aws:s3:::spectralmc-models-prod/*"
      ]
    }
  ]
}
```

**Inference Node Policy** (read-only access):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::spectralmc-models-prod",
        "arn:aws:s3:::spectralmc-models-prod/*"
      ]
    }
  ]
}
```

**Garbage Collection Policy** (delete access):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::spectralmc-models-prod",
        "arn:aws:s3:::spectralmc-models-prod/*"
      ]
    }
  ]
}
```

### Credentials Management

**Option 1: IAM Roles** (Recommended for EC2/ECS):
```python
# No explicit credentials needed - uses instance role
from spectralmc.storage import AsyncBlockchainModelStore

async with AsyncBlockchainModelStore("spectralmc-models-prod") as store:
    # Automatically uses IAM role credentials
    pass
```

**Option 2: AWS Secrets Manager**:
```bash
# Store credentials in Secrets Manager
aws secretsmanager create-secret \
    --name spectralmc/s3-credentials \
    --secret-string '{
        "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    }'
```

```python
import boto3
import json

# Retrieve credentials from Secrets Manager
secrets_client = boto3.client('secretsmanager')
response = secrets_client.get_secret_value(SecretId='spectralmc/s3-credentials')
credentials = json.loads(response['SecretString'])

# Set environment variables
import os
os.environ['AWS_ACCESS_KEY_ID'] = credentials['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['AWS_SECRET_ACCESS_KEY']
```

---

## Multi-Environment Strategy

### Storage Backend Selection

Choose the right storage backend for your deployment:

```mermaid
flowchart TB
  Start{Deployment Type?}
  Cloud[Cloud Deployment]
  OnPrem[On-Premise or Private Cloud]

  CloudProvider{Cloud Provider?}
  AWS[AWS Infrastructure]
  OtherCloud[GCP, Azure, or Other]

  UseS3[Use AWS S3]
  UseMinIO[Use MinIO S3-Compatible]

  Start -->|Cloud-based| Cloud
  Start -->|On-premise or air-gapped| OnPrem

  Cloud --> CloudProvider
  CloudProvider -->|AWS| AWS
  CloudProvider -->|Non-AWS| OtherCloud

  AWS --> UseS3
  OtherCloud --> UseMinIO
  OnPrem --> UseMinIO

  UseS3 --> S3Config[endpoint_url=null - Use AWS SDK default]
  UseMinIO --> MinIOConfig[endpoint_url=http://minio:9000]
```

**AWS S3 Advantages**:
- Fully managed service (no infrastructure maintenance)
- 99.999999999% durability guarantee
- Automatic scaling and global availability
- Native IAM integration
- Lower operational overhead

**MinIO Advantages**:
- On-premise deployment (data sovereignty)
- No egress costs for internal traffic
- Air-gapped environment support
- Same S3 API (drop-in replacement)
- Full control over infrastructure

### Bucket Naming Convention

Use separate buckets per environment:
```
spectralmc-models-dev        # Development
spectralmc-models-staging    # Staging
spectralmc-models-prod       # Production
```

### Environment Configuration

**Development** (`config/dev.yaml`):
```yaml
storage:
  bucket_name: spectralmc-models-dev
  region: us-west-2
  endpoint_url: null  # Use AWS S3

inference:
  version_counter: null  # Track latest (tracking mode)
  poll_interval: 10.0

garbage_collection:
  keep_versions: 5
  run_interval: daily
```

**Production** (`config/prod.yaml`):
```yaml
storage:
  bucket_name: spectralmc-models-prod
  region: us-west-2
  endpoint_url: null

inference:
  version_counter: 42  # Pin to specific version (pinned mode)
  poll_interval: 60.0

garbage_collection:
  keep_versions: 20
  protect_tags: [5, 12, 20, 42]  # Protect production releases
  run_interval: weekly
```

---

## MinIO Production Setup

For on-premise or private cloud deployments, use MinIO:

### Docker Compose (High Availability)

```yaml
version: '3.8'

services:
  minio1:
    image: minio/minio:latest
    command: server --console-address ":9001" http://minio{1...4}/data{1...2}
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
      MINIO_VOLUMES: "/data1 /data2"
    volumes:
      - minio1-data1:/data1
      - minio1-data2:/data2
    networks:
      - minio-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  minio2:
    image: minio/minio:latest
    command: server --console-address ":9001" http://minio{1...4}/data{1...2}
    # ... similar configuration

  minio3:
    image: minio/minio:latest
    command: server --console-address ":9001" http://minio{1...4}/data{1...2}
    # ... similar configuration

  minio4:
    image: minio/minio:latest
    command: server --console-address ":9001" http://minio{1...4}/data{1...2}
    # ... similar configuration

  nginx:
    image: nginx:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - minio-net
    depends_on:
      - minio1
      - minio2
      - minio3
      - minio4

networks:
  minio-net:
    driver: bridge

volumes:
  minio1-data1:
  minio1-data2:
  minio2-data1:
  minio2-data2:
  minio3-data1:
  minio3-data2:
  minio4-data1:
  minio4-data2:
```

### MinIO Configuration

**Enable Erasure Coding** (data redundancy):
```bash
mc admin config set myminio/ storage_class standard=EC:4
```

**Configure Replication** (multi-site):
```bash
mc admin bucket remote add myminio/spectralmc-models-prod \
    https://dr-site.example.com/spectralmc-models-prod \
    --service replication
```

---

## Monitoring & Observability

### Key Metrics to Track

**Storage Metrics**:
- `spectralmc_storage_size_bytes` - Total storage used
- `spectralmc_version_count` - Number of versions
- `spectralmc_commit_rate` - Commits per second
- `spectralmc_commit_latency_seconds` - Time to commit

**Inference Metrics**:
- `spectralmc_inference_client_version` - Current version being served
- `spectralmc_model_load_time_seconds` - Time to load model
- `spectralmc_inference_requests_total` - Request count

**GC Metrics**:
- `spectralmc_gc_versions_deleted_total` - Versions deleted
- `spectralmc_gc_bytes_freed_total` - Storage freed
- `spectralmc_gc_run_duration_seconds` - GC execution time

### CloudWatch Logs

Enable CloudWatch logging for S3:
```bash
aws s3api put-bucket-logging \
    --bucket spectralmc-models-prod \
    --bucket-logging-status file://logging.json
```

`logging.json`:
```json
{
  "LoggingEnabled": {
    "TargetBucket": "spectralmc-logs",
    "TargetPrefix": "s3-access-logs/"
  }
}
```

### Alerting

**CloudWatch Alarms**:

1. **High Storage Growth**:
```bash
aws cloudwatch put-metric-alarm \
    --alarm-name spectralmc-high-storage-growth \
    --alarm-description "Alert when storage grows >10GB/day" \
    --metric-name BucketSizeBytes \
    --namespace AWS/S3 \
    --statistic Average \
    --period 86400 \
    --threshold 10737418240 \
    --comparison-operator GreaterThanThreshold
```

2. **Chain Corruption Detected**:
   - Set up custom metric for corruption detection
   - Alert on any non-zero corruption events

---

## Backup & Disaster Recovery

### Backup Strategy

**1. S3 Cross-Region Replication**:
```bash
aws s3api put-bucket-replication \
    --bucket spectralmc-models-prod \
    --replication-configuration file://replication.json
```

`replication.json`:
```json
{
  "Role": "arn:aws:iam::123456789012:role/s3-replication-role",
  "Rules": [{
    "Status": "Enabled",
    "Priority": 1,
    "Destination": {
      "Bucket": "arn:aws:s3:::spectralmc-models-dr",
      "ReplicationTime": {
        "Status": "Enabled",
        "Time": {"Minutes": 15}
      }
    }
  }]
}
```

**2. Export to Glacier** (long-term):
```bash
# Use lifecycle policy (see S3 Configuration section)
```

**3. Local Backup** (critical versions):
```bash
# Export specific versions
python -m spectralmc.storage export spectralmc-models-prod \
    --versions 42,50,60 \
    --output backup-critical-versions.tar.gz
```

### Disaster Recovery Runbook

**Disaster Recovery Workflow** (S3 Bucket Deletion):

```mermaid
flowchart TB
  Incident[Incident Detected - Bucket Missing]
  Assess{Backup Available?}
  NoBackup[CRITICAL - No backup - Data loss]
  HasBackup[Cross-region replica exists]

  Restore[1. Restore from DR bucket]
  Verify[2. Verify chain integrity]
  CheckIntegrity{Chain Valid?}
  Corrupted[3a. Identify corruption point]
  Valid[3b. Chain verified OK]
  UpdateClients[4. Update inference clients]
  Monitor[5. Monitor for 24 hours]
  Complete[Recovery Complete]

  Incident --> Assess
  Assess -->|No backup| NoBackup
  Assess -->|Replica available| HasBackup
  HasBackup --> Restore
  Restore --> Verify
  Verify --> CheckIntegrity
  CheckIntegrity -->|Corruption found| Corrupted
  CheckIntegrity -->|All valid| Valid
  Corrupted --> UpdateClients
  Valid --> UpdateClients
  UpdateClients --> Monitor
  Monitor --> Complete
```

**Recovery Steps**:

**Scenario: S3 Bucket Accidentally Deleted**

1. **Restore from cross-region replica**:
```bash
aws s3 sync s3://spectralmc-models-dr s3://spectralmc-models-prod
```

2. **Verify chain integrity**:
```bash
python -m spectralmc.storage verify spectralmc-models-prod --detailed
```

3. **Update inference clients** to point to restored bucket

**RTO**: < 1 hour
**RPO**: < 15 minutes (replication lag)

---

## Cost Optimization

### Storage Classes

**Recommendation**:
- **STANDARD**: Recent versions (last 7 days)
- **STANDARD_IA**: Older versions (7-90 days)
- **GLACIER**: Archived versions (>90 days)

**Estimated Costs** (us-west-2, 1TB total):
- STANDARD: $0.023/GB = $23/month
- STANDARD_IA: $0.0125/GB = $12.50/month (savings: 46%)
- GLACIER: $0.004/GB = $4/month (savings: 83%)

### Garbage Collection Policy

Choose GC policy based on cost vs safety tradeoffs:

```mermaid
flowchart TB
  Start{Storage Cost Priority?}
  LowCost[Low cost priority]
  Balance[Balanced approach]
  HighSafety[High safety priority]

  CheckExpType{Experiment Type?}
  Production[Production models only]
  Research[Research and experiments]

  Aggressive[Aggressive GC - keep_versions=5]
  Moderate[Moderate GC - keep_versions=20]
  Conservative[Conservative GC - keep_versions=50]

  Start -->|Minimize storage cost| LowCost
  Start -->|Balance cost and safety| Balance
  Start -->|Maximize version history| HighSafety

  LowCost --> CheckExpType
  CheckExpType -->|Production only| Aggressive
  CheckExpType -->|Mix of both| Moderate

  Balance --> Moderate
  HighSafety --> Conservative

  Aggressive --> AggressiveConfig[keep_min=3 - protect_tags=prod only]
  Moderate --> ModerateConfig[keep_min=5 - protect_tags=releases]
  Conservative --> ConservativeConfig[keep_min=10 - protect_tags=all]
```

**Policy Comparison**:

| Policy | keep_versions | keep_min | Storage Cost | Risk | Use Case |
|--------|--------------|----------|--------------|------|----------|
| **Aggressive** | 5 | 3 | Low | High | Production only, tight budget |
| **Moderate** | 20 | 5 | Medium | Medium | Mixed workloads, standard |
| **Conservative** | 50 | 10 | High | Low | Research, experiment tracking |

**Aggressive** (low storage cost, higher risk):
```yaml
keep_versions: 5
keep_min_versions: 3
protect_tags: [latest_production]
```

**Conservative** (higher storage cost, lower risk):
```yaml
keep_versions: 50
keep_min_versions: 10
protect_tags: [all_production_releases]
```

### Request Optimization

**Batch Operations**: Use `get_versions_batch()` instead of individual `get_version()` calls

**Caching**: Cache frequently accessed versions locally

---

## Troubleshooting

### Common Issues

**1. "OutOfMemoryError" during commit**:
- **Cause**: Large model checkpoint
- **Solution**: Increase instance memory or use model compression

**2. "ConflictError: Not fast-forward"**:
- **Cause**: Concurrent commits from multiple trainers
- **Solution**: Implement commit retry logic or use separate buckets per trainer

**3. "ChainCorruptionError: Broken Merkle chain"**:
- **Cause**: Manual S3 modification or bit rot
- **Solution**: Restore from backup, run `verify --detailed` to identify corruption point

**4. Slow inference client startup**:
- **Cause**: Large checkpoint download
- **Solution**: Use lazy loading or cache checkpoints locally

### Debug Commands

```bash
# Verify chain integrity
python -m spectralmc.storage verify my-bucket --detailed

# Find corruption
python -m spectralmc.storage find-corruption my-bucket

# List all versions
python -m spectralmc.storage list-versions my-bucket

# Inspect specific version
python -m spectralmc.storage inspect my-bucket v0000000042

# Check storage usage
aws s3 ls s3://my-bucket --recursive --summarize --human-readable
```

---

## Contact & Support

For issues or questions:
- Email: matt@resolvefintech.com
- Documentation: See CLAUDE.md
- GitHub Issues: https://github.com/Tuee22/SpectralMC/issues
