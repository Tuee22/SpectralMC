# File: documents/engineering/blockchain_storage.md
# Blockchain Model Versioning (Deprecated)

**Status**: Deprecated  
**Supersedes**: Prior blockchain storage notes  
**Referenced by**: None

> **Purpose**: Legacy pointer for the former blockchain storage framing.
> **📖 Authoritative Reference**: [Object Store Model Versioning](object_store_storage.md)

## Summary

SpectralMC no longer describes storage as a blockchain. The authoritative storage
model is an append-only, content-addressed object store with explicit S3/MinIO
assumptions and a tamper-evident manifest chain. See
[Object Store Model Versioning](object_store_storage.md) for the current SSoT.
