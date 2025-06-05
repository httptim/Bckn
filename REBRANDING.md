# Bckn Rebranding Documentation

This document describes the comprehensive rebranding from Krist to Bckn (Bacon-themed cryptocurrency).

## Overview

Bckn is a fork of Krist that has been completely rebranded with a bacon theme. The rebranding includes changes to:
- Currency name: Krist → Bckn
- Currency symbol: KST → BCN
- Address prefix: k → b
- Name suffix: .kst → .bacon
- Domain: krist.dev → bckn.dev
- WebSocket domain: ws.krist.dev → ws.bckn.dev

## Technical Changes

### 1. Address Format Changes
- **Address prefix**: Changed from `k` to `b`
  - Files modified: `src/utils/crypto.ts`, `src/utils/validationBckn.ts`
  - New address format: `b[a-z0-9]{9}` (e.g., `babcdef123`)
  - Legacy v1 addresses remain unchanged: `[a-f0-9]{10}`

### 2. Name System Changes
- **Name suffix**: Changed from `.kst` to `.bacon`
  - Files modified: `src/utils/validationBckn.ts`
  - New name format: `[a-z0-9]{1,64}.bacon`
  - Metaname format: `[a-z0-9-_]{1,32}@[a-z0-9]{1,64}.bacon`

### 3. Currency Configuration
- **Currency name**: "Bacon"
- **Currency symbol**: "BCN"
- **Configuration location**: `src/krist/motd.ts`

### 4. Core Class and Function Renames
- `KristError` → `BcknError` (file renamed from `KristError.ts` to `BcknError.ts`)
- `isValidKristAddress` → `isValidBcknAddress`
- `isValidKristAddressList` → `isValidBcknAddressList`
- `validationKrist.ts` → `validationBckn.ts`

### 5. Database Configuration
- Default database name: `krist` → `bckn`
- Default database user: `krist` → `bckn`
- Test database: `test_krist` → `test_bckn`
- Redis prefix: `krist:` → `bckn:`

### 6. Environment Variables
- No changes to environment variable names for backward compatibility
- Default values updated where applicable

### 7. Scripts and Tools
- `krist-miner-simple.py` → `bckn-miner-simple.py`
- `krist-miner-gpu.py` → `bckn-miner-gpu.py`
- `krist-tools.sh` → `bckn-tools.sh`
- Mining scripts updated to use new address format and API endpoints

### 8. Web Assets and Documentation
- Updated all references in README.md
- Updated package.json metadata
- Updated HTML templates and error pages
- Updated API documentation references
- Domain references changed from krist.dev to bckn.dev

## Migration Guide

### For Node Operators

1. **Update your domain configuration**:
   - Main API: `krist.dev` → `bckn.dev`
   - WebSocket: `ws.krist.dev` → `ws.bckn.dev`

2. **Update database names** (if using defaults):
   ```sql
   ALTER DATABASE krist RENAME TO bckn;
   ALTER USER krist RENAME TO bckn;
   ```

3. **Update Redis keys** (optional):
   - Keys are prefixed with `bckn:` instead of `krist:`
   - Old keys will need to be migrated or the node resynced

4. **Update environment variables**:
   - `PUBLIC_URL`: Update to use bckn.dev
   - `PUBLIC_WS_URL`: Update to use ws.bckn.dev
   - Database connection strings if using default names

### For Wallet Developers

1. **Address validation**:
   - Update regex patterns to accept `b` prefix for v2 addresses
   - Address format: `/^b[a-z0-9]{9}$/`

2. **Name system**:
   - Update name suffix from `.kst` to `.bacon`
   - Name format: `/^[a-z0-9]{1,64}\.bacon$/i`

3. **API endpoints**:
   - Update base URL from `https://krist.dev` to `https://bckn.dev`
   - All endpoint paths remain the same

4. **Currency display**:
   - Symbol: BCN (instead of KST)
   - Name: Bacon (instead of Krist)

### For Miners

1. **Address generation**:
   - New addresses start with `b` instead of `k`
   - Use the updated `makeV2Address` function or equivalent

2. **Mining endpoints**:
   - Update API URL in mining software to `https://bckn.dev`
   - Mining algorithm remains unchanged (SHA-256)

## Backward Compatibility

The following items were intentionally NOT changed to maintain compatibility:
- API endpoint paths (only domain changes)
- WebSocket protocol and message formats
- Database schema
- Mining algorithm
- Transaction format
- Environment variable names

## Legal and Attribution

- Original Krist created by 3d6 and Lemmmy
- Currently maintained by tmpim
- Licensed under GPL-3.0
- Bckn fork created and maintained by httptim

The copyright notices in source files retain the original "part of Krist" attribution as required by the license, while the functional code has been rebranded to Bckn.

## Repository Information

- Original repository: https://github.com/tmpim/krist
- Fork repository: https://github.com/httptim/Bckn