# IT Procurement Audit Benchmark

> **Codename:** AuditBench  
> **Purpose:** A single end-to-end evaluation scenario that stress-tests every Tri-Mem component — MSA, Visual Bus, RAG, Entropy Router, and Multi-Agent Orchestration — in a realistic corporate IT audit setting.

---

## Table of Contents

1. [Scenario Overview](#scenario-overview)
2. [Why This Task](#why-this-task)
3. [Environment Design](#environment-design)
4. [The IT Policy & Procurement Guide (MSA Corpus)](#the-it-policy--procurement-guide-msa-corpus)
5. [Audit Records & Data (RAG Corpus)](#audit-records--data-rag-corpus)
6. [Planted Violations (Ground Truth)](#planted-violations-ground-truth)
7. [Agent Roles (Multi-Agent)](#agent-roles-multi-agent)
8. [Action Space](#action-space)
9. [Memory Layer Stress Map](#memory-layer-stress-map)
10. [Entropy Router Trigger Points](#entropy-router-trigger-points)
11. [Scoring Rubric](#scoring-rubric)
12. [Metrics](#metrics)
13. [Baseline Comparisons](#baseline-comparisons)
14. [Implementation Plan](#implementation-plan)
15. [Demo Script](#demo-script)

---

## 1. Scenario Overview

**Setting:** NovaCorp, a mid-size SaaS company (~400 employees), is undergoing its annual IT procurement and security audit. The audit must review 6 months of procurement records, verify credential hygiene, check vendor compliance, and flag policy violations.

**The challenge:** The audit spans **50+ procurement records**, references a **40-page IT Policy & Procurement Guide**, and requires retrieving **exact alphanumeric strings** (API keys, license serials, IP addresses, PO numbers) across a **50-60 turn** investigation.

**The agents:**
- **Compliance Auditor** — knows the policy guide, checks each record against corporate rules
- **Forensics Analyst** — hunts exact credentials, cross-references data, verifies technical details

Both agents share a common episodic timeline and fact store, but have different rulebooks loaded into their semantic memory.

---

## 2. Why This Task

| TriMem Component | Why This Task *Requires* It (Not Optional) |
|---|---|
| **MSA** | A 40-page policy guide with 12 sections of rules. A baseline agent that re-reads this every turn burns ~8,000 tokens/turn and still loses middle-section rules to context rot by turn 20. MSA pre-computes it once. |
| **RAG** | API keys like `sk-NvC-4f8a2b1c9d3e7f6a0b`, license serials like `LIC-2024-NVC-00847`, IPs like `10.42.17.203`. One wrong character = missed violation or false finding. Visual compression *will* blur these. |
| **Visual Bus** | 50+ turn audit. At turn 42, the agent needs to recall that a vendor flagged at turn 8 just appeared in a new PO. Text history at turn 42 = ~25,000 tokens of raw logs. Visual Bus compresses this to a few image tiles. |
| **Entropy Router** | The agent is confident about a budget threshold rule (low entropy → MSA). Uncertain whether a vendor was already checked (medium → Visual Bus). Needs the exact API key for cross-reference (high → RAG). All three bands fire naturally. |
| **Multi-Agent** | Compliance checks require policy expertise. Forensic cross-referencing requires data retrieval expertise. Neither agent alone can complete the audit efficiently. They must share findings without sharing full transcripts. |

---

## 3. Environment Design

### Interface Contract

Same pattern as ALFWorld — drop-in compatible:

```python
class AuditSim:
    def reset() -> str                              # Returns initial briefing
    def step(action: str) -> (str, bool, bool)      # (observation, done, success)
```

### Environment State

```
NovaCorp Audit Environment
├── procurement_records/          # 50 records (POs, invoices, contracts)
│   ├── PO-2025-0001 ... PO-2025-0050
│   └── Each record: vendor, amount, approver, date, department, items
├── credential_vault/             # API keys, tokens, certificates
│   ├── 30 active credentials with rotation dates
│   └── Each: service_name, key_value, created_date, last_rotated, owner
├── vendor_registry/              # 20 vendors with compliance status
│   ├── Each: name, contract_id, certifications, data_residency, tier
│   └── Contract terms, SLA details
├── network_inventory/            # Servers, services, firewall rules
│   ├── 15 servers with IPs, ports, services
│   └── Firewall rule sets
├── employee_directory/           # Approvers with roles and authority levels
│   ├── 25 employees with titles, departments, approval limits
│   └── Delegation chains
└── email_archive/                # 10 flagged emails (audit-relevant correspondence)
    └── Each: from, to, date, subject, body (some contain smoking guns)
```

### Observation Format

Observations return structured text resembling real audit artifacts:

```
--- PROCUREMENT RECORD: PO-2025-0287 ---
Vendor:         CloudSync Solutions
Contract ID:    CSC-NVC-2024-0091
Amount:         $47,500.00
Approved By:    Sarah Chen (Engineering Manager)
Approval Date:  2025-09-14
Department:     Engineering
Items:          CloudSync Enterprise License (50 seats) + Premium API Access
Payment Terms:  Net 30
Budget Code:    ENG-2025-CLOUD-04
Status:         Active
---
```

---

## 4. The IT Policy & Procurement Guide (MSA Corpus)

This is the ~40-page corporate document loaded into MSA. The full text will be in `benchmarks/data/novacorp_it_policy.md`. Below is the section outline and key rules the agents must internalize.

### Section Outline

```
NovaCorp IT Policy & Procurement Guide v4.2
Effective: January 1, 2025
Last Revised: March 15, 2025

PART I — PROCUREMENT GOVERNANCE
  Section 1: Procurement Authority & Approval Thresholds
  Section 2: Vendor Selection & Evaluation
  Section 3: Competitive Bidding Requirements
  Section 4: Contract Terms & SLA Standards
  Section 5: Budget Allocation & Departmental Limits

PART II — SECURITY & COMPLIANCE
  Section 6: Vendor Security Requirements
  Section 7: Credential & API Key Management
  Section 8: Data Residency & Privacy
  Section 9: Network Security & Access Control
  Section 10: Incident Response & Escalation

PART III — OPERATIONAL CONTROLS
  Section 11: License Management & Software Compliance
  Section 12: Audit Trail & Documentation Requirements
```

### Critical Rules (Audit-Relevant Excerpts)

**Section 1 — Approval Thresholds:**
```
1.3.1  Purchases under $5,000: Department Manager approval sufficient.
1.3.2  Purchases $5,000 – $24,999: Director-level approval required.
1.3.3  Purchases $25,000 – $99,999: VP-level approval required.
         Must include written justification memo (Form IT-PRC-03).
1.3.4  Purchases $100,000+: CTO approval + Board Finance Committee review.
         Requires 3 competitive bids (Section 3.2).
1.4.1  No single approver may authorize a purchase within their own
         department exceeding $10,000 without a secondary cross-department sign-off.
1.5.1  Split-purchase detection: Multiple POs to the same vendor within a
         30-day window that collectively exceed a threshold tier are treated
         as a single purchase for approval purposes.
```

**Section 3 — Competitive Bidding:**
```
3.2.1  All purchases exceeding $50,000 require minimum 3 written competitive bids.
3.2.2  Bids must be documented in the Procurement Portal with timestamps.
3.2.3  Sole-source exceptions require VP approval + written justification
         filed within 5 business days of PO issuance.
3.4.1  Vendor relationships exceeding 24 consecutive months must undergo
         a re-evaluation before renewal.
```

**Section 6 — Vendor Security:**
```
6.1.1  All vendors handling NovaCorp data must hold current SOC 2 Type II
         certification or ISO 27001 certification.
6.1.2  Vendors processing PII (as defined in Section 8) must additionally
         hold current SOC 2 + HIPAA BAA if health data is involved.
6.2.1  Vendor security assessments must be conducted annually. Assessments
         older than 14 months are considered lapsed.
6.3.1  Vendors classified as Tier 1 (critical infrastructure) must maintain
         99.95% uptime SLA. Tier 2 vendors: 99.9%. Tier 3: 99.5%.
```

**Section 7 — Credential Management:**
```
7.1.1  All API keys and service tokens must be rotated every 90 calendar days.
7.1.2  Keys not rotated within 90 days must be flagged as NON-COMPLIANT and
         escalated to the Security team within 24 hours of detection.
7.2.1  API keys must be stored in the approved vault (HashiCorp Vault or
         AWS Secrets Manager). Keys stored in environment variables,
         config files, or code repositories are NON-COMPLIANT.
7.3.1  Deprovisioned employee accounts must have all associated API keys
         revoked within 48 hours of termination.
7.4.1  Service accounts must follow naming convention: svc-<team>-<service>-<env>
         (e.g., svc-eng-datasync-prod). Non-conforming names are flagged.
```

**Section 8 — Data Residency:**
```
8.1.1  All PII data must be stored within US or EU data centers.
8.1.2  Vendors with data processing in APAC regions require explicit DPA
         (Data Processing Agreement) approved by Legal.
8.2.1  Any data transfer to a vendor's sub-processor must be documented.
8.3.1  Vendors processing EU citizen data must be GDPR-compliant with
         a valid Data Processing Agreement on file.
```

**Section 9 — Network Security:**
```
9.1.1  All internal services must operate on approved ports: 443 (HTTPS),
         5432 (PostgreSQL), 6379 (Redis), 8080 (internal APIs), 22 (SSH — jump box only).
9.1.2  Any service running on a non-approved port requires Security team
         exception documented in the Network Exception Registry.
9.2.1  Database ports (5432, 3306, 6379) must NOT be exposed to public internet.
9.3.1  All external-facing services require WAF (Web Application Firewall) protection.
9.4.1  VPN access logs must be retained for 12 months minimum.
```

**Section 11 — License Management:**
```
11.1.1 All software deployments must have a valid, non-expired license
         on file before production deployment.
11.1.2 License renewals must be initiated 30 days before expiration.
11.2.1 Seat count must not exceed licensed quantity. Over-deployment by
         even 1 seat is a compliance violation.
11.3.1 Open-source software must be reviewed against the Approved OSS List.
         GPL-licensed dependencies in commercial products require Legal review.
```

---

## 5. Audit Records & Data (RAG Corpus)

All exact values below are the ground truth stored in RAG. The agents must retrieve these *exactly* — no approximation.

### 5.1 Procurement Records (Selected — 12 of 50)

```
RECORD   VENDOR                    AMOUNT       APPROVER                    DATE        DEPT
PO-2025-0042  DataVault Inc.       $8,200.00    Mark Torres (Eng Manager)   2025-07-22  Engineering
PO-2025-0087  CloudSync Solutions  $47,500.00   Sarah Chen (Eng Manager)    2025-09-14  Engineering
PO-2025-0088  CloudSync Solutions  $32,000.00   Sarah Chen (Eng Manager)    2025-10-02  Engineering
PO-2025-0103  NexGen Analytics     $124,750.00  David Park (VP Engineering) 2025-08-30  Engineering
PO-2025-0119  SecureNet Corp       $18,500.00   Lisa Wang (IT Director)     2025-11-05  IT Operations
PO-2025-0156  Quantum DB           $6,800.00    James Liu (Data Manager)    2025-10-18  Data Science
PO-2025-0178  AeroHost Ltd         $95,000.00   Priya Sharma (VP Infra)     2025-09-01  Infrastructure
PO-2025-0201  MindBridge AI        $52,000.00   Carlos Reyes (ML Director)  2025-11-20  Machine Learning
PO-2025-0223  SwiftLog Systems     $3,200.00    Amy Park (DevOps Manager)   2025-12-01  DevOps
PO-2025-0287  CloudSync Solutions  $47,500.00   Sarah Chen (Eng Manager)    2026-01-08  Engineering
PO-2025-0291  DataVault Inc.       $14,500.00   Mark Torres (Eng Manager)   2026-01-15  Engineering
PO-2025-0305  NexGen Analytics     $28,000.00   Rachel Kim (Eng Director)   2026-02-01  Engineering
```

### 5.2 Credential Vault (Selected — 10 of 30)

```
SERVICE               KEY VALUE                              CREATED      LAST ROTATED   OWNER           STORAGE
DataSync API          sk-NvC-4f8a2b1c9d3e7f6a0b             2025-06-15   2025-08-20     svc-eng-datasync-prod    Vault
CloudSync Primary     cs-api-7Kx9mP2qR4wL8nBv               2025-04-01   2025-12-10     svc-eng-cloudsync-prod   Vault
NexGen Ingest         ng-prod-aB3cD4eF5gH6iJ7k              2025-08-30   2025-09-15     svc-data-nexgen-prod     Vault
Quantum DB Admin      qdb-root-X8y7Z6w5V4u3T2s1             2025-03-10   2025-06-01     james.liu                .env file
AeroHost CDN          ah-cdn-mN9oP0qR1sT2uV3w               2025-09-01   2026-03-15     svc-infra-aerohost-prod  Vault
SecureNet Firewall    sn-fw-4A5b6C7d8E9f0G1h                2025-11-05   2025-11-05     svc-it-securenet-prod    Vault
MindBridge ML API     mb-ml-zZ9yY8xX7wW6vV5u                2025-11-20   2025-11-20     carlos.reyes.personal    config.yaml
SwiftLog Ingest       sl-ing-2K3l4M5n6O7p8Q9r               2025-12-01   2026-02-28     svc-devops-swiftlog-prod Vault
Internal Auth Token   int-auth-hH7gG6fF5eE4dD3c             2025-01-10   2025-04-02     svc-eng-auth-prod        Vault
Monitoring Service    mon-svc-1A2b3C4d5E6f7G8h              2025-07-01   2025-10-15     svc-ops-monitor-prod     Vault
```

### 5.3 Vendor Registry (Selected — 8 of 20)

```
VENDOR                CONTRACT ID          CERTS              DATA RESIDENCY   TIER  RELATIONSHIP START  LAST ASSESSED
CloudSync Solutions   CSC-NVC-2024-0091    SOC2 Type II       US-East          1     2023-03-15         2024-11-20
NexGen Analytics      NGA-NVC-2024-0044    ISO 27001          US-West, APAC    2     2024-08-01         2025-01-10
DataVault Inc.        DVI-NVC-2023-0012    SOC2 Type I        US-East          2     2022-06-01         2024-06-15
SecureNet Corp        SNC-NVC-2025-0003    SOC2 Type II       US-East, EU      1     2025-01-01         2025-01-01
AeroHost Ltd          AHL-NVC-2024-0078    ISO 27001          EU, APAC         1     2024-01-10         2025-02-15
Quantum DB            QDB-NVC-2024-0055    None               US-West          3     2024-10-01         (never)
MindBridge AI         MBA-NVC-2025-0009    SOC2 Type II       US-East, EU      2     2025-06-01         2025-06-01
SwiftLog Systems      SLS-NVC-2025-0011    SOC2 Type II       US-East          3     2025-10-01         2025-10-01
```

### 5.4 Network Inventory (Selected — 6 of 15)

```
HOSTNAME              IP ADDRESS       SERVICES                    PORTS OPEN         EXTERNAL?  WAF?
db-primary-01         10.42.17.201     PostgreSQL                  5432               No         N/A
db-replica-02         10.42.17.202     PostgreSQL                  5432, 3306         No         N/A
api-gateway-01        10.42.17.203     NovaCorp API                443, 8443          Yes        Yes
cache-prod-01         10.42.17.204     Redis                       6379               No         N/A
ml-inference-01       10.42.17.210     MindBridge ML Endpoint      443, 9090          Yes        No
nexgen-collector-01   10.42.17.215     NexGen Data Collector       443, 5432, 27017   Yes        No
```

### 5.5 Flagged Emails (Selected — 4 of 10)

```
EMAIL #1
From: sarah.chen@novacorp.com
To: procurement@novacorp.com
Date: 2025-09-12
Subject: Urgent — CloudSync renewal
Body: "Can we push this through fast? David is on PTO and we can't
       wait for VP sign-off. I'll approve it myself — it's only slightly
       over my limit. We need CloudSync live before the Q4 push."

EMAIL #2
From: james.liu@novacorp.com
To: devops@novacorp.com
Date: 2025-10-16
Subject: RE: Quantum DB access
Body: "I put the admin key in the .env on the data-science server
       for now. I know it should be in Vault but the onboarding
       process takes forever. Will move it next sprint."

EMAIL #3
From: security-alerts@novacorp.com
To: all-engineering@novacorp.com
Date: 2026-01-20
Subject: Credential rotation reminder
Body: "The following service keys are overdue for rotation (>90 days):
       DataSync API (last rotated: 2025-08-20, 153 days ago)
       Internal Auth Token (last rotated: 2025-04-02, 293 days ago)
       NexGen Ingest (last rotated: 2025-09-15, 127 days ago)"

EMAIL #4
From: legal@novacorp.com
To: priya.sharma@novacorp.com
Date: 2025-08-28
Subject: AeroHost — APAC data processing
Body: "Priya — we still don't have a signed DPA for AeroHost's APAC
       data centers. Until that's in place, no PII workloads should
       be routed through their Singapore nodes. Please escalate."
```

---

## 6. Planted Violations (Ground Truth)

These are the violations the agents must discover. Each is tagged with the **primary memory layer** that detects it and the **policy section** it violates.

### 6.1 MSA-Dependent Violations (Require Policy Knowledge)

| # | Violation | Records | Policy Section | Details |
|---|-----------|---------|----------------|---------|
| V-01 | **Approval threshold breach** | PO-2025-0087 | §1.3.3 | $47,500 purchase approved by Engineering Manager (Sarah Chen). Requires VP-level approval. |
| V-02 | **Approval threshold breach** | PO-2025-0088 | §1.3.3 | $32,000 purchase approved by Engineering Manager. Requires VP-level approval. |
| V-03 | **Split-purchase circumvention** | PO-2025-0087 + PO-2025-0088 | §1.5.1 | Two CloudSync POs within 18 days totaling $79,500 — treated as single $79,500 purchase, requiring VP approval. |
| V-04 | **Missing competitive bids** | PO-2025-0103 | §3.2.1 | $124,750 purchase — requires 3 competitive bids. None on file. |
| V-05 | **Missing competitive bids** | PO-2025-0201 | §3.2.1 | $52,000 purchase — requires 3 competitive bids. None on file. |
| V-06 | **Vendor re-evaluation overdue** | CloudSync Solutions | §3.4.1 | Relationship since 2023-03-15 (>24 months). No re-evaluation documented before renewal PO-2025-0287. |
| V-07 | **Self-department high-value approval** | PO-2025-0087, PO-2025-0088 | §1.4.1 | Sarah Chen approved >$10,000 in her own department without cross-department sign-off. |
| V-08 | **Unapproved port** | ml-inference-01 | §9.1.1, §9.1.2 | Port 9090 is not on the approved list. No Security exception on file. |
| V-09 | **Missing WAF** | ml-inference-01 | §9.3.1 | External-facing service without WAF protection. |
| V-10 | **Database port exposed externally** | nexgen-collector-01 | §9.2.1 | Port 5432 (PostgreSQL) and 27017 (MongoDB) open on an external-facing server. |

### 6.2 RAG-Dependent Violations (Require Exact String Retrieval)

| # | Violation | Records | Policy Section | Exact Values Required |
|---|-----------|---------|----------------|-----------------------|
| V-11 | **API key rotation overdue** | DataSync API | §7.1.1 | Key: `sk-NvC-4f8a2b1c9d3e7f6a0b`, last rotated: `2025-08-20` (153+ days) |
| V-12 | **API key rotation overdue** | Internal Auth Token | §7.1.1 | Key: `int-auth-hH7gG6fF5eE4dD3c`, last rotated: `2025-04-02` (293+ days) |
| V-13 | **API key rotation overdue** | NexGen Ingest | §7.1.1 | Key: `ng-prod-aB3cD4eF5gH6iJ7k`, last rotated: `2025-09-15` (127+ days) |
| V-14 | **Improper key storage** | Quantum DB Admin | §7.2.1 | Key: `qdb-root-X8y7Z6w5V4u3T2s1` stored in `.env file` (not approved vault) |
| V-15 | **Improper key storage** | MindBridge ML API | §7.2.1 | Key: `mb-ml-zZ9yY8xX7wW6vV5u` stored in `config.yaml` (not approved vault) |
| V-16 | **Non-conforming service account** | MindBridge ML API | §7.4.1 | Owner: `carlos.reyes.personal` — violates `svc-<team>-<service>-<env>` naming convention |
| V-17 | **Non-conforming service account** | Quantum DB Admin | §7.4.1 | Owner: `james.liu` — personal account, not a service account |
| V-18 | **Expired license** | — | §11.1.1 | License `LIC-2024-NVC-00847` for Quantum DB expired `2025-12-31` (still deployed) |

### 6.3 Visual Bus-Dependent Violations (Require Episodic Recall)

These violations can only be detected by connecting information discovered at different turns.

| # | Violation | Discovery Pattern | Details |
|---|-----------|-------------------|---------|
| V-19 | **Email corroborates approval breach** | Email #1 (early turn) + PO-2025-0087 (later turn) | Sarah Chen's email explicitly states she bypassed VP approval. Agent must connect email to PO. |
| V-20 | **Email corroborates storage violation** | Email #2 (early turn) + Quantum DB credential record (later turn) | James Liu's email admits storing key in .env. Agent must connect email to credential record. |
| V-21 | **Repeated vendor pattern** | PO-2025-0087 (turn ~10) + PO-2025-0088 (turn ~15) + PO-2025-0287 (turn ~40) | Three CloudSync POs across the audit — the third at turn 40 triggers the re-evaluation rule (§3.4.1). Agent must recall the earlier POs. |
| V-22 | **Missing DPA despite active PII routing** | Email #4 about AeroHost DPA (turn ~12) + AeroHost vendor record showing APAC residency (turn ~30) + PO-2025-0178 still active (turn ~35) | Legal flagged missing DPA. Vendor still active with APAC data processing. Three data points across 20+ turns. |

### 6.4 Cross-Layer Violations (Require Multiple Memory Modalities)

| # | Violation | Layers Required | Details |
|---|-----------|-----------------|---------|
| V-23 | **Vendor without required certification** | MSA (§6.1.1) + RAG (vendor registry) | Quantum DB has `None` for certifications but handles NovaCorp data. MSA says SOC2/ISO required. |
| V-24 | **Lapsed vendor assessment** | MSA (§6.2.1) + RAG (last assessed dates) | DataVault Inc. last assessed `2024-06-15` — over 14 months ago. Policy requires annual assessment within 14 months. |
| V-25 | **Over-threshold without board review** | MSA (§1.3.4) + RAG (PO-2025-0103 details) | $124,750 purchase by VP — but §1.3.4 requires CTO + Board Finance Committee for $100K+. Only VP signed. |
| V-26 | **NexGen APAC without DPA** | MSA (§8.1.2) + RAG (vendor data residency: "US-West, APAC") | NexGen processes data in APAC. Policy requires explicit DPA. None on file. |

**Total planted violations: 26**

---

## 7. Agent Roles (Multi-Agent)

### 7.1 Compliance Auditor

**Specialization:** Policy interpretation and rule application  
**MSA Corpus:** Full IT Policy & Procurement Guide (all 12 sections)  
**Primary Memory:** MSA (rules) + Visual Bus (what has been reviewed)  
**Responsibilities:**
- Review procurement records against approval thresholds
- Verify competitive bidding compliance
- Check vendor re-evaluation schedules
- Validate budget and departmental controls
- Flag policy violations with section references

**Agent-local rules (MSA):** All of Part I (Procurement Governance) and Part III (Operational Controls)

### 7.2 Forensics Analyst

**Specialization:** Technical verification and exact-data cross-referencing  
**MSA Corpus:** Part II (Security & Compliance) of the policy guide  
**Primary Memory:** RAG (exact values) + Visual Bus (investigation timeline)  
**Responsibilities:**
- Audit credential rotation dates and storage locations
- Verify vendor security certifications
- Check network configurations against policy
- Cross-reference emails with procurement records
- Retrieve and validate exact API keys, IPs, license serials

**Agent-local rules (MSA):** All of Part II (Security & Compliance)

### 7.3 Inter-Agent Communication

```
Compliance Auditor                    Forensics Analyst
      │                                      │
      │──── Shared Visual Bus ───────────────│
      │     (compressed episodic timeline    │
      │      of both agents' discoveries)    │
      │                                      │
      │──── Shared RAG Store ────────────────│
      │     (exact facts: keys, IPs, IDs,   │
      │      amounts, dates, vendor data)    │
      │                                      │
      ├─ Agent-local MSA ─┤  ├─ Agent-local MSA ─┤
      │ Part I + III       │  │ Part II            │
      │ (Procurement +     │  │ (Security +        │
      │  Operations)       │  │  Compliance)       │
      └───────────────────┘  └────────────────────┘
```

### 7.4 Delegation via Entropy Router

- Compliance Auditor encounters a credential storage question (high entropy on security topic) → **delegates to Forensics Analyst**
- Forensics Analyst finds a suspicious PO amount but is uncertain about threshold rules → **delegates to Compliance Auditor**
- Either agent is uncertain about what the other has already investigated → **consults shared Visual Bus**

---

## 8. Action Space

```
# Navigation / Record Access
review record <PO-number>              # Open and read a procurement record
review credential <service-name>       # Pull credential details from vault
review vendor <vendor-name>            # Pull vendor registry entry
review network <hostname>              # Pull network inventory for a server
review email <email-number>            # Read a flagged email
review license <license-id>            # Check license status

# Investigation
cross-reference <item-A> <item-B>      # Compare two records for connections
search records <query>                 # Search procurement DB by keyword
search credentials <query>             # Search credential vault
check policy <section-number>          # Consult a specific policy section (MSA)
calculate days-since <date>            # Compute elapsed time

# Findings
flag violation <record> <policy-section> <severity> <description>
    # severity: critical | high | medium | low

# Control
delegate <agent-role> <query>          # Route question to other agent
submit audit report                    # Finalize (only when confident all violations found)
```

### Syntax Strictness (Mirrors ALFWorld)

- `review record PO-2025-0087` → ✅ Returns record
- `review record PO2025-0087` → ❌ "Record not found." (missing hyphen)
- `flag violation PO-2025-0087 §1.3.3 high "Approval threshold breach"` → ✅
- `flag violation PO-2025-0087 section 1.3.3 high "..."` → ❌ "Invalid policy reference format."

Exact syntax enforcement mirrors the syntactic action gap — RAG must provide the precise PO numbers and policy section references.

---

## 9. Memory Layer Stress Map

### Turn-by-Turn Stress Pattern (Designed)

```
Turn  1-5:   Initial briefing, review emails          → Visual Bus (building timeline)
Turn  6-15:  Review procurement records batch 1        → RAG (exact PO amounts, approvers)
                                                       + MSA (checking threshold rules)
Turn 16-20:  Review credential vault                   → RAG (exact key values, dates)
                                                       + MSA (rotation/storage policies)
Turn 21-30:  Review vendor registry + network          → RAG (exact certs, IPs, ports)
                                                       + MSA (security requirements)
Turn 31-40:  Cross-reference earlier findings           → Visual Bus (recall turn 8 email
                                                         connecting to turn 35 PO)
Turn 41-50:  Deep investigation — connect the dots     → ALL THREE simultaneously
              Split-purchase detection                  → Visual Bus (recall earlier POs)
              Exact key for cross-reference             → RAG
              Policy rule for combined threshold        → MSA
Turn 51-55:  Compile findings, submit report           → Visual Bus (complete audit trail)
                                                       + RAG (exact values for citations)
```

### Context Rot Trap (Baseline Killer)

By turn 40, a baseline agent has accumulated ~30,000+ tokens of audit history. The emails reviewed at turns 3-5 are now in the "lost middle." The baseline will:
- Forget Sarah Chen's email admitting she bypassed VP approval
- Forget James Liu's email about .env storage
- Fail to connect the third CloudSync PO to the first two
- Hallucinate API key characters when citing them in the report

TriMem survives because:
- Emails are in the Visual Bus as a compressed episodic timeline
- Exact key values are in RAG, retrieved losslessly on demand
- Policy rules never left MSA — zero context competition

---

## 10. Entropy Router Trigger Points

| Turn Range | Agent State | Expected Entropy | Routed To | Why |
|---|---|---|---|---|
| 6 | Reading PO amount, checking threshold | Low | MSA | Agent is confident about the dollar-to-threshold mapping |
| 8 | "What was in that email from Sarah?" | Medium | Visual Bus | Agent is uncertain about episodic detail from turn 3 |
| 12 | "What's the exact key value for DataSync?" | High | RAG | Agent has no internal knowledge of exact alphanumeric string |
| 22 | "Does Quantum DB have SOC2?" | Medium→High | RAG | Agent vaguely recalls "no certs" but needs exact confirmation |
| 35 | "Was there a previous CloudSync PO?" | Medium | Visual Bus | Agent is uncertain about a pattern across earlier turns |
| 38 | "What amount was PO-2025-0087?" | High | RAG | Needs exact dollar amount for split-purchase calculation |
| 42 | "Is port 9090 on the approved list?" | Low | MSA | Policy section 9.1.1 has the definitive list |
| 48 | "What did the Legal email say about AeroHost DPA?" | Medium | Visual Bus | Episodic recall of specific email content from turn ~12 |
| 50 | "What's the exact license serial for Quantum DB?" | High | RAG | Exact alphanumeric retrieval for the audit report |

---

## 11. Scoring Rubric

### 11.1 Primary Score: Violation Detection (F1)

```
Precision = (correctly flagged violations) / (total flagged violations)
Recall    = (correctly flagged violations) / (total planted violations)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

**Correct flag requires:**
1. Correct record/entity identified
2. Correct policy section cited
3. Correct severity assigned
4. Description mentions the right factual details

### 11.2 Fidelity Score: Exact Value Accuracy

For each violation that requires citing an exact value (API key, IP, PO number, dollar amount, date):

```
Fidelity = (citations with exactly correct values) / (total citations requiring exact values)
```

This isolates RAG's contribution. A system that finds the right violation but cites `sk-NvC-4f8a2b1c9d3e7f6a0b` as `sk-NvC-4f8a2b1c9d3e7f6a` (truncated) gets the detection point but loses the fidelity point.

### 11.3 Efficiency Scores

```
Token Efficiency   = baseline_total_tokens / trimem_total_tokens  (higher = better compression)
Turn Efficiency    = baseline_turns / trimem_turns                (higher = faster audit)
Cost Efficiency    = baseline_USD_cost / trimem_USD_cost          (higher = cheaper)
```

### 11.4 Cross-Turn Recall Score

Specifically for Visual Bus-dependent violations (V-19 through V-22):

```
Cross-Turn Recall = (correctly connected cross-turn violations) / 4
```

This is the metric that proves episodic memory works. Baselines should score near 0/4 at 50+ turns due to context rot.

### 11.5 Severity Grading

| Severity | Point Value | Description |
|----------|-------------|-------------|
| Critical | 3 points | Security breach, data exposure risk, massive policy violation |
| High | 2 points | Significant non-compliance, financial risk |
| Medium | 1 point | Process violation, documentation gap |
| Low | 0.5 point | Minor non-conformance, naming convention |

```
Weighted Score = Σ (correctly flagged severity × point value) / Σ (planted severity × point value)
```

---

## 12. Metrics

Every benchmark run reports:

| Metric | What It Measures | Primary Layer Tested |
|---|---|---|
| **Violation F1** | Detection accuracy | Overall system |
| **Fidelity Score** | Exact value citation accuracy | RAG |
| **Cross-Turn Recall** | Connecting findings across 20+ turn gaps | Visual Bus |
| **Policy Rule Accuracy** | Correct section cited per violation | MSA |
| **Token Efficiency** | Compression ratio vs baseline | Visual Bus + MSA |
| **Turn Efficiency** | Audit speed (fewer turns = less confusion) | Visual Bus + MSA |
| **Cost Efficiency** | USD savings | Token economics |
| **Router Accuracy** | Did entropy router pick correct modality? | Entropy Router |
| **Delegation Accuracy** | Did inter-agent routing improve outcome? | Multi-Agent |
| **False Positive Rate** | Violations flagged that don't exist | Overall calibration |
| **Spatial Recall Rate** | % of cross-turn facts correctly recalled | Visual Bus |
| **Syntactic Error Rate** | Malformed commands (wrong PO format, etc.) | RAG |
| **Latency per Turn** | Time per decision including routing overhead | Router efficiency |
| **Memory Source Distribution** | % turns using MSA vs Visual Bus vs RAG | Entropy Router balance |

---

## 13. Baseline Comparisons

### 13.1 Agents to Compare

```
Agent A — Baseline:         Full text history, policy in system prompt, no RAG
Agent B — RAG-only:         Full text history + ChromaDB for exact facts
Agent C — TriMem (single):  MSA + Visual Bus + RAG + Entropy Router (one agent)
Agent D — TriMem (multi):   Two agents with shared Visual Bus + RAG (full system)
Agent E — Transcript-pass:  Two agents passing raw transcripts (multi-agent baseline)
```

### 13.2 Expected Results Pattern

```
                    Violation F1  Fidelity  Cross-Turn  Token Cost   Turns
Baseline            0.30-0.40     0.20      0/4         ~150K tokens  55+
RAG-only            0.45-0.55     0.70      0/4         ~170K tokens  50+
TriMem (single)     0.65-0.75     0.85      2/4         ~60K tokens   40-45
TriMem (multi)      0.80-0.90     0.90      3-4/4       ~45K tokens   35-40
Transcript-pass     0.50-0.60     0.65      1/4         ~200K tokens  50+
```

### 13.3 Hero Figures for Paper/Demo

1. **Token Cost Curve:** X = turn number, Y = cumulative tokens. Baseline grows linearly. TriMem curves and plateaus.
2. **Violation Discovery Timeline:** X = turn, Y = cumulative violations found. TriMem frontloads discoveries. Baseline misses late-stage cross-references.
3. **Fidelity Heatmap:** For each violation requiring exact values, show correct/incorrect per agent type. RAG agents show solid green. Baselines show red (hallucinated characters).
4. **Memory Modality Sunburst:** Per-turn which modality was used, color-coded. Shows the entropy router dynamically shifting.
5. **Multi-Agent Communication Graph:** Nodes = agents, edges = delegations. Width = information value. Compare transcript-pass (thick, wasteful) vs TriMem (thin, targeted).

---

## 14. Implementation Plan

### Phase A: Audit Simulator (`benchmarks/audit_sim.py`)

```python
class AuditSim:
    """
    NovaCorp IT Procurement Audit environment.
    Interface mirrors ALFWorldSim for drop-in compatibility.
    """
    def __init__(self, difficulty="standard"):
        self.records = load_procurement_records()      # 50 POs
        self.credentials = load_credential_vault()     # 30 keys
        self.vendors = load_vendor_registry()          # 20 vendors
        self.network = load_network_inventory()        # 15 servers
        self.emails = load_email_archive()             # 10 emails
        self.licenses = load_license_data()            # 15 licenses
        self.violations_found = []
        self.ground_truth = PLANTED_VIOLATIONS         # 26 violations

    def reset(self) -> str:
        """Return initial audit briefing."""

    def step(self, action: str) -> tuple[str, bool, bool]:
        """Parse action, return observation."""

    def score(self) -> dict:
        """Compare violations_found against ground_truth. Return F1, fidelity, etc."""
```

### Phase B: Policy Document (`benchmarks/data/novacorp_it_policy.md`)

Full 40-page document expanding all 12 sections above into realistic corporate policy language. This becomes the MSA corpus.

### Phase C: Audit Data Files (`benchmarks/data/`)

```
benchmarks/data/
├── novacorp_it_policy.md           # MSA corpus (40 pages)
├── procurement_records.json        # 50 PO records
├── credential_vault.json           # 30 credentials
├── vendor_registry.json            # 20 vendors
├── network_inventory.json          # 15 servers
├── email_archive.json              # 10 emails
├── license_data.json               # 15 licenses
└── ground_truth_violations.json    # 26 planted violations with scoring metadata
```

### Phase D: Audit Agents

```
agents/
├── audit_compliance_agent.py       # Compliance Auditor (MSA-heavy)
├── audit_forensics_agent.py        # Forensics Analyst (RAG-heavy)
└── audit_orchestrator.py           # Multi-agent coordinator
```

### Phase E: Dashboard Extension

- Add "Audit Benchmark" tab to existing dashboard
- Audit-specific visualizations: violation discovery timeline, fidelity heatmap
- Two-agent split view showing both agents' turns interleaved

---

## 15. Demo Script

### Live Demo Flow (~10 minutes)

**Minute 0-1: Setup**
> "NovaCorp has 50 procurement records, 30 API credentials, 20 vendors, and a 40-page policy guide. There are 26 planted violations. Let's see how different agent architectures handle this audit."

**Minute 1-3: Baseline Run (fast-forward)**
> Show token cost climbing. Show the agent at turn 35 failing to recall an email from turn 5. Show it hallucinating an API key (`sk-NvC-4f8a2b1c...` becomes `sk-NvC-4f8a2b1d...`). Final score: 9/26 violations found, 0.20 fidelity.

**Minute 3-5: TriMem Single-Agent Run**
> Show entropy router in real-time: orange (MSA) when checking thresholds, cyan (RAG) when pulling exact keys, purple (Visual Bus) when recalling earlier findings. Token curve flattens. Score: 18/26, 0.85 fidelity.

**Minute 5-8: TriMem Multi-Agent Run**
> Show both agents working. Compliance Auditor flags PO-2025-0087 as over-threshold. Forensics Analyst finds the email corroborating it. They share the finding via Visual Bus. Delegation fires: Compliance asks Forensics "is Quantum DB certified?" — Forensics checks RAG, responds "None." Score: 23/26, 0.90 fidelity.

**Minute 8-10: Comparison Dashboard**
> Side-by-side bar charts. Token cost: baseline 150K vs TriMem-multi 45K. Fidelity: 0.20 vs 0.90. Cross-turn recall: 0/4 vs 3/4. "This is what modality-matched memory buys you."

---

## Appendix: Difficulty Scaling

| Difficulty | Records | Credentials | Violations | Max Turns | Policy Size |
|---|---|---|---|---|---|
| **Easy** | 15 | 10 | 8 | 30 | 10 pages |
| **Standard** | 50 | 30 | 26 | 55 | 40 pages |
| **Hard** | 100 | 60 | 50+ | 80 | 40 pages + amendments |
| **Adversarial** | 100 | 60 | 50+ (with 10 near-miss non-violations) | 80 | 40 pages + contradictory amendments |

The adversarial tier adds records that *look* like violations but are technically compliant — tests false positive calibration and forces careful MSA consultation.
