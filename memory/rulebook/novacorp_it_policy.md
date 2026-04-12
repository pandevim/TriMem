# NovaCorp IT Policy & Procurement Guide

_Revision 7.3 — Issued by the Office of the Chief Information Security Officer._
_This document is the authoritative source for all NovaCorp IT auditors. Deviation from documented SOPs will be flagged during quarterly review._

---

## 1. Auditor Mandate and Scope

NovaCorp auditors operate as read/write agents across the corporate network from the `vpn_gateway` entry point. Every audit session is logged. Auditors are expected to complete tasks by issuing exactly **one terminal command per turn** and interpreting the environment's response before issuing the next command. Speculative or chained commands will be rejected by the gateway parser.

Auditors hold read-privileged credentials to every system listed in Section 10 (System Glossary). Write operations (upload, revoke, run, scan) require that the target system has been explicitly `access`ed in the same session; the gateway enforces this prerequisite even when an auditor's role-based access controls would otherwise permit the action.

The canonical failure modes returned by the gateway are documented in Section 9. An auditor who sees a failure response should treat it as diagnostic — it always identifies which class of violation occurred and therefore which corrective action is warranted.

---

## 2. Terminal Command Reference

The NovaCorp gateway accepts seven terminal verbs. Every valid command matches one of the following grammars exactly. Whitespace is significant; extra tokens, punctuation, or reasoning text inside the command slot will trigger a `Syntax error` response.

- `access <system>` — establishes a session with `<system>`. Must precede any other action that targets `<system>`.
- `query <system>` — lists the visible records or endpoints inside `<system>`. Required for any SOP that enumerates records before downloading.
- `download <record> from <system>` — pulls `<record>` into the auditor's local workspace.
- `upload <record> to <system>` — transmits `<record>` from the local workspace to `<system>`. Requires that `<record>` has been previously downloaded.
- `revoke <token> with <system>` — terminates `<token>` using `<system>` as the revocation authority.
- `scan <record> with <system>` — runs a threat scan of `<record>` using `<system>` as the scanning tool.
- `run <script>` — executes `<script>` on the currently accessed system.

Auditors must use the literal record and system identifiers returned by the environment (for example `procurement_db`, `invoice_1`, `token_1`). Synonyms, plural forms, or natural-language paraphrases will be rejected as syntax errors.

---

## 3. SOP 3.1 — Vendor Invoice Audit

Applies when the task is to review a vendor invoice and upload it to the compliance dashboard. This is the most common procurement audit and must follow the canonical six-step sequence below without reordering.

1. `access procurement_db` — establish a session with the procurement database.
2. `query procurement_db` — enumerate invoices visible at this privilege level.
3. `download invoice_1 from procurement_db` — pull the invoice into the local workspace.
4. `access compliance_dashboard` — switch session to the compliance dashboard.
5. `query compliance_dashboard` — confirm the dashboard is ready to receive an upload.
6. `upload invoice_1 to compliance_dashboard` — complete the audit transfer.

Skipping the `query` on either system, or attempting to upload before switching to the dashboard, will trigger an `Access denied or prerequisite not met` response. The prerequisite is structural, not semantic — the dashboard's ingestion API refuses writes from sessions that have not queried it first.

---

## 4. SOP 3.2 — Stale Token Revocation

Applies when the task is to identify a stale user token and revoke it via the admin portal. This is a four-step security SOP; there is no query step because `auth_logs` exposes tokens directly on access.

1. `access auth_logs` — connect to the authentication log service.
2. `download token_1 from auth_logs` — retrieve the stale token record.
3. `access admin_portal` — switch session to the admin portal, which is the revocation authority.
4. `revoke token_1 with admin_portal` — complete the revocation.

Note the `with` preposition in the final command: the revoke verb binds the token to the revocation authority, not to a location. Using `from` instead of `with` is the most common syntactic failure for this SOP and will produce a `Syntax error` response.

---

## 5. SOP 3.3 — SOC2 Compliance Verification

Applies when the task is to retrieve a vendor SOC2 report and upload it to the audit tracker. Five steps, with a mandatory `query` on the vendor portal (SOC2 reports are not enumerated on access for confidentiality reasons).

1. `access vendor_portal` — connect to the third-party vendor portal.
2. `query vendor_portal` — enumerate SOC2 reports available at this privilege level.
3. `download soc2_report_1 from vendor_portal` — pull the report to the local workspace.
4. `access audit_tracker` — switch session to the audit tracker.
5. `upload soc2_report_1 to audit_tracker` — file the report.

The audit tracker does not require a pre-upload query. It is a write-only compliance sink.

---

## 6. SOP 3.4 — Vulnerability Patch Deployment

Applies when the task is to download a patch script and run it on the main server. The `run` verb has no `with` or `from` suffix — it executes `<script>` on the currently accessed system, which must be `main_server` at the moment of execution.

1. `access code_repo` — connect to the source code repository.
2. `query code_repo` — enumerate patch scripts.
3. `download patch_script_1 from code_repo` — pull the script to the local workspace.
4. `access main_server` — switch session to the target server (this also sets the execution context for `run`).
5. `run patch_script_1` — execute the patch.

Attempting to `run patch_script_1 on main_server` will be rejected as a syntax error; the `on` preposition is not part of the grammar. The currently accessed system is implicit.

---

## 7. SOP 3.5 — Phishing Email Analysis

Applies when the task is to download a suspicious email and scan it for threats. Four steps, no query step (the email server exposes the flagged record on access).

1. `access email_server` — connect to the mail service.
2. `download email_1 from email_server` — pull the suspect message.
3. `access security_scanner` — switch session to the scanning tool.
4. `scan email_1 with security_scanner` — complete the analysis.

The `scan` verb uses `with` like `revoke` — it binds the target to the scanning tool, not a location. Auditors who have just completed SOP 3.2 often reflexively type `scan email_1 from security_scanner`; this is the second-most-common syntax error in phishing analysis tasks.

---

## 8. SOP 3.6 — Expense Policy Violation

Applies when the task is to find an unapproved expense and flag it in the compliance dashboard. Structurally identical to SOP 3.1 but with different record and source identifiers.

1. `access expense_system` — connect to the corporate expense system.
2. `query expense_system` — enumerate expense reports.
3. `download expense_report_1 from expense_system` — pull the flagged report.
4. `access compliance_dashboard` — switch session to the dashboard.
5. `query compliance_dashboard` — confirm dashboard readiness.
6. `upload expense_report_1 to compliance_dashboard` — complete the flag.

Like SOP 3.1, both the source-side and the destination-side `query` steps are mandatory. Skipping either will produce an `Access denied or prerequisite not met` response.

---

## 9. Error Response Codes

The gateway returns exactly three failure response strings. Each one maps unambiguously to a category of violation, which in turn dictates the corrective action the auditor should take.

- `Access denied or prerequisite not met. Follow standard operating procedures.` — The command's syntax is valid and the target exists, but the SOP prerequisite ordering was violated. Corrective action: re-examine the relevant SOP in Sections 3-8 and execute the earlier step that was skipped (typically an `access` or `query`).
- `Command executed but returned no results or failed.` — The command's syntax is valid but the target system or record does not exist in the current context. Corrective action: re-check the system identifier against the System Glossary (Section 10) or the record identifier from the most recent `query` response.
- `Syntax error: Unrecognized terminal command.` — The command's grammar did not match any of the seven verbs in Section 2. Corrective action: re-issue the command using the exact grammar; common causes are extra words, wrong prepositions (`from` vs `with`), or chain-of-thought reasoning text leaking into the command slot.

Auditors who receive the same failure response on two consecutive turns should suspect a loop and escalate per Section 11 rather than retry with superficial edits.

---

## 10. System Glossary

The following systems are reachable from `vpn_gateway` during standard audit sessions. Identifiers are case-insensitive but must be spelled exactly as listed.

- `procurement_db` — primary procurement database; stores vendor invoices (`invoice_*`).
- `compliance_dashboard` — write-sink for audit artefacts; requires a pre-upload query.
- `auth_logs` — authentication log service; exposes tokens (`token_*`) on access.
- `admin_portal` — revocation authority for user credentials and tokens.
- `vendor_portal` — third-party vendor surface; hosts SOC2 reports (`soc2_report_*`).
- `audit_tracker` — write-only compliance sink for filed reports.
- `code_repo` — source code repository; hosts patch scripts (`patch_script_*`).
- `main_server` — the production application server; execution context for the `run` verb.
- `email_server` — corporate mail service; exposes flagged messages (`email_*`).
- `security_scanner` — threat-scanning tool; used with the `scan` verb.
- `expense_system` — corporate expense reporting system (`expense_report_*`).

Systems such as `hr_database`, `slack_archives`, `legacy_crm`, `azure_blob`, and `contract_vault` are sometimes visible at the gateway but are **not** part of any current audit SOP. Accessing them during a documented audit task is a common distractor failure and wastes a turn.

---

## 11. Escalation and Loop Detection

If the auditor issues the same failing command twice in a row, the correct response is to stop and consult the relevant SOP rather than iterate. Repeated identical failures almost always indicate a missing prerequisite (Section 9, class 1) or a wrong preposition (Section 9, class 3), not a transient gateway error.

If a task exceeds 15 turns without completion, the auditor should re-read the SOP section applicable to the task type and verify that every step has been executed in order. The NovaCorp gateway does not support partial rollback; once a step is completed it cannot be undone, but re-executing an already-completed step is a no-op and safe.

---

## 12. Change Control and Document Authority

This policy document is revised quarterly. SOP command sequences (Sections 3-8) are the contractual source of truth and override any contradictory guidance the auditor may recall from previous revisions. When in doubt, the command sequences in this document are correct; the auditor's memory is not.
