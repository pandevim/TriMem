"""
Simulated NovaCorp IT Audit environment for Tri-Mem benchmarking.

Mirrors the ALFWorld-style strict command interface but with IT audit vocabulary:
- Strict command syntax: "access procurement_db", "download invoice_1 from procurement_db"
- Digital asset IDs (token_1, soc2_report_1, etc.)
- System exploration with databases that must be queried
- Multi-step audit tasks requiring planning and memory

Command mapping from ALFWorld:
    go to [location]            → access [system]
    open [container]            → query [database]
    take [item] from [location] → download [record] from [system]
    put [item] in/on [location] → upload [record] to [system]

Designed as a drop-in replacement; swap back for ALFWorld when ready.
"""
import random


# ── Task templates ──────────────────────────────────────────────

TASK_TEMPLATES = [
    {
        "id": "audit_vendor_invoice",
        "type": "audit",
        "goal": "Review a vendor invoice and upload it to the compliance dashboard.",
        "solution_steps": [
            "access procurement_db", "query procurement_db", "download invoice_1 from procurement_db",
            "access compliance_dashboard", "query compliance_dashboard", "upload invoice_1 to compliance_dashboard",
        ],
    },
    {
        "id": "revoke_stale_token",
        "type": "security",
        "goal": "Identify a stale user token and revoke it via the admin portal.",
        "solution_steps": [
            "access auth_logs", "download token_1 from auth_logs",
            "access admin_portal", "revoke token_1 with admin_portal",
        ],
    },
    {
        "id": "verify_soc2",
        "type": "compliance",
        "goal": "Retrieve a vendor SOC2 report and upload it to the audit tracker.",
        "solution_steps": [
            "access vendor_portal", "query vendor_portal", "download soc2_report_1 from vendor_portal",
            "access audit_tracker", "upload soc2_report_1 to audit_tracker",
        ],
    },
    {
        "id": "patch_vulnerability",
        "type": "patch",
        "goal": "Download a patch script and run it on the main server.",
        "solution_steps": [
            "access code_repo", "query code_repo", "download patch_script_1 from code_repo",
            "access main_server", "run patch_script_1",
        ],
    },
    {
        "id": "analyze_phishing_email",
        "type": "analysis",
        "goal": "Download a suspicious email and scan it for threats.",
        "solution_steps": [
            "access email_server", "download email_1 from email_server",
            "access security_scanner", "scan email_1 with security_scanner",
        ],
    },
    {
        "id": "flag_policy_violation",
        "type": "compliance",
        "goal": "Find an unapproved expense and flag it in the compliance dashboard.",
        "solution_steps": [
            "access expense_system", "query expense_system", "download expense_report_1 from expense_system",
            "access compliance_dashboard", "query compliance_dashboard", "upload expense_report_1 to compliance_dashboard",
        ],
    },
]


class NovaCorpAuditSim:
    """Simulated Corporate IT Audit environment with strict syntax parsing."""

    def __init__(self, task: dict, shuffle_distractors: bool = True):
        self.task = task
        self.goal = task["goal"]
        self.task_id = task["id"]
        self.task_type = task["type"]
        self.solution = list(task["solution_steps"])

        # State tracking
        self.current_system = "vpn_gateway"
        self.local_workspace = []  # Replaces 'inventory'
        self.completed_steps = []
        self.step_index = 0
        self.done = False
        self.won = False
        self.turn_count = 0

        # Build valid commands at each step
        self._valid_at_step = {i: cmd for i, cmd in enumerate(self.solution)}

        # Systems in the environment for initial description
        self._systems = self._extract_systems()

    def _extract_systems(self) -> list[str]:
        """Pull unique system names from solution steps."""
        sys_list = []
        for step in self.solution:
            if step.startswith("access "):
                system = step[7:]
                if system not in sys_list:
                    sys_list.append(system)
        return sys_list

    def reset(self) -> str:
        """Return initial observation."""
        sys_list = ", ".join(self._systems[:6])
        extra = random.sample(
            ["hr_database", "slack_archives", "legacy_crm", "azure_blob", "contract_vault"],
            k=min(3, 5),
        )
        all_systems = sys_list + ", " + ", ".join(extra)
        return (
            f"You are logged into the NovaCorp central IT gateway. "
            f"Available systems on your network are: {all_systems}.\n\n"
            f"Available commands: access <system>, query <system>, download <file>, "
            f"upload <file>, scan <target>, run <script>, revoke <permission>.\n\n"
            f"Your audit task is to: {self.goal}"
        )

    def step(self, action: str) -> tuple[str, bool, bool]:
        """
        Execute an action. Returns (observation, done, success).
        Mirrors strict syntax parsing for IT Audit commands.
        """
        self.turn_count += 1
        action = action.strip().lower()

        if self.done:
            return "Task already completed.", True, self.won

        # Check if action matches the next expected step
        expected = self._valid_at_step.get(self.step_index, "").lower()

        if action == expected:
            obs = self._success_obs(action)
            self.completed_steps.append(action)
            self.step_index += 1

            # Check if task is complete
            if self.step_index >= len(self.solution):
                self.done = True
                self.won = True
                return obs + "\n\nAudit task completed successfully!", True, True

            return obs, False, False

        # Check if it matches a FUTURE step (out of order but valid)
        for future_idx in range(self.step_index + 1, len(self.solution)):
            if action == self._valid_at_step[future_idx].lower():
                return "Access denied or prerequisite not met. Follow standard operating procedures.", False, False

        # Check if it's a valid command format but wrong target
        if any(action.startswith(prefix) for prefix in
               ["access ", "download ", "upload ", "query ", "revoke ", "scan ", "run "]):
            return "Command executed but returned no results or failed.", False, False

        return "Syntax error: Unrecognized terminal command.", False, False

    def _success_obs(self, action: str) -> str:
        """Generate a plausible success observation."""
        if action.startswith("access "):
            target = action[7:]
            # Reveal what's at this system (next action hints)
            next_step = self._valid_at_step.get(self.step_index + 1, "")
            items = []
            if "download " in next_step:
                obj = next_step.split("from")[0].replace("download ", "").strip()
                items.append(obj)
            extras = random.sample(["user_manual.pdf", "system_log_bak", "config.json"], k=1)
            all_items = items + extras
            return f"Connection established to {target}. Visible endpoints/files: {', '.join(all_items)}."

        if action.startswith("query "):
            target = action[6:]
            next_step = self._valid_at_step.get(self.step_index + 1, "")
            items = []
            if "download " in next_step and "from" in next_step:
                obj = next_step.split("from")[0].replace("download ", "").strip()
                items.append(obj)
            if not items:
                items = ["no anomalous records"]
            return f"Running query on {target}... Query returned: {', '.join(items)}."

        if action.startswith("download "):
            obj = action.split("from")[0].replace("download ", "").strip()
            self.local_workspace.append(obj)
            return f"Successfully downloaded {obj} to your secure local workspace."

        if action.startswith("upload "):
            obj = action.split("to")[0].replace("upload ", "").strip()
            if obj in self.local_workspace:
                self.local_workspace.remove(obj)
            return f"Data transfer complete. {obj} uploaded."

        if action.startswith("revoke "):
            obj = action.split("with")[0].replace("revoke ", "").strip()
            return f"Credentials for {obj} have been revoked."

        if action.startswith("scan "):
            obj = action.split("with")[0].replace("scan ", "").strip()
            return f"Scan complete for {obj}. Threat logged."

        if action.startswith("run "):
            return f"Executing {action[4:]}... Script finished with exit code 0."

        return "Command successful."

    @property
    def is_syntactic_error(self):
        """Check if last obs was a strict syntax fail."""
        return False


def get_tasks(n: int = 10, seed: int = 42) -> list[dict]:
    """Get n tasks, cycling through templates."""
    rng = random.Random(seed)
    tasks = []
    templates = list(TASK_TEMPLATES)
    for i in range(n):
        t = templates[i % len(templates)].copy()
        t["run_id"] = f"{t['id']}_{i}"
        tasks.append(t)
    return tasks
