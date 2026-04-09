"""
Simulated ALFWorld-style environment for Tri-Mem benchmarking.

Mirrors the real ALFWorld text interface:
- Strict command syntax: "go to countertop 1", "take apple 1 from countertop 1"
- Object IDs (apple 1, knife 2, etc.)
- Room exploration with containers that must be opened
- Multi-step tasks requiring planning and memory

Designed as a drop-in; swap this for real ALFWorld when ready.
"""
import random
from dataclasses import dataclass, field
from typing import Optional


# ── Task templates ──────────────────────────────────────────────

TASK_TEMPLATES = [
    {
        "id": "heat_apple",
        "type": "heat",
        "goal": "Heat an apple and put it on the countertop.",
        "solution_steps": [
            "go to fridge 1", "open fridge 1", "take apple 1 from fridge 1",
            "go to microwave 1", "open microwave 1", "put apple 1 in/on microwave 1",
            "close microwave 1", "heat apple 1 with microwave 1",
            "open microwave 1", "take apple 1 from microwave 1",
            "go to countertop 1", "put apple 1 in/on countertop 1",
        ],
    },
    {
        "id": "clean_mug",
        "type": "clean",
        "goal": "Clean a mug and put it in the cabinet.",
        "solution_steps": [
            "go to coffeemachine 1", "take mug 1 from coffeemachine 1",
            "go to sinkbasin 1", "clean mug 1 with sinkbasin 1",
            "go to cabinet 1", "open cabinet 1", "put mug 1 in/on cabinet 1",
        ],
    },
    {
        "id": "cool_tomato",
        "type": "cool",
        "goal": "Cool a tomato and put it on the dining table.",
        "solution_steps": [
            "go to countertop 2", "take tomato 1 from countertop 2",
            "go to fridge 1", "open fridge 1", "put tomato 1 in/on fridge 1",
            "close fridge 1", "cool tomato 1 with fridge 1",
            "open fridge 1", "take tomato 1 from fridge 1",
            "go to diningtable 1", "put tomato 1 in/on diningtable 1",
        ],
    },
    {
        "id": "find_pen",
        "type": "pick_and_place",
        "goal": "Put a pen on the desk.",
        "solution_steps": [
            "go to drawer 1", "open drawer 1", "take pen 1 from drawer 1",
            "go to desk 1", "put pen 1 in/on desk 1",
        ],
    },
    {
        "id": "slice_bread",
        "type": "slice",
        "goal": "Slice a bread loaf and put a slice on the plate.",
        "solution_steps": [
            "go to countertop 1", "take knife 1 from countertop 1",
            "go to countertop 2", "take bread 1 from countertop 2",
            "use knife 1", "slice bread 1 with knife 1",
            "go to plate 1", "put breadslice 1 in/on plate 1",
        ],
    },
    {
        "id": "examine_book",
        "type": "examine",
        "goal": "Find a book and examine it under the desklamp.",
        "solution_steps": [
            "go to shelf 1", "take book 1 from shelf 1",
            "go to desk 1", "use desklamp 1",
        ],
    },
    {
        "id": "heat_egg",
        "type": "heat",
        "goal": "Heat an egg and put it on the plate.",
        "solution_steps": [
            "go to fridge 1", "open fridge 1", "take egg 1 from fridge 1",
            "go to microwave 1", "open microwave 1", "put egg 1 in/on microwave 1",
            "close microwave 1", "heat egg 1 with microwave 1",
            "open microwave 1", "take egg 1 from microwave 1",
            "go to countertop 1", "take plate 1 from countertop 1",
            "go to diningtable 1", "put egg 1 in/on plate 1",
        ],
    },
    {
        "id": "clean_plate",
        "type": "clean",
        "goal": "Clean a plate and put it in the cabinet.",
        "solution_steps": [
            "go to diningtable 1", "take plate 1 from diningtable 1",
            "go to sinkbasin 1", "clean plate 1 with sinkbasin 1",
            "go to cabinet 2", "open cabinet 2", "put plate 1 in/on cabinet 2",
        ],
    },
    {
        "id": "cool_potato",
        "type": "cool",
        "goal": "Cool a potato and put it on the countertop.",
        "solution_steps": [
            "go to stoveburner 1", "take potato 1 from stoveburner 1",
            "go to fridge 1", "open fridge 1", "put potato 1 in/on fridge 1",
            "close fridge 1", "cool potato 1 with fridge 1",
            "open fridge 1", "take potato 1 from fridge 1",
            "go to countertop 1", "put potato 1 in/on countertop 1",
        ],
    },
    {
        "id": "find_remote",
        "type": "pick_and_place",
        "goal": "Put a remote control on the sofa.",
        "solution_steps": [
            "go to coffeetable 1", "take remotecontrol 1 from coffeetable 1",
            "go to sofa 1", "put remotecontrol 1 in/on sofa 1",
        ],
    },
]


class ALFWorldSim:
    """Simulated ALFWorld environment with strict syntax parsing."""

    def __init__(self, task: dict, shuffle_distractors: bool = True):
        self.task = task
        self.goal = task["goal"]
        self.task_id = task["id"]
        self.task_type = task["type"]
        self.solution = list(task["solution_steps"])

        # State tracking
        self.current_location = "kitchen"
        self.inventory = []
        self.completed_steps = []
        self.step_index = 0
        self.done = False
        self.won = False
        self.turn_count = 0

        # Build valid commands at each step
        self._valid_at_step = {i: cmd for i, cmd in enumerate(self.solution)}

        # Objects in the environment for initial description
        self._locations = self._extract_locations()

    def _extract_locations(self) -> list[str]:
        """Pull unique location names from solution steps."""
        locs = []
        for step in self.solution:
            if step.startswith("go to "):
                loc = step[6:]
                if loc not in locs:
                    locs.append(loc)
        return locs

    def reset(self) -> str:
        """Return initial observation."""
        loc_list = ", ".join(self._locations[:6])
        extra = random.sample(
            ["garbagecan 1", "stoveburner 2", "shelf 3", "drawer 2", "toaster 1"],
            k=min(3, 5),
        )
        all_things = loc_list + ", " + ", ".join(extra)
        return (
            f"You are in the middle of a room. Looking quickly around you, "
            f"you see: {all_things}.\n\n"
            f"Your task is to: {self.goal}"
        )

    def step(self, action: str) -> tuple[str, bool, bool]:
        """
        Execute an action. Returns (observation, done, success).
        Mirrors ALFWorld's strict syntax parsing.
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
                return obs + "\n\nTask completed successfully!", True, True

            return obs, False, False

        # Check if it matches a FUTURE step (out of order but valid)
        for future_idx in range(self.step_index + 1, len(self.solution)):
            if action == self._valid_at_step[future_idx].lower():
                return "Nothing happens.", False, False  # syntactic error - wrong order

        # Check if it's a valid command format but wrong target
        if any(action.startswith(prefix) for prefix in
               ["go to ", "take ", "put ", "open ", "close ", "clean ",
                "heat ", "cool ", "use ", "slice ", "examine "]):
            return "Nothing happens.", False, False  # syntactic error

        return "I don't understand that command.", False, False

    def _success_obs(self, action: str) -> str:
        """Generate a plausible success observation."""
        if action.startswith("go to "):
            target = action[6:]
            # Reveal what's at this location (next action hints)
            next_step = self._valid_at_step.get(self.step_index + 1, "")
            items = []
            if "take " in next_step:
                obj = next_step.split("from")[0].replace("take ", "").strip()
                items.append(obj)
            extras = random.sample(["dishsponge 1", "saltshaker 1", "spatula 1", "cup 2"], k=1)
            all_items = items + extras
            return f"On the {target}, you see: {', '.join(all_items)}."

        if action.startswith("open "):
            target = action[5:]
            next_step = self._valid_at_step.get(self.step_index + 1, "")
            items = []
            if "take " in next_step and "from" in next_step:
                obj = next_step.split("from")[0].replace("take ", "").strip()
                items.append(obj)
            if not items:
                items = ["nothing useful"]
            return f"You open the {target}. Inside, you see: {', '.join(items)}."

        if action.startswith("take "):
            obj = action.split("from")[0].replace("take ", "").strip()
            self.inventory.append(obj)
            return f"You pick up the {obj}."

        if action.startswith("put "):
            obj = action.split("in/on")[0].replace("put ", "").strip()
            if obj in self.inventory:
                self.inventory.remove(obj)
            return f"You put the {obj} down."

        if action.startswith("clean "):
            obj = action.split("with")[0].replace("clean ", "").strip()
            return f"You clean the {obj}."

        if action.startswith("heat "):
            obj = action.split("with")[0].replace("heat ", "").strip()
            return f"You heat the {obj}."

        if action.startswith("cool "):
            obj = action.split("with")[0].replace("cool ", "").strip()
            return f"You cool the {obj}."

        if action.startswith("close "):
            return f"You close the {action[6:]}."

        if action.startswith("use "):
            return f"You turn on the {action[4:]}."

        if action.startswith("slice "):
            obj = action.split("with")[0].replace("slice ", "").strip()
            return f"You slice the {obj}."

        return "OK."

    @property
    def is_syntactic_error(self):
        """Check if last obs was 'Nothing happens' (strict syntax fail)."""
        return False  # tracked per-turn externally


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
