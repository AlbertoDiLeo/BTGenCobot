"""Query rewriting module to transform simple commands into detailed behavioral descriptions"""
import os
import re
import logging
from typing import Optional
from litellm import completion

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a robot behavior planner. Transform commands into structured output for behavior tree generation.

AVAILABLE ACTIONS (use ONLY these exact names):
- SpinLeft: Rotate left. Parameters: spin_dist (radians: 1.57=90°, 3.14=180°), time_allowance
- SpinRight: Rotate right. Parameters: spin_dist (radians), time_allowance
- DriveOnHeading: Move forward. Parameters: dist_to_travel (meters), speed (m/s), time_allowance
- BackUp: Move backward. Parameters: backup_dist (meters), backup_speed (m/s), time_allowance
- Wait: Pause. Parameters: wait_duration (seconds)
- ComputePathToPose: Plan path to goal. Parameters: goal, path (output)
- FollowPath: Execute planned path. Parameters: path
- NavigateToPose: Navigate directly to pose. Parameters: goal
- DetectObject: Find object visually. Parameters: object_description (REQUIRED), target_pose (output)
- PickObject: Pick up object. Parameters: object_description (REQUIRED)
- PlaceObject: Place held object. Parameters: place_description (REQUIRED, e.g., "table", "bin", "box")
- ClearEntireCostmap: Clear navigation costmap

AVAILABLE CONDITIONS (use when the task requires checking state):
- GoalReached, IsStuck, IsBatteryLow, TimeExpired, DistanceTraveled, GoalUpdated

CONTROL STRUCTURES:
- Sequence: Execute in order, all must succeed
- Fallback: Try alternatives until one succeeds
- ReactiveSequence: Re-check conditions each tick
- Retry: RetryUntilSuccessful with num_attempts
- Repeat: Repeat with num_cycles

OUTPUT FORMAT - Output EXACTLY these 3 lines, nothing else:
Actions: <comma-separated action names>
Structure: <control structure>
Description: <what the tree does>

EXAMPLES:

Command: "rotate left 90 degrees"
Actions: SpinLeft
Structure: Sequence
Description: The behavior tree performs a left rotation of 90 degrees using SpinLeft with spin_dist=1.57 radians.

Command: "move forward 2 meters then wait 5 seconds"
Actions: DriveOnHeading, Wait
Structure: Sequence
Description: The behavior tree executes a sequence where the robot first moves forward 2 meters using DriveOnHeading with dist_to_travel=2.0, then pauses for 5 seconds using Wait with wait_duration=5.

Command: "pick up the red cup"
Actions: DetectObject, ComputePathToPose, FollowPath, PickObject
Structure: Sequence
Description: The behavior tree orchestrates a pick-up operation. First, DetectObject locates the red cup and outputs target_pose. Then ComputePathToPose plans a path to target_pose. FollowPath executes the navigation. Finally, PickObject grasps the red cup.

Command: "try to pick up the cube, retry 3 times if it fails"
Actions: PickObject, BackUp
Structure: Retry
Description: The behavior tree wraps PickObject in a RetryUntilSuccessful decorator with num_attempts=3, with BackUp as recovery action.

Command: "navigate to the kitchen, if stuck back up and try again"
Actions: NavigateToPose, BackUp
Structure: Fallback
Description: The behavior tree uses a Fallback containing NavigateToPose to the kitchen. If navigation fails, it executes BackUp to recover, then retries.

Command: "go to charging station, but if battery is not low just wait"
Actions: NavigateToPose, Wait, IsBatteryLow
Structure: Fallback
Description: The behavior tree uses a Fallback. First checks IsBatteryLow condition - if true, NavigateToPose to charging station. If battery is fine, just Wait.

RULES:
1. Output ONLY the 3 lines (Actions, Structure, Description) - no explanations, no options, no questions
2. Actions must be from the available list - approximate if needed, never refuse
3. If a requested condition isn't available, use the closest match or omit it
4. Always produce valid output - never ask for clarification"""

REWRITE_USER_TEMPLATE = """Command: {command}"""


def _extract_location(text: str) -> str:
    """Extract a location/goal name from a phrase."""
    for strip in ["go to", "navigate to", "move to", "travel to", "go back to",
                  "return to", "back to", "head to"]:
        text = text.replace(strip, "").strip()
    # remove leading 'the' only once
    text = re.sub(r'^the\s+', '', text.strip(), flags=re.IGNORECASE)
    return text.strip(" .,") or "goal"


def _local_rewrite(command: str) -> str:
    """
    Offline keyword-based fallback for query rewriting.
    Handles simple sequences AND complex Fallback/Recovery/Retry patterns.
    Produces the structured format (Actions / Structure / Description)
    that the fine-tuned model expects.
    """
    cmd = command.lower()

    # ── COMPLEX PATTERNS (checked first) ───────────────────────────────────

    # Pattern: retry N times  →  RetryUntilSuccessful
    retry_match = re.search(r"retr(?:y|ies?)(?: up to| at most)?\s*(\d+)\s*times?", cmd)
    if retry_match:
        n = retry_match.group(1)
        # What are we retrying? Extract the base action from the part before "retry"
        base_cmd = command[:retry_match.start()].strip(" ,;")
        base = _local_rewrite(base_cmd) if base_cmd else None
        if base:
            actions_line = base.split("\n")[0]  # "Actions: ..."
            desc_line = base.split("Description:")[-1].strip()
            actions = actions_line.replace("Actions:", "").strip()
            return (
                f"Actions: {actions}, BackUp\n"
                f"Structure: Retry\n"
                f"Description: The behavior tree wraps the following operation in a "
                f"RetryUntilSuccessful decorator with num_attempts={n}: {desc_line} "
                f"If it fails, BackUp is used as recovery before retrying."
            )

    # Pattern: "if fails / if it fails / if blocked / if stuck → do Y" or
    #          "try X, otherwise Y" / "X else Y" / "X or else go to Y"
    _fallback_splits = [
        "if that fails", "if it fails", "if fails",
        "otherwise", "or else", ", else", "if blocked",
        "if you can't", "if unable", "if not possible",
        "if the door is closed", "if you cannot",
    ]
    split_keyword = None
    split_pos = -1
    for kw in _fallback_splits:
        pos = cmd.find(kw)
        if pos != -1 and (split_pos == -1 or pos < split_pos):
            split_keyword = kw
            split_pos = pos

    if split_keyword and split_pos > 5:
        main_part = command[:split_pos].strip(" ,;")
        recovery_part = command[split_pos + len(split_keyword):].strip(" ,;")

        main_desc = _local_rewrite(main_part) if main_part else None
        recovery_desc = _local_rewrite(recovery_part) if recovery_part else None

        if main_desc and recovery_desc:
            main_actions = main_desc.split("\n")[0].replace("Actions:", "").strip()
            main_text = main_desc.split("Description:")[-1].strip()
            rec_actions = recovery_desc.split("\n")[0].replace("Actions:", "").strip()
            rec_text = recovery_desc.split("Description:")[-1].strip()

            all_actions = ", ".join(dict.fromkeys(
                [a.strip() for a in (main_actions + ", " + rec_actions).split(",") if a.strip()]
            ))
            return (
                f"Actions: {all_actions}\n"
                f"Structure: Fallback\n"
                f"Description: The behavior tree uses a Fallback node with two branches. "
                f"Primary branch: {main_text} "
                f"If the primary branch fails, the recovery branch executes: {rec_text}"
            )

    # Pattern: "X, if stuck → back up"  (with or without leading 'and')
    stuck_match = re.search(r"(?:,\s*|\band\s+)(?:if (?:it (?:gets? )?)?stuck|if (?:it (?:is )?)?blocked)", cmd)
    if stuck_match:
        main_part = command[:stuck_match.start()].strip(" ,;")
        main_desc = _local_rewrite(main_part) if main_part else None
        if main_desc:
            main_actions = main_desc.split("\n")[0].replace("Actions:", "").strip()
            main_text = main_desc.split("Description:")[-1].strip()
            return (
                f"Actions: {main_actions}, ClearEntireCostmap, BackUp\n"
                f"Structure: Fallback\n"
                f"Description: The behavior tree uses a Fallback node. "
                f"Primary branch: {main_text} "
                f"If stuck, the recovery branch clears the costmap with ClearEntireCostmap "
                f"and backs up with BackUp."
            )

    # Pattern: "go to A, if fails go to B" without explicit keyword — detect
    # two NavigateToPose targets separated by comma/then
    nav_targets = re.findall(
        r"(?:go to|navigate to|move to|head to|go back to|return to)\s+([\w\s]+?)(?:,|\.|;|$)",
        cmd
    )
    if len(nav_targets) >= 2:
        loc_a = nav_targets[0].strip()
        loc_b = nav_targets[1].strip()
        # Only use Fallback if there's a conditional hint between them
        between = cmd[cmd.find(loc_a) + len(loc_a):cmd.find(loc_b)]
        if any(w in between for w in ["if", "or", "else", "can't", "cannot", "fail", "instead"]):
            return (
                f"Actions: NavigateToPose, NavigateToPose\n"
                f"Structure: Fallback\n"
                f"Description: The behavior tree uses a Fallback node. "
                f"Primary branch: NavigateToPose drives the robot to {loc_a}. "
                f"If that fails, the recovery branch uses NavigateToPose to go to {loc_b} instead."
            )

    # ── SIMPLE PATTERNS ─────────────────────────────────────────────────────

    # ------------------------------------------------------------------ pick
    if any(w in cmd for w in ["pick up", "pick", "grab", "grasp", "take"]):
        # try to find the object name after the trigger word
        for trigger in ["pick up", "pick", "grab", "grasp", "take"]:
            idx = cmd.find(trigger)
            if idx != -1:
                obj = command[idx + len(trigger):].strip().lstrip("the ").strip() or "object"
                break
        place = None
        for prep in ["and place", "and put", "and set", "onto", "on the", "on"]:
            pi = cmd.find(prep)
            if pi != -1:
                place = command[pi:].strip().lstrip("and ").strip()
                break
        if place:
            return (
                f"Actions: DetectObject, ComputePathToPose, FollowPath, PickObject, PlaceObject\n"
                f"Structure: Sequence\n"
                f"Description: The behavior tree orchestrates a pick-and-place operation. "
                f"DetectObject locates the {obj} and outputs target_pose. "
                f"ComputePathToPose plans a path to target_pose. "
                f"FollowPath executes the navigation. "
                f"PickObject grasps the {obj}. "
                f"PlaceObject places it {place}."
            )
        return (
            f"Actions: DetectObject, ComputePathToPose, FollowPath, PickObject\n"
            f"Structure: Sequence\n"
            f"Description: The behavior tree orchestrates a pick-up operation. "
            f"DetectObject locates the {obj} and outputs target_pose. "
            f"ComputePathToPose plans a path to target_pose. "
            f"FollowPath executes the navigation. "
            f"PickObject grasps the {obj}."
        )

    # ------------------------------------------------------------------ back up / back away
    if re.match(r'^back\s*(?:up|away|off)?$', cmd.strip()):
        return (
            "Actions: BackUp\nStructure: Sequence\n"
            "Description: The behavior tree backs the robot up using BackUp with backup_dist=0.5."
        )

    # ----------------------------------------------------------------- place
    if any(w in cmd for w in ["place", "put", "set down", "deposit"]):
        for trigger in ["place", "put", "set down", "deposit"]:
            idx = cmd.find(trigger)
            if idx != -1:
                dest = command[idx + len(trigger):].strip().lstrip("the ").strip() or "table"
                break
        return (
            f"Actions: PlaceObject\n"
            f"Structure: Sequence\n"
            f"Description: The behavior tree places the held object. "
            f"PlaceObject deposits the object at the {dest}."
        )

    # ---------------------------------------------------------------- navigate
    if any(w in cmd for w in ["navigate", "go to", "move to", "travel to"]):
        dest = cmd.split("to")[-1].strip() if "to" in cmd else "goal"
        dest = re.sub(r'^the\s+', '', dest, flags=re.IGNORECASE).strip(" .,") or "goal"
        return (
            f"Actions: NavigateToPose\n"
            f"Structure: Sequence\n"
            f"Description: The behavior tree navigates the robot. "
            f"NavigateToPose drives the robot to the {dest}."
        )

    # ----------------------------------------------------------------- rotate
    if any(w in cmd for w in ["spin", "rotat", "turn left", "turn right"]):
        if "right" in cmd:
            return (
                "Actions: SpinRight\nStructure: Sequence\n"
                "Description: The behavior tree rotates the robot to the right using SpinRight with spin_dist=1.57."
            )
        return (
            "Actions: SpinLeft\nStructure: Sequence\n"
            "Description: The behavior tree rotates the robot to the left using SpinLeft with spin_dist=1.57."
        )

    # ------------------------------------------------------------------ move
    if any(w in cmd for w in ["move forward", "drive", "go forward"]):
        dist = "1.0"
        for part in cmd.split():
            try:
                dist = str(float(part))
                break
            except ValueError:
                pass
        return (
            f"Actions: DriveOnHeading\nStructure: Sequence\n"
            f"Description: The behavior tree drives the robot forward {dist} meters using DriveOnHeading."
        )

    # ----------------------------------------------------------------- detect
    if any(w in cmd for w in ["detect", "find", "look for", "search"]):
        obj = cmd.split("for")[-1].strip() if "for" in cmd else "object"
        return (
            f"Actions: DetectObject\nStructure: Sequence\n"
            f"Description: The behavior tree detects an object. "
            f"DetectObject locates the {obj} and outputs target_pose."
        )

    # ----------------------------------------------------------------- wait
    if "wait" in cmd:
        duration = "5"
        for part in cmd.split():
            try:
                duration = str(int(float(part)))
                break
            except ValueError:
                pass
        return (
            f"Actions: Wait\nStructure: Sequence\n"
            f"Description: The behavior tree pauses the robot for {duration} seconds using Wait."
        )

    # ---------------------------------------------------------------- generic
    return (
        f"Actions: NavigateToPose\nStructure: Sequence\n"
        f"Description: The behavior tree executes the command: {command}. "
        f"NavigateToPose drives the robot to the goal."
    )


class QueryRewriter:
    """Rewrites simple commands into detailed behavioral descriptions"""

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-haiku-4.5",
        api_base: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 500,
        temperature: float = 0.3
    ):
        """
        Initialize query rewriter

        Args:
            model: Model to use via LiteLLM (OpenRouter format)
            api_base: API base URL for OpenRouter
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
        """
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Check for API key
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.warning("OPENROUTER_API_KEY not found in environment variables")

    def rewrite_query(self, command: str) -> Optional[str]:
        """
        Rewrite a simple command into detailed behavioral description

        Args:
            command: Simple robot command (e.g., "rotate left")

        Returns:
            Rewritten text ready to use as model input, or None if failed
        """
        try:
            user_prompt = REWRITE_USER_TEMPLATE.format(command=command)

            logger.info(f"Rewriting query: {command}")

            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=self.api_base,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            content = response.choices[0].message.content.strip()
            logger.info(f"Rewrite response: {content}")

            # Return raw response - no parsing
            return content

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            logger.info("Falling back to local keyword-based rewriter")
            return _local_rewrite(command)


# Global rewriter instance
_rewriter: Optional[QueryRewriter] = None


def initialize_rewriter(**kwargs) -> QueryRewriter:
    """Initialize the global query rewriter"""
    global _rewriter
    _rewriter = QueryRewriter(**kwargs)
    return _rewriter


def get_rewriter() -> Optional[QueryRewriter]:
    """Get the global query rewriter instance"""
    return _rewriter


def rewrite_command(command: str) -> Optional[str]:
    """
    Convenience function to rewrite a command using the global rewriter

    Args:
        command: Simple robot command

    Returns:
        Rewritten text ready to use as model input, or None if failed
    """
    if _rewriter is None:
        logger.warning("Query rewriter not initialized")
        return None

    return _rewriter.rewrite_query(command)
