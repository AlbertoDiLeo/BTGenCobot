"""BehaviorTree Generation with CFG-Constrained LLM Inference"""
import time
import logging
import re
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_restricted_grammar(allowed_actions: List[str], structure: Optional[str] = None, max_depth: int = 5) -> str:
    """
    Generate a restricted EBNF grammar using EXPLICIT syntax: <Action ID="NodeType" .../>.
    Uses depth-limited hierarchy to prevent degenerate infinite nesting.
    Defines specific ports per action type to prevent hallucination.

    Args:
        allowed_actions: List of allowed action names (e.g., ["SpinLeft", "BackUp", "DetectObject"])
        structure: Optional structure hint (unused, kept for API compatibility)
        max_depth: Maximum nesting depth (default 5 levels)

    Returns:
        Complete EBNF grammar string with explicit Action/Condition syntax
    """
    # All known action nodes with their valid ports
    ACTION_PORTS = {
        "SpinLeft": ["spin_dist", "time_allowance"],
        "SpinRight": ["spin_dist", "time_allowance"],
        "DriveOnHeading": ["dist_to_travel", "speed", "time_allowance"],
        "BackUp": ["backup_dist", "backup_speed", "time_allowance"],
        "Wait": ["wait_duration"],
        "ComputePathToPose": ["goal", "path", "planner_id"],
        "FollowPath": ["path", "controller_id"],
        "NavigateToPose": ["goal"],
        "DetectObject": ["object_description", "target_pose"],
        "PickObject": ["object_description"],
        "PlaceObject": ["place_description"],
        "ClearEntireCostmap": [],
    }

    KNOWN_ACTIONS = set(ACTION_PORTS.keys())

    # All known condition nodes with their valid ports
    CONDITION_PORTS = {
        "GoalReached": [],
        "IsStuck": [],
        "IsBatteryLow": [],
        "GoalUpdated": [],
        "TimeExpired": ["seconds"],
        "DistanceTraveled": ["distance"],
    }

    KNOWN_CONDITIONS = set(CONDITION_PORTS.keys())

    if not allowed_actions:
        raise ValueError("No actions provided in allowed_actions")

    # Separate actions and conditions from input (deduplicate)
    unique_actions = []
    unique_conditions = []

    for item in allowed_actions:
        if item in KNOWN_ACTIONS:
            if item not in unique_actions:
                unique_actions.append(item)
        elif item in KNOWN_CONDITIONS:
            if item not in unique_conditions:
                unique_conditions.append(item)
        else:
            logger.warning(f"Unknown action/condition: {item}, skipping")

    if not unique_actions:
        raise ValueError(f"No valid actions found in allowed_actions: {allowed_actions}")

    # Only use conditions if explicitly specified - don't default to all
    # Conditions in sequences cause failures if their state doesn't match

    logger.info(f"Generated explicit grammar for actions: {unique_actions}, conditions: {unique_conditions}")

    # Collect all unique ports needed (to avoid duplicate rule definitions)
    all_ports = set()
    for action in unique_actions:
        all_ports.update(ACTION_PORTS.get(action, []))
    for condition in unique_conditions:
        all_ports.update(CONDITION_PORTS.get(condition, []))

    # Base grammar with explicit syntax
    grammar = r"""// BehaviorTree XML Grammar - EXPLICIT SYNTAX (Restricted)
// Uses <Action ID="NodeType" .../> format to match model output

start: xml_decl? root

xml_decl: "<?xml" WS "version" WS? "=" WS? version_val WS? "?>"
version_val: "\"1.0\""

root: "<root" " " "BTCPP_format=\"4\"" " " main_tree_attr " "? ">" WS? behavior_tree WS? "</root>"
main_tree_attr: "main_tree_to_execute=\"" tree_id "\""
tree_id: /[A-Za-z_][A-Za-z0-9_]*/

behavior_tree: "<BehaviorTree" " " bt_id_attr " "? ">" WS? bt_content WS? "</BehaviorTree>"
bt_id_attr: "ID=\"" tree_id "\""

bt_content: node_l1

"""

    # Build node definitions for each level
    has_conditions = len(unique_conditions) > 0
    
    for level in range(1, max_depth + 1):
        is_leaf_level = (level == max_depth)
        next_level = level + 1

        node_options = ["action_node"]
        if has_conditions:
            node_options.append("condition_node")

        if not is_leaf_level:
            node_options.extend([f"control_l{level}", f"decorator_l{level}"])

        grammar += f"// Level {level}" + (" (leaf level)" if is_leaf_level else "") + "\n"
        grammar += f"node_l{level}: {' | '.join(node_options)}\n\n"

        if not is_leaf_level:
            child_node = f"node_l{next_level}"
            child_list = f"node_list_l{next_level}"

            # Control nodes - with optional name attribute only
            grammar += f'control_l{level}: sequence_l{level} | fallback_l{level} | parallel_l{level} | reactive_seq_l{level} | reactive_fb_l{level}\n'
            grammar += f'sequence_l{level}: "<Sequence" name_attr? ">" WS? {child_list} WS? "</Sequence>"\n'
            grammar += f'fallback_l{level}: "<Fallback" name_attr? ">" WS? {child_list} WS? "</Fallback>"\n'
            grammar += f'parallel_l{level}: "<Parallel" name_attr? ">" WS? {child_list} WS? "</Parallel>"\n'
            grammar += f'reactive_seq_l{level}: "<ReactiveSequence" name_attr? ">" WS? {child_list} WS? "</ReactiveSequence>"\n'
            grammar += f'reactive_fb_l{level}: "<ReactiveFallback" name_attr? ">" WS? {child_list} WS? "</ReactiveFallback>"\n\n'

            # Decorator nodes - use generic attribute pattern like static grammar
            grammar += f'decorator_l{level}: inverter_l{level} | force_success_l{level} | force_failure_l{level} | repeat_l{level} | retry_l{level} | keep_running_l{level}\n'
            grammar += f'inverter_l{level}: "<Inverter" ((" " | "\\t") attribute)* " "? ">" WS? {child_node} WS? "</Inverter>"\n'
            grammar += f'force_success_l{level}: "<ForceSuccess" ((" " | "\\t") attribute)* " "? ">" WS? {child_node} WS? "</ForceSuccess>"\n'
            grammar += f'force_failure_l{level}: "<ForceFailure" ((" " | "\\t") attribute)* " "? ">" WS? {child_node} WS? "</ForceFailure>"\n'
            grammar += f'repeat_l{level}: "<Repeat" ((" " | "\\t") attribute)* " "? ">" WS? {child_node} WS? "</Repeat>"\n'
            grammar += f'retry_l{level}: "<RetryUntilSuccessful" ((" " | "\\t") attribute)* " "? ">" WS? {child_node} WS? "</RetryUntilSuccessful>"\n'
            grammar += f'keep_running_l{level}: "<KeepRunningUntilFailure" ((" " | "\\t") attribute)* " "? ">" WS? {child_node} WS? "</KeepRunningUntilFailure>"\n\n'

            # Node list
            grammar += f'{child_list}: {child_node} (WS? {child_node})*\n\n'

    # Generate specific action rules with their exact ports
    grammar += "// Action nodes - each with specific allowed ports\n"
    action_alternatives = []
    for action in unique_actions:
        ports = ACTION_PORTS.get(action, [])
        rule_name = f'{action.lower()}_action'
        action_alternatives.append(rule_name)
        if ports:
            # Build optional port attributes (each can appear 0 or 1 time)
            port_attrs = " ".join([f'{p}_attr?' for p in ports])
            grammar += f'{rule_name}: "<Action" " " "ID=\\"{action}\\"" " "? {port_attrs} "/>"\n'
        else:
            grammar += f'{rule_name}: "<Action" " " "ID=\\"{action}\\"" " "? "/>"\n'

    grammar += f"action_node: {' | '.join(action_alternatives)}\n\n"

    # Generate specific condition rules with their exact ports (only if conditions exist)
    if unique_conditions:
        grammar += "// Condition nodes - each with specific allowed ports\n"
        condition_alternatives = []
        for condition in unique_conditions:
            ports = CONDITION_PORTS.get(condition, [])
            rule_name = f'{condition.lower()}_condition'
            condition_alternatives.append(rule_name)
            if ports:
                port_attrs = " ".join([f'{p}_attr?' for p in ports])
                grammar += f'{rule_name}: "<Condition" " " "ID=\\"{condition}\\"" " "? {port_attrs} "/>"\n'
            else:
                grammar += f'{rule_name}: "<Condition" " " "ID=\\"{condition}\\"" " "? "/>"\n'

        grammar += f"condition_node: {' | '.join(condition_alternatives)}\n\n"

    # Define all port attribute rules once (no duplicates)
    grammar += "// Port attribute definitions\n"
    for port in sorted(all_ports):
        grammar += f'{port}_attr: " " "{port}=\\"" attr_value "\\""\n'

    grammar += """
// Generic attribute - any name="value" pair (matches static grammar)
attribute: att_name " "? "=" " "? "\\"" att_value_content "\\""
att_name: /[a-zA-Z_][a-zA-Z0-9_]*/
att_value_content: /[^<&"]*/

// Common attributes
name_attr: " " "name=\\"" attr_value "\\""

// Attribute value (no quotes, angle brackets, or ampersands)
attr_value: /[^<&"]*/

WS: /[ \\t\\n\\r]+/
"""

    return grammar


def parse_allowed_actions(rewritten_input: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Parse the 'Actions:' and 'Structure:' lines from rewritten input.
    
    Args:
        rewritten_input: Rewritten input string from query rewriter
        
    Returns:
        Tuple of (list of action names, structure hint string) or (None, None) if not found
        
    Example:
        Input: "Structure: Fallback[...]\nActions: Spin, BackUp\nTask: ..."
        Output: (["Spin", "BackUp"], "Fallback")
    """
    if not rewritten_input:
        return None, None
    
    actions = None
    structure = None
    
    # Look for "Actions:" line
    actions_match = re.search(r'Actions:\s*([^\n]+)', rewritten_input, re.IGNORECASE)
    if actions_match:
        actions_str = actions_match.group(1).strip()
        # Split by comma and clean up
        actions = [action.strip() for action in actions_str.split(',')]
        actions = [action for action in actions if action]  # Remove empty strings
    
    # Look for "Structure:" line to understand control flow needs
    structure_match = re.search(r'Structure:\s*([^\n]+)', rewritten_input, re.IGNORECASE)
    if structure_match:
        structure_str = structure_match.group(1).strip()
        # Extract main control structure type
        if 'Fallback' in structure_str:
            structure = 'Fallback'
        elif 'RetryUntilSuccessful' in structure_str or 'Retry' in structure_str:
            structure = 'Retry'
        elif 'Repeat' in structure_str:
            structure = 'Repeat'
        elif 'ReactiveSequence' in structure_str:
            structure = 'ReactiveSequence'
        elif 'ReactiveFallback' in structure_str:
            structure = 'ReactiveFallback'
        elif 'Sequence' in structure_str:
            structure = 'Sequence'
        else:
            structure = structure_str
    
    if actions:
        logger.info(f"Parsed allowed actions: {actions}, structure: {structure}")
    
    return actions, structure


class BTGenerator:
    """BehaviorTree generator using Transformers + Outlines CFG constraints"""

    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        """Initialize the BT generator

        Args:
            model_path: Path to the base model directory
            adapter_path: Optional path to the LoRA adapter directory
        """
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.model = None
        self.tokenizer = None
        self.outlines_model = None
        self.xml_pattern = None
        self.model_loaded = False

        logger.info(f"Initializing with model: {self.model_path}")
        if self.adapter_path:
            logger.info(f"Using adapter: {self.adapter_path}")

    def load_model(self):
        """Load the model with transformers backend"""
        try:
            logger.info("Loading model with transformers...")
            start_time = time.time()

            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import outlines

            logger.info(f"Loading base model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Using device: {device}")

            if device in {"cuda", "mps"}:
                model_dtype = torch.float16
            else:
                model_dtype = torch.float32

            if device == "cuda":
                device_map = "cuda:0"
            elif device == "mps":
                device_map = {"": "mps"}
            else:
                device_map = "cpu"

            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                dtype=model_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True
            )

            if self.adapter_path:
                logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
                logger.info("Adapter loaded successfully")
            else:
                self.model = base_model
                logger.info("No adapter specified, using base model only")

            logger.info("Creating Outlines wrapper with transformers backend...")
            self.outlines_model = outlines.from_transformers(
                self.model,  # type: ignore
                self.tokenizer
            )

            grammar_path = Path(__file__).parent.parent / "bt_grammar.ebnf"
            logger.info(f"Loading grammar from: {grammar_path}")

            if not grammar_path.exists():
                raise FileNotFoundError(f"Grammar file not found: {grammar_path}")

            bt_grammar = grammar_path.read_text()

            logger.info("Creating CFG pattern for XML...")
            try:
                from outlines.types import CFG
                self.xml_pattern = CFG(bt_grammar)
                logger.info("XML pattern created successfully (using outlines.types.CFG)")
            except Exception as e:
                logger.warning(f"Failed to create XML CFG pattern: {e}")
                self.xml_pattern = None

            load_time = time.time() - start_time
            self.model_loaded = True

            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"XML pattern available: {self.xml_pattern is not None}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate_xml(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        custom_grammar: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate BT XML using CFG-constrained generation

        Args:
            prompt: Complete prompt for generation
            max_tokens: Maximum tokens to generate (default 2048, set to None for unlimited)
            temperature: Sampling temperature (default 0.3, lower=more deterministic)
            custom_grammar: Optional custom grammar to use instead of default

        Returns:
            Tuple of (generated_xml, error_message)
        """
        if not self.model_loaded:
            return None, "Model not loaded"

        # Determine which grammar to use
        if custom_grammar:
            grammar_to_use = custom_grammar
            logger.info("Using custom restricted grammar")
        elif self.xml_pattern is not None:
            # Use the default grammar pattern
            grammar_to_use = self.xml_pattern
            logger.info("Using default grammar")
        else:
            return None, "No grammar available"
        
        if self.outlines_model is None:
            return None, "Outlines model not available"

        try:
            start_time = time.time()

            # If custom_grammar is a string, create CFG from it
            if isinstance(grammar_to_use, str):
                from outlines.types import CFG
                logger.info("Creating CFG pattern from custom grammar...")
                cfg_creation_start = time.time()
                cfg_pattern = CFG(grammar_to_use)
                logger.info(f"CFG pattern created in {time.time() - cfg_creation_start:.2f}s")
            else:
                cfg_pattern = grammar_to_use

            # Call the Outlines model directly with the CFG output type
            logger.info("Starting model generation...")
            result = self.outlines_model(
                prompt,
                cfg_pattern,
                max_new_tokens=max_tokens,
                temperature=temperature
            )

            gen_time = time.time() - start_time
            logger.info(f"XML generated in {gen_time:.2f}s")
            
            if not result:
                logger.warning("Model returned empty result")

            return result, None

        except Exception as e:
            error_msg = f"XML generation failed: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, error_msg

    def generate_bt(
        self,
        command: str,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        prompt_format: str = "chat",
        rewritten_input: Optional[str] = None,
        custom_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate BehaviorTree XML from natural language command

        Args:
            command: Natural language command
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            prompt_format: Format of prompt - "chat" (Llama chat) or "alpaca" (instruction format)
            rewritten_input: Rewritten input from query rewriter (used directly in prompt)
            custom_instruction: Optional custom instruction to override default alpaca_instruction.txt

        Returns:
            Dictionary with:
            {
                "bt_xml": str,
                "generation_time_ms": int,
                "method_used": str,
                "success": bool,
                "error": Optional[str]
            }
        """
        if not self.model_loaded:
            return {
                "bt_xml": None,
                "generation_time_ms": 0,
                "method_used": None,
                "success": False,
                "error": "Model not loaded"
            }

        from prompts import build_prompt, build_alpaca_prompt, extract_xml_from_response
        from validation.validator import validate_bt_xml, validate_action_space, validate_semantic_structure
        from validation.post_processor import create_default_filter

        start_time = time.time()

        if self.xml_pattern is not None:
            logger.info(f"Generating BT for: {command}")

            # Parse allowed actions from rewritten input if available
            custom_grammar = None
            if rewritten_input:
                allowed_actions, structure = parse_allowed_actions(rewritten_input)
                if allowed_actions:
                    try:
                        grammar_str = generate_restricted_grammar(allowed_actions, structure)
                        custom_grammar = grammar_str
                        logger.info(f"Using restricted grammar with actions: {allowed_actions}, structure: {structure}")
                        logger.debug(f"Generated grammar:\n{grammar_str}")
                    except Exception as e:
                        logger.warning(f"Failed to generate restricted grammar: {e}, using default")
                        custom_grammar = None

            if prompt_format == "alpaca":
                prompt = build_alpaca_prompt(command, rewritten_input=rewritten_input, custom_instruction=custom_instruction)
            else:
                prompt = build_prompt(command, output_format="xml", tokenizer=self.tokenizer)

            logger.info(f"=== PROMPT ===\n{prompt}\n=== END PROMPT ===")
            xml_result, error = self.generate_xml(prompt, max_tokens, temperature, custom_grammar=custom_grammar)

            if xml_result:
                xml_result = extract_xml_from_response(xml_result)

                filter_obj = create_default_filter()
                filtered_xml, was_modified, filter_reason = filter_obj.apply_filters(xml_result)

                if was_modified:
                    logger.info(f"Post-processing applied: {filter_reason}")
                    xml_result = filtered_xml

                is_valid, val_error = validate_bt_xml(xml_result, strict=False)

                if not is_valid:
                    logger.warning(f"Validation failed: {val_error}")
                    gen_time_ms = int((time.time() - start_time) * 1000)
                    return {
                        "bt_xml": xml_result,
                        "generation_time_ms": gen_time_ms,
                        "method_used": "cfg",
                        "success": False,
                        "error": f"Validation failed: {val_error}"
                    }

                action_valid, action_issues = validate_action_space(xml_result)
                if not action_valid:
                    logger.warning(f"Invalid actions: {'; '.join(action_issues)}")
                    gen_time_ms = int((time.time() - start_time) * 1000)
                    return {
                        "bt_xml": xml_result,
                        "generation_time_ms": gen_time_ms,
                        "method_used": "cfg",
                        "success": False,
                        "error": f"Invalid actions: {'; '.join(action_issues)}"
                    }

                semantic_valid, semantic_warnings = validate_semantic_structure(xml_result)

                gen_time_ms = int((time.time() - start_time) * 1000)
                logger.info(f"✓ BT generated successfully in {gen_time_ms}ms")

                all_warnings = []
                if was_modified:
                    all_warnings.append(filter_reason)
                if semantic_warnings:
                    all_warnings.extend(semantic_warnings)

                warning_msg = "; ".join(all_warnings) if all_warnings else None

                return {
                    "bt_xml": xml_result,
                    "generation_time_ms": gen_time_ms,
                    "method_used": "cfg",
                    "success": True,
                    "error": None,
                    "warning": warning_msg
                }
            else:
                logger.warning(f"XML generation failed: {error}")
                gen_time_ms = int((time.time() - start_time) * 1000)
                return {
                    "bt_xml": None,
                    "generation_time_ms": gen_time_ms,
                    "method_used": "cfg",
                    "success": False,
                    "error": error
                }

        gen_time_ms = int((time.time() - start_time) * 1000)
        return {
            "bt_xml": None,
            "generation_time_ms": gen_time_ms,
            "method_used": None,
            "success": False,
            "error": "All generation methods failed"
        }


# Global generator instance
_generator: Optional[BTGenerator] = None


def get_generator() -> BTGenerator:
    """Get or create the global generator instance"""
    global _generator
    if _generator is None:
        from core.config import get_model_path, get_adapter_path
        _generator = BTGenerator(get_model_path(), get_adapter_path())
    return _generator


def initialize_generator():
    """Initialize and load the generator"""
    generator = get_generator()
    if not generator.model_loaded:
        generator.load_model()
    return generator
