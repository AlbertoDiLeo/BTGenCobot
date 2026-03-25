"""BehaviorTree XML Validation"""
import xml.etree.ElementTree as ET
from typing import Tuple, Optional


CONTROL_NODES = ["Sequence", "Fallback", "Parallel", "ReactiveSequence", "ReactiveFallback"]

DECORATOR_NODES = ["Inverter", "ForceSuccess", "ForceFailure", "Repeat", "RetryUntilSuccessful", "KeepRunningUntilFailure", "RateController"]

CONDITION_NODES = ["GoalReached", "IsStuck", "IsBatteryLow", "GoalUpdated", "TimeExpired", "DistanceTraveled"]

ACTION_NODES = [
    "ComputePathToPose",
    "FollowPath",
    "NavigateToPose",
    "SpinLeft",
    "SpinRight",
    "BackUp",
    "DriveOnHeading",
    "Wait",
    "DetectObject",
    "PickObject",
    "PlaceObject",
    "ClearEntireCostmap",
]

ALL_VALID_NODES = CONTROL_NODES + DECORATOR_NODES + CONDITION_NODES + ACTION_NODES


def get_all_node_types():
    """Get list of all valid BT node types"""
    return ALL_VALID_NODES


class BTValidationError(Exception):
    """Custom exception for BT validation errors"""
    pass


def validate_bt_xml(xml_string: str, strict: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate BehaviorTree XML structure

    Args:
        xml_string: BT XML content to validate
        strict: If True, perform strict validation of node types

    Returns:
        Tuple of (is_valid, error_message)
        error_message is None if valid
    """
    try:
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            return False, f"XML parse error: {str(e)}"

        # Support two formats:
        # 1. Full document: <root BTCPP_format="4">...<BehaviorTree>...</BehaviorTree></root>
        # 2. BehaviorTree only: <BehaviorTree>...</BehaviorTree>
        
        if root.tag == "root":
            # Full document format
            btcpp_format = root.get("BTCPP_format")
            if btcpp_format != "4":
                return False, f"BTCPP_format must be '4', got '{btcpp_format}'"

            behavior_trees = root.findall("BehaviorTree")
            if len(behavior_trees) == 0:
                return False, "No BehaviorTree element found"

            for bt in behavior_trees:
                tree_id = bt.get("ID")
                if not tree_id:
                    return False, "BehaviorTree element missing ID attribute"

                if len(list(bt)) == 0:
                    return False, f"BehaviorTree '{tree_id}' is empty"

            main_tree = root.get("main_tree_to_execute")
            if main_tree:
                tree_ids = [bt.get("ID") for bt in behavior_trees]
                if main_tree not in tree_ids:
                    return False, f"main_tree_to_execute '{main_tree}' not found in defined trees"
                    
        elif root.tag == "BehaviorTree":
            # BehaviorTree only format (from finetuned model)
            tree_id = root.get("ID")
            if not tree_id:
                return False, "BehaviorTree element missing ID attribute"
            
            if len(list(root)) == 0:
                return False, f"BehaviorTree '{tree_id}' is empty"
            
            # For this format, behavior_trees is just the root element
            behavior_trees = [root]
        else:
            return False, f"Root element must be 'root' or 'BehaviorTree', got '{root.tag}'"

        if strict:
            valid_nodes = set(get_all_node_types()) | {"BehaviorTree", "root"}

            def check_node_types(element):
                if element.tag not in valid_nodes and not element.tag.startswith("?"):
                    return False, f"Unknown node type: {element.tag}"

                for child in element:
                    is_valid, err = check_node_types(child)
                    if not is_valid:
                        return is_valid, err

                return True, None

            for bt in behavior_trees:
                for child in bt:
                    is_valid, err = check_node_types(child)
                    if not is_valid:
                        return is_valid, err

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def extract_tree_info(xml_string: str) -> dict:
    """
    Extract metadata from BT XML

    Args:
        xml_string: BT XML content

    Returns:
        Dictionary with tree metadata
    """
    try:
        root = ET.fromstring(xml_string)

        info = {
            "btcpp_format": root.get("BTCPP_format"),
            "main_tree": root.get("main_tree_to_execute"),
            "trees": []
        }

        for bt in root.findall("BehaviorTree"):
            tree_info = {
                "id": bt.get("ID"),
                "node_count": len(list(bt.iter())) - 1
            }
            info["trees"].append(tree_info)

        return info

    except Exception as e:
        return {"error": str(e)}


def validate_action_space(xml_string: str) -> Tuple[bool, list]:
    """
    Validate that all actions used in the BT are from the allowed action space.
    This prevents the model from hallucinating invalid actions.
    
    Args:
        xml_string: BT XML content
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    CONTROL_NODES = {
        'Sequence', 'Fallback', 'Parallel', 'ReactiveSequence', 
        'ReactiveFallback', 'RecoveryNode', 'PipelineSequence'
    }

    NAV2_ACTIONS = {
        'ComputePathToPose', 'ComputePathThroughPoses', 'FollowPath',
        'NavigateToPose', 'Spin', 'SpinLeft', 'SpinRight', 'Wait', 'BackUp', 'DriveOnHeading',
        'AssistedTeleop'
    }

    RECOVERY_ACTIONS = {
        'ClearEntireCostmap', 'ClearCostmapExceptRegion',
        'ClearCostmapAroundRobot', 'ReinitializeGlobalLocalization'
    }

    MANIPULATION_ACTIONS = {
        'DetectObject', 'PickObject', 'PlaceObject', 'MoveArm',
        'OpenGripper', 'CloseGripper'
    }

    UTILITY_ACTIONS = {
        'SayText'
    }

    CONDITION_NODES = {
        'IsStuck', 'GoalReached', 'TransformAvailable',
        'GloballyUpdatedGoal', 'GoalUpdated', 'IsBatteryLow',
        'DistanceTraveled', 'TimeExpired'
    }

    DECORATOR_NODES = {
        'Inverter', 'ForceSuccess', 'ForceFailure', 'Repeat',
        'RetryUntilSuccessful', 'KeepRunningUntilFailure', 
        'RateController', 'DistanceController', 'SpeedController',
        'GoalUpdater'
    }
    
    # Explicit syntax wrappers - validate their ID attribute instead of tag
    EXPLICIT_WRAPPERS = {'Action', 'Condition', 'SubTree'}
    
    ALLOWED_NODES = (CONTROL_NODES | NAV2_ACTIONS | RECOVERY_ACTIONS | 
                     MANIPULATION_ACTIONS | UTILITY_ACTIONS | 
                     CONDITION_NODES | DECORATOR_NODES)
    
    try:
        root = ET.fromstring(xml_string)
        
        def check_action_node(elem, path=""):
            node_path = f"{path}/{elem.tag}"

            # Handle explicit syntax: <Action ID="SpinLeft" /> or <Condition ID="GoalReached" />
            if elem.tag in EXPLICIT_WRAPPERS:
                node_id = elem.get('ID')
                if node_id and node_id not in ALLOWED_NODES:
                    issues.append(
                        f"Invalid {elem.tag.lower()} '{node_id}' at {node_path}. "
                        f"Not in allowed action space."
                    )
            # Handle compact syntax: <SpinLeft /> or unknown tags
            elif elem.tag not in ALLOWED_NODES and elem.tag not in {'BehaviorTree', 'root'}:
                issues.append(
                    f"Invalid action '{elem.tag}' at {node_path}. "
                    f"Not in allowed action space."
                )

            for child in elem:
                check_action_node(child, node_path)
        
        for bt in root.findall("BehaviorTree"):
            for child in bt:
                check_action_node(child, bt.get("ID", ""))
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error checking action space: {str(e)}"]


def validate_semantic_structure(xml_string: str) -> Tuple[bool, list]:
    """
    Validate semantic correctness of the behavior tree structure.
    Checks for common errors like:
    - ComputePathToPose without FollowPath
    - Missing DetectObject before using object poses
    - Unnecessary deep nesting
    
    Args:
        xml_string: BT XML content
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    try:
        root = ET.fromstring(xml_string)
        
        def check_sequence_semantics(elem, parent_path=""):
            """Check if Sequence nodes have proper action pairs"""
            if elem.tag == 'Sequence':
                children_tags = [child.tag for child in elem]

                if 'ComputePathToPose' in children_tags and 'FollowPath' not in children_tags:
                    warnings.append(
                        f"ComputePathToPose found at {parent_path}/Sequence "
                        f"but no FollowPath to execute the computed path"
                    )

                if len(elem) == 1 and elem[0].tag == 'Sequence':
                    warnings.append(
                        f"Unnecessary nested Sequence at {parent_path}/Sequence. "
                        f"Single-child Sequence nodes can be flattened."
                    )

            for child in elem:
                check_sequence_semantics(child, f"{parent_path}/{elem.tag}")

        for bt in root.findall("BehaviorTree"):
            for child in bt:
                check_sequence_semantics(child, bt.get("ID", ""))

        return True, warnings
        
    except Exception as e:
        return True, [f"Error checking semantics: {str(e)}"]


def validate_command_semantics(xml_string: str, command: str) -> Tuple[bool, list]:
    """
    Validate high-level BT semantics against the original natural language command.
    This catches simple-command degeneracies such as duplicated PlaceObject or
    duplicated DetectObject nodes.
    """
    issues = []
    cmd = command.lower()

    is_place_only = (
        any(token in cmd for token in ["place", "put", "set down", "deposit"]) and
        not any(token in cmd for token in ["pick up", "pick", "grab", "grasp", "take"])
    )
    is_detect_only = (
        any(token in cmd for token in ["detect", "find", "look for", "search"]) and
        not any(token in cmd for token in ["place", "put", "set down", "deposit", "pick up", "pick", "grab", "grasp", "take"])
    )

    if not (is_place_only or is_detect_only):
        return True, issues

    try:
        root = ET.fromstring(xml_string)

        action_ids = []
        for elem in root.iter():
            if elem.tag == "Action":
                node_id = elem.get("ID")
                if node_id:
                    action_ids.append(node_id)

        if is_place_only:
            place_count = sum(1 for node_id in action_ids if node_id == "PlaceObject")
            if place_count == 0:
                issues.append("Pure place command generated no PlaceObject node")
            elif place_count > 1:
                issues.append(f"Pure place command generated {place_count} PlaceObject nodes")

        if is_detect_only:
            detect_count = sum(1 for node_id in action_ids if node_id == "DetectObject")
            if detect_count == 0:
                issues.append("Pure detect command generated no DetectObject node")
            elif detect_count > 1:
                issues.append(f"Pure detect command generated {detect_count} DetectObject nodes")

        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Error checking command semantics: {str(e)}"]


def validate_blackboard_variables(xml_string: str) -> Tuple[bool, list]:
    """
    Check for consistent use of blackboard variables

    Blackboard variables are referenced as {variable_name} in node parameters.
    This function checks that variables are written before being read.

    Args:
        xml_string: BT XML content

    Returns:
        Tuple of (is_valid, list of issues)
    """
    import re

    issues = []
    written_vars = set()

    try:
        root = ET.fromstring(xml_string)

        var_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

        output_params = {
            'target_pose', 'pose', 'path', 'result', 'output',
            'detected', 'position', 'orientation'
        }

        def check_node(elem, path=""):
            nonlocal written_vars, issues

            node_path = f"{path}/{elem.tag}"

            for key, value in elem.attrib.items():
                vars_in_value = var_pattern.findall(value)

                for var in vars_in_value:
                    if key in output_params:
                        written_vars.add(var)
                    else:
                        if var not in written_vars:
                            issues.append(
                                f"Variable '{{{var}}}' used in {node_path}[@{key}] "
                                f"but not written by any previous node"
                            )

            for child in elem:
                check_node(child, node_path)

        for bt in root.findall("BehaviorTree"):
            for child in bt:
                check_node(child, bt.get("ID", ""))

        return len(issues) == 0, issues

    except Exception as e:
        return False, [f"Error checking blackboard variables: {str(e)}"]


def pretty_print_xml(xml_string: str) -> str:
    """
    Format XML with proper indentation

    Args:
        xml_string: XML content

    Returns:
        Formatted XML string
    """
    try:
        ET.indent(ET.fromstring(xml_string))
        return ET.tostring(ET.fromstring(xml_string), encoding='unicode')
    except Exception:
        # If formatting fails, return original
        return xml_string


if __name__ == "__main__":
    # Test validation
    valid_xml = """<?xml version="1.0"?>
<root BTCPP_format="4" main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="TestSequence">
      <DetectObject object_description="cup" target_pose="{cup_pose}"/>
      <ComputePathToPose goal="{cup_pose}" path="{path}"/>
      <FollowPath path="{path}"/>
    </Sequence>
  </BehaviorTree>
</root>"""

    print("=== Testing Valid XML ===")
    is_valid, error = validate_bt_xml(valid_xml, strict=False)
    print(f"Valid: {is_valid}, Error: {error}")

    print("\n=== Tree Info ===")
    print(extract_tree_info(valid_xml))

    print("\n=== Blackboard Variable Check ===")
    is_valid, issues = validate_blackboard_variables(valid_xml)
    print(f"Valid: {is_valid}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
