import os
import subprocess
import sys
import argparse
import re

def get_task_scripts(scripts_dir):
    """List and sort all Task_*.py files in the Scripts directory."""
    files = [f for f in os.listdir(scripts_dir) if f.startswith('Task_') and f.endswith('.py')]
    # Numerical sort based on the first number found after 'Task_'
    def get_number(filename):
        match = re.search(r'Task_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    
    return sorted(files, key=get_number)

def run_script(script_path, root_dir):
    """Run a Python script using subprocess with PYTHONPATH set to the project root."""
    print(f"\n{'='*20}")
    print(f"Executing: {os.path.basename(script_path)}")
    print(f"{'='*20}\n")
    
    # Update environment with PYTHONPATH to include the project root and its parent
    # This supports both 'import utilities' and 'import ML4HC_Project1.utilities'
    env = os.environ.copy()
    parent_dir = os.path.dirname(root_dir)
    additional_paths = [root_dir, parent_dir]
    
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(additional_paths + current_pythonpath.split(os.pathsep))
    else:
        env["PYTHONPATH"] = os.pathsep.join(additional_paths)

    try:
        # We use sys.executable to ensure we use the same Python environment as main.py
        subprocess.run([sys.executable, script_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error: Script {os.path.basename(script_path)} failed with exit code {e.returncode}")
    except Exception as e:
        print(f"\n[!] Unexpected error while running {os.path.basename(script_path)}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Task scripts from the Scripts directory.")
    parser.add_argument('--Scripts', type=str, help="Specify a precise suffix (e.g. '1', '2_3a') or a range (e.g. '2-4' or '2\u20134').")
    args = parser.parse_args()

    # Scripts directory is relative to this main.py file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, 'Scripts')

    if not os.path.isdir(scripts_dir):
        print(f"Error: 'Scripts' directory not found at {scripts_dir}")
        sys.exit(1)

    all_scripts = get_task_scripts(scripts_dir)
    scripts_to_run = []

    if args.Scripts:
        # Check for range: '2-4' or '2-4' (supporting both hyphen and en-dash)
        # Using unicode for en-dash to be safe: \u2013
        range_match = re.match(r'^(\d+)[\-\u2013](\d+)$', args.Scripts)
        
        if range_match:
            start, end = map(int, range_match.groups())
            # Ensure start <= end for a valid range
            if start > end:
                start, end = end, start
                
            for script in all_scripts:
                num_match = re.search(r'Task_(\d+)', script)
                if num_match:
                    num = int(num_match.group(1))
                    if start <= num <= end:
                        scripts_to_run.append(script)
        else:
            # Precise file case: e.g. '1' -> 'Task_1.py', '2_3a' -> 'Task_2_3a.py'
            precise_file = f"Task_{args.Scripts}.py"
            if precise_file in all_scripts:
                scripts_to_run.append(precise_file)
            else:
                # Fallback check: maybe the user didn't include 'Task_' in their expectation but it's there
                print(f"Error: Could not find script '{precise_file}' in Scripts/ directory.")
                print(f"Available scripts: {', '.join(all_scripts)}")
                sys.exit(1)
    else:
        # No Scripts argument provided: run all
        scripts_to_run = all_scripts

    if not scripts_to_run:
        print("No scripts found matching the criteria.")
        return

    # Execute selected scripts
    for script_name in scripts_to_run:
        script_full_path = os.path.join(scripts_dir, script_name)
        run_script(script_full_path, base_dir)

if __name__ == "__main__":
    main()
