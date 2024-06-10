import json
import os
import sys


class DummyResultTracker:
    def start_code_block(self, syntax=""):
        pass

    def end_code_block(self):
        pass

    def write_command_line(self):
        pass

    def write_command_line_section(self):
        pass

    def write_section(self, title, level=0):
        pass

    def write_line(self, line=""):
        pass

    def write_result(self, result):
        pass

    def write_exception(self, lines):
        pass

    def finish(self):
        pass


class ResultTracker:
    def __init__(self, is_train, result_prefix, include_motion, task_name, run_name):
        self.is_train = is_train
        self.include_motion = include_motion
        self.task_name = task_name
        self.run_name = run_name

        # Init output dir and result file
        results_folder = "results"
        if result_prefix is not None:
            results_folder = os.path.join(results_folder, result_prefix)
        mode_folder = "train" if is_train else "predict"
        motion_folder = "motion" if include_motion else "no_motion"
        self.output_dir = os.path.join(results_folder, mode_folder, task_name, motion_folder)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        output_md_file = os.path.join(self.output_dir, f"{run_name}.md")
        self.tracker_file = open(output_md_file, "w")

    def start_code_block(self, syntax=""):
        self.tracker_file.write(f"```{syntax}\n")

    def end_code_block(self):
        self.tracker_file.write("```\n\n")

    def write_command_line(self):
        self.tracker_file.write("python3")

        for arg in sys.argv:
            self.tracker_file.write(" ")
            if arg.startswith("--"):
                self.tracker_file.write("\\\n    ")
            self.tracker_file.write(arg)
        self.tracker_file.write("\n")

    def write_command_line_section(self):
        self.write_section("Command")
        self.start_code_block("bash")
        self.write_command_line()
        self.end_code_block()

    def write_section(self, title, level=0):
        section_prefix = "#" * (level + 1)
        self.tracker_file.write(f"{section_prefix} {title}\n")

    def write_line(self, line="", ending="\n"):
        self.tracker_file.write(f"{line}{ending}")

    def write_result(self, result):
        self.write_section("Results")
        self.start_code_block("json")
        result_dump = json.dumps(result, indent=4)
        self.tracker_file.write(result_dump)
        self.tracker_file.write("\n")
        self.end_code_block()

        output_json_file = os.path.join(self.output_dir, f"{self.run_name}.json")
        with open(output_json_file, "w") as result_file:
            result_file.write(result_dump)

    def write_exception(self, lines):
        self.write_section("Exception")
        self.start_code_block()
        for line in lines:
            self.write_line(line, ending="")
        self.write_line()
        self.end_code_block()

    def finish(self):
        if not self.tracker_file.closed:
            self.tracker_file.close()
