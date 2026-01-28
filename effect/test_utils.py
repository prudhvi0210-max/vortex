import os
import importlib.util

utils_dir = os.path.dirname(__file__)
utils_path = os.path.join(utils_dir, "utils.py")

print("Looking for utils at:", utils_path)
print("Exists?", os.path.exists(utils_path))

spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

print("run_estimator exists:", hasattr(utils, "run_estimator"))
