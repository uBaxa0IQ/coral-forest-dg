import json
from nir_dg.reporting import build_reports
if __name__ == "__main__":
    print(json.dumps(build_reports(), indent=2))
