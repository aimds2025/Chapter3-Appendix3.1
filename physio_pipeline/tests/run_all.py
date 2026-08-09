"""Run every test suite in one shot (no pytest needed): python tests/run_all.py"""
import runpy, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# ensure synthetic data exists for the data-dependent suites
if not (ROOT / "data" / "synthetic" / "manifest.csv").exists():
    runpy.run_path(str(ROOT / "data" / "generate_synthetic.py"), run_name="__main__")
suites = ["test_smoke", "test_synthetic_cohorts", "test_streaming", "test_mcp_platform"]
fails = 0
for s in suites:
    print(f"\n===== {s} =====")
    try:
        runpy.run_path(str(ROOT / "tests" / f"{s}.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        fails += 1; print(f"[SUITE ERROR] {s}: {e}")
print(f"\n{'ALL SUITES OK' if not fails else str(fails)+' suite(s) errored'}")
