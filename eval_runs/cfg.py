"""Toggle Kalypso ablation switches for the budget-fix eval runs.

Usage: python3 cfg.py <command> [args]
  virtual on|off          virtual_pinning default in query_processor.py
  priority on|off         off = send priority=-1 from both request builders
  plan reset              restore plan.py (fixed-ratio blocks commented out)
  plan2 F1 F2             activate the fixed 2-stage split
  plan3 F1 F2 F3          activate the fixed 3-stage split
"""
import subprocess
import sys

ROOT = "/home/hojaeson_umass/kalypso/"
PKG = ROOT + "vllm/kalypso/"
BASE_COMMIT = "644260269"  # budget fix; plan.py has both fixed-ratio blocks commented out


def write_if_changed(path, new):
    with open(path) as f:
        old = f.read()
    if old != new:
        with open(path, "w") as f:
            f.write(new)


def virtual(on):
    path = PKG + "query_processor.py"
    s = open(path).read()
    want = f"virtual_pinning: bool = {on},"
    other = f"virtual_pinning: bool = {not on},"
    if want not in s:
        assert s.count(other) == 1, "virtual_pinning default not found"
        s = s.replace(other, want)
    write_if_changed(path, s)


def priority(on):
    path = PKG + "execution/vllm_executor.py"
    s = open(path).read()
    real = "priority=priority,\n            vllm_xargs"
    flat = "priority=-1,\n            vllm_xargs"
    s = s.replace(flat, real) if on else s.replace(real, flat)
    assert s.count(real if on else flat) == 2, "expected both request builders"
    write_if_changed(path, s)


def plan_base():
    return subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{BASE_COMMIT}:vllm/kalypso/controller/plan.py"], text=True
    )


def plan_reset():
    write_if_changed(PKG + "controller/plan.py", plan_base())


def _active(n, fractions):
    return (
        f"            elif num_stages == {n}:\n"
        f"                fixed_fractions = ({', '.join(str(x) for x in fractions)})\n"
        "                for sid, fraction in zip(stage_ids, fixed_fractions):\n"
        "                    kv.register_stage(\n"
        "                        sid,\n"
        "                        fraction,\n"
        "                        min_fraction=fraction,\n"
        "                        max_fraction=fraction,\n"
        "                    )\n"
    )


def plan_fixed(n, fractions):
    assert abs(sum(fractions) - 1.0) < 1e-6 and len(fractions) == n
    s = plan_base()
    start3 = s.index("            # elif num_stages == 3:\n")
    start2 = s.index("            # elif num_stages == 2:\n")
    end2 = s.index("            elif num_stages > 1:\n")
    if n == 3:
        s = s[:start3] + _active(3, fractions) + s[start2:]
    else:
        s = s[:start2] + _active(2, fractions) + s[end2:]
    write_if_changed(PKG + "controller/plan.py", s)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "virtual":
        virtual(args[0] == "on")
    elif cmd == "priority":
        priority(args[0] == "on")
    elif cmd == "plan" and args[0] == "reset":
        plan_reset()
    elif cmd == "plan2":
        plan_fixed(2, [float(x) for x in args])
    elif cmd == "plan3":
        plan_fixed(3, [float(x) for x in args])
    else:
        sys.exit(__doc__)
