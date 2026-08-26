import shutil
f = "hora/tasks/allegro_hand_hora.py"
s = open(f).read()
reps = [
 ("primitive_list = self.object_type.split('+')\n",
  "primitive_list = self.object_type.split('+')\n        _aroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')\n"),
 ("cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))",
  "cuboids = sorted(glob(f'{_aroot}/assets/cuboid/{subset_name}/*.urdf'))"),
 ("self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')",
  "self.asset_files_dict[f'cuboid_{i}'] = os.path.relpath(name, _aroot)"),
 ("cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))",
  "cylinders = sorted(glob(f'{_aroot}/assets/cylinder/{subset_name}/*.urdf'))"),
 ("self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')",
  "self.asset_files_dict[f'cylinder_{i}'] = os.path.relpath(name, _aroot)"),
]
if "_aroot = os.path.join" in s:
    print("already patched"); raise SystemExit
for a, b in reps:
    assert s.count(a) == 1, ("bad anchor count", s.count(a), a[:40])
shutil.copy(f, f + ".bak_glob")
for a, b in reps:
    s = s.replace(a, b)
open(f, "w").write(s)
print("PATCHED glob branches (CWD-safe absolute paths)")
