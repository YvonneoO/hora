import shutil
f = "hora/tasks/allegro_hand_grasp.py"
s = open(f).read()
old = ("            np.save(name, self.saved_grasping_states[:_tgt].cpu().numpy())\n"
       "            print('SAVED', name, 'n=', len(self.saved_grasping_states[:_tgt]))\n"
       "            exit()")
new = ("            _arr = self.saved_grasping_states[:_tgt].cpu().numpy()\n"
       "            import os as _os\n"
       "            np.save(name + '.tmp', _arr)\n"
       "            _os.replace(name + '.tmp.npy', name)\n"
       "            print('SAVED', name, 'n=', len(_arr), flush=True)\n"
       "            exit()")
if "'.tmp'" in s:
    print("already atomic"); raise SystemExit
assert s.count(old) == 1, ("anchor count", s.count(old))
shutil.copy(f, f + ".bak_atomic")
s = s.replace(old, new)
open(f, "w").write(s)
print("PATCHED atomic save")
