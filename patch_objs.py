f = "hora/tasks/allegro_hand_hora.py"
s = open(f).read()
anchor = "            'cube': 'assets/cube.urdf',\n"
assert s.count(anchor) == 1, ("anchor", s.count(anchor))
add = ""
for k, v in [("cyl", "assets/cylinder.urdf"), ("box0", "assets/cuboid/default/0000.urdf")]:
    if f"'{k}'" not in s:
        add += f"            '{k}': '{v}',\n"
if add:
    s = s.replace(anchor, anchor + add)
    open(f, "w").write(s)
    print("PATCHED:", add.strip())
else:
    print("already present")
