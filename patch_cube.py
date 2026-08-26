f = "hora/tasks/allegro_hand_hora.py"
s = open(f).read()
anchor = "            'simple_tennis_ball': 'assets/ball.urdf',\n"
assert s.count(anchor) == 1, ("anchor count", s.count(anchor))
if "'cube'" not in s:
    import shutil; shutil.copy(f, f + ".bak_cube")
    s = s.replace(anchor, anchor + "            'cube': 'assets/cube.urdf',\n")
    open(f, "w").write(s)
    print("PATCHED cube added")
else:
    print("already has cube")
