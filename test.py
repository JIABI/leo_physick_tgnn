import leo_pg.kernels.physick.coeff_head as m
import inspect
print("coeff_head loaded from:", m.__file__)
print("--- forward() head ---")
src = inspect.getsource(m.CoeffHead.forward).splitlines()
for i,l in enumerate(src[:40], 1):
    print(f"{i:02d} {l}")


