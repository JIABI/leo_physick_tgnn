from leo_pg.kernels.physick import KernelBank, PhysiCKMessage
from leo_pg.physics import KernelBank as CompatibilityKernelBank
from leo_pg.physics import PhysiCKMessage as CompatibilityPhysiCKMessage


def test_legacy_physics_imports_resolve_to_active_physick():
    assert CompatibilityPhysiCKMessage is PhysiCKMessage
    assert CompatibilityKernelBank is KernelBank
