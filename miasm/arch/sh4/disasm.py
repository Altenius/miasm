from future.utils import viewvalues

from miasm.core.asmblock import AsmConstraint, disasmEngine
from miasm.arch.sh4.arch import mn_sh4

class dis_sh4b(disasmEngine):
    attrib = 'b'
    def __init__(self, bs=None, **kwargs):
        super(dis_sh4b, self).__init__(mn_sh4, self.attrib, bs, **kwargs)