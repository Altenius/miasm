#-*- coding:utf-8 -*-

from miasm.expression.expression import ExprAssign, ExprOp
from miasm.ir.ir import IRBlock, AssignBlock
from miasm.ir.analysis import ira
from miasm.arch.sh4.sem import ir_sh4b

class ir_a_sh4(ir_sh4b, ira):
    def __init__(self, loc_db=None):
        ir_sh4b.__init__(self, loc_db)
        self.ret_reg = self.arch.regs.R0
    
    """
    def call_effects(self, ad, instr):
        call_assignblk = AssignBlock(
            [
                ExprAssign(
                    self.ret_reg,
                    ExprOp(
                        'call_func_ret',
                        ad,
                        self.arch.regs.R0,
                        self.arch.regs.R1,
                        self.arch.regs.R2,
                        self.arch.regs.R3,
                    )
                ),
            ],
            instr
        )

        return [call_assignblk], []
    """

    def get_out_regs(self, _):
        return set([self.ret_reg])

    def sizeof_char(self):
        return 8

    def sizeof_short(self):
        return 16

    def sizeof_int(self):
        return 32

    def sizeof_long(self):
        return 32

    def sizeof_pointer(self):
        return 32