import miasm.expression.expression as m2_expr
from miasm.expression.expression import *
from miasm.ir.ir import IntermediateRepresentation, IRBlock, AssignBlock
from miasm.arch.sh4.arch import mn_sh4
from miasm.arch.sh4.regs import PC, R0, SR, tf, qf, mf, sf, MACH, MACL, GBR, PR, SSR, SPC, VBR, TRA, FPUL
from miasm.core.sembuilder import SemBuilder
from miasm.jitter.csts import EXCEPT_DIV_BY_ZERO


def update_flag_subwc_cf(arg1, arg2, arg3):
    return [ExprAssign(tf, ExprOp("FLAG_SUBWC_CF", arg1, arg2, arg3))]

def update_flag_sub_of(arg1, arg2):
    return [ExprAssign(tf, ExprOp("FLAG_SUB_OF", arg1, arg2))]

def update_flag_add_cf(arg1, arg2, arg3):
    """
    Compute overflow flag for (arg1 + arg2 + tf) and sets T
    """
    return [ExprAssign(tf, ExprOp("FLAG_EQ_ADDWC", arg1, arg2, arg3))]

def update_flag_add_of(op1, op2):
    return [ExprAssign(tf, ExprOp("FLAG_ADD_OF", op1, op2))]

def update_flag_eq(op1, op2):
    return [ExprAssign(tf, ExprOp("FLAG_EQ_CMP", op1, op2))]

# SemBuilder context
ctx = {
    "PC": PC,
    "PR": PR,
    "R0": R0,
    "SR": SR,
    "sf": sf,
    "tf": tf,
    "qf": qf,
    "mf": mf,
    "MACH": MACH,
    "MACL": MACL,
    "SSR": SSR,
    "SPC": SPC,
    "VBR": VBR,
    "FPUL": FPUL,
}

sbuild = SemBuilder(ctx)

@sbuild.parse
def mov(m, n):
     n = m.signExtend(n.size)

@sbuild.parse
def mov_b(m, n):
    n = m[:8].signExtend(n.size)

@sbuild.parse
def mov_w(m, n):
    n = m[:16].signExtend(n.size)

@sbuild.parse
def mov_l(m, n):
    n = m.signExtend(n.size)

@sbuild.parse
def movt(n):
    n = tf.zeroExtend(n.size)

@sbuild.parse
def swapb(m, n):
    temp0 = m & 0xFFFF0000
    temp1 = (m & 0x000000FF) << 8
    n = (m & 0x0000FF00) >> 8
    n = n | temp1 | temp0

@sbuild.parse
def swapw(m, n):
    temp1 = (m & 0x0000FFFF) << 16
    n = (m & 0xFFFF0000) >> 16
    n = n | temp1

@sbuild.parse
def xtrct(m, n):
    high = (m << 16) & 0xFFFF0000
    low = (n >> 16) & 0x0000FFFF
    n = high | low

@sbuild.parse
def add(m, n):
    n = n + m

def addc(ir, instr, m, n):
    e = []
    r = n + (m + tf.zeroExtend(m.size))
    e.append(ExprAssign(n, r))
    e += update_flag_add_cf(m, n, tf)
    
    return e, []

def addv(ir, instr, m, n):
    e = []
    r = n + m
    e.append(ExprAssign(n, r))
    e += update_flag_add_of(m, n)
    
    return e, []

def cmpeq(ir, instr, m, n):
    return update_flag_eq(m, n), []

def cmphs(ir, instr, m, n):
    return [ExprAssign(tf, ExprOp("<=u", m, n))], []

def cmpge(ir, instr, m, n):
    return [ExprAssign(tf, ExprOp("<=s", m, n))], []

def cmphi(ir, instr, m, n):
    return [ExprAssign(tf, ExprOp("<u", m, n))], []

def cmpgt(ir, instr, m, n):
    return [ExprAssign(tf, ExprOp("<s", m, n))], []

def cmppl(ir, instr, n):
    return [ExprAssign(tf, ExprOp("<s", tf.zeroExtend(n.size), n))], []

def cmppz(ir, instr, n):
    return [ExprAssign(tf, ExprOp("<=s", tf.zeroExtend(n.size), n))], []

@sbuild.parse
def cmpstr(m, n):
    temp = m ^ n
    hh = temp & i32(0xFF)
    hl = (temp & i32(0xFF00)) >> i32(8)
    lh = (temp & i32(0xFF0000)) >> i32(16)
    ll = (temp & i32(0xFF000000)) >> i32(24)
    tf = '=='(hh & hl & lh & ll, i32(0))

@sbuild.parse
def div0s(m, n):
    qf = 0 if (n & 0x80000000) else 1
    mf = 0 if (m & 0x80000000) else 1
    tf = 0 if '=='(qf, mf) else 1

@sbuild.parse
def div0u(m, n):
    qf = 0
    mf = 0
    tf = 0


def div1(ir, instr, m, n):
    e = []
    e.append(ExprAssign(n, ExprOp("div1", n, m)))
    return e, []

def dmuls(ir, instr, m, n):
    e = []
    res = ExprOp("*", n.signExtend(64), m.signExtend(64))
    e.append(ExprAssign(MACL, res[:32]))
    e.append(ExprAssign(MACH, res[32:64]))

    return e, []

def dmulu(ir, instr, m, n):
    e = []
    res = ExprOp("*", n.zeroExtend(64), m.zeroExtend(64))
    e.append(ExprAssign(MACL, res[:32]))
    e.append(ExprAssign(MACH, res[32:64]))

    return e, []

def dt(ir, instr, n):
    e = []
    res = n - ExprInt(1, 32)
    e.append(ExprAssign(n, res))
    e.append(ExprAssign(tf, ExprOp("FLAG_EQ", res, ExprInt(0, 32))))

    return e, []

def extsb(ir, instr, m, n):
    e = [ExprAssign(n, m[:8].signExtend(32))]
    return e, []

def extsw(ir, instr, m, n):
    e = [ExprAssign(n, m[:16].signExtend(32))]
    return e, []

def extub(ir, instr, m, n):
    e = [ExprAssign(n, m[:8].zeroExtend(32))]
    return e, []

def extuw(ir, instr, m, n):
    e = [ExprAssign(n, m[:16].zeroExtend(32))]
    return e, []

def mac_l(ir, instr, src, dst):
    """
    Untested
    """
    e = []
    res = src.signExtend(64) * dst.signExtend(64)
    tempml = res[:32].zeroExtend(64) + MACL.zeroExtend(64)
    tempmh = res[32:64] + MACH + tempml[32:64]
    e.append(ExprAssign(MACL, tempml[:32]))
    e.append(ExprAssign(MACH, tempmh))
    return e, []

def mac_w(ir, instr, src, dst):
    """
    Untested
    """
    e = []
    res = src[:16].signExtend(64) * dst[:16].signExtend(64)
    tempml = res[:32].zeroExtend(64) + MACL.zeroExtend(64)
    tempmh = res[32:64] + MACH + tempml[32:64]
    e.append(ExprAssign(MACL, tempml[:32]))
    e.append(ExprAssign(MACH, tempmh))
    return e, []

def mull(ir, instr, m, n):
    res = m * n
    e = [ExprAssign(MACL, res)]
    return e, []

def mulsw(ir, instr, m, n):
    res = m[:16].signExtend(32) * n[:16].signExtend(32)
    e = [ExprAssign(MACL, res)]
    return e, []

def muluw(ir, instr, m, n):
    res = m[:16].zeroExtend(32) * n[:16].zeroExtend(32)
    e = [ExprAssign(MACL, res)]
    return e, []

def neg(ir, instr, m, n):
    e = [ExprAssign(n, -m)]
    return e, []

def negc(ir, instr, m, n):
    e = []
    temp = -m
    res = temp - tf.zeroExtend(32)
    e.append(ExprAssign(n, res))
    e += update_flag_subwc_cf(ExprInt(0, 32), m, tf.zeroExtend(32))
    return e, []

@sbuild.parse
def sub(m, n):
    n = n - m

def subc(ir, instr, m, n):
    e = []
    e.append(ExprAssign(n, n - m - tf.zeroExtend(32)))
    e += update_flag_subwc_cf(n, m, tf.zeroExtend(32))
    return e, []

def subv(ir, instr, m, n):
    e = []
    e.append(ExprAssign(n, n - m))
    e += update_flag_sub_of(n, m)
    return e, []

@sbuild.parse
def l_and(m, n):
    n = n & m

@sbuild.parse
def and_b(imm, r0gbr):
    """
    Untested
    """
    res = imm & r0gbr[:8]
    r0gbr = res

@sbuild.parse
def l_not(m, n):
    n = ~m

@sbuild.parse
def l_or(m, n):
    n = n | m

@sbuild.parse
def or_b(imm, r0gbr):
    """
    Untested
    """
    r0gbr = imm | r0gbr

@sbuild.parse
def tas_b(n):
    tf = i1(1) if n[:8] == i8(0) else i1(0)
    n = n | 0x80

@sbuild.parse
def tst(m, n):
    tf = '=='('&'(n, m), i32(0))

@sbuild.parse
def tst_b(imm, r0gbr):
    """Untested"""
    tf = '=='(imm & r0gbr, i8(0))

@sbuild.parse
def xor(m, n):
    n = n ^ m

@sbuild.parse
def xor_b(imm, r0gbr):
    """Untested"""
    r0gbr = imm ^ r0gbr

@sbuild.parse
def rotl(n):
    trans = i1(0) if ('=='(n & i32(0x80000000), i32(0))) else i1(1)
    res = n << i32(1)
    n = (res | i32(0x1)) if trans else (res & ~i32(1))
    tf = trans

@sbuild.parse
def rotr(n):
    trans = i1(0) if ('=='(n & i32(0x00000001), i32(0))) else i1(1)
    res = n >> i32(1)
    n = (res | i32(0x80000000)) if trans else (res & ~i32(0x80000000))
    tf = trans

@sbuild.parse
def rotcl(n):
    trans = i1(0) if ('=='(n & i32(0x80000000), i32(0))) else i1(1)
    res = n << i32(1)
    n = (res | i32(0x1)) if tf else (res & ~i32(1))
    tf = trans

@sbuild.parse
def rotcr(n):
    trans = i1(0) if ('=='(n & i32(0x00000001), i32(0))) else i1(1)
    res = n >> i32(1)
    n = (res | i32(0x80000000)) if tf else (res & ~i32(0x80000000))
    tf = trans

@sbuild.parse
def shad(m, n):
    is_pos = '=='(m & i32(0x80000000), i32(0))
    n = (n << (m & i32(0x1F))) if is_pos else 'a>>'(n, -m & i32(0x1F))

@sbuild.parse
def shal(n):
    n = n << i32(1)
    tf = i1(0) if '=='(n & i32(0x80000000), i32(0)) else i1(1)

@sbuild.parse
def shar(n):
    n = (n >> i32(1)) | (n & i32(0x80000000))
    tf = i1(0) if '=='(n & i32(0x00000001), i32(0)) else i1(1)

@sbuild.parse
def shld(m, n):
    is_pos = '=='(m & i32(0x80000000), i32(0))
    n = (n << (m & i32(0x1F))) if is_pos else '>>'(n, -m & i32(0x1F))

@sbuild.parse
def shll(n):
    tf = n[31:32]
    n = n << i32(1)

@sbuild.parse
def shlr(n):
    tf = n[0:1]
    n = n >> i32(1)

@sbuild.parse
def shll2(n):
    n = n << i32(2)

@sbuild.parse
def shlr2(n):
    n = n >> i32(2)

@sbuild.parse
def shll8(n):
    n = n << i32(8)

@sbuild.parse
def shlr8(n):
    n = n >> i32(8)

@sbuild.parse
def shll16(n):
    n = n << i32(16)

@sbuild.parse
def shlr16(n):
    n = n >> i32(16)

@sbuild.parse
def bf(d):
    offset_expr = i32(instr.offset)
    loc_next_expr = ExprLoc(ir.get_next_loc_key(instr), ir.IRDst.size)

    addr = d * ExprInt(2, 32) + ExprInt(4, 32) + PC
    dst = loc_next_expr if tf else d
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def bfs(d):
    loc_next_expr = ExprLoc(ir.get_next_break_loc_key(instr), ir.IRDst.size)

    dst = loc_next_expr if tf else d
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def bt(d):
    loc_next_expr = ExprLoc(ir.get_next_loc_key(instr), ir.IRDst.size)

    dst = d if tf else loc_next_expr
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def bts(d):
    loc_next_expr = ExprLoc(ir.get_next_break_loc_key(instr), ir.IRDst.size)

    dst = d if tf else loc_next_expr
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def bra(d):
    dst = d
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def braf(m):
    dst = m + i32(4) + PC
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def bsr(d):
    loc_next_expr = ExprLoc(ir.get_next_break_loc_key(instr), ir.IRDst.size)
    dst = d
    PR = loc_next_expr
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def bsrf(m):
    loc_next_expr = ExprLoc(ir.get_next_break_loc_key(instr), ir.IRDst.size)
    dst = m + i32(4) + PC
    PR = loc_next_expr
    PC = dst
    ir.IRDst = dst

@sbuild.parse
def jmp_l(m):
    PC = m
    ir.IRDst = m

def jsr_l(ir, instr, m):
    e = []
    loc_next_expr = ExprLoc(ir.get_next_break_loc_key(instr), ir.IRDst.size)
    PR = loc_next_expr
    PC = m.arg
    ir.IRDst = m.arg

@sbuild.parse
def rts():
    PC = PR
    ir.IRDst = PR

@sbuild.parse
def clrmac():
    MACL = i32(0)
    MACH = i32(0)

@sbuild.parse
def clrs():
    sf = i1(0)

@sbuild.parse
def clrt():
    tf = i1(0)

def ldc(ir, instr, m, dst):
    e = []
    if m == SR:
        e.append(ExprAssign(dst, m & ExprInt(0x700083F3, 32)))
    else:
        e.append(ExprAssign(dst, m))
    return e, []

def ldc_l(ir, instr, m, dst):
    e = []
    if m == SR:
        e.append(ExprAssign(dst, m & ExprInt(0x700083F3, 32)))
    else: 
        e.append(ExprAssign(dst, m))
    # Get raw register (this is dirty...)
    reg = instr.args[0]._ptr._args[0]
    # Post increment
    e.append(ExprAssign(reg, reg + ExprInt(4, 32)))

    e.append(ExprAssign(instr.args[0], instr.args[0] + ExprInt(4, 32)))
    return e, []

@sbuild.parse
def lds(m, dst):
    dst = m

@sbuild.parse
def lds_l(m, dst):
    dst = m
    # Get raw register (this is dirty...)
    reg = instr.args[0]._ptr._args[0]
    # Post increment
    reg = reg + ExprInt(4, 32)

def ldtbl(ir, instr):
    raise NotImplementedError("LDTBL is not implemented")

@sbuild.parse
def movca_l(r0, n):
    n = r0

def nop(ir, instr):
    return [], []

def ocbi_l(ir, instr, n):
    # Invalidate operand cache block
    return [], []

def ocbp_l(it, instr, n):
    # Write back and invalidate operand cache block
    return [], []

def ocbwb_l(ir, instr, n):
    # Write back operand cache block
    return [], []

def pref_l(ir, instr, n):
    return [], []

@sbuild.parse
def rte():
    SR = SSR
    PC = SPC

@sbuild.parse
def sets():
    sf = i1(1)

@sbuild.parse
def sett():
    tf = i1(1)

def sleep(ir, instr):
    pass

@sbuild.parse
def stc(src, n):
    n = src

@sbuild.parse
def stc_l(src, n):
    n = src
    # Get raw register (this is dirty...)
    reg = instr.args[1]._ptr._args[1]
    # Post decrement
    reg = reg - ExprInt(4, 32)

@sbuild.parse
def sts(src, n):
    n = src

@sbuild.parse
def sts_l(src, n):
    n = src
    # Get raw register (this is dirty...)
    reg = instr.args[1].ptr.args[-1]
    # Post decrement
    reg = reg - ExprInt(4, 32)

def trapa(ir, instr, imm):
    raise NotImplementedError("TRAPA is incomplete")
"""
    loc_next_expr = ExprLoc(ir.get_next_break_loc_key(instr), ir.IRDst.size)

    SPC = loc_next_expr
    SSR = SR
    TRA = imm * i32(4)
    PC = VBR + i32(0x100)
    # TODO: Set SR and EXPEVT
"""

@sbuild.parse
def fldi0(n):
    n = i32(0)

@sbuild.parse
def fldi1(n):
    n = i32(0x3F800000)

@sbuild.parse
def fmov(m, n):
    n = m

@sbuild.parse
def fmov_s(m, n):
    n = m

@sbuild.parse
def flds(m, fpul):
    fpul = m

@sbuild.parse
def fsts(fpul, m):
    m = fpul

@sbuild.parse
def fabs(n):
    n = n & i32(0x7FFFFFFF)

@sbuild.parse
def fneg(n):
    n = n ^ i32(0x80000000)

@sbuild.parse
def fadd(m, n):
    n = 'fadd'(n, m)

@sbuild.parse
def fsub(m, n):
    n = 'fsub'(n, m)

@sbuild.parse
def fcmpeq(m, n):
    # TODO: VERIFY
    tf = 'fcom_c0'(n, m)[0:1]

@sbuild.parse
def fcmpgt(m, n):
    # TODO: VERIFY
    tf = 'fcom_c3'(n, m)[0:1]

@sbuild.parse
def fdiv(m, n):
    n = 'fdiv'(n, m)

@sbuild.parse
def float(fpul, n):
    n = 'fpconvert_fp32'(fpul)

@sbuild.parse
def fmac(fr0, m, n):
    n = 'fadd'('fmul'(fr0, m), n)

@sbuild.parse
def fmul(m, n):
    n = 'fmul'(n, m)

@sbuild.parse
def fsqrt(n):
    n = 'fsqrt'(n)

@sbuild.parse
def ftrc(m, fpul):
    fpul = 'fp_to_sint32'(m)

mnemo_func = sbuild.functions
mnemo_func.update({
        'mov.b': mov_b,
        'mov.w': mov_w,
        'mov.l': mov_l,

        "addc": addc,
        "addv": addv,
        "cmpeq": cmpeq,
        "cmphs": cmphs,
        "cmpge": cmpge,
        "cmphi": cmphi,
        "cmpgt": cmpgt,
        "cmppl": cmppl,
        "cmppz": cmppz,

        "div1": div1,
        "dmuls": dmuls,
        "dmulu": dmulu,
        "dt": dt,
        "extsb": extsb,
        "extsw": extsw,
        "extub": extub,
        "extuw": extuw,
        "mac.l": mac_l,
        "mac.w": mac_w,
        "mull": mull,
        "mulsw": mulsw,
        "muluw": muluw,
        "neg": neg,
        "negc": negc,
        "subc": subc,
        "subv": subv,
        "and": l_and,
        "and.b": and_b,
        "not": l_not,
        "or": l_or,
        "or.b": or_b,
        "tas.b": tas_b,
        "tst.b": tst_b,
        "xor.b": xor_b,
        "ldc": ldc,
        "ldc.l": ldc_l,
        "lds.l": lds_l,
        "ldtbl": ldtbl,
        "movca.l": movca_l,
        "nop": nop,
        "ocbi.l": ocbi_l,
        "ocbp.l": ocbp_l,
        "ocbwb.l": ocbwb_l,
        "pref.l": pref_l,
        "sleep": sleep,
        "stc.l": stc_l,
        "sts.l": sts_l,
        "fmov.s": fmov_s,
        "jsr.l": jsr_l,
        "jmp.l": jmp_l,
})

def get_mnemo_expr(ir, instr, *args):
    instr, extra_ir = mnemo_func[instr.name.lower()](ir, instr, *args)
    return instr, extra_ir


class ir_sh4b(IntermediateRepresentation):
    def __init__(self, loc_db=None):
        IntermediateRepresentation.__init__(self, mn_sh4, "b", loc_db)
        self.pc = mn_sh4.getpc()
        self.IRDst = ExprId('IRDst', 32)
        self.addrsize = 32

    def mod_pc(self, instr, instr_ir, extra_ir):
        # fix PC (+0 for SH4)
        pc_fixed = {self.pc: ExprInt(instr.offset, 32)}

        for i, expr in enumerate(instr_ir):
            instr_ir[i] = ExprAssign(expr.dst, expr.src.replace_expr(pc_fixed))

        for idx, irblock in enumerate(extra_ir):
            extra_ir[idx] = irblock.modify_exprs(lambda expr: expr.replace_expr(pc_fixed) \
                                                 if expr != self.pc else expr,
                                                 lambda expr: expr.replace_expr(pc_fixed))
    
    def get_ir(self, instr):
        args = instr.args
        instr_ir, extra_ir = get_mnemo_expr(self, instr, *args)

        self.mod_pc(instr, instr_ir, extra_ir)
        return instr_ir, extra_ir
    
    def get_next_instr(self, instr):
        return self.loc_db.get_or_create_offset_location(instr.offset  + 2)
    
    def get_next_break_loc_key(self, instr):
        return self.loc_db.get_or_create_offset_location(instr.offset  + 4)

