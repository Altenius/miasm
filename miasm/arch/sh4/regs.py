from builtins import range
from miasm.expression.expression import *
from miasm.core.cpu import reg_info, gen_reg

# GP
gpregs_str = ['R%d' % r for r in range(0x10)]
gpregs_expr = [ExprId(x, 32) for x in gpregs_str]
gpregs = reg_info(gpregs_str, gpregs_expr)

bgpregs_str = ['R%d_BANK' % r for r in range(0x8)]
bgpregs_expr = [ExprId(x, 32) for x in bgpregs_str]
bgpregs = reg_info(bgpregs_str, bgpregs_expr)

fregs_str = ['FR%d' % r for r in range(0x10)]
fregs_expr = [ExprId(x, 32) for x in fregs_str]
fregs = reg_info(fregs_str, fregs_expr)

dregs_str = ['DR%d' % r for r in range(0x8)]
dregs_expr = [ExprId(x, 32) for x in dregs_str]
dregs = reg_info(dregs_str, dregs_expr)


PC, reg_info_pc = gen_reg('PC')
PR, reg_info_pr = gen_reg('PR')
R0, reg_info_r0 = gen_reg('R0')
FR0, reg_info_fr0 = gen_reg('FR0')
GBR, reg_info_gbr = gen_reg('GBR')
SR, reg_info_sr = gen_reg('SR')
VBR, reg_info_vbr = gen_reg('VBR')
SSR, reg_info_ssr = gen_reg('SSR')
SPC, reg_info_spc = gen_reg('SPC')
SGR, reg_info_sgr = gen_reg('SGR')
DBR, reg_info_dbr = gen_reg('DBR')
MACH, reg_info_mach = gen_reg('MACH')
MACL, reg_info_macl = gen_reg('MACL')
FPUL, reg_info_fpul = gen_reg('FPUL')
TRA, reg_info_tra = gen_reg('TRA')

R0 = gpregs_expr[0]
R1 = gpregs_expr[1]
R2 = gpregs_expr[2]
R3 = gpregs_expr[3]
R4 = gpregs_expr[4]
R5 = gpregs_expr[5]
R6 = gpregs_expr[6]
R7 = gpregs_expr[7]
R8 = gpregs_expr[8]
R9 = gpregs_expr[9]
R10 = gpregs_expr[10]
R11 = gpregs_expr[11]
R12 = gpregs_expr[12]
R13 = gpregs_expr[13]
R14 = gpregs_expr[14]
R15 = gpregs_expr[15]

FR0 = fregs_expr[0]
FR1 = fregs_expr[1]
FR2 = fregs_expr[2]
FR3 = fregs_expr[3]
FR4 = fregs_expr[4]
FR5 = fregs_expr[5]
FR6 = fregs_expr[6]
FR7 = fregs_expr[7]
FR8 = fregs_expr[8]
FR9 = fregs_expr[9]
FR10 = fregs_expr[10]
FR11 = fregs_expr[11]
FR12 = fregs_expr[12]
FR13 = fregs_expr[13]
FR14 = fregs_expr[14]
FR15 = fregs_expr[15]


reg_tf = 'tf'
reg_mf = 'mf'
reg_qf = 'qf'
reg_sf = 'sf'

tf = ExprId(reg_tf, size=1)
mf = ExprId(reg_mf, size=1)
qf = ExprId(reg_qf, size=1)
sf = ExprId(reg_sf, size=1)

all_regs_ids = [
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15,
    FR0, FR1, FR2, FR3, FR4, FR5, FR6, FR7, FR8, FR9, FR10, FR11, FR12, FR13, FR14, FR15,
    tf, mf, qf, sf,

    PC, PR, GBR, SR, VBR, SSR, SPC,
    SGR, DBR, MACH, MACL, FPUL, FR0, TRA]

all_regs_ids_no_alias = all_regs_ids

all_regs_ids_byname = dict([(x.name, x) for x in all_regs_ids])

all_regs_ids_init = [ExprId("%s_init" % x.name, x.size) for x in all_regs_ids]

regs_init = {}
for i, r in enumerate(all_regs_ids):
    regs_init[r] = all_regs_ids_init[i]

regs_flt_expr = []
