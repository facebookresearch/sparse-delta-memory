# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Op-level test grid of the SDM write/read against the float64 reference.

Factors (``FACTORS``): writes W, reads R, chunk size C, tokens per stream T (relative to C,
so ragged last chunks are covered), head_dim, bank dtype, snapshot quantization,
key-weighted decay, layout and slot distribution. Tiers:

  default  a pairwise covering array: every pair of factor values is in some case
  slow     a strength-3 covering array: every triple
  anchors  adversarial cases: one-factor sweeps around the base and the worst geometry,
           W/R extremes, extreme gates and betas, ragged T at a non-power-of-two C,
           quantization with odd head_dim and key-weighted decay, and the four cotangent
           layouts

Both arrays are committed literally (``_PAIRWISE`` and ``_STRENGTH3``, one base-36 digit
per factor); ``test_op_grid.py`` checks their coverage with ``uncovered``, and
``--generate STRENGTH`` builds a new array.

Every case runs the op forward and backward on inputs cast to the bank dtype and compares
the readings, the six gradients and the final bank with ``sdm_testlib.oracle_write_read``
on the same values:
  fp32 bank   GATE_FP32 on every tensor;
  bf16 bank   bf16_gate_failures (ceilings, and the same inputs on an fp32 bank);
  quantized   quant_gate_failures on the k_val, g and q_val gradients (against the same
              case unquantized); the other tensors take the bank's gate.

Layouts: ``3d_p2`` and ``3d_p1`` are [P, T, *] streams with disjoint slot ranges; ``2d``
is one [2T, *] sequence over one bank of 2 * slots (num_memory_slots=0). Slot
distributions draw distinct slots per token from a bank of ``SLOTS[kind]`` slots:
  uniform       uniform
  collide       uniform over a small bank: heavy cross-token collisions
  sticky        each slot of the previous token is kept with probability 0.9
  single_hot    one slot is in every token's write and read sets
  kq_overlap    each token reads its own write slots (then uniform slots if R > W)
  hot_boundary  one slot is written and read by the two tokens on each side of every
                chunk boundary

Usage (against the kernels on PYTHONPATH):
  python tests/op_grid.py --tier default --out grid_default.json
  python tests/op_grid.py --generate 2
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
import traceback
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_testlib import (  # noqa: E402
    BF16_BANK_COLLISION_CEILINGS,
    BF16_COLLISION_WRITES,
    OP_GRADS,
    OP_INPUTS,
    QUANT_GRADS,
    bf16_gate_failures,
    bits_equal,
    deviations,
    fp32_gate_failures,
    make_cotangent,
    oracle_write_read,
    profiled_kernel_names,
    quant_gate_failures,
)

FACTORS = {
    "W": (1, 16, 32, 48, 64, 65, 128, 256),
    "R": (1, 16, 63, 64, 100, 255, 256),
    "C": (16, 17, 32, 48, 64, 96, 127, 128, 255, 256),
    "T": ("C-1", "C", "2C-1", "2C", "2C+1", "4C+3"),
    "hd": (64, 65, 128, 129, 256, 512, 960, 961, 1024, 1025, 1920, 2048),
    "dtype": ("float32", "bfloat16"),
    "quant": ("none", "fp8", "int4"),
    "kwd": (False, True),
    "layout": ("3d_p2", "3d_p1", "2d"),
    "slots": ("uniform", "collide", "sticky", "single_hot", "kq_overlap", "hot_boundary"),
}
SLOTS = {"uniform": 65536, "collide": 512, "sticky": 4096, "single_hot": 4096,
         "kq_overlap": 4096, "hot_boundary": 4096}
STICKY_KEEP = 0.9
BASE = dict(W=64, R=64, C=64, T="2C", hd=128, dtype="bfloat16", quant="none", kwd=False,
            layout="3d_p2", slots="uniform")
WORST = dict(BASE, W=256, R=256, C=256, hd=1024, dtype="float32")
TENSORS = ("readings", "final_memory") + OP_GRADS
_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"

# Covering arrays, one row per case, one base-36 digit per factor (in FACTORS order).
_PAIRWISE = """
3654412110 7403900121 4134201002 0272311024 6311610103 1065702013 5540b01115 2080010020
5593112001 2192500114 7421812015 1225a10102 6601a01020 3315401021 0383302112 4662210125
6140912004 4271002111 1153801123 7032b12100 5552710022 5024601024 2474701105 0600512003
0445100020 3460310002 7214111113 2613b02122 1581401104 6233510125 6184b00011 0091901125
6455001004 3105602115 5375800100 2535a12013 4523212110 7363a01024 3041511122 4482400103
1630610111 2322911011 4350111125 0564802121 7170410022 1394000023 5161312120 3290711110
4695812104 1412a00125 3073201003 6002812002 7685300125 2251211021 4044312101 3032111104
5284912113 0110202004 2243400115 7642012103 4515501120 0331700021 7593600112 1575910100
4204b02004 6520311103 3503012105 0050a00111 2262612120 5434512001 6671100002 7623712024
6495202003 3325b01113 0124011002 5610911012 1344200124 3184a02010 2302311105 3210812100
0491b11104 6061402024 7352501101 1105112120 0483610013 4015711124 5201400021 6145700102
1473511111 7481212120 1455300021 0520411013 5045a00103 1063b01111 4293a02011 3231901024
4055601000 4364910020 2483801003 5310000120 2384100101 5204211024 1241810003 0392412125
1542610110 7270b02121 1665111010 1560000015 4471611025 3654b01114 7284711101 5473a11022
2650910001 7030010123 5163500012 3120111122 7094301023 7120511110 2133811110 1430400113
1002711105 1534311122 3185501021 4113312021
"""
_STRENGTH3 = """
0272900002 7231b11011 0344711123 6121410105 1514300114 5085812022 7393a02101 4040201005
3672612114 6400102024 4613510023 2135901110 2564412000 5251600120 1592011025 3305310001
1453111102 0361002015 1625602003 2190610012 6082500111 3661a01122 5474801011 7600700115
2023b11124 5430010104 4495b00113 0115a11004 6154212021 5243312115 7575110120 4112802120
7483301000 4281212103 3550502012 0260912121 1041a10010 6312401013 2284100025 3014002103
5163700003 5602511100 3343800112 4501911014 2332702004 1370b02020 6295811104 0533601105
7541902023 7422212012 3465511025 0624101010 1005401121 0032310023 2411112111 6504b12102
3520200121 4653001111 5094910105 2651810001 6070a11103 4294701022 6633a12020 5645400024
1583712011 6355712110 2204a02013 7154201114 1462a00105 0191502124 4330412112 3210b10000
1265010012 5120302025 7031100102 7684512004 0180311112 1160811113 3142112022 2373210115
4452611023 7103012123 6491200020 2680001023 5391111023 2542301101 6223000004 6415901122
7362411124 0444812005 0655b10125 7014610021 6575302015 3471710004 3384a11015 3093901011
0053402004 3483402120 1234501123 3102b01015 5175412101 4174010020 4063712125 3555911103
2692002102 5512710112 1381900124 7250112005 0403610110 1324211100 4501500105 6341511002
6064601100 4225a10011 5535201022 2021802113 5590a01110 6640910111 4325a02124 2424500122
3185600102 2202210124 1612912015 4562102011 1671311122 1230400001 5322b00121 0593810021
3245001110 1173100013 5011701020 1494312004 4344602111 2510602125 7065b02004 3634210005
0613b02012 7212502110 0682112103 2445b12003 0295202013 2451401015 5631302113 0421011101
6074400012 5220611014 6272811125 5303901022 6481612015 2273502001 7525701005 4350300103
7224810123 3330801124 7440511121 1141b00104 4382810000 0540212014 0132510011 6131701101
0360500000 4690712000 5052a12014 4022401002 7161310000 4675501114 6643201123 4584b01001
5305102115 2000512020 0614411121 0485700021 3091302115 1101202112 7613801105 2184912002
7392010014 3223712102 5584000113 4420900000 0075212120 1033111014 2553a10122 2165000121
2463711114 6565610024 7173601022 3051010024 3492410003 6334b00114 1155500000 5663212010
1080012105 0200811003 7435002025 4140111100 6110011012 7182a00123 1623110121 4215311024
4035610000 7590402123 6523912110 1203412025 0311210102 6264112103 4402302111 2581b00110
5354311002 0274a00124 6452b01000 5571802104 7242610005 2361201021 3605712023 5475a12022
7060a12012 7375912103 6105800110 6045a02125 3123511113 0452800114 7301611004 5670b10002
1512201001 7285411014 4664900023 3534812011 1150701024 4441300022 3062800101 3274b01123
2215101012 2622310010 1692601121 5244012021 4183200104 4433a01015 5560b11115 7253911113
4132b12022 1242700013 3321602020 5404211103 1315b01105 0001602022 1411800023 6390512125
7691b12025 2530710123 4100a01121 2143410011 4263801010 0542400110 1174712115 1461512111
3312100104 0151a02113 4155900025 2470601100 7050410120 0572511123 3204301120 2003910105
7374001102 6024311001 6611911004 3232a01102 6280710100 6313302120 0022700115 1640812022
2394501011 1604000004 5045502013 6582701024 3470911015 1054b12111 7543100014 7655302011
1193401110 6662302102 4681801115 5413602004 0030102021 1363610013 0092901120 1122811004
7072712121 5241411102 5310a00111 2591a12004 5194612103 7454710000 3231912020 0293310105
5181101110 7502802022 3695a10120 3244510104 6385012123 6473b10021 7033210103 4573401123
4071902111 1475200004 4414400005 7641012000 6694802013 3164912114 0525312122 4560001104
6551110113 5465110120 3013211012 5292102101 4605110022 4054511104 5124a11000 3352202025
2395400122 7630602114 0410712003 5073300110 2044801024 2162200103 7383b10122 0515000120
1443901100 2075002014 7195211125 6201a00111 5482201005 3521412124 2574910122 4191700111
4271112005 2353012100 2255612004 3424b02015 5452912101 1514a02022 7493500103 3040601113
5133800005 1320901112 3563301013 7084311125 2634701012 2505101101 3541702115 2010310004
3690100114 6250211002 4001b10113 0163b11003 7323502005 1332310015 6663400111 6053802025
5503200011 6442010013 4511601012 2490311021 7170800111 7615412002 3100411004 2320100003
0554111025 0464802102 1682b01022 6002011010 1095700102 1261112024 0271402010 5285b12115
3254412111 0335611101 3603a01004 1434612122 0043002022 1650a11011 5155011002 2685202021
1530311000 5114500115 5334411010 4280512022 5080901113 2613102120 4551202010 3611511001
6014100025 6125102121 4165400123 4642a01004 0380410005 7512b12023 6515811111 4624610125
3682310124 6644b01120 6653611112 7304902010 0351301124 4363101122 5624712024 4521002012
7110900121 0302a11022 2261b10122 3673000125 3191811012 3072111120 2642502025 6132412024
7431502100 3522a02003 0235910114 3400000102 5383511124 1502711100 2152100024 3130212010
2083611004 2480a11004 4545801103 0673701010 6434911003 5061901102 2365812125 7205500112
5581312101 2240802120 3211201125 2044200012 3081711123 7380700021 4331a00013 2171a01015
5193b11104 6103300023 5075600115 7594101122 0631801010 7564a11001 3035b11102 1052200123
5372801113 0084201010 6092201114 1283a12100 0660602004 7400b01024 4432210021 5213410123
4232012115 6530500104 0561701022 1131011013 5601602101 7145710102 6620b00103 5652411015
7270312014 0394b00120 0104002005 1144102115 4601400010 0345100121 2012a12111 5440412104
2374302003 5525511001 7671211123 1340011011 0583911005 1151601025 0121912123 3435110013
7632900112 5421a12103 6195b02101 7125b10120 1113312021 7621411011 4342900115 2031410015
1300510003 4243b10105 3280610121 7351811112 6173902004 1484402003 0220a10125 0693210102
0142612120 2233301112 7020000020 1610611110 1073500122 5162511022 6493700022 6683101022
6004702014 3502910023 5341212114 0040b11111 2672b10014 3444a01021 2291912003 0290010123
3182012021 7552011011 4493111015 3594611000 2214212110 6425412000 6635010001 5550810120
3375402005 1065301123 3362700120 4304810101 4085101124 3425800020 7211712014 5251511023
1025911111 6654a00004 7412a10110 2483810023 6371600111 2401302005 5362611102 1585500010
3153b10024 5264700125 7662101025 4492501104 2422912024 2524010025 3115211003 7461601023
4090811125 5695901102 1521802115 0374110014 4372700102 1552411022 6460212015 1390200113
5532911121 0172200001 5570102012 0015501005 3651912010 7580200102 6174510100 5262310014
0034a02021 6320810122 4134302124 1535a12023 5493002010 2354902121 0505b10114 5611011025
2423601021 5143a11023 5062000021 2050700013 3574212022 0282b00004 7095811000 1544612124
6212600113 6231801102 6332101010 3201100120 3674112021 7044402115 6605201125 6030b01115
0274612113 0252710101 7472300103 4203612012 5411b02102 0323401003 3411311112 5100910114
1214912013 3514710111 2641712011 1221300023 2125711120 2222400023 3333511100 2114b12011
4595300120 6145302014 3351b01123 0620501122 0404501101 5042b00020 4610200014 4123a10022
1450001115 1573010021 2382411001 3265201014 2393702025 0465401002 0641310020 7605a11005
1254811013 7531a00124 1291411024 6061911015 0281011102 7422111024 5275712004 6184811014
1352500015 6300601005 1672412124 6192a02003 5023110022 2345a00120 7663811004 7001211111
1245200121 4425712113 0364300121 6222500110 1665702101 7433411101 0050901122 4623301014
7181512125 0314901100 0455a00010 6540a01022 1564b12020 0470202125 5402701001 2104612023
4315002011 2315510003 1433b01120 4250b11012 1021511024 3295500100 2435310011 2600411003
2095102020 2602810102 0111100013 4094a11100 6590412105 2682610000 3000300112 0192701124
7134101121 1275812101 2563512014 1355110000 4480800102 4461310115 1321700011 4470012013
2560902010 0212012014 3655810013 7265a10100 5582611123 2300b12115 4004100003 1014810100
0513a00125 5094410103 6612701115 7295600114 7490901104 6653502103 4160610101 4395212012
0434011002 2663a10103 1341402013 0405900121 3660b02101 7642210010 7224901015 2114412104
5585411011 0223210114 5534b10023 0243512025 4243910102 4140500120 6183412022 1403800024
1435711125 7071502105 5615310015 7552610001 2500201115 2453201123 3493612115 0035812123
7380102001 3621101105 3100711102 5634111004 3152300020 5053111121 0045912014 2231602123
6162901115 4603b12100 0062100012 5230a00115 3543b00102 6664012010 6241100103 1510100014
0421b10012 6671012115 0233100105 4551712124 5681a01012 7593700123 6285911010 3073a00111
5201010021 7392312122 1623211025 6281301101 7340300113 6132602124 2445602122 1180b11003
3210812021 6690302113 3001801005 6525212023 2141000104 4014b12014 3394010121 5673902120
2055300025 5440701020 0604300104 4673801104 4564210102 6415510122 1372a10120 3583110021
1022601102 5035501115 7564511120 1510410000 5470500011 4043711114 3060412022 4155410024
5623800021 0494a12012 4630502001 4033701010 5653711025 4585a10001 3632412123 6571702103
3033000020 3363912112 3170300001 2451511021 3350a01115 4694400111 6361a12014 5322012023
3463010103 7481902021 4091000013 4211a01003 5261400005 1510510114 5120202124 6331211114
7333900105 2142811001 4091601104 3002500104 1201b02001 1502100023 0672000001 1483000020
3290b10012 0404402122 6254011113 7312312103 1413010102 4184700122 5002a01020 4385300023
2201710102 6281b01020 3641610012 1392801114 3315610015 0312800012 2594201111 0494101122
4575b00015 6453311120 7432811025 7554b10012 1624a10014 4362b12103 2685701115 3525010003
3483512013 1691502112 1630900023 4533102023 7384612010 3442210110 1583802101 5502410120
3011411101 5543611125 6103500015 7103101123 7500302123 3044300012 6195010021 7071b11113
4281411122 0350600110 5272201101 3261602015 2133502012 6354400123 2065502023 4013901021
6561800020 3454612001 1424411021 3021201102 1293a01015 3302211010 1502612024 0071810010
5241812011 1144911025 1092b11002 0535410105 0303710004 5055210004 4542001025 7283012004
1593911125 6390901020 7105411023 1465912113 2572811102 2160110111 3682902105 4313702113
1551901014 3154701022 7234202003 7470402111 1115711002 3371500103 2305001102 5383700104
2160300110 6020700100 7160712020 5120412101 1480100115 3005612115 6502001105 5494310011
2070702005 4223101013 3085b12120 6493811025 0172b11013 4570600111 7122601115 5115801014
5092500005 0643a11000 4323500002 6463b02014 0600901010 6203402014 0125802102 2332001011
2414001120 1120000025 7572a12001 6384200000 5245901123 1545512003 4521b02121 2641911101
5163a01002 2444110121 2393602011 5271000011 4664500115 7314211000 0410300020 7063900125
1284611004 2023a02110 2591b12015 1130a12020 0190412110 3114a01101 6475102110 3624900111
5042110113 4190912113 4075310122 6481a01025 3655112120 6265502123 4673a01111 4035902003
4152802102 6191101111 0285802005 1402a02120 1550300004 6611a01124 0115612115 0652210110
7060211021 1223b00023 3233710001 0520601115 7422311110 4251a11015 3321311004 2510911025
7243a12013 5094802004 2233002010 0633b02111 1204900003 6415202123 0081a11120 7453a01011
3081800025 3192701022 2295802003 7345b11110 7333302012 0301411010 1375601105 5480b00100
0362212014 2471400101 3520101120 7534710013 6051612004 4601202114 4643111103 5331701111
5252310115 4590502123 6135310102 6422a10123 0151511025 5082400102 3625000113 2585312015
5502802020 5195310003 4530002115 1681412022 4680010104 7214100121 6052911110 1134810122
4654100100 3432610125 0292601124 5142212025 6490600011 1503a12110 6570202114 6613b02010
2532b02124 0450002112 7265000120 0332200122 2472101021 6361700111 0360a00110 7653511002
5611111122 4434810004 7515011102 0400110125 5625901122 5055b00123 5131610023 4040410020
2633812024 6442801013 5195712122 2234102123 3281212015 5200b12014 3660111015 5214912112
1662412001 4475200010 4481912111 4103002112 1090111005 3645402010 5394211111 2683512122
0572300024 2434a01021 1480310000 6040702113 0504201011 6604900005 0183510102 5500310004
2330200104 5562810000 4410110101 2251b12124 1383211003 4164a11114 4302711123 0381610110
6045402000 1365212102 1401311103 5511200100 1143312020 0135701112 4625810004 6024601113
3272802110 7022a00105 6243610012 4184612000 3492202000 7582812121 1375900023 4624201024
2610702004 2311810011 3414900115 7042812005 2425200101 2163601015 1235211121 3231310101
3243911125
"""


# ------------------------------------------------------------------------------ cases
def decode(encoded):
    names = list(FACTORS)
    return [{n: FACTORS[n][int(ch, 36)] for n, ch in zip(names, row)}
            for row in encoded.split()]


def case_key(case):
    """Stable identifier of a case; non-default T and extras are appended."""
    parts = [f"W{case['W']}", f"R{case['R']}", f"C{case['C']}", f"hd{case['hd']}",
             case["dtype"], case["quant"], f"kwd{int(case['kwd'])}", case["layout"],
             case["slots"]]
    if case["T"] != "2C":
        parts.append("T" + case["T"])
    for extra, default in (("gates", "moderate"), ("cotangent", "dense")):
        if case.get(extra, default) != default:
            parts.append(case[extra])
    return "-".join(parts)


def tokens(spec, C):
    """Tokens per stream for a T factor value such as "2C+1"."""
    k, _, d = spec.partition("C")
    return int(k or 1) * C + int(d or 0)


def anchors():
    cases = []
    for base in (BASE, WORST):
        cases.append(dict(base))
        for name, values in FACTORS.items():
            cases += [dict(base, **{name: v}) for v in values if v != base[name]]
    cases += [dict(BASE, W=1, R=256), dict(BASE, W=256, R=1)]
    cases += [dict(BASE, gates=g) for g in ("g-20", "beta0", "beta1", "beta2")]
    cases += [dict(BASE, C=17, T=t) for t in ("2C-1", "2C+1")]
    cases += [dict(BASE, quant=q, hd=hd, kwd=kwd)
              for q in ("fp8", "int4") for hd in (65, 961) for kwd in (False, True)]
    cases += [dict(BASE, cotangent=c) for c in ("dense2", "expanded", "noncontig")]
    # The largest deviations of the strength-3 tier: heavy bf16 bank collisions, int4 with
    # per-write gates, and a slot shared across every chunk boundary.
    cases += [dict(BASE, W=256, R=63, C=48, hd=2048, quant="fp8", layout="3d_p1",
                   slots="collide", T="C"),
              dict(BASE, W=65, R=63, C=17, hd=1025, quant="int4", kwd=True, layout="3d_p1",
                   slots="sticky", T="2C+1"),
              dict(BASE, W=1, R=16, C=256, hd=256, quant="int4", kwd=True, layout="3d_p1",
                   T="C-1"),
              dict(BASE, W=1, R=16, C=96, hd=1920, dtype="float32", quant="int4", kwd=True,
                   layout="3d_p1", slots="single_hot", T="C"),
              dict(BASE, W=32, R=1, C=32, hd=1024, dtype="float32", quant="int4", kwd=True,
                   layout="3d_p1", slots="single_hot", T="C"),
              dict(BASE, W=32, R=1, C=16, hd=1025, kwd=True, slots="hot_boundary")]
    unique = {}
    for c in cases:
        unique.setdefault(case_key(c), c)
    return list(unique.values())


def tier(name):
    if name == "default":
        return decode(_PAIRWISE)
    if name == "slow":
        return decode(_STRENGTH3)
    if name == "anchors":
        return anchors()
    raise ValueError(name)


# -------------------------------------------------------------------- covering arrays
def _tuples(strength):
    sizes = [len(v) for v in FACTORS.values()]
    return {c: set(itertools.product(*(range(sizes[i]) for i in c)))
            for c in itertools.combinations(range(len(sizes)), strength)}


def _index_rows(cases):
    return [tuple(FACTORS[n].index(c[n]) for n in FACTORS) for c in cases]


def uncovered(cases, strength):
    """Number of factor-value tuples of the given strength that no case covers."""
    need = _tuples(strength)
    for row in _index_rows(cases):
        for c, left in need.items():
            left.discard(tuple(row[i] for i in c))
    return sum(len(left) for left in need.values())


def generate(strength, seed=0, candidates=20):
    """Greedy covering array: each row starts from an uncovered tuple and fills the other
    factors with the value covering the most new tuples (ties broken at random, so levels
    stay balanced); the best of ``candidates`` rows is kept. Returns the encoded rows."""
    rng = random.Random(seed)
    sizes = [len(v) for v in FACTORS.values()]
    need = _tuples(strength)
    by_factor = {f: [c for c in need if f in c] for f in range(len(sizes))}
    rows = []
    while any(need.values()):
        open_combos = [c for c, left in need.items() if left]
        best, best_gain = None, -1
        for _ in range(candidates):
            combo = open_combos[int(rng.random() * len(open_combos))]
            start = sorted(need[combo])[int(rng.random() * len(need[combo]))]
            row = dict(zip(combo, start))
            rest = [f for f in range(len(sizes)) if f not in row]
            rng.shuffle(rest)
            for f in rest:
                def gain(v):
                    row[f] = v
                    g = sum(tuple(row[i] for i in c) in need[c] for c in by_factor[f]
                            if all(i in row for i in c))
                    del row[f]
                    return g
                row[f] = max(range(sizes[f]), key=lambda v: (gain(v), rng.random()))
            full = tuple(row[i] for i in range(len(sizes)))
            total = sum(tuple(full[i] for i in c) in left for c, left in need.items())
            if total > best_gain:
                best, best_gain = full, total
        rows.append(best)
        for c, left in need.items():
            left.discard(tuple(best[i] for i in c))
    return " ".join("".join(_DIGITS[v] for v in r) for r in rows)


# ----------------------------------------------------------------------------- inputs
def _distinct(keys, width):
    return keys.topk(width, dim=-1).indices.sort(dim=-1).values


def _slot_rows(kind, S, T, W, R, sph, C, gen):
    """k_idx [S, T, W], q_idx [S, T, R]: distinct local slots per token."""
    def rand(rows):
        return torch.rand(rows, sph, generator=gen)

    if kind == "sticky":
        def chain(width):
            out = torch.empty(S, T, width, dtype=torch.int64)
            out[:, 0] = _distinct(rand(S), width)
            for t in range(1, T):
                keys = rand(S)
                keep = torch.rand(S, width, generator=gen) < STICKY_KEEP
                prev = out[:, t - 1]
                keys.scatter_(1, prev, torch.where(keep, 2.0, keys.gather(1, prev)))
                out[:, t] = _distinct(keys, width)
            return out
        return chain(W), chain(R)
    kk, kq = rand(S * T), rand(S * T)
    if kind in ("single_hot", "hot_boundary"):
        hot = torch.randint(0, sph, (S, 1, 1), generator=gen).expand(S, T, 1).reshape(-1, 1)
        rows = torch.ones(T, dtype=torch.bool)
        if kind == "hot_boundary":
            offset = torch.arange(T).view(-1, 1) - torch.arange(C, max(T, C), C).view(1, -1)
            rows = ((offset >= -2) & (offset < 2)).any(1)
        rows = rows.expand(S, T).reshape(-1, 1)
        for keys in (kk, kq):
            keys.scatter_(1, hot, torch.where(rows, 2.0, keys.gather(1, hot)))
    k_idx = _distinct(kk, W)
    if kind == "kq_overlap":
        own = k_idx[:, :min(W, R)]
        kq.scatter_(1, own, 2.0)
    return k_idx.view(S, T, W), _distinct(kq, R).view(S, T, R)


def make_inputs(case):
    """float32 master inputs on the GPU as [S, T, *] streams, with S, T and the bank size.

    The ``2d`` layout is one stream of 2T tokens over a bank of 2 * slots."""
    C, W, R, hd = case["C"], case["W"], case["R"], case["hd"]
    T = tokens(case["T"], C)
    S = 2 if case["layout"] == "3d_p2" else 1
    sph = SLOTS[case["slots"]]
    if case["layout"] == "2d":
        T, sph = 2 * T, 2 * sph
    gen = torch.Generator().manual_seed(zlib.crc32(case_key(case).encode()))
    k_idx, q_idx = _slot_rows(case["slots"], S, T, W, R, sph, C, gen)
    offset = torch.arange(S).view(S, 1, 1) * sph

    def randn(*shape, scale):
        return torch.randn(*shape, generator=gen) * scale

    gates = case.get("gates", "moderate")
    g = torch.empty(S, T, W if case["kwd"] else 1).uniform_(-0.5, -0.01, generator=gen)
    beta = torch.rand(S, T, 1, generator=gen) * 0.3
    if gates == "g-20":
        g.fill_(-20.0)
    elif gates.startswith("beta"):
        beta.fill_(float(gates[len("beta"):]))
    d = dict(memory=randn(S * sph, hd, scale=0.1), k_idx=k_idx + offset, q_idx=q_idx + offset,
             k_val=randn(S, T, W, scale=0.3), v=randn(S, T, hd, scale=0.1), beta=beta, g=g,
             q_val=randn(S, T, R, scale=0.3))
    return {k: v.cuda() for k, v in d.items()}, S, T, sph


# ------------------------------------------------------------------------------- runs
def run_op(d, case, S, T, sph, dtype, quant, G, kernels=None):
    """Op forward + backward on bank-dtype copies of ``d`` with the cotangent values ``G``
    in the case's memory layout: {tensor name: result}. When ``kernels`` is a set, the GPU
    op names launched by the op's forward and backward are added to it."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    def profiled(fn, *args, **kwargs):
        if kernels is None:
            return fn(*args, **kwargs)
        out, names = profiled_kernel_names(fn, *args, **kwargs)
        kernels.update(names)
        return out

    leaves = {n: d[n].to(dtype, copy=True).requires_grad_(True) for n in OP_INPUTS}
    two_d = case["layout"] == "2d"

    def lay(t):
        return t.reshape(S * T, -1) if two_d else t

    readings = profiled(
        sdm_write_read,
        leaves["memory"], lay(d["k_idx"]), lay(leaves["k_val"]), lay(leaves["v"]),
        lay(leaves["beta"]), lay(leaves["g"]), lay(d["q_idx"]), lay(leaves["q_val"]),
        chunk_size=case["C"], num_memory_slots=0 if two_d else sph,
        num_parallel=1 if two_d else S, snapshot_quant=quant,
        key_weighted_decay=case["kwd"])
    # The op advances the bank in place and its backward restores it.
    final = leaves["memory"].detach().clone()
    mode = case.get("cotangent", "dense")
    cotangent = make_cotangent(readings.detach(), mode)
    if mode != "expanded":  # all ones, exact in every dtype
        cotangent.copy_(G.reshape(cotangent.shape))
    grads = profiled(torch.autograd.grad, readings, [leaves[n] for n in OP_INPUTS],
                     grad_outputs=cotangent)
    out = dict(zip(OP_GRADS, grads), readings=readings.detach().reshape(S, T, -1),
               final_memory=final, restored_memory=leaves["memory"].detach())
    return {k: v.float() for k, v in out.items()}


def writes_per_slot(k_idx, C):
    """The largest mean number of writes per written slot over the chunks of every stream."""
    S, T, _ = k_idx.shape
    worst = 1.0
    for s in range(S):
        for t0 in range(0, T, C):
            rows = k_idx[s, t0:t0 + C].reshape(-1)
            worst = max(worst, rows.numel() / rows.unique().numel())
    return worst


def _quant_tail_class(case, S, T, dtype):
    """The class with which the case's training call is refused before any work (fp8 / int4 rows of a quantized tail
    head, or the device / int4 refusals on this device: memory_ops.quant_tail_class), or None."""
    from sparse_delta_memory.memory_ops import quant_tail_class
    P, toks = (1, S * T) if case["layout"] == "2d" else (S, T)
    return quant_tail_class(dtype, case["hd"], case["quant"], P * -(-toks // case["C"]) * case["C"] * case["W"],
                            device=torch.device("cuda", torch.cuda.current_device()))


def run_case(case, profile=False):
    """Run one case; returns its record (status "pass", "refused", "fail" or "error"). With
    ``profile``, the record's "kernels" lists the GPU ops of the case's own run. A case whose
    training call is refused (fp8 / int4 rows of a quantized tail head) records the refusal
    ("refused": its class) instead of the comparison."""
    from sparse_delta_memory.memory_ops import last_dispatch, reset_dispatch

    t0 = time.perf_counter()
    rec = {"key": case_key(case), "case": case, "status": "error", "failures": {}}
    try:
        master, S, T, sph = make_inputs(case)
        dtype = getattr(torch, case["dtype"])
        # The reference and every run see the bank-dtype values.
        d = {k: v.to(dtype).float() if v.is_floating_point() else v for k, v in master.items()}
        template = torch.empty(S * T, case["hd"], device="cuda", dtype=dtype)
        G = make_cotangent(template, case.get("cotangent", "dense")).float().reshape(S, T, -1)
        cls = _quant_tail_class(case, S, T, dtype)
        if cls is not None:
            from sparse_delta_memory.memory_ops import UnsupportedConfigurationError
            reset_dispatch()
            try:
                run_op(d, case, S, T, sph, dtype, case["quant"], G)
            except UnsupportedConfigurationError as e:
                rec.update(refused=e.tail_class, refusal=str(e))
            rec["dispatch"] = last_dispatch()
            if rec.get("refused") != cls or rec["dispatch"]:
                rec["failures"]["refusal"] = f"expected a class {cls} refusal before any work, got {rec.get('refused')}"
            rec["status"] = "fail" if rec["failures"] else "refused"
            rec["seconds"] = time.perf_counter() - t0
            return rec
        written = d["k_idx"].reshape(-1).unique()

        def with_update(res):
            """Adds the change of the written bank rows, which a full-bank metric dilutes."""
            res["bank_update"] = (res["final_memory"][written].double()
                                  - d["memory"][written].double())
            return res

        readings, grads, final = oracle_write_read(d, G, sph=sph)
        ref = with_update(dict(grads, readings=readings, final_memory=final))
        reset_dispatch()
        kernels = set() if profile else None
        got = with_update(run_op(d, case, S, T, sph, dtype, case["quant"], G, kernels))
        rec["dispatch"] = last_dispatch()
        if profile:
            rec["kernels"] = sorted(kernels)
        dev = deviations(ref, got)
        rec["deviations"] = dev
        quant, fp32 = case["quant"] != "none", dtype == torch.float32
        own = [k for k in TENSORS if not (quant and k in QUANT_GRADS)]
        if fp32:
            rec["failures"] = fp32_gate_failures(dev, own + ["bank_update"])
        else:
            dev32 = deviations(ref, with_update(
                run_op(d, case, S, T, sph, torch.float32, case["quant"], G)))
            rec["deviations_fp32_bank"] = dev32
            rec["writes_per_slot"] = writes_per_slot(d["k_idx"], case["C"])
            ceilings = ({"final_memory": BF16_BANK_COLLISION_CEILINGS}
                        if rec["writes_per_slot"] > BF16_COLLISION_WRITES else None)
            rec["failures"] = bf16_gate_failures(dev, dev32, own, ceilings)
        if quant:
            dev_none = deviations(ref, with_update(run_op(d, case, S, T, sph, dtype, "none", G)))
            rec["deviations_unquantized"] = dev_none
            rec["failures"].update(quant_gate_failures(dev, dev_none, case["quant"], fp32))
        untouched = torch.ones(d["memory"].shape[0], dtype=torch.bool, device="cuda")
        untouched[written] = False
        if not torch.equal(got["final_memory"][untouched], d["memory"][untouched]):
            rec["failures"]["untouched_rows"] = "a bank row that no token writes changed"
        if not quant and not bits_equal(got["restored_memory"], d["memory"]):
            rec["failures"]["bank_restore"] = "the backward did not restore the bank bitwise"
        rec["status"] = "fail" if rec["failures"] else "pass"
    except Exception as e:  # a crash is a recorded failing point, not a harness error
        rec["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["seconds"] = time.perf_counter() - t0
    torch.cuda.empty_cache()
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=("default", "slow", "anchors"))
    p.add_argument("--match", default="", help="run only cases whose key contains this")
    p.add_argument("--kernels", action="store_true", help="record the op's GPU op names")
    p.add_argument("--out")
    p.add_argument("--generate", type=int, metavar="STRENGTH",
                   help="print a covering array of this strength and exit")
    a = p.parse_args(argv)
    if a.generate:
        print(generate(a.generate))
        return 0
    if not (a.tier and a.out):
        p.error("--tier and --out are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [c for c in tier(a.tier) if a.match in case_key(c)]
    records = []
    for i, case in enumerate(cases):
        rec = run_case(case, a.kernels)
        records.append(rec)
        why = rec.get("error") or "; ".join(f"{k}: {v}" for k, v in rec["failures"].items())
        print(f"[{i + 1}/{len(cases)}] {rec['status']:5s} {rec['key']} "
              f"({rec['seconds']:.1f} s) {why[:300]}", flush=True)
        with open(a.out, "w") as f:
            json.dump({"tier": a.tier, "records": records}, f, indent=1, default=str)
    counts = {s: sum(r["status"] == s for r in records) for s in ("pass", "refused", "fail", "error")}
    print(json.dumps(counts))
    return 0 if counts["pass"] + counts["refused"] == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
