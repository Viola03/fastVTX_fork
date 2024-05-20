IS_MC = True
IS_EE = True
IS_SS = False
IS_EMU = False
year = "2018"

from CommonParticles.Utils import DefaultTrackingCuts
from DecayTreeTuple.Configuration import *

from GaudiKernel.SystemOfUnits import *
from Gaudi.Configuration import *
from Configurables import (
    DecayTreeTuple,
    TupleToolKinematic,
)
from Configurables import (
    TupleToolPid,
)

from Configurables import DecayTreeTuple
from Configurables import DaVinci

config_electron = {
    "stripping_line": "Bu2LLK_eeLine2",
    "decayname": "B2Kee",
    "decay": "[B+ -> ^(J/psi(1S)->^e+ ^e-) ^K+]CC",
    "branches": {
        "B_plus": "[ B+ -> (J/psi(1S)->e+ e-)  K+]CC",
        "K_Kst": "[ B+ -> (J/psi(1S)->e+ e-) ^K+]CC",
        "e_minus": "[(B+ -> (J/psi(1S)->e+ ^e-) K+), (B- -> (J/psi(1S)->^e- e+) K-)]",
        "e_plus": "[(B+ -> (J/psi(1S)->^e+ e-) K+), (B- -> (J/psi(1S)->e- ^e+) K-)]",
        "J_psi_1S": "[ B+ -> ^(J/psi(1S)->e+ e-) K+]CC",
    },
    # "decay": "[B+ -> ^(J/psi(1S)->^e+ ^e-) ^K+]CC",
    # "branches": {
    #     "M": "[ B+ -> (J/psi(1S)->e+ e-)  K+]CC",
    #     "A": "[ B+ -> (J/psi(1S)->e+ e-) ^K+]CC",
    #     "B": "[(B+ -> (J/psi(1S)->e+ ^e-) K+), (B- -> (J/psi(1S)->^e- e+) K-)]",
    #     "C": "[(B+ -> (J/psi(1S)->^e+ e-) K+), (B- -> (J/psi(1S)->e- ^e+) K-)]",
    # },
}

# DefaultTrackingCuts().Cuts = {"Chi2Cut": [0, 3], "CloneDistCut": [5000, 9e99]}

from StrippingConf.Configuration import StrippingConf, StrippingStream

# from StrippingArchive.Stripping34.StrippingRD.StrippingBu2LLK import (
#     Bu2LLKConf as builder,
# )
# from StrippingArchive.Stripping34.StrippingRD.StrippingBu2LLK import (
#     default_config as stripping_config,
# )


__all__ = ("Bu2LLKConf", "default_config")

default_config = {
    "NAME": "Bu2LLK",
    "BUILDERTYPE": "Bu2LLKConf",
    "CONFIG": {
        "DaughterPT": 250.0,
        # "KaonPT": 250.0,
        # "DiHadronMass": 9999.0,
        # "KaonIPCHI2": 0.0,
    },
    "WGs": ["RD"],
    "STREAMS": ["Leptonic"],
}

from Gaudi.Configuration import *
from GaudiConfUtils.ConfigurableGenerators import (
    FilterDesktop,
    CombineParticles,
    DaVinci__N3BodyDecays,
)
from PhysSelPython.Wrappers import (
    Selection,
    DataOnDemand,
    MergedSelection,
    AutomaticData,
)
from StrippingConf.StrippingLine import StrippingLine
from StrippingUtils.Utils import LineBuilder


class Bu2LLKConf(LineBuilder):
    """
    Builder for R_X measurements
    """

    # now just define keys. Default values are fixed later
    __configuration_keys__ = (
        "DaughterPT",
        # "KaonPT",
        # "DiHadronMass",
        # "KaonIPCHI2",
    )

    def __init__(self, name, config):
        LineBuilder.__init__(self, name, config)

        self._name = name

        eeXLine_name = name + "_ee"

        from StandardParticles import StdLooseElectrons as Electrons
        from StandardParticles import StdLooseMuons as Muons
        from StandardParticles import StdLoosePions as Pions
        from StandardParticles import StdLooseKaons as Kaons

        # 1 : Make K, Ks, K*, K1, Phi and Lambdas

        SelElectrons = self._filterHadron(
            name="ElectronsFor" + self._name, sel=Electrons, params=config
        )

        SelMuons = self._filterHadron(
            name="MuonsFor" + self._name, sel=Muons, params=config
        )

        SelPions = self._filterHadron(
            name="PionsFor" + self._name, sel=Pions, params=config
        )

        SelKaons = self._filterHadron(
            name="KaonsFor" + self._name, sel=Kaons, params=config
        )

        SelB2eeXFromTracks = self._makeB2LLX(
            eeXLine_name + "2",
            daughters=[
                SelElectrons,
                SelMuons,
                SelPions,
                SelKaons,
            ],
            params=config,
        )

        self.B2eeXFromTracksLine = StrippingLine(
            eeXLine_name + "Line2",
            prescale=1,
            postscale=1,
            selection=SelB2eeXFromTracks,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.registerLine(self.B2eeXFromTracksLine)

    #####################################################
    def _filterHadron(self, name, sel, params):
        """
        Filter for all hadronic final states
        """

        _Code = "(PT > %(DaughterPT)s *MeV)" % params

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[sel])

    #####################################################
    def _makeB2LLX(self, name, daughters, params):
        """
        CombineParticles / Selection for the B
        """

        # _Decays = [
        #     "[ B+ -> J/psi(1S) K+ ]cc",
        #     "[ B+ -> J/psi(1S) pi+ ]cc",
        #     "[ B+ -> J/psi(1S) K*(892)+ ]cc",
        #     "[ B+ -> J/psi(1S) K_1(1270)+ ]cc",
        #     "[ B0 -> J/psi(1S) KS0 ]cc",
        #     "[ B0 -> J/psi(1S) K*(892)0 ]cc",
        #     "[ B_s0 -> J/psi(1S) phi(1020) ]cc",
        #     "[ Lambda_b0 -> J/psi(1S) Lambda0 ]cc",
        #     "[ Lambda_b0 -> J/psi(1S) Lambda(1520)0 ]cc",
        # ]

        # _Decays = []
        # particle_list = ["e", "mu", "K", "pi"]
        # for particle_i in particle_list:
        #     for particle_j in particle_list:
        #         for particle_k in particle_list:
        #             _Decays.append(
        #                 "[ B+ -> %s+ %s+ %s- ]cc" % (particle_i, particle_j, particle_k)
        #             )
        _Decays = ["[ B+ -> K+ e+ e- ]cc"]

        _Combine = CombineParticles(DecayDescriptors=_Decays)

        _Merge = MergedSelection("Merge" + name, RequiredSelections=daughters)

        return Selection(name, Algorithm=_Combine, RequiredSelections=[_Merge])


#####################################################

# from StrippingArchive.Stripping34.StrippingRD.StrippingBu2LLK import (
#     Bu2LLKConf as builder,
# )
# from StrippingArchive.Stripping34.StrippingRD.StrippingBu2LLK import (
#     default_config as stripping_config,
# )
builder = Bu2LLKConf
stripping_config = default_config

mod_stripping_config = stripping_config["CONFIG"]

# # remaining cuts
# mod_stripping_config["LeptonPT"] = 250.0
# mod_stripping_config["KaonPT"] = 250.0

# # removed cuts
# mod_stripping_config["BFlightCHI2"] = 0.0
# mod_stripping_config["BDIRA"] = 0.0
# mod_stripping_config["BIPCHI2"] = 9999.0
# mod_stripping_config["BVertexCHI2"] = 9999.0
# mod_stripping_config["DiLeptonPT"] = 0.0
# mod_stripping_config["DiLeptonFDCHI2"] = 0.0
# mod_stripping_config["DiLeptonIPCHI2"] = 0.0
# # (VFASPF(VCHI2/VDOF) < 9)
# mod_stripping_config["LeptonIPCHI2"] = 0.0
# mod_stripping_config["TauPT"] = 0.0
# mod_stripping_config["TauVCHI2DOF"] = 9999.0
# mod_stripping_config["KaonIPCHI2"] = 0.0
# mod_stripping_config["KstarPVertexCHI2"] = 9999.0
# mod_stripping_config["KstarPMassWindow"] = 9999.0
# mod_stripping_config["KstarPADOCACHI2"] = 9999.0
# mod_stripping_config["DiHadronMass"] = 9999.0
# mod_stripping_config["UpperMass"] = 9999.0
# mod_stripping_config["BMassWindow"] = 9999.0
# mod_stripping_config["BMassWindowTau"] = 9999.0
# mod_stripping_config["PIDe"] = -9999.0
# mod_stripping_config["Trk_Chi2"] = 9999.0
# mod_stripping_config["Trk_GhostProb"] = 9999.0
# mod_stripping_config["K1_MassWindow_Lo"] = 0.0
# mod_stripping_config["K1_MassWindow_Hi"] = 9999.0
# mod_stripping_config["K1_VtxChi2"] = 9999.0
# mod_stripping_config["K1_SumPTHad"] = 0.0
# mod_stripping_config["K1_SumIPChi2Had"] = 0.0

builder_name = "Bu2LLK"

lb = builder(builder_name, mod_stripping_config)

stream = StrippingStream("MyStream")

cfg = config_electron

for line in lb.lines():
    if line.name() == "Stripping" + cfg["stripping_line"]:
        stream.appendLines([line])

from Configurables import ProcStatusCheck

bad_events_filter = ProcStatusCheck()

sc = StrippingConf(
    Streams=[stream],
    MaxCandidates=2000,
    AcceptBadEvents=False,
    BadEventSelection=bad_events_filter,
)

trigger_list = [
    "L0HadronDecision",
    "L0ElectronDecision",
    "L0ElectronHiDecision",
    "L0MuonDecision",
    "L0DiMuonDecision",
    "L0MuonHighDecision",
    "L0PhotonDecision",
]

path_to_look = "Phys/" + cfg["stripping_line"] + "/Particles"

from Configurables import DecayTreeTuple

tuple = DecayTreeTuple(cfg["decayname"] + "_Tuple")
tuple.Inputs = [path_to_look]
tuple.Decay = cfg["decay"]

tuple.addBranches(cfg["branches"])

tuple.ToolList += [
    "TupleToolAngles",
    "TupleToolEventInfo",
    "TupleToolPrimaries",
    "TupleToolPropertime",
    "TupleToolTrackInfo",
    "TupleToolBremInfo",
    "TupleToolRecoStats",
    "TupleToolMuonPid",
    "TupleToolMCBackgroundInfo",
    "TupleToolL0Data",
    "TupleToolANNPID",
]

tuple.ToolList += ["TupleToolPid"]
tuple.addTool(TupleToolPid, name="TupleToolPid")
tuple.TupleToolPid.Verbose = True
tuple.ToolList += ["TupleToolMCTruth"]

from Configurables import TupleToolKinematic

tuple.ToolList += ["TupleToolKinematic"]
tuple.addTool(TupleToolKinematic, name="TupleToolKinematic")
tuple.TupleToolKinematic.Verbose = True


from Configurables import DaVinci

from Configurables import EventNodeKiller

eventNodeKiller = EventNodeKiller("Stripkiller")
eventNodeKiller.Nodes = ["/Event/AllStreams", "/Event/Strip"]

DaVinci()
# DaVinci().EvtMax = -1
# DaVinci().PrintFreq = 1000
DaVinci().EvtMax = 45
DaVinci().PrintFreq = 1
DaVinci().Simulation = IS_MC
DaVinci().Lumi = not IS_MC
if IS_MC:
    DaVinci().appendToMainSequence([eventNodeKiller])
    DaVinci().appendToMainSequence([sc.sequence()])
DaVinci().UserAlgorithms = [tuple]
DaVinci().VerboseMessages = True
if year != "2018":
    DaVinci().DataType = year
else:
    DaVinci().DataType = "2017"
DaVinci().TupleFile = "DVntuple.root"

print("Local INFO    DaVinci configurations")
print("Local INFO    Data Type  = {}".format(DaVinci().DataType))
print("Local INFO    Simulation = {}".format(DaVinci().Simulation))

from GaudiConf import IOHelper

IOHelper().inputFiles(
    [
        "/eos/home-m/marshall/DL-Advocate2/Kee.dst"
        # "/eos/home-m/marshall/DL-Advocate2/00140982_00000034_7.AllStreams.dst"
    ],
    clear=True,
)
