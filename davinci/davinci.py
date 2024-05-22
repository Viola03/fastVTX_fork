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
    # "decay": "[B+ -> ^(J/psi(1S)->^e+ ^e-) ^K+]CC",
    # "branches": {
    #     "B_plus": "[ B+ -> (J/psi(1S)->e+ e-)  K+]CC",
    #     "K_Kst": "[ B+ -> (J/psi(1S)->e+ e-) ^K+]CC",
    #     "e_minus": "[(B+ -> (J/psi(1S)->e+ ^e-) K+), (B- -> (J/psi(1S)->^e- e+) K-)]",
    #     "e_plus": "[(B+ -> (J/psi(1S)->^e+ e-) K+), (B- -> (J/psi(1S)->e- ^e+) K-)]",
    #     "J_psi_1S": "[ B+ -> ^(J/psi(1S)->e+ e-) K+]CC",
    # },
    # "decay": "[Beauty -> X+ X+ X-]CC",
    # "branches": {
    #     "B_plus": "[Beauty -> X+ X+ X-]CC",
    # },
    "decay": "[B+ -> ^K+ ^e+ ^e-]CC",
    "branches": {
        "M": "[B+ -> K+ e+ e-]CC",
        "A": "[B+ -> ^K+ e+ e-]CC",
        "B": "[B+ -> K+ ^e+ e-]CC",
        "C": "[B+ -> K+ e+ ^e-]CC",
    },
}

from StrippingConf.Configuration import StrippingConf, StrippingStream

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

__all__ = ("Bu2LLKConf", "default_config")

default_config = {
    "NAME": "Bu2LLK",
    "BUILDERTYPE": "Bu2LLKConf",
    "CONFIG": {
        "DaughterPT":250.0,
        "UpperMass": 99999.0,
        "BMassWindow":99999.0,
        "BVertexCHI2": 9999.0,
        "BIPCHI2": 9999.0,
        "BDIRA": 0.0,
        "BFlightCHI2": 0.0,
    },
    "WGs": ["RD"],
    "STREAMS": ["Leptonic"],
}

from Gaudi.Configuration import *
from GaudiConfUtils.ConfigurableGenerators import (
    FilterDesktop,
    CombineParticles,
)
from PhysSelPython.Wrappers import (
    Selection,
    MergedSelection,
)
from StrippingConf.StrippingLine import StrippingLine
from StrippingUtils.Utils import LineBuilder


class Bu2LLKConf(LineBuilder):
    """
    Builder for R_X measurements
    """
    __configuration_keys__ = (
        "DaughterPT",
        "UpperMass",
        "BMassWindow",
        "BVertexCHI2",
        "BIPCHI2",
        "BDIRA",
        "BFlightCHI2",
    )

    def __init__(self, name, config):
        LineBuilder.__init__(self, name, config)

        self._name = name

        eeXLine_name = name + "_ee"

        from StandardParticles import StdLooseElectrons as Electrons
        from StandardParticles import StdLooseMuons as Muons
        from StandardParticles import StdLoosePions as Pions
        from StandardParticles import StdLooseKaons as Kaons

        SelElectrons = self._filterHadron(
            name="ElectronsFor" + self._name, sel=Electrons, params=config
        )

        SelMuons = self._filterHadron(
            name="MuonsFor" + self._name, sel=Muons, params=config
        )

        SelKaons = self._filterHadron(
            name="KaonsFor" + self._name, sel=Kaons, params=config
        )

        SelPions = self._filterHadron(
            name="PionsFor" + self._name, sel=Pions, params=config
        )

        SelB2eeXFromTracks = self._makeB2LLX(
            eeXLine_name + "2",
            hadrons=[
                SelElectrons,
                SelMuons,
                SelPions,
                SelKaons,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
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
        _Code = (
            "(PT > %(DaughterPT)s *MeV)" % params
        )

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[sel])

    
    #####################################################
    def _makeB2LLX(
        self, name, hadrons, params, masscut
    ):
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

        # _Decays = [
        #     "[ B+ -> J/psi(1S) K+ ]cc",
        # ]

        # _Decays = []
        # particle_list = ["e", "mu", "K", "pi"]
        # for particle_i in particle_list:
        #     for particle_j in particle_list:
        #         for particle_k in particle_list:
        #             _Decays.append(
        #                 "[ B+ -> %s+ %s+ %s- ]cc" % (particle_i, particle_j, particle_k)
        #             )
        # _Decays = ["[ B+ -> K+ e+ e- ]cc", "[ B+ -> K+ mu+ mu- ]cc", "[ B+ -> K+ pi+ pi- ]cc"] # PID cuts in StdLoose are wide enough to avoid all patterns and avoid repeated candidates?
        _Decays = ["[ B+ -> K+ e+ e- ]cc"] # PID cuts in StdLoose are wide enough to avoid all patterns and avoid repeated candidates?


        _Cut = (
            "((VFASPF(VCHI2/VDOF) < %(BVertexCHI2)s) "
            "& (BPVIPCHI2() < %(BIPCHI2)s) "
            "& (BPVDIRA > %(BDIRA)s) "
            "& (BPVVDCHI2 > %(BFlightCHI2)s))" % params
        )

        _Combine = CombineParticles(
            DecayDescriptors=_Decays, CombinationCut=masscut, MotherCut=_Cut
        )

        _Merge = MergedSelection("Merge" + name, RequiredSelections=hadrons)

        return Selection(
            # name, Algorithm=_Combine, RequiredSelections=[dilepton, _Merge]
            name, Algorithm=_Combine, RequiredSelections=[_Merge]
        )

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

builder = Bu2LLKConf
stripping_config = default_config

mod_stripping_config = stripping_config["CONFIG"]

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
    MaxCandidates=4000,
    AcceptBadEvents=False,
    BadEventSelection=bad_events_filter,
)


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
    # "TupleToolBremInfo",
    "TupleToolRecoStats",
    # "TupleToolMuonPid",
    "TupleToolMCBackgroundInfo",
    # "TupleToolL0Data",
    # "TupleToolANNPID",
]

# tuple.ToolList += ["TupleToolPid"]
# tuple.addTool(TupleToolPid, name="TupleToolPid")
# tuple.TupleToolPid.Verbose = True
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
# DaVinci().EvtMax = 200
DaVinci().EvtMax = 25
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
        # "/eos/home-m/marshall/DL-Advocate2/Kee.dst"
        # "/eos/home-m/marshall/DL-Advocate2/00140982_00000034_7.AllStreams.dst"
        "/afs/cern.ch/work/m/marshall/fast_vertexing_variables/davinci/00113947_00000003_7.AllStreams.dst"
    ],
    clear=True,
)
