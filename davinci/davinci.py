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

###############################################################################
# (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
__author__ = "Patrick Koppenburg, Alex Shires, Thomas Blake, Luca Pescatore, Simone Bifani, Yasmine Amhis, Paula Alvarez Cartelle, Harry Cliff, Rafael Silva Coutinho, Guido Andreassi, Mick Mulder"
__date__ = "16/06/2014"
__version__ = "$Revision: 3 $"

__all__ = ("Bu2LLKConf", "default_config")

default_config = {
    "NAME": "Bu2LLK",
    "BUILDERTYPE": "Bu2LLKConf",
    "CONFIG": {
        "BFlightCHI2": 100,
        "BDIRA": 0.9995,
        "BIPCHI2": 25,
        "BVertexCHI2": 9,
        "DiLeptonPT": 0,
        "DiLeptonFDCHI2": 16,
        "DiLeptonIPCHI2": 0,
        "LeptonIPCHI2": 9,
        "LeptonPT": 300,
        "TauPT": 0,
        "TauVCHI2DOF": 150,
        "KaonIPCHI2": 9,
        "KaonPT": 400,
        "KstarPVertexCHI2": 25,
        "KstarPMassWindow": 300,
        "KstarPADOCACHI2": 30,
        "DiHadronMass": 2600,
        "UpperMass": 5500,
        "BMassWindow": 1500,
        "BMassWindowTau": 5000,
        "PIDe": 0,
        "Trk_Chi2": 3,
        "Trk_GhostProb": 0.4,
        "K1_MassWindow_Lo": 0,
        "K1_MassWindow_Hi": 6000,
        "K1_VtxChi2": 12,
        "K1_SumPTHad": 800,
        "K1_SumIPChi2Had": 48.0,
        "Bu2eeLinePrescale": 1,
        "Bu2eeLine2Prescale": 1,
        "Bu2eeLine3Prescale": 1,
        "Bu2mmLinePrescale": 1,
        "Bu2meLinePrescale": 1,
        "Bu2meSSLinePrescale": 1,
        "Bu2mtLinePrescale": 1,
        "Bu2mtSSLinePrescale": 1,
        "RelatedInfoTools": [],
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
        "BFlightCHI2",
        "BDIRA",
        "BIPCHI2",
        "BVertexCHI2",
        "DiLeptonPT",
        "DiLeptonFDCHI2",
        "DiLeptonIPCHI2",
        "LeptonIPCHI2",
        "LeptonPT",
        "TauPT",
        "TauVCHI2DOF",
        "KaonIPCHI2",
        "KaonPT",
        "KstarPVertexCHI2",
        "KstarPMassWindow",
        "KstarPADOCACHI2",
        "DiHadronMass",
        "UpperMass",
        "BMassWindow",
        "BMassWindowTau",
        "PIDe",
        "Trk_Chi2",
        "Trk_GhostProb",
        "K1_MassWindow_Lo",
        "K1_MassWindow_Hi",
        "K1_VtxChi2",
        "K1_SumPTHad",
        "K1_SumIPChi2Had",
        "Bu2eeLinePrescale",
        "Bu2eeLine2Prescale",
        "Bu2eeLine3Prescale",
        "Bu2mmLinePrescale",
        "Bu2meLinePrescale",
        "Bu2meSSLinePrescale",
        "Bu2mtLinePrescale",
        "Bu2mtSSLinePrescale",
        "RelatedInfoTools",
    )

    def __init__(self, name, config):
        LineBuilder.__init__(self, name, config)

        self._name = name

        eeXLine_name = name + "_ee"

        from StandardParticles import StdLoosePions as Pions
        from StandardParticles import StdLooseKaons as Kaons
        from StandardParticles import StdLooseKstar2Kpi as Kstars
        from StandardParticles import StdLoosePhi2KK as Phis
        from StandardParticles import StdLooseKsLL as KshortsLL
        from StandardParticles import StdLooseKsDD as KshortsDD
        from StandardParticles import StdLooseLambdaLL as LambdasLL
        from StandardParticles import StdLooseLambdaDD as LambdasDD
        from StandardParticles import StdLooseLambdastar2pK as Lambdastars

        # 1 : Make K, Ks, K*, K1, Phi and Lambdas

        SelKaons = self._filterHadron(
            name="KaonsFor" + self._name, sel=Kaons, params=config
        )

        SelPions = self._filterHadron(
            name="PionsFor" + self._name, sel=Pions, params=config
        )

        SelKshortsLL = self._filterHadron(
            name="KshortsLLFor" + self._name, sel=KshortsLL, params=config
        )

        SelKshortsDD = self._filterHadron(
            name="KshortsDDFor" + self._name, sel=KshortsDD, params=config
        )

        SelKstars = self._filterHadron(
            name="KstarsFor" + self._name, sel=Kstars, params=config
        )

        SelKstarsPlusLL = self._makeKstarPlus(
            name="KstarsPlusLLFor" + self._name,
            kshorts=KshortsLL,
            pions=Pions,
            params=config,
        )

        SelKstarsPlusDD = self._makeKstarPlus(
            name="KstarsPlusDDFor" + self._name,
            kshorts=KshortsDD,
            pions=Pions,
            params=config,
        )

        SelK1s = self._makeK1(
            name="K1For" + self._name, kaons=Kaons, pions=Pions, params=config
        )

        SelPhis = self._filterHadron(
            name="PhisFor" + self._name, sel=Phis, params=config
        )

        SelLambdasLL = self._filterHadron(
            name="LambdasLLFor" + self._name, sel=LambdasLL, params=config
        )

        SelLambdasDD = self._filterHadron(
            name="LambdasDDFor" + self._name, sel=LambdasDD, params=config
        )

        SelLambdastars = self._filterHadron(
            name="LambdastarsFor" + self._name, sel=Lambdastars, params=config
        )
        from StandardParticles import StdDiElectronFromTracks as DiElectronsFromTracks

        DiElectronID = "(PIDe > -9999.)" % config

        SelDiElectronFromTracks = self._filterDiLepton(
            "SelDiElectronFromTracksFor" + self._name,
            dilepton=DiElectronsFromTracks,
            params=config,
            idcut=DiElectronID,
        )

        SelB2eeXFromTracks = self._makeB2LLX(
            eeXLine_name + "2",
            dilepton=SelDiElectronFromTracks,
            hadrons=[
                SelPions,
                SelKaons,
                SelKstars,
                SelPhis,
                SelKshortsLL,
                SelKshortsDD,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelKstarsPlusLL,
                SelKstarsPlusDD,
                SelK1s,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        SPDFilter = {
            "Code": " ( recSummary(LHCb.RecSummary.nSPDhits,'Raw/Spd/Digits') < 600 )",
            "Preambulo": [
                "from LoKiNumbers.decorators import *",
                "from LoKiCore.basic import LHCb",
            ],
        }

        self.B2eeXFromTracksLine = StrippingLine(
            eeXLine_name + "Line2",
            prescale=config["Bu2eeLine2Prescale"],
            postscale=1,
            selection=SelB2eeXFromTracks,
            RelatedInfoTools=config["RelatedInfoTools"],
            # FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.registerLine(self.B2eeXFromTracksLine)

    #####################################################
    def _filterHadron(self, name, sel, params):
        """
        Filter for all hadronic final states
        """

        # requires all basic particles to have IPCHI2 > KaonIPCHI2
        # and hadron PT > KaonPT
        # need to add the ID here
        _Code = (
            "(PT > %(KaonPT)s *MeV) & "
            "(M < %(DiHadronMass)s*MeV) & "
            "((ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s)) | "
            "(NDAUGHTERS == NINTREE(ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s))))"
            % params
        )
        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[sel])

    #####################################################
    def _filterDiLepton(self, name, dilepton, params, idcut=None):
        """
        Handy interface for dilepton filter
        """

        _Code = (
            "(ID=='J/psi(1S)') & "
            "(PT > %(DiLeptonPT)s *MeV) & "
            "(MM < %(UpperMass)s *MeV) & "
            "(MINTREE(ABSID<14,PT) > %(LeptonPT)s *MeV) & "
            "(MINTREE(ABSID<14,MIPCHI2DV(PRIMARY)) > %(LeptonIPCHI2)s) & "
            # "(VFASPF(VCHI2/VDOF) < 9) & (BPVVDCHI2 > %(DiLeptonFDCHI2)s) & "
            "(BPVVDCHI2 > %(DiLeptonFDCHI2)s) & "
            "(MIPCHI2DV(PRIMARY) > %(DiLeptonIPCHI2)s)" % params
        )

        # add additional cut on PID if requested
        if idcut:
            _Code += " & " + idcut

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[dilepton])

    #####################################################
    def _makeKstarPlus(self, name, kshorts, pions, params):
        """
        Make a K*(892)+ -> KS0 pi+
        """

        _Decays = "[K*(892)+ -> KS0 pi+]cc"

        _CombinationCut = (
            "(APT > %(KaonPT)s *MeV) & "
            "(ADAMASS('K*(892)+') < %(KstarPMassWindow)s *MeV) & "
            "(ADOCACHI2CUT( %(KstarPADOCACHI2)s  , ''))" % params
        )

        _MotherCut = "(VFASPF(VCHI2) < %(KstarPVertexCHI2)s)" % params

        _KshortCut = (
            "(PT > %(KaonPT)s *MeV) & "
            "(M < %(DiHadronMass)s*MeV) & "
            "(NDAUGHTERS == NINTREE(ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s)))"
            % params
        )

        _PionCut = (
            "(PT > %(KaonPT)s *MeV) & "
            "(M < %(DiHadronMass)s*MeV) & "
            "(ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s))" % params
        )

        _Combine = CombineParticles(
            DecayDescriptor=_Decays,
            CombinationCut=_CombinationCut,
            MotherCut=_MotherCut,
        )

        _Combine.DaughtersCuts = {"KS0": _KshortCut, "pi+": _PionCut}

        return Selection(name, Algorithm=_Combine, RequiredSelections=[kshorts, pions])

    #####################################################
    def _makeK1(self, name, kaons, pions, params):
        """
        Make a K1 -> K+pi-pi+
        """

        _Decays = "[K_1(1270)+ -> K+ pi+ pi-]cc"

        # define all the cuts
        _K1Comb12Cuts = (
            "(AM > %(K1_MassWindow_Lo)s*MeV) & (AM < %(K1_MassWindow_Hi)s*MeV)" % params
        )
        _K1CombCuts = (
            "(AM > %(K1_MassWindow_Lo)s*MeV) & (AM < %(K1_MassWindow_Hi)s*MeV) & ((APT1+APT2+APT3) > %(K1_SumPTHad)s*MeV)"
            % params
        )

        _K1MotherCuts = (
            "(VFASPF(VCHI2) < %(K1_VtxChi2)s) & (SUMTREE(MIPCHI2DV(PRIMARY),((ABSID=='K+') | (ABSID=='K-') | (ABSID=='pi+') | (ABSID=='pi-')),0.0) > %(K1_SumIPChi2Had)s)"
            % params
        )
        _daughtersCuts = (
            "(TRCHI2DOF < %(Trk_Chi2)s) & (TRGHOSTPROB < %(Trk_GhostProb)s)" % params
        )

        _Combine = DaVinci__N3BodyDecays()

        _Combine.DecayDescriptor = _Decays

        _Combine.DaughtersCuts = {"K+": _daughtersCuts, "pi+": _daughtersCuts}

        _Combine.Combination12Cut = _K1Comb12Cuts
        _Combine.CombinationCut = _K1CombCuts
        _Combine.MotherCut = _K1MotherCuts

        # make and store the Selection object
        return Selection(name, Algorithm=_Combine, RequiredSelections=[kaons, pions])

    #####################################################
    def _makeB2LLX(
        self, name, dilepton, hadrons, params, masscut="(ADAMASS('B+')< 1500 *MeV"
    ):
        """
        CombineParticles / Selection for the B
        """

        _Decays = [
            "[ B+ -> J/psi(1S) K+ ]cc",
            "[ B+ -> J/psi(1S) pi+ ]cc",
            "[ B+ -> J/psi(1S) K*(892)+ ]cc",
            "[ B+ -> J/psi(1S) K_1(1270)+ ]cc",
            "[ B0 -> J/psi(1S) KS0 ]cc",
            "[ B0 -> J/psi(1S) K*(892)0 ]cc",
            "[ B_s0 -> J/psi(1S) phi(1020) ]cc",
            "[ Lambda_b0 -> J/psi(1S) Lambda0 ]cc",
            "[ Lambda_b0 -> J/psi(1S) Lambda(1520)0 ]cc",
        ]

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
            name, Algorithm=_Combine, RequiredSelections=[dilepton, _Merge]
        )


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

# remaining cuts
mod_stripping_config["LeptonPT"] = 250.0
mod_stripping_config["KaonPT"] = 250.0

# removed cuts
mod_stripping_config["BFlightCHI2"] = 0.0
mod_stripping_config["BDIRA"] = 0.0
mod_stripping_config["BIPCHI2"] = 9999.0
mod_stripping_config["BVertexCHI2"] = 9999.0
mod_stripping_config["DiLeptonPT"] = 0.0
mod_stripping_config["DiLeptonFDCHI2"] = 0.0
mod_stripping_config["DiLeptonIPCHI2"] = 0.0
# (VFASPF(VCHI2/VDOF) < 9)
mod_stripping_config["LeptonIPCHI2"] = 0.0
mod_stripping_config["TauPT"] = 0.0
mod_stripping_config["TauVCHI2DOF"] = 9999.0
mod_stripping_config["KaonIPCHI2"] = 0.0
mod_stripping_config["KstarPVertexCHI2"] = 9999.0
mod_stripping_config["KstarPMassWindow"] = 9999.0
mod_stripping_config["KstarPADOCACHI2"] = 9999.0
mod_stripping_config["DiHadronMass"] = 9999.0
mod_stripping_config["UpperMass"] = 9999.0
mod_stripping_config["BMassWindow"] = 9999.0
mod_stripping_config["BMassWindowTau"] = 9999.0
mod_stripping_config["PIDe"] = -9999.0
mod_stripping_config["Trk_Chi2"] = 9999.0
mod_stripping_config["Trk_GhostProb"] = 9999.0
mod_stripping_config["K1_MassWindow_Lo"] = 0.0
mod_stripping_config["K1_MassWindow_Hi"] = 9999.0
mod_stripping_config["K1_VtxChi2"] = 9999.0
mod_stripping_config["K1_SumPTHad"] = 0.0
mod_stripping_config["K1_SumIPChi2Had"] = 0.0

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
        # '/eos/home-m/marshall/DL-Advocate2/Kee.dst'
        "/eos/home-m/marshall/DL-Advocate2/00140982_00000034_7.AllStreams.dst"
    ],
    clear=True,
)
