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

import numpy as np

stripping_line = "Bu2LLK_eeLine_Generalised"


negative_particles = [11, 13, -211, -321]
positive_particles = [-11, -13, 211, 321]
particle_dict = {
11:"e-", -11:"e+",
13:"mu-", -13:"mu+",
-211:"pi-", 211:"pi+",
-321:"K-", 321:"K+",
}
combinations = np.empty((0,3))
for p1 in positive_particles:
    for p2 in positive_particles:
        for n1 in negative_particles:
            combinations = np.append(combinations, [[p1,p2,n1]], axis=0)
print(np.shape(combinations))
# Sort each row to make the order of particles irrelevant
sorted_combinations = np.sort(combinations, axis=1)
# Extract unique rows
unique_combinations = np.unique(sorted_combinations, axis=0)
print(np.shape(unique_combinations))
full_list_of_decays = []
for idx, combination in enumerate(unique_combinations):
    full_list_of_decays.append("[ B+ -> %s %s %s ]cc"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]))





config_electron = []

for idx, combination in enumerate(unique_combinations):

    config_electron.append({
        "decayname": "B_%s%s%s"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),
        'decay': "[B+ -> ^%s ^%s ^%s]CC"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),
        'branches': {
            "MOTHER": "[ B+ -> %s %s %s]CC"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),
            "DAUGHTER1": "[ B+ ->  ^%s %s %s]CC"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),
            "DAUGHTER2": "[B+ ->  %s ^%s %s]CC"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),
            "DAUGHTER3": "[B+ ->  %s %s ^%s]CC"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]])},
    })

from StrippingConf.Configuration import StrippingConf, StrippingStream


__all__ = ("Bu2LLKConf", "default_config")

default_config = {
    "CONFIG": {
        "DaughterPT":250.0,
        "UpperMass": 99999.0,
        "BMassWindow":99999.0,
        "BVertexCHI2": 9999.0,
        "BIPCHI2": 9999.0,
        "BDIRA": 0.0,
        "BFlightCHI2": 0.0,
        "DiLeptonPT": 0.0,
    },
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


###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

class GeneralConf(LineBuilder):
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
        "DiLeptonPT",
    )

    def __init__(self, name, config):
        LineBuilder.__init__(self, name, config)

        self._name = name

        eeXLine_name = name + "_ee"
        # eeXLine_name = name

        from StandardParticles import StdLooseElectrons as Electrons
        from StandardParticles import StdLooseMuons as Muons
        from StandardParticles import StdLoosePions as Pions
        from StandardParticles import StdLooseKaons as Kaons

        # from StandardParticles import StdAllNoPIDsElectrons as Electrons
        # from StandardParticles import StdAllNoPIDsMuons as Muons
        # from StandardParticles import StdAllNoPIDsPions as Pions
        # from StandardParticles import StdAllNoPIDsKaons as Kaons

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
            eeXLine_name + "_Generalised",
            # eeXLine_name,
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
            eeXLine_name + "Line_Generalised",
            # eeXLine_name + "Line",
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

    
    # #####################################################
    # def _filterDiLepton( self, name, dilepton, params ) :
    #     """
    #     Handy interface for dilepton filter
    #     """

    #     _Code = (
    #         "(PT > %(DiLeptonPT)s *MeV)" % params
    #     )

    #     _Filter = FilterDesktop( Code = _Code )

    #     return Selection(name, Algorithm = _Filter, RequiredSelections = [ dilepton ] )


    #####################################################
    def _makeB2LLX(
        self, name, hadrons, params, masscut
    ):
        """
        CombineParticles / Selection for the B
        """

        _Decays = full_list_of_decays

        _Cut = (
            "((mcMatch('B+')) | (mcMatch('B-')) | (mcMatch('B0')) | (mcMatch('B~0')) | (mcMatch('B_s0')) | (mcMatch('B_s~0'))  | (mcMatch('B_c+')) | (mcMatch('B_c-')))"
        )

        _Combine = CombineParticles(
            Preambulo = ["from LoKiPhysMC.decorators import *","from LoKiPhysMC.functions import mcMatch"],
            DecayDescriptors=_Decays, CombinationCut=masscut, MotherCut=_Cut
        )

        _Merge = MergedSelection("Merge" + name, RequiredSelections=hadrons)

        return Selection(
            name, Algorithm=_Combine, RequiredSelections=[_Merge]
        )

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################


builder = GeneralConf
stripping_config = default_config

mod_stripping_config = stripping_config["CONFIG"]

builder_name = "Bu2LLK"
# builder_name = "B2XMuMu"
# builder_name = "General"

lb = builder(builder_name, mod_stripping_config)

stream = StrippingStream("MyStream")

for line in lb.lines():
    print(line.name())
    if line.name() == "Stripping" + stripping_line:
        stream.appendLines([line])

from Configurables import ProcStatusCheck

bad_events_filter = ProcStatusCheck()

sc = StrippingConf(
    Streams=[stream],
    MaxCandidates=4000,
    AcceptBadEvents=False,
    BadEventSelection=bad_events_filter,
)

from Configurables import DecayTreeTuple

tuples = []
for config_electron_i in config_electron:
    tuples.append(DecayTreeTuple(config_electron_i["decayname"] + "_Tuple"))

for tuple in tuples:
    tuple.Inputs = ["Phys/Bu2LLK_eeLine_Generalised/Particles"]

for idx, config_electron_i in enumerate(config_electron):
    tuples[idx].Decay = config_electron_i["decay"]
    tuples[idx].addBranches(config_electron_i["branches"])

from Configurables import TupleToolMCTruth
from Configurables import TupleToolKinematic

mc_truth = TupleToolMCTruth()
mc_truth.ToolList = ["MCTupleToolKinematic", "MCTupleToolHierarchy"]

for tuple in tuples:
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

    tuple.ToolList += ["TupleToolMCTruth"]
    tuple.ToolList += ["MCTupleToolEventType"]


    tuple.addTool(mc_truth)
    tuple.TupleToolMCTruth.Verbose = True

    # MCTupleToolHierarchyOtherTracks
    # https://gitlab.cern.ch:8443/lhcb/Analysis/-/merge_requests/707/diffs#diff-content-279d6abf536ade7c3044e046262dba6dd103778c


    tuple.ToolList += ["TupleToolKinematic"]
    tuple.addTool(TupleToolKinematic, name="TupleToolKinematic")
    tuple.TupleToolKinematic.Verbose = True


from Configurables import DaVinci

from Configurables import EventNodeKiller

eventNodeKiller = EventNodeKiller("Stripkiller")
eventNodeKiller.Nodes = ["/Event/AllStreams", "/Event/Strip"]

DaVinci()
# DaVinci().EvtMax = -1
# DaVinci().PrintFreq = 250
DaVinci().EvtMax = 100
DaVinci().PrintFreq = 25
DaVinci().Simulation = IS_MC
DaVinci().Lumi = not IS_MC
if IS_MC:
    DaVinci().appendToMainSequence([eventNodeKiller])
    DaVinci().appendToMainSequence([sc.sequence()])
# DaVinci().UserAlgorithms = [tuple]
DaVinci().UserAlgorithms = tuples
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
        # "/afs/cern.ch/work/m/marshall/fast_vertexing_variables/davinci/00113947_00000003_7.AllStreams.dst"
        "/afs/cern.ch/work/m/marshall/fast_vertexing_variables/davinci/Kee.dst"
        # "/afs/cern.ch/work/m/marshall/fast_vertexing_variables/davinci/Kmumu.dst"
    ],
    clear=True,
)
