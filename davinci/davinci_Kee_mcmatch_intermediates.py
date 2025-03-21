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

negative_particles = [11, 13, -211, -321]
positive_particles = [-11, -13, 211, 321]
particle_dict = {
11:"e-", -11:"e+",
13:"mu-", -13:"mu+",
-211:"pi-", 211:"pi+",
-321:"K-", 321:"K+",
}

################################################################################
B_string = 'B+'

full_list_of_decays = []
config_electron = []
full_list_of_decays.append("[ %s -> %s %s %s ]cc"%(B_string,particle_dict[321],particle_dict[-11],particle_dict[11]))
full_list_of_decays.append("[ %s -> %s J/psi(1S) ]cc"%(B_string,particle_dict[321]))
# full_list_of_decays = ["[ B+ -> K+ e+ e- ]cc", "[ B+ -> K+ J/psi(1S) ]cc"]

config_electron.append({
    "stripping_line": "Bu2LLK_eeLine2",
    "decayname": "%s_%sJpsi(%s%s)"%(B_string,particle_dict[321],particle_dict[-11],particle_dict[11]),
    'decay': "[B+ -> ^(J/psi(1S)->^e+ ^e-) ^K+]CC",
    'branches': {
        "MOTHER": "[ B+ -> (J/psi(1S)->e+ e-)  K+]CC",
        "DAUGHTER1": "[ B+ -> (J/psi(1S)->e+ e-) ^K+]CC",
        "DAUGHTER2": "[(B+ -> (J/psi(1S)->e+ ^e-) K+), (B- -> (J/psi(1S)->^e- e+) K-)]",
        "DAUGHTER3": "[(B+ -> (J/psi(1S)->^e+ e-) K+), (B- -> (J/psi(1S)->e- ^e+) K-)]",
        "INTERMEDIATE": "[ B+ -> ^(J/psi(1S)->e+ e-) K+]CC"},
        'intermediate_daughters':[particle_dict[-11],particle_dict[11]],
        'full_list_of_decays_i':full_list_of_decays,
})
################################################################################


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
# make new stripping line with mcMATCH for each decay string

from Gaudi.Configuration import *
from Configurables import CombineParticles
from CommonParticles.Utils import *
import GaudiKernel.SystemOfUnits as Units
from Configurables            import DataOnDemandSvc

_particles = {}

def updateDoD ( alg , hat = 'Phys/' ) :
    """
    Update Data-On-Demand service
    """
    _parts = { hat+alg.name()+'/Particles' : alg } 
    _particles.update ( _parts ) 
    
    dod = DataOnDemandSvc()
    dod.AlgMap.update(
        { hat + alg.name() + '/Particles' : alg.getFullName() }
        )
    return _parts 
    

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

    def __init__(self, name, config, intermediate_daughters, full_list_of_decays_i):
        LineBuilder.__init__(self, name, config)

        self._name = name

        eeXLine_name = name + "_ee"

        # # outdated
        # from StandardParticles import StdLooseElectrons as Electrons
        # from StandardParticles import StdLooseMuons as Muons
        # from StandardParticles import StdLoosePions as Pions
        # from StandardParticles import StdLooseKaons as Kaons

        # # Use to get closer to davinci_intermediates
        from StandardParticles import StdAllLooseElectrons as Electrons
        from StandardParticles import StdAllLooseMuons as Muons
        from StandardParticles import StdAllLoosePions as Pions
        from StandardParticles import StdAllLooseKaons as Kaons

        # # # Even more candidates? - might be taking a long long time to run
        # from StandardParticles import StdAllNoPIDsElectrons as Electrons
        # from StandardParticles import StdAllNoPIDsMuons as Muons
        # from StandardParticles import StdAllNoPIDsPions as Pions
        # from StandardParticles import StdAllNoPIDsKaons as Kaons

        SelElectrons = self._filterHadron(
            name="ElectronsFor" + self._name, sel=Electrons, params=config, mcMatch="e"
        )

        SelMuons = self._filterHadron(
            name="MuonsFor" + self._name, sel=Muons, params=config, mcMatch="mu"
        )

        SelKaons = self._filterHadron(
            name="KaonsFor" + self._name, sel=Kaons, params=config, mcMatch="K"
        )

        SelPions = self._filterHadron(
            name="PionsFor" + self._name, sel=Pions, params=config, mcMatch="pi"
        )

        from Configurables import CombineParticles
        
        _CombineParticles = CombineParticles(
            "StdLooseIntermediate"+self._name,
            DecayDescriptor='J/psi(1S) -> %s %s'%(intermediate_daughters[0], intermediate_daughters[1]),
            CombinationCut="(ADOCACHI2CUT(99999, ''))",
            MotherCut="(VFASPF(VCHI2) < 99999)",
            ) 

        RequiredSelections_list = []
        if 'e+' in intermediate_daughters or 'e-' in intermediate_daughters:
            RequiredSelections_list.append(SelElectrons)
        if 'mu+' in intermediate_daughters or 'mu-' in intermediate_daughters:
            RequiredSelections_list.append(SelMuons)
        if 'K+' in intermediate_daughters or 'K-' in intermediate_daughters:
            RequiredSelections_list.append(SelKaons)
        if 'pi+' in intermediate_daughters or 'pi-' in intermediate_daughters:
            RequiredSelections_list.append(SelPions)

        intermediate_combinations = Selection(
            'Sel_StdLooseIntermediate'+self._name,
            Algorithm=_CombineParticles,
            RequiredSelections=RequiredSelections_list
        )


        SelB2eeXFromTracks = self._makeB2LLX(
            eeXLine_name + "_Generalised",
            intermediate_combinations,
            hadrons=[
                SelElectrons,
                SelMuons,
                SelPions,
                SelKaons,
            ],
            full_list_of_decays_i=full_list_of_decays_i,
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        self.B2eeXFromTracksLine = StrippingLine(
            eeXLine_name + "Line_Generalised",
            prescale=1,
            postscale=1,
            selection=SelB2eeXFromTracks,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.registerLine(self.B2eeXFromTracksLine)


    #####################################################
    def _filterHadron(self, name, sel, params, mcMatch=''):
        """
        Filter for all hadronic final states
        """
        if mcMatch != '':
            params["mcMatch"] = mcMatch
            _Code = (
                "(PT > %(DaughterPT)s *MeV)"
                " & (mcMatch('%(mcMatch)s+') | mcMatch('%(mcMatch)s-'))" % params
            )
        
        else:
            _Code = (
                "(PT > %(DaughterPT)s *MeV)" % params
            )


        _Filter = FilterDesktop(Preambulo = ["from LoKiPhysMC.decorators import *","from LoKiPhysMC.functions import mcMatch"], Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[sel])


    #####################################################
    def _makeB2LLX(
        self, name, intermediates, hadrons, full_list_of_decays_i, params, masscut
    ):
        """
        CombineParticles / Selection for the B
        """

        _Decays = full_list_of_decays_i
        
        _Cut = (
            "((mcMatch('B+')) | (mcMatch('B-')) | (mcMatch('B0')) | (mcMatch('B~0')) | (mcMatch('B_s0')) | (mcMatch('B_s~0'))  | (mcMatch('B_c+')) | (mcMatch('B_c-')))"
        )

        _Combine = CombineParticles(
            'Generalised_combine_Particles',
            Preambulo = ["from LoKiPhysMC.decorators import *","from LoKiPhysMC.functions import mcMatch"],
            DecayDescriptors=_Decays, CombinationCut=masscut, MotherCut=_Cut
        )

        _Merge = MergedSelection("Merge" + name, RequiredSelections=hadrons)

        return Selection(
            name, Algorithm=_Combine, RequiredSelections=[intermediates, _Merge]
        )

###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

from Configurables import ProcStatusCheck


from Configurables import DecayTreeTuple

tuples = []
sc = []
for config_electron_i in config_electron:
    name = config_electron_i["decayname"]
    tuple_i = DecayTreeTuple(name + "_Tuple")
    print('\n\n\n')
    print(name, config_electron_i["intermediate_daughters"])
    mod_stripping_config = default_config["CONFIG"]
    builder_name = "Bu2LLK_%s"%name
    try:
        del mod_stripping_config['mcMatch']
    except:
        pass
    lb = GeneralConf(builder_name, mod_stripping_config, intermediate_daughters = config_electron_i["intermediate_daughters"], full_list_of_decays_i=config_electron_i["full_list_of_decays_i"])
    stream = StrippingStream("MyStream_%s"%name)

    stripping_line = 'Bu2LLK_%s_eeLine_Generalised'%name
    for line in lb.lines():
        if line.name() == "Stripping" + stripping_line:
            stream.appendLines([line])
    bad_events_filter = ProcStatusCheck()
    sc_i = StrippingConf(
        name='Conf_%s'%name,
        Streams=[stream],
        MaxCandidates=4000,
        AcceptBadEvents=False,
        BadEventSelection=bad_events_filter,
    )
    sc.append(sc_i)

    tuple_i.Inputs = ["Phys/%s/Particles"%stripping_line]
    tuples.append(tuple_i)


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
DaVinci().EvtMax = 250
DaVinci().PrintFreq = 25
DaVinci().Simulation = IS_MC
DaVinci().Lumi = not IS_MC
DaVinci().UserAlgorithms = []
if IS_MC:
    list_to_appendToMainSequence = []
    for idx, sc_i in enumerate(sc):
        DaVinci().appendToMainSequence([eventNodeKiller])
        DaVinci().appendToMainSequence([sc_i.sequence()])
        DaVinci().UserAlgorithms += [tuples[idx]]

DaVinci().VerboseMessages = True
# DaVinci().VerboseMessages = False
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
        "/afs/cern.ch/work/m/marshall/fast_vertexing_variables/davinci/Kee.dst"
        # "/afs/cern.ch/work/m/marshall/fast_vertexing_variables/davinci/Kmumu.dst"
    ],
    clear=True,
)
