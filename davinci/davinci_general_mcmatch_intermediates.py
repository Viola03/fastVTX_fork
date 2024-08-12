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
combinations = np.empty((0,3))
for p1 in positive_particles:
    for p2 in positive_particles:
        for n1 in negative_particles:
            combinations = np.append(combinations, [[p1,p2,n1]], axis=0)
            combinations = np.append(combinations, [[-1*p1,-1*p2,-1*n1]], axis=0)
            
sorted_combinations = np.sort(combinations, axis=1)
unique_combinations = np.unique(sorted_combinations, axis=0)

full_list_of_decays = []
for idx, combination in enumerate(combinations):
    # full_list_of_decays.append(
    # ["[ B+ -> %s %s %s ]cc"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),
    #                "[ B+ -> %s J/psi(1S) ]cc"%(particle_dict[combination[0]])])
    # full_list_of_decays.append( "[ B+ -> %s J/psi(1S) ]cc"%(particle_dict[combination[0]]))
    full_list_of_decays.append(["B+ -> %s J/psi(1S)"%(particle_dict[combination[0]])])
    

config_electron = []

for idx, combination in enumerate(combinations):

    # config_electron.append({
    #     "decayname": "B_%sJpsi(%s%s)"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),

    #     'decay': "[B+ -> ^(J/psi(1S)->^%s ^%s) ^%s]CC"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #     'branches': {
    #         "MOTHER": "[ B+ -> (J/psi(1S)->%s %s)  %s]CC"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #         "DAUGHTER1": "[ B+ -> (J/psi(1S)->%s %s) ^%s]CC"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #         "DAUGHTER2": "[(B+ -> (J/psi(1S)->%s ^%s) %s), (B- -> (J/psi(1S)->^%s %s) %s)]"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]],particle_dict[combination[2]],particle_dict[combination[1]],particle_dict[combination[0]]),
    #         "DAUGHTER3": "[(B+ -> (J/psi(1S)->^%s %s) %s), (B- -> (J/psi(1S)->%s ^%s) %s)]"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]],particle_dict[combination[2]],particle_dict[combination[1]],particle_dict[combination[0]]),
    #         "INTERMEDIATE": "[ B+ -> ^(J/psi(1S)->%s %s) %s]CC"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]])},
    #     'intermediate_daughters':[particle_dict[combination[1]],particle_dict[combination[2]]],
    #     'full_list_of_decays_i':full_list_of_decays[idx],
    # })

    config_electron.append({
        "decayname": "B_%sJpsi(%s%s)"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),

        'decay': "B+ -> ^(J/psi(1S)->^%s ^%s) ^%s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
        'branches': {
            "MOTHER": "B+ -> (J/psi(1S)->%s %s)  %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
            "DAUGHTER1": "B+ -> (J/psi(1S)->%s %s) ^%s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
            "DAUGHTER2": "B+ -> (J/psi(1S)->%s ^%s) %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
            "DAUGHTER3": "B+ -> (J/psi(1S)->^%s %s) %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
            "INTERMEDIATE": "B+ -> ^(J/psi(1S)->%s %s) %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]])},
        'intermediate_daughters':[particle_dict[combination[1]],particle_dict[combination[2]]],
        'full_list_of_decays_i':full_list_of_decays[idx],
    })

    # config_electron.append({
    #     "decayname": "B_%sJpsi(%s%s)"%(particle_dict[combination[0]],particle_dict[combination[1]],particle_dict[combination[2]]),

    #     'decay': "B+ ->  ^%s ^%s ^%s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #     'branches': {
    #         "MOTHER": "B+ -> %s %s  %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #         "DAUGHTER1": "B+ -> %s %s ^%s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #         "DAUGHTER2": "B+ -> %s ^%s %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]]),
    #         "DAUGHTER3": "B+ -> ^%s %s %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]])},
    #         # "INTERMEDIATE": "B+ ->  %s %s %s"%(particle_dict[combination[1]],particle_dict[combination[2]],particle_dict[combination[0]])},
    #     'intermediate_daughters':[particle_dict[combination[1]],particle_dict[combination[2]]],
    #     'full_list_of_decays_i':full_list_of_decays[idx],
    # })

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

        from StandardParticles import StdLooseElectrons as Electrons
        from StandardParticles import StdLooseMuons as Muons
        from StandardParticles import StdLoosePions as Pions
        from StandardParticles import StdLooseKaons as Kaons

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
        
        # if intermediate_daughters[0].replace('+','').replace('-','') == intermediate_daughters[1].replace('+','').replace('-',''):
        _CombineParticles = CombineParticles(
            "StdLooseIntermediate"+self._name,
            DecayDescriptor='J/psi(1S) -> %s %s'%(intermediate_daughters[0], intermediate_daughters[1]),
            CombinationCut="(ADOCACHI2CUT(99999, ''))",
            MotherCut="(VFASPF(VCHI2) < 99999)"        
            ) 
        # else:
        #     _CombineParticles = CombineParticles(
        #         "StdLooseIntermediate"+self._name,
        #         DecayDescriptor='[J/psi(1S) -> %s %s]cc'%(intermediate_daughters[0], intermediate_daughters[1]),
        #         CombinationCut="(ADOCACHI2CUT(99999, ''))",
        #         MotherCut="(VFASPF(VCHI2) < 99999)"        
        #         ) 

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
            # name, Algorithm=_Combine, RequiredSelections=[_Merge]
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
# DaVinci().EvtMax = -1
# DaVinci().PrintFreq = 250
DaVinci().EvtMax = 100
DaVinci().PrintFreq = 25
DaVinci().Simulation = IS_MC
DaVinci().Lumi = not IS_MC
DaVinci().UserAlgorithms = []
if IS_MC:
    # DaVinci().appendToMainSequence([eventNodeKiller])
    list_to_appendToMainSequence = []
    for idx, sc_i in enumerate(sc):
        DaVinci().appendToMainSequence([eventNodeKiller])
        DaVinci().appendToMainSequence([sc_i.sequence()])
        DaVinci().UserAlgorithms += [tuples[idx]]
        # list_to_appendToMainSequence.append(sc_i.sequence())
        # list_to_appendToMainSequence.append(sc_i.sequence())
    # DaVinci().appendToMainSequence(list_to_appendToMainSequence)
# DaVinci().UserAlgorithms = [tuple]
# DaVinci().UserAlgorithms = tuples
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
